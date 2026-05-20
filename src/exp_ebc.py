"""Influential Node Analysis - Section 6.5 of Local Dampening paper.

This script implements the EBC (Egocentric Betweenness Centrality) experiments
comparing differentially private top-k selection mechanisms at a common
**rho-zCDP** budget. Every mechanism here is a *peeling* top-k: it selects one
node at a time, and after each pick the chosen node is dropped so the next round
selects from the reduced set of remaining candidates.

The sweep variable `rho` is the total rho-zCDP budget for one top-k call, split
evenly as rho/k per peeling round. Per-method conversions:

  * `report_noisy_max`, `shifted_ld` (pure eps-DP, peeling):
    per-iteration eps = sqrt(2 * (rho / k)) via
    `src/dp_conv.py::rho_zcdp_to_eps_for_pure_dp`. By Lemma 9 of Bun & Steinke,
    each call is (rho/k)-zCDP; zCDP composes additively over k peeling rounds.
  * eSNM-T (peeling): Student's-T smooth-sensitivity noise is *pure eps-DP*
    (Bun & Steinke 2019, Thm 31 -- polynomial tails, like Cauchy), so the
    optimizer receives eps = sqrt(2 * (rho / k)) and the round is (rho/k)-zCDP
    by the pure-DP => (1/2 eps^2)-zCDP bound (Bun & Steinke 2016).
  * eSNM-LLN (peeling): Laplace-log-normal noise is NOT pure-DP -- it is
    directly (1/2 eps^2)-CDP (Bun & Steinke 2019, Prop. 3). Since (1/2 eps^2)-CDP
    equals rho-zCDP with rho = eps^2/2, the optimizer receives the *same*
    eps = sqrt(2 * (rho / k)), but for the CDP-native reason.
    Each round runs the mechanism on the reduced candidate set. Because the
    optimizer is strictly per-candidate (a node's parameters depend only on its
    own local-sensitivity row and the budget), parameters are computed once per
    per-round eps over all nodes and sliced to the surviving candidates -- this
    is identical to re-optimizing on the reduced set, but avoids k optimizer calls.

All five mechanisms therefore spend exactly rho-zCDP per top-k call (k rounds of
(rho/k)-zCDP, composed by Lemma 9 of Bun & Steinke), so accuracy at a fixed rho
is a fair head-to-head comparison.
"""

import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from esnm.mechanism import esnm_lln, esnm_t

from dp_conv import rho_zcdp_to_cdp_eps, rho_zcdp_to_eps_for_pure_dp
from ebc import (
    adj_list_to_numba,
    build_ls_matrix,
    compute_ebc_all,
    get_degrees,
    get_max_degree,
    global_sensitivity_ebc,
    load_edge_list,
)
from local_dampening import ld, shifted_ld
from optimize_params import optimize_params_lln, optimize_params_tdist
from standard_selection import report_noisy_max


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def load_or_compute_ebc(
    graph_path: Path,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, dict[int, set], int]:
    """Load EBC scores from cache or compute them.

    Args:
        graph_path: Path to the edge list file.
        cache_dir: Directory to store cache files. If None, uses graph_path.parent.

    Returns:
        Tuple of (ebc_scores, adj_list, n_nodes).
    """
    if cache_dir is None:
        cache_dir = graph_path.parent

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{graph_path.stem}_ebc.npy"

    # Load graph
    adj_list, n_nodes, n_edges = load_edge_list(graph_path)
    print(f"  Nodes: {n_nodes}, Edges: {n_edges}")

    # Try to load from cache
    if cache_file.exists():
        print(f"  Loading EBC scores from cache: {cache_file}")
        ebc_scores = np.load(cache_file)
        if len(ebc_scores) == n_nodes:
            return ebc_scores, adj_list, n_nodes
        print("  Cache size mismatch, recomputing...")

    # Compute EBC
    print("  Computing EBC scores...")
    neighbors_list, n_nodes = adj_list_to_numba(adj_list)
    ebc_scores = compute_ebc_all(neighbors_list, n_nodes)

    # Save to cache
    np.save(cache_file, ebc_scores)
    print(f"  EBC scores cached to: {cache_file}")

    return ebc_scores, adj_list, n_nodes


class SelectionMethod:
    """Base class for selection methods."""

    def __call__(self, u: np.ndarray, selected_mask: np.ndarray) -> int:
        """Select a single element.

        Args:
            u: Utility array (EBC scores).
            selected_mask: Boolean mask of already selected elements.

        Returns:
            Index of the selected element.
        """
        raise NotImplementedError


@dataclass
class RNM(SelectionMethod):
    """Report Noisy Max selection mechanism."""

    eps: float
    gs: float

    def __call__(self, u: np.ndarray, selected_mask: np.ndarray) -> int:
        # Mask out already selected elements
        u_masked = u.copy()
        u_masked[selected_mask] = -np.inf
        return report_noisy_max(u_masked, self.gs, self.eps)


@dataclass
class LocalDampening(SelectionMethod):
    """Local Dampening selection mechanism."""

    eps: float
    gs: float
    ls: np.ndarray

    def __call__(self, u: np.ndarray, selected_mask: np.ndarray) -> int:
        # Mask out already selected elements
        u_masked = u.copy()
        u_masked[selected_mask] = -np.inf

        # Create a reduced problem for unselected elements
        valid_indices = np.where(~selected_mask)[0]
        u_valid = u[valid_indices]
        ls_valid = self.ls[valid_indices]

        selected = ld(u_valid, self.gs, self.eps, ls_valid)
        return valid_indices[selected]


@dataclass
class ShiftedLocalDampening(SelectionMethod):
    """Shifted Local Dampening selection mechanism."""

    eps: float
    gs: float
    ls: np.ndarray

    def __call__(self, u: np.ndarray, selected_mask: np.ndarray) -> int:
        # Create a reduced problem for unselected elements
        valid_indices = np.where(~selected_mask)[0]
        u_valid = u[valid_indices]
        ls_valid = self.ls[valid_indices]

        selected = shifted_ld(u_valid, self.gs, self.eps, ls_valid)
        return valid_indices[selected]


class ESNMParamsCache:
    """Cache for ESNM optimized parameters keyed on the per-round epsilon.

    A (rho/k)-zCDP round uses eps = sqrt(2 * rho/k) for both noise families:
    Student's-T is pure-eps-DP (=> (1/2 eps^2)-zCDP) and LLN is directly
    (1/2 eps^2)-CDP, so the two map to the *same* eps. Parameters are
    per-candidate, so caching one full-node optimization per eps lets each
    round slice the surviving candidates.
    """

    def __init__(
        self,
        local_sensitivity: np.ndarray,
        rho_values: np.ndarray,
        k_values: list[int],
    ):
        self.local_sensitivity = local_sensitivity
        self._cache_t: dict[float, tuple] = {}
        self._cache_lln: dict[float, tuple] = {}

        # Each (rho/k)-zCDP peeling round uses pure-DP eps = sqrt(2 * rho/k).
        per_call_eps: set[float] = {
            rho_zcdp_to_eps_for_pure_dp(float(rho) / int(k))
            for rho in rho_values
            for k in k_values
        }

        print("  Pre-computing ESNM-T parameters...")
        for eps in sorted(per_call_eps):
            self._compute_t_params(eps)

        print("  Pre-computing ESNM-LLN parameters...")
        for eps in sorted(per_call_eps):
            self._compute_lln_params(eps)

    def _compute_t_params(self, eps: float) -> None:
        """Compute and cache ESNM-T parameters for given per-round pure-DP eps."""
        if eps in self._cache_t:
            return

        degree_freedom = 3
        t_candidates = np.linspace(0, eps / (degree_freedom + 1), 150)
        t, s, _, ss = optimize_params_tdist(
            eps, degree_freedom, t_candidates, self.local_sensitivity
        )
        self._cache_t[eps] = (
            np.ascontiguousarray(t),
            np.ascontiguousarray(s),
            np.ascontiguousarray(ss),
        )

    def _compute_lln_params(self, eps: float) -> None:
        """Compute and cache ESNM-LLN parameters for given per-round pure-DP eps."""
        if eps in self._cache_lln:
            return

        t_candidates = np.logspace(-9, 10, 150)
        t, sigmas, s, _, ss = optimize_params_lln(
            eps, t_candidates, self.local_sensitivity
        )
        self._cache_lln[eps] = (
            np.ascontiguousarray(t),
            np.ascontiguousarray(sigmas),
            np.ascontiguousarray(s),
            np.ascontiguousarray(ss),
        )

    def get_t_params(self, eps: float) -> tuple:
        """Get cached ESNM-T parameters for per-round pure-DP eps."""
        if eps not in self._cache_t:
            self._compute_t_params(eps)
        return self._cache_t[eps]

    def get_lln_params(self, eps: float) -> tuple:
        """Get cached ESNM-LLN parameters for per-round pure-DP eps."""
        if eps not in self._cache_lln:
            self._compute_lln_params(eps)
        return self._cache_lln[eps]


class Selection_ESNM_T(SelectionMethod):
    """ESNM-T peeling selection (pure eps-DP per round).

    A (rho/k)-zCDP round uses pure-DP eps = sqrt(2 * rho/k); k rounds compose to
    rho-zCDP. Each round selects over the reduced set of unselected candidates;
    the per-candidate parameters are precomputed once over all nodes (for this
    per-round eps) and sliced to the surviving candidates, then mapped back.
    """

    def __init__(self, cache: ESNMParamsCache, rho: float, k: int):
        # Student's-T is pure-eps-DP; eps = sqrt(2 rho/k) => (rho/k)-zCDP.
        _, s, ss = cache.get_t_params(rho_zcdp_to_eps_for_pure_dp(rho / k))
        self.s = np.ascontiguousarray(s)
        self.ss = np.ascontiguousarray(ss)

    def __call__(self, u: np.ndarray, selected_mask: np.ndarray) -> int:
        valid_indices = np.where(~selected_mask)[0]
        u_valid = np.ascontiguousarray(u[valid_indices])
        s_valid = np.ascontiguousarray(self.s[valid_indices])
        ss_valid = np.ascontiguousarray(self.ss[valid_indices])

        selected = esnm_t(u_valid, ss_valid, s_valid, 3.0)
        return valid_indices[selected]


class Selection_ESNM_LLN(SelectionMethod):
    """ESNM-LLN peeling selection ((1/2 eps^2)-CDP per round).

    LLN noise is not pure-DP; it is directly (1/2 eps^2)-CDP (Bun & Steinke 2019,
    Prop. 3), which equals (rho/k)-zCDP at eps = sqrt(2 * rho/k); k rounds compose
    to rho-zCDP. Each round selects over the reduced set of unselected candidates;
    the per-candidate parameters are precomputed once over all nodes (for this
    per-round eps) and sliced to the surviving candidates, then mapped back.
    """

    def __init__(self, cache: ESNMParamsCache, rho: float, k: int):
        # LLN is (1/2 eps^2)-CDP (not pure-DP); eps = sqrt(2 rho/k) => (rho/k)-zCDP.
        _, sigmas, s, ss = cache.get_lln_params(rho_zcdp_to_cdp_eps(rho / k))
        self.s = np.ascontiguousarray(s)
        self.ss = np.ascontiguousarray(ss)
        self.sigma = np.ascontiguousarray(sigmas)

    def __call__(self, u: np.ndarray, selected_mask: np.ndarray) -> int:
        valid_indices = np.where(~selected_mask)[0]
        u_valid = np.ascontiguousarray(u[valid_indices])
        s_valid = np.ascontiguousarray(self.s[valid_indices])
        ss_valid = np.ascontiguousarray(self.ss[valid_indices])
        sigma_valid = np.ascontiguousarray(self.sigma[valid_indices])

        selected = esnm_lln(u_valid, ss_valid, s_valid, sigma_valid)
        return valid_indices[selected]


def priv_topk(
    ebc_scores: np.ndarray,
    k: int,
    method: SelectionMethod,
) -> np.ndarray:
    """Algorithm 3: PrivTopk - iteratively select k nodes.

    Args:
        ebc_scores: EBC scores for all nodes.
        k: Number of top nodes to select.
        method: Selection method instance.

    Returns:
        Array of indices of selected top-k nodes.
    """
    # Peeling: iteratively select k nodes
    n = len(ebc_scores)
    selected = np.zeros(k, dtype=np.int64)
    selected_mask = np.zeros(n, dtype=bool)

    for i in range(k):
        idx = method(ebc_scores, selected_mask)
        selected[i] = idx
        selected_mask[idx] = True

    return selected


def compute_accuracy(true_topk: np.ndarray, predicted_topk: np.ndarray) -> float:
    """Compute accuracy as overlap between true and predicted top-k.

    Args:
        true_topk: Indices of true top-k nodes.
        predicted_topk: Indices of predicted top-k nodes.

    Returns:
        Accuracy = |true_topk ∩ predicted_topk| / k
    """
    true_set = set(true_topk)
    predicted_set = set(predicted_topk)
    overlap = len(true_set & predicted_set)
    return overlap / len(true_topk)


def run_experiment_for_dataset(
    dataset_name: str,
    data_dir: Path,
    results_dir: Path,
    k_values: list[int],
    rho_values: np.ndarray,
    n_runs: int,
) -> None:
    """Run EBC experiments for a single dataset.

    Args:
        dataset_name: Name of the dataset (without extension).
        data_dir: Directory containing graph files.
        results_dir: Directory to save results.
        k_values: List of k values to test.
        rho_values: Array of rho-zCDP budgets to test.
        n_runs: Number of runs per configuration.
    """
    print(f"\n{'=' * 60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'=' * 60}")

    # Load dataset and EBC scores (with caching)
    data_path = data_dir / f"{dataset_name}.txt"
    if not data_path.exists():
        print(f"  WARNING: Dataset file not found: {data_path}")
        return

    ebc_scores, adj_list, n_nodes = load_or_compute_ebc(data_path)
    print(f"  EBC computed for {n_nodes} nodes")
    print(f"  Max EBC: {ebc_scores.max():.2f}, Mean EBC: {ebc_scores.mean():.2f}")

    # Build sensitivity structures
    print("Building sensitivity structures...")
    degrees = get_degrees(adj_list)
    max_degree = get_max_degree(adj_list)
    gs = global_sensitivity_ebc(max_degree)
    print(f"  Max degree: {max_degree}, Global sensitivity: {gs:.2f}")

    max_distance = 100
    ls_matrix = build_ls_matrix(degrees, max_distance)

    # Pre-compute ESNM parameters for every per-round pure-DP eps the experiment
    # needs (each (rho/k)-zCDP peeling round uses eps = sqrt(2 * rho/k)).
    print("Pre-computing ESNM parameters...")
    esnm_cache = ESNMParamsCache(ls_matrix, rho_values, k_values)

    # Get true top-k for each k
    true_topk_dict = {}
    for k in k_values:
        true_topk_dict[k] = np.argsort(ebc_scores)[::-1][:k]

    # Methods to test. `rho` is the total rho-zCDP budget for one top-k call.
    # Every method spends pure-DP eps = sqrt(2 * rho/k) per peeling round, so each
    # round is (rho/k)-zCDP and the k rounds compose to exactly rho-zCDP -- an
    # equal-budget head-to-head comparison.
    methods_config = {
        # "rnm": lambda rho, k: RNM(eps=rho_zcdp_to_eps_for_pure_dp(rho / k), gs=gs),
        # "ld": lambda rho, k: LocalDampening(
        #     eps=rho_zcdp_to_eps_for_pure_dp(rho / k), gs=gs, ls=ls_matrix
        # ),
        # "shifted_ld": lambda rho, k: ShiftedLocalDampening(
        #     eps=rho_zcdp_to_eps_for_pure_dp(rho / k), gs=gs, ls=ls_matrix
        # ),
        "esnm_t": lambda rho, k: Selection_ESNM_T(esnm_cache, rho, k),
        "esnm_lln": lambda rho, k: Selection_ESNM_LLN(esnm_cache, rho, k),
    }

    for method_name, method_factory in methods_config.items():
        print(f"\nRunning {method_name}...")
        result_file = results_dir / f"{dataset_name}_{method_name}.txt"

        with open(result_file, "w") as f:
            print("method\tk\trho\tmean_acc\tstd_acc\tmean_time", file=f)

            for k in k_values:
                true_topk = true_topk_dict[k]

                for rho in rho_values:
                    accuracies = []
                    times = []

                    for run in range(n_runs):
                        try:
                            method = method_factory(rho, k)

                            start_time = time.perf_counter()
                            predicted_topk = priv_topk(ebc_scores, k, method)
                            elapsed = time.perf_counter() - start_time

                            acc = compute_accuracy(true_topk, predicted_topk)
                            accuracies.append(acc)
                            times.append(elapsed)
                        except Exception as e:
                            print(f"  Error at rho={rho:.2e}, k={k}: {e}")
                            break

                    if accuracies:
                        mean_acc = np.mean(accuracies)
                        std_acc = np.std(accuracies)
                        mean_time = np.mean(times)

                        print(
                            f"{method_name}\t{k}\t{rho:.2e}\t{mean_acc:.4f}\t{std_acc:.4f}\t{mean_time:.4f}",
                            file=f,
                        )
                        print(
                            f"  k={k}, rho={rho:.2e}: acc={mean_acc:.4f} ± {std_acc:.4f}"
                        )

        print(f"  Results saved to {result_file}")


def main():
    """Run EBC experiments on multiple datasets."""
    set_seeds(42)

    # Datasets to process
    datasets = [
        "enron",
        "dblp",
        "github",
    ]

    # Experiment parameters
    k_values = [10, 50, 100]
    rho_values = np.logspace(-3, 4, num=20)
    n_runs = 10

    # Directories
    data_dir = Path.cwd() / "data/graph"
    results_dir = Path.cwd() / "results/ebc"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running EBC experiments on {len(datasets)} datasets: {datasets}")

    # Iterate over all datasets
    for dataset_name in datasets:
        run_experiment_for_dataset(
            dataset_name=dataset_name,
            data_dir=data_dir,
            results_dir=results_dir,
            k_values=k_values,
            rho_values=rho_values,
            n_runs=n_runs,
        )

    print(f"\n{'=' * 60}")
    print("All experiments completed!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
