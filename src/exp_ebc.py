"""Influential Node Analysis - Section 6.5 of Local Dampening paper.

This script implements the EBC (Egocentric Betweenness Centrality) experiments
using various differentially private mechanisms for top-k selection.
"""

import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from esnm.mechanism import esnm_lln, esnm_lln_topk, esnm_t, esnm_t_topk

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


def get_rho_from_approx_dp(target_epsilon: float, target_delta: float) -> float:
    """Calculate zCDP rho from approximate DP (epsilon, delta).

    Formula from Bun & Steinke (2016): epsilon = rho + 2*sqrt(rho * ln(1/delta))
    """
    log_inv_delta = np.log(1 / target_delta)
    term1 = np.sqrt(log_inv_delta + target_epsilon)
    term2 = np.sqrt(log_inv_delta)
    return (term1 - term2) ** 2


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

    is_oneshot: bool = False

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
    """Cache for ESNM optimized parameters.

    Pre-computes parameters for each epsilon value to avoid redundant computation.
    """

    def __init__(
        self,
        local_sensitivity: np.ndarray,
        eps_values: np.ndarray,
        precompute_oneshot: bool = False,
    ):
        self.local_sensitivity = local_sensitivity
        self._cache_t: dict[float, tuple] = {}
        self._cache_lln: dict[float, tuple] = {}
        self._cache_t_oneshot: dict[float, tuple] = {}
        self._cache_lln_oneshot: dict[float, tuple] = {}

        # Pre-compute parameters for all epsilon values
        print("  Pre-computing ESNM-T parameters...")
        for eps in eps_values:
            self._compute_t_params(eps)

        print("  Pre-computing ESNM-LLN parameters...")
        for eps in eps_values:
            self._compute_lln_params(eps)

        if precompute_oneshot:
            print("  Pre-computing ESNM-T one-shot parameters...")
            for eps in eps_values:
                self._compute_t_params_oneshot(eps)

            print("  Pre-computing ESNM-LLN one-shot parameters...")
            for eps in eps_values:
                self._compute_lln_params_oneshot(eps)

    def _compute_t_params(self, eps: float) -> None:
        """Compute and cache ESNM-T parameters for given epsilon."""
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
        """Compute and cache ESNM-LLN parameters for given epsilon."""
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
        """Get cached ESNM-T parameters."""
        if eps not in self._cache_t:
            self._compute_t_params(eps)
        return self._cache_t[eps]

    def get_lln_params(self, eps: float) -> tuple:
        """Get cached ESNM-LLN parameters."""
        if eps not in self._cache_lln:
            self._compute_lln_params(eps)
        return self._cache_lln[eps]

    def _compute_t_params_oneshot(self, eps: float) -> None:
        """Compute and cache ESNM-T one-shot parameters (full epsilon)."""
        if eps in self._cache_t_oneshot:
            return

        degree_freedom = 3
        t_candidates = np.linspace(0, eps / (degree_freedom + 1), 150)
        t, s, _, ss = optimize_params_tdist(
            eps, degree_freedom, t_candidates, self.local_sensitivity
        )
        self._cache_t_oneshot[eps] = (
            np.ascontiguousarray(t),
            np.ascontiguousarray(s),
            np.ascontiguousarray(ss),
        )

    def _compute_lln_params_oneshot(self, eps: float) -> None:
        """Compute and cache ESNM-LLN one-shot parameters (full epsilon)."""
        if eps in self._cache_lln_oneshot:
            return

        t_candidates = np.logspace(-9, 10, 150)
        t, sigmas, s, _, ss = optimize_params_lln(
            eps, t_candidates, self.local_sensitivity
        )
        self._cache_lln_oneshot[eps] = (
            np.ascontiguousarray(t),
            np.ascontiguousarray(sigmas),
            np.ascontiguousarray(s),
            np.ascontiguousarray(ss),
        )

    def get_t_params_oneshot(self, eps: float) -> tuple:
        """Get cached ESNM-T one-shot parameters."""
        if eps not in self._cache_t_oneshot:
            self._compute_t_params_oneshot(eps)
        return self._cache_t_oneshot[eps]

    def get_lln_params_oneshot(self, eps: float) -> tuple:
        """Get cached ESNM-LLN one-shot parameters."""
        if eps not in self._cache_lln_oneshot:
            self._compute_lln_params_oneshot(eps)
        return self._cache_lln_oneshot[eps]


class Selection_ESNM_T(SelectionMethod):
    """ESNM-T selection mechanism with cached parameters."""

    def __init__(self, cache: ESNMParamsCache, eps: float, k: int):
        t, s, ss = cache.get_t_params(eps)
        self.eps = eps / k
        self.t = t
        self.s = s
        self.ss = ss

    def __call__(self, u: np.ndarray, selected_mask: np.ndarray) -> int:
        # Mask out already selected elements
        u_masked = u.copy()
        u_masked[selected_mask] = -np.inf

        selected_idx = esnm_t(
            np.ascontiguousarray(u_masked),
            self.ss.copy(),
            self.s.copy(),
            3.0,
        )
        return selected_idx


class Selection_ESNM_LLN(SelectionMethod):
    """ESNM-LLN selection mechanism with cached parameters."""

    def __init__(self, cache: ESNMParamsCache, eps: float, delta: float, k: int):
        t, sigmas, s, ss = cache.get_lln_params(eps)

        # Convert to zCDP
        log_d = np.log(1 / delta)
        self.eps = np.max(
            [
                eps / k,
                (np.sqrt(2) / np.sqrt(k)) * (np.sqrt(eps + log_d) - np.sqrt(log_d)),
            ]
        )
        self.t = t
        self.s = s
        self.ss = ss
        self.sigma = sigmas

    def __call__(self, u: np.ndarray, selected_mask: np.ndarray) -> int:
        # Mask out already selected elements
        u_masked = u.copy()
        u_masked[selected_mask] = -np.inf

        selected_idx = esnm_lln(
            np.ascontiguousarray(u_masked),
            self.ss.copy(),
            self.s.copy(),
            self.sigma.copy(),
        )
        return selected_idx


class Selection_ESNM_T_Oneshot(SelectionMethod):
    """ESNM-T one-shot selection mechanism for top-k."""

    is_oneshot = True

    def __init__(self, cache: ESNMParamsCache, eps: float):
        t, s, ss = cache.get_t_params_oneshot(eps)
        self.eps = eps  # Full epsilon, NO division by k
        self.t = t
        self.s = s
        self.ss = ss

    def select_topk(self, u: np.ndarray, k: int) -> np.ndarray:
        """Select top-k elements in one shot."""
        selected_indices = esnm_t_topk(
            np.ascontiguousarray(u),
            self.ss.copy(),
            self.s.copy(),
            3.0,
            k,
        )
        return np.array(selected_indices, dtype=np.int64)


class Selection_ESNM_LLN_Oneshot(SelectionMethod):
    """ESNM-LLN one-shot selection mechanism for top-k."""

    is_oneshot = True

    def __init__(self, cache: ESNMParamsCache, eps: float, delta: float):
        t, sigmas, s, ss = cache.get_lln_params_oneshot(eps)
        # Convert to zCDP rho (NO division by k for one-shot)
        self.eps = get_rho_from_approx_dp(eps, delta)
        self.t = t
        self.s = s
        self.ss = ss
        self.sigma = sigmas

    def select_topk(self, u: np.ndarray, k: int) -> np.ndarray:
        """Select top-k elements in one shot."""
        selected_indices = esnm_lln_topk(
            np.ascontiguousarray(u),
            self.ss.copy(),
            self.s.copy(),
            self.sigma.copy(),
            k,
        )
        return np.array(selected_indices, dtype=np.int64)


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
    # One-shot methods select all k elements at once
    if method.is_oneshot:
        return method.select_topk(ebc_scores, k)

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
    eps_values: np.ndarray,
    n_runs: int,
) -> None:
    """Run EBC experiments for a single dataset.

    Args:
        dataset_name: Name of the dataset (without extension).
        data_dir: Directory containing graph files.
        results_dir: Directory to save results.
        k_values: List of k values to test.
        eps_values: Array of epsilon values to test.
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

    # Pre-compute ESNM parameters for all epsilon values (major speedup)
    print("Pre-computing ESNM parameters...")
    esnm_cache = ESNMParamsCache(ls_matrix, eps_values, precompute_oneshot=True)

    # Get true top-k for each k
    true_topk_dict = {}
    for k in k_values:
        true_topk_dict[k] = np.argsort(ebc_scores)[::-1][:k]

    # Methods to test
    methods_config = {
        "rnm": lambda eps, k: RNM(eps=eps / k, gs=gs),
        # "ld": lambda eps, k: LocalDampening(eps=eps / k, gs=gs, ls=ls_matrix),
        "shifted_ld": lambda eps, k: ShiftedLocalDampening(
            eps=eps / k, gs=gs, ls=ls_matrix
        ),
        "esnm_t": lambda eps, k: Selection_ESNM_T(esnm_cache, eps, k),
        "esnm_lln": lambda eps, k: Selection_ESNM_LLN(esnm_cache, eps, delta=1e-6, k=k),
        "esnm_t_oneshot": lambda eps, k: Selection_ESNM_T_Oneshot(esnm_cache, eps),
        "esnm_lln_oneshot": lambda eps, k: Selection_ESNM_LLN_Oneshot(
            esnm_cache, eps, delta=1e-6
        ),
    }

    for method_name, method_factory in methods_config.items():
        print(f"\nRunning {method_name}...")
        result_file = results_dir / f"{dataset_name}_{method_name}.txt"

        with open(result_file, "w") as f:
            print("method\tk\teps\tmean_acc\tstd_acc\tmean_time", file=f)

            for k in k_values:
                true_topk = true_topk_dict[k]

                for eps in eps_values:
                    accuracies = []
                    times = []

                    for run in range(n_runs):
                        try:
                            method = method_factory(eps, k)

                            start_time = time.perf_counter()
                            predicted_topk = priv_topk(ebc_scores, k, method)
                            elapsed = time.perf_counter() - start_time

                            acc = compute_accuracy(true_topk, predicted_topk)
                            accuracies.append(acc)
                            times.append(elapsed)
                        except Exception as e:
                            print(f"  Error at eps={eps:.2e}, k={k}: {e}")
                            break

                    if accuracies:
                        mean_acc = np.mean(accuracies)
                        std_acc = np.std(accuracies)
                        mean_time = np.mean(times)

                        print(
                            f"{method_name}\t{k}\t{eps:.2e}\t{mean_acc:.4f}\t{std_acc:.4f}\t{mean_time:.4f}",
                            file=f,
                        )
                        print(
                            f"  k={k}, eps={eps:.2e}: acc={mean_acc:.4f} ± {std_acc:.4f}"
                        )

        print(f"  Results saved to {result_file}")


def main():
    """Run EBC experiments on multiple datasets."""
    set_seeds(42)

    # Datasets to process
    datasets = [
        "enron",
        # "dblp",
        # "github",
    ]

    # Experiment parameters
    k_values = [10, 50, 100]
    eps_values = np.logspace(-3, 4, num=20)
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
            eps_values=eps_values,
            n_runs=n_runs,
        )

    print(f"\n{'=' * 60}")
    print("All experiments completed!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
