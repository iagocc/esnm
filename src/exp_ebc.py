"""Influential Node Analysis - Section 6.5 of Local Dampening paper.

This script implements the EBC (Egocentric Betweenness Centrality) experiments
comparing differentially private top-k selection mechanisms at a common
**pure eps-DP** budget. Every mechanism here is a *peeling* top-k: it selects one
node at a time, and after each pick the chosen node is dropped so the next round
selects from the reduced set of remaining candidates.

The sweep variable `eps` is the total pure-(eps,0)-DP budget for one top-k call.
Every mechanism is pure-DP, so by **basic composition** the budget splits evenly
as eps/k per peeling round and the k rounds compose to exactly eps-DP:

  * `report_noisy_max`, `ld`, `shifted_ld`: each round runs at eps/k. These are
    pure-eps-DP selection mechanisms.
  * eSNM-T / eSNM-GCP / eSNM-LCP (peeling): Student's-T, Gaussian-core
    Pareto-tail, and Laplace-core Pareto-tail smooth-sensitivity noises are all
    *pure eps-DP* (polynomial tails => finite shift and dilation max-divergence),
    so the optimizer receives eps/k directly. Each round runs the mechanism on
    the reduced candidate set. Because the optimizer is strictly per-candidate (a
    node's parameters depend only on its own local-sensitivity row and the
    budget), parameters are computed once per per-round eps over all nodes and
    sliced to the surviving candidates -- identical to re-optimizing on the
    reduced set, but avoids k optimizer calls.

All mechanisms therefore spend exactly eps-DP per top-k call (k rounds of
(eps/k)-DP composed by basic composition), so accuracy at a fixed eps is a fair
head-to-head comparison.

Utility (selectable via `build_utility`):

  * "value": the raw EBC score. Its element local sensitivity is degree-based
    (Definition 20 / Lemma 12; see docs/ebc.md) with global sensitivity ~ Delta^2/4,
    large and heterogeneous, so the smooth-sensitivity mechanisms gain little.
  * "rank_margin" (default): u(v) = -sqrt(1 + rank(v)), rank by descending EBC
    (0 = highest). Under the rank-Lipschitz model (one neighbouring edge moves a
    node's rank by at most 1, as in the percentile rank utility), the local
    sensitivity at gap g and distance k is sqrt(1+max(0,g-k)) - sqrt(max(0,g-k)),
    the global sensitivity is sqrt(2)-1, and the smooth sensitivity decays with the
    margin to the boundary. This bounded, near-homogeneous element-wise sensitivity
    is what lets the eSNM variants beat the competitors. The utility is monotone in
    EBC, so argmax still recovers the true EBC top-k.
"""

import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from esnm.mechanism import esnm_gcp, esnm_lcp, esnm_t

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
from optimize_params import (
    optimize_params_gcp,
    optimize_params_lcp,
    optimize_params_tdist,
)
from standard_selection import report_noisy_max
from topk.esnm_joint import _build_local_sensitivity

# Tail exponent for the Gaussian-core Pareto-tail (GCP) noise (gamma > 2).
_GCP_GAMMA = 4.5
# Tail exponent for the Laplace-core Pareto-tail (LCP) noise (gamma > 2).
_LCP_GAMMA = 4.5


def _round_up_pow10(value: int) -> int:
    """Smallest power of 10 that is >= value (>= 1).

    Used as the degree bound D for the "value" utility's GLOBAL sensitivity: the
    observed max degree is rounded up to the next power of 10 (e.g. 500 -> 1000,
    1383 -> 10000, 1000 -> 1000). Because D >= every degree, the bound is a valid
    upper bound on all node sensitivities and clipping to it is a no-op (no
    distortion), while the rounding coarsens it to a stable round number that
    changes only when the max degree crosses a power-of-10 boundary -- so it does
    not track the exact max degree. Integer loop to avoid log10 float edge cases.
    """
    bound = 1
    while bound < value:
        bound *= 10
    return bound


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


def build_utility(
    utility: str,
    ebc_scores: np.ndarray,
    degrees: np.ndarray,
    max_degree_bound: int,
    max_distance: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (u, ls_matrix, gs) for the chosen utility.

    Args:
        utility: "value" (raw EBC score) or "rank_margin" (sqrt rank margin).
        ebc_scores: EBC score for every node.
        degrees: Node degrees (used by the "value" sensitivity).
        max_degree_bound: degree bound D used by the "value" global sensitivity
            (bounded-degree edge-DP), the observed max degree rounded up to a
            power of 10 (see `_round_up_pow10`).
        max_distance: Distance horizon for the local-sensitivity matrix.

    Returns:
        u: utility array passed to the selection mechanisms.
        ls_matrix: element local-sensitivity matrix, shape (n_nodes, max_distance+1).
        gs: global sensitivity of the utility.

    "value" uses the degree-based EBC sensitivity (Definition 20 / Lemma 12, see
    docs/ebc.md). EBC global sensitivity is unbounded over arbitrary graphs, so it
    is taken w.r.t. a maximum-degree bound D (bounded-degree edge-DP):
    gs = max(D*(D-1)/4, D). D is the max degree rounded up to a power of 10, a
    coarse round bound that does not track the exact max degree. Since D >= every
    degree, gs upper-bounds all node sensitivities and the clip-to-D is a no-op.

    "rank_margin" uses u(v) = -sqrt(1 + rank(v)) with rank by descending EBC
    (0 = highest). Under the rank-Lipschitz model (one neighbouring edge moves a
    node's rank by at most 1) its local sensitivity is the bounded, margin-decaying
    curve LS(g') = sqrt(1+g') - sqrt(g') <= sqrt(2)-1 built by
    `topk.esnm_joint._build_local_sensitivity`. Its global sensitivity sqrt(2)-1 is
    already data-independent. Monotone in EBC, so the argmax still recovers the
    true EBC top-k.
    """
    if utility == "value":
        # D >= every degree (it is the max degree rounded up to a power of 10),
        # so this clip is a no-op safety net and gs upper-bounds all node
        # sensitivities under the same bounded-degree assumption.
        degrees_bounded = np.minimum(degrees, max_degree_bound)
        return (
            ebc_scores,
            build_ls_matrix(degrees_bounded, max_distance),
            global_sensitivity_ebc(max_degree_bound),
        )
    if utility == "rank_margin":
        n = len(ebc_scores)
        # argsort is deterministic, so this order matches the true_topk ordering
        # computed from the same ebc_scores in run_experiment_for_dataset.
        order = np.argsort(ebc_scores)[::-1]
        rank = np.empty(n, dtype=np.int64)
        rank[order] = np.arange(n)
        gap = rank.astype(np.float64)
        u = -np.sqrt(1.0 + gap)
        ls_matrix = np.ascontiguousarray(
            _build_local_sensitivity(gap, max_distance).astype(np.float64)
        )
        return u, ls_matrix, float(np.sqrt(2.0) - 1.0)
    raise ValueError(f"unknown utility {utility!r}")


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

    Every peeling round runs at pure-DP eps/k (basic composition). All three
    noise families (Student's-T, GCP, LCP) are pure-eps-DP, so the optimizer
    receives that per-round eps directly. Parameters are per-candidate, so
    caching one full-node optimization per eps lets each round slice the
    surviving candidates.
    """

    def __init__(
        self,
        local_sensitivity: np.ndarray,
        eps_values: np.ndarray,
        k_values: list[int],
    ):
        self.local_sensitivity = local_sensitivity
        self._cache_t: dict[float, tuple] = {}
        self._cache_gcp: dict[float, tuple] = {}
        self._cache_lcp: dict[float, tuple] = {}

        # Each peeling round uses pure-DP eps/k (basic composition over k rounds).
        per_call_eps: set[float] = {
            float(eps) / int(k) for eps in eps_values for k in k_values
        }

        print("  Pre-computing ESNM-T parameters...")
        for eps in sorted(per_call_eps):
            self._compute_t_params(eps)

        print("  Pre-computing ESNM-GCP parameters...")
        for eps in sorted(per_call_eps):
            self._compute_gcp_params(eps)

        print("  Pre-computing ESNM-LCP parameters...")
        for eps in sorted(per_call_eps):
            self._compute_lcp_params(eps)

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

    def get_t_params(self, eps: float) -> tuple:
        """Get cached ESNM-T parameters for per-round pure-DP eps."""
        if eps not in self._cache_t:
            self._compute_t_params(eps)
        return self._cache_t[eps]

    def _compute_gcp_params(self, eps: float) -> None:
        """Compute and cache ESNM-GCP parameters for given per-round pure-DP eps."""
        if eps in self._cache_gcp:
            return

        # GCP feasibility: t < eps / gamma (sigma fixed to 1, WLOG).
        t_candidates = np.linspace(0, eps / _GCP_GAMMA, 150)
        t, s, _, ss = optimize_params_gcp(
            eps, _GCP_GAMMA, t_candidates, self.local_sensitivity
        )
        self._cache_gcp[eps] = (
            np.ascontiguousarray(t),
            np.ascontiguousarray(s),
            np.ascontiguousarray(ss),
        )

    def get_gcp_params(self, eps: float) -> tuple:
        """Get cached ESNM-GCP parameters for per-round pure-DP eps."""
        if eps not in self._cache_gcp:
            self._compute_gcp_params(eps)
        return self._cache_gcp[eps]

    def _compute_lcp_params(self, eps: float) -> None:
        """Compute and cache ESNM-LCP parameters for given per-round pure-DP eps."""
        if eps in self._cache_lcp:
            return

        # LCP feasibility: t < eps / gamma (sigma fixed to 1, WLOG).
        t_candidates = np.linspace(0, eps / _LCP_GAMMA, 150)
        t, s, _, ss = optimize_params_lcp(
            eps, _LCP_GAMMA, t_candidates, self.local_sensitivity
        )
        self._cache_lcp[eps] = (
            np.ascontiguousarray(t),
            np.ascontiguousarray(s),
            np.ascontiguousarray(ss),
        )

    def get_lcp_params(self, eps: float) -> tuple:
        """Get cached ESNM-LCP parameters for per-round pure-DP eps."""
        if eps not in self._cache_lcp:
            self._compute_lcp_params(eps)
        return self._cache_lcp[eps]


class Selection_ESNM_T(SelectionMethod):
    """ESNM-T peeling selection (pure eps-DP per round).

    Each peeling round runs at pure-DP eps/k; k rounds compose to eps-DP by basic
    composition. Each round selects over the reduced set of unselected candidates;
    the per-candidate parameters are precomputed once over all nodes (for this
    per-round eps) and sliced to the surviving candidates, then mapped back.
    """

    def __init__(self, cache: ESNMParamsCache, eps: float, k: int):
        _, s, ss = cache.get_t_params(eps / k)
        self.s = np.ascontiguousarray(s)
        self.ss = np.ascontiguousarray(ss)

    def __call__(self, u: np.ndarray, selected_mask: np.ndarray) -> int:
        valid_indices = np.where(~selected_mask)[0]
        u_valid = np.ascontiguousarray(u[valid_indices])
        s_valid = np.ascontiguousarray(self.s[valid_indices])
        ss_valid = np.ascontiguousarray(self.ss[valid_indices])

        selected = esnm_t(u_valid, ss_valid, s_valid, 3.0)
        return valid_indices[selected]


class Selection_ESNM_GCP(SelectionMethod):
    """ESNM-GCP peeling selection (pure eps-DP per round).

    GCP (Gaussian-core Pareto-tail) noise is pure-eps-DP (polynomial tail =>
    finite shift and dilation max-divergence). Each peeling round runs at pure-DP
    eps/k; k rounds compose to eps-DP by basic composition. Each round selects
    over the reduced set of unselected candidates; the per-candidate parameters
    are precomputed once over all nodes (for this per-round eps) and sliced to
    the surviving candidates.
    """

    def __init__(self, cache: ESNMParamsCache, eps: float, k: int):
        _, s, ss = cache.get_gcp_params(eps / k)
        self.s = np.ascontiguousarray(s)
        self.ss = np.ascontiguousarray(ss)

    def __call__(self, u: np.ndarray, selected_mask: np.ndarray) -> int:
        valid_indices = np.where(~selected_mask)[0]
        u_valid = np.ascontiguousarray(u[valid_indices])
        s_valid = np.ascontiguousarray(self.s[valid_indices])
        ss_valid = np.ascontiguousarray(self.ss[valid_indices])

        selected = esnm_gcp(u_valid, ss_valid, s_valid, _GCP_GAMMA)
        return valid_indices[selected]


class Selection_ESNM_LCP(SelectionMethod):
    """ESNM-LCP peeling selection (pure eps-DP per round).

    LCP (Laplace-core Pareto-tail) noise is pure-eps-DP (polynomial tail =>
    finite shift and dilation max-divergence). Each peeling round runs at pure-DP
    eps/k; k rounds compose to eps-DP by basic composition. Each round selects
    over the reduced set of unselected candidates; the per-candidate parameters
    are precomputed once over all nodes (for this per-round eps) and sliced to
    the surviving candidates.
    """

    def __init__(self, cache: ESNMParamsCache, eps: float, k: int):
        _, s, ss = cache.get_lcp_params(eps / k)
        self.s = np.ascontiguousarray(s)
        self.ss = np.ascontiguousarray(ss)

    def __call__(self, u: np.ndarray, selected_mask: np.ndarray) -> int:
        valid_indices = np.where(~selected_mask)[0]
        u_valid = np.ascontiguousarray(u[valid_indices])
        s_valid = np.ascontiguousarray(self.s[valid_indices])
        ss_valid = np.ascontiguousarray(self.ss[valid_indices])

        selected = esnm_lcp(u_valid, ss_valid, s_valid, _LCP_GAMMA)
        return valid_indices[selected]


def priv_topk(
    utility_scores: np.ndarray,
    k: int,
    method: SelectionMethod,
) -> np.ndarray:
    """Algorithm 3: PrivTopk - iteratively select k nodes.

    Args:
        utility_scores: Per-node utility array (e.g. raw EBC or the rank-margin
            utility from `build_utility`).
        k: Number of top nodes to select.
        method: Selection method instance.

    Returns:
        Array of indices of selected top-k nodes.
    """
    # Peeling: iteratively select k nodes
    n = len(utility_scores)
    selected = np.zeros(k, dtype=np.int64)
    selected_mask = np.zeros(n, dtype=bool)

    for i in range(k):
        idx = method(utility_scores, selected_mask)
        selected[i] = idx
        selected_mask[idx] = True

    return selected


def compute_metrics(
    predicted_topk: np.ndarray,
    true_topk: np.ndarray,
    true_rank: np.ndarray,
    u: np.ndarray,
) -> tuple[float, float, float, float]:
    """Quality metrics for one private top-k selection.

    Plain overlap@k is value- and order-blind: picking the rank-(k+1) node (almost
    as influential as rank-k) scores like picking a junk node, and it ignores
    placement. These metrics fix that.

    Args:
        predicted_topk: selected node indices in peeling order (pick 0 first).
        true_topk: true top-k node indices (descending EBC).
        true_rank: 0-based rank of every node by descending EBC (0 = best).
        u: utility array the mechanism optimises (monotone in EBC).

    Returns (rank_err, ndcg, util_regret, overlap):
      * rank_err: mean true-rank of the selected nodes minus the ideal mean
        (k-1)/2. 0 = selected exactly the true top-k (any order); larger means
        picks came from deeper in the ranking. Scale-free and heavy-tail-robust.
      * ndcg: NDCG@k with graded relevance max(0, k - true_rank), scored on the
        peeling order. 1.0 = true top-k recovered in exact rank order. Rewards
        putting the most influential nodes first; bounded and scale-free.
      * util_regret: mean per-pick utility gap u(true top-k) - u(selected). This
        is exactly the quantity the DP selection bound controls (selected utility
        vs optimal). 0 = optimal.
      * overlap: |true ∩ predicted| / k -- the familiar accuracy anchor.
    """
    k = len(true_topk)
    sel_ranks = true_rank[predicted_topk].astype(np.float64)

    rank_err = float(sel_ranks.mean() - (k - 1) / 2.0)

    # NDCG@k with rank-graded relevance (true #1 -> k, #2 -> k-1, ..., outside -> 0).
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(np.maximum(0.0, k - sel_ranks) * discounts))
    idcg = float(np.sum(np.arange(k, 0, -1, dtype=np.float64) * discounts))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    util_regret = float((u[true_topk].sum() - u[predicted_topk].sum()) / k)

    overlap = len(set(true_topk.tolist()) & set(predicted_topk.tolist())) / k

    return rank_err, ndcg, util_regret, overlap


def run_experiment_for_dataset(
    dataset_name: str,
    data_dir: Path,
    results_dir: Path,
    k_values: list[int],
    eps_values: np.ndarray,
    n_runs: int,
    utility: str = "rank_margin",
    max_degree_bound: int | None = None,
) -> None:
    """Run EBC experiments for a single dataset.

    Args:
        dataset_name: Name of the dataset (without extension).
        data_dir: Directory containing graph files.
        results_dir: Directory to save results.
        k_values: List of k values to test.
        eps_values: Array of total pure-eps-DP budgets to test.
        n_runs: Number of runs per configuration.
        utility: Selection utility, "value" or "rank_margin" (see `build_utility`).
        max_degree_bound: degree bound for the "value" global sensitivity. If
            None, the observed max degree is rounded up to a power of 10
            (`_round_up_pow10`). Unused by "rank_margin".
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

    # Build the utility and its element local-sensitivity structures. The "value"
    # global sensitivity uses a degree bound D = max degree rounded up to a power
    # of 10 (coarse, does not track the exact max degree).
    print(f"Building sensitivity structures (utility={utility})...")
    degrees = get_degrees(adj_list)
    observed_max_degree = get_max_degree(adj_list)
    degree_bound = (
        max_degree_bound
        if max_degree_bound is not None
        else _round_up_pow10(observed_max_degree)
    )
    max_distance = 100
    u, ls_matrix, gs = build_utility(
        utility, ebc_scores, degrees, degree_bound, max_distance
    )
    print(
        f"  Degree bound D: {degree_bound} "
        f"(observed max degree {observed_max_degree} rounded up to a power of 10)"
    )
    print(f"  Global sensitivity: {gs:.4f}, max element LS: {ls_matrix.max():.4f}")

    # Pre-compute ESNM parameters for every per-round pure-DP eps the experiment
    # needs (each peeling round uses eps/k under basic composition).
    print("Pre-computing ESNM parameters...")
    esnm_cache = ESNMParamsCache(ls_matrix, eps_values, k_values)

    # True ranking by descending EBC (0 = highest). `true_rank[node]` is its
    # 0-based rank; the per-k true top-k is the prefix of the same order.
    desc_order = np.argsort(ebc_scores)[::-1]
    true_rank = np.empty(n_nodes, dtype=np.int64)
    true_rank[desc_order] = np.arange(n_nodes)
    true_topk_dict = {k: desc_order[:k] for k in k_values}

    # Methods to test. `eps` is the total pure-eps-DP budget for one top-k call.
    # Every method is pure-DP and spends eps/k per peeling round, so the k rounds
    # compose to exactly eps-DP by basic composition -- an equal-budget
    # head-to-head comparison.
    methods_config = {
        "rnm": lambda eps, k: RNM(eps=eps / k, gs=gs),
        # "ld": lambda eps, k: LocalDampening(eps=eps / k, gs=gs, ls=ls_matrix),
        "shifted_ld": lambda eps, k: ShiftedLocalDampening(
            eps=eps / k, gs=gs, ls=ls_matrix
        ),
        # "esnm_t": lambda eps, k: Selection_ESNM_T(esnm_cache, eps, k),
        # "esnm_gcp": lambda eps, k: Selection_ESNM_GCP(esnm_cache, eps, k),
        # "esnm_lcp": lambda eps, k: Selection_ESNM_LCP(esnm_cache, eps, k),
    }

    for method_name, method_factory in methods_config.items():
        print(f"\nRunning {method_name}...")
        result_file = results_dir / f"{dataset_name}_{utility}_{method_name}.txt"

        with open(result_file, "w") as f:
            print(
                "method\tk\teps"
                "\tmean_rank_err\tstd_rank_err\tmean_ndcg\tstd_ndcg"
                "\tmean_util_regret\tstd_util_regret\tmean_overlap\tstd_overlap"
                "\tmean_time",
                file=f,
            )

            for k in k_values:
                true_topk = true_topk_dict[k]

                for eps in eps_values:
                    metrics = []  # rows of (rank_err, ndcg, util_regret, overlap)
                    times = []

                    for run in range(n_runs):
                        try:
                            method = method_factory(eps, k)

                            start_time = time.perf_counter()
                            predicted_topk = priv_topk(u, k, method)
                            elapsed = time.perf_counter() - start_time

                            metrics.append(
                                compute_metrics(predicted_topk, true_topk, true_rank, u)
                            )
                            times.append(elapsed)
                        except Exception as e:
                            print(f"  Error at eps={eps:.2e}, k={k}: {e}")
                            break

                    if metrics:
                        arr = np.asarray(metrics)  # (n_runs, 4)
                        mean_m = arr.mean(axis=0)
                        std_m = arr.std(axis=0)
                        mean_time = float(np.mean(times))
                        re_m, nd_m, ur_m, ov_m = mean_m
                        re_s, nd_s, ur_s, ov_s = std_m

                        print(
                            f"{method_name}\t{k}\t{eps:.2e}"
                            f"\t{re_m:.4f}\t{re_s:.4f}\t{nd_m:.4f}\t{nd_s:.4f}"
                            f"\t{ur_m:.4f}\t{ur_s:.4f}\t{ov_m:.4f}\t{ov_s:.4f}"
                            f"\t{mean_time:.4f}",
                            file=f,
                        )
                        print(
                            f"  k={k}, eps={eps:.2e}: rank_err={re_m:.1f} "
                            f"ndcg={nd_m:.3f} util_reg={ur_m:.3f} overlap={ov_m:.2f}"
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
    eps_values = np.logspace(0, 1.5, num=20)
    n_runs = 10
    utility = "value"  # "value" for the raw-EBC baseline

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
            utility=utility,
        )

    print(f"\n{'=' * 60}")
    print("All experiments completed!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
