import numpy as np

from esnm.mechanism import (
    esnm_gcp as _esnm_gcp_cpp,
    esnm_lln as _esnm_lln_cpp,
    esnm_t as _esnm_t_cpp,
)
from topk.joint import (
    compute_log_diff_counts,
    get_diffs_to_positions,
    make_diff_matrix,
    sequence_from_diff,
)
from optimize_params import (
    optimize_params_gcp,
    optimize_params_lln,
    optimize_params_tdist,
)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_inputs(
    counts: np.ndarray, epsilon: float, k: int, sensitivity: int
) -> None:
    if counts.ndim != 1:
        raise ValueError("counts must be a 1-D array.")
    if counts.size == 0:
        raise ValueError("counts must be non-empty.")
    if k <= 0 or k > counts.size:
        raise ValueError("k must be in [1, len(counts)].")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0.")
    if sensitivity <= 0:
        raise ValueError("sensitivity must be > 0.")


# ---------------------------------------------------------------------------
# Local sensitivity for sqrt utility u = -sqrt(1 + gap)
# ---------------------------------------------------------------------------

# Max Hamming distance for smooth sensitivity computation.  Beyond this
# distance LS saturates at sqrt(2)-1 and e^{-tk} decay makes contributions
# negligible, so capping K saves memory: table is (max_gap+1, _MAX_DISTANCE+1)
# instead of (max_gap+1)^2.
_MAX_DISTANCE = 100


def _build_local_sensitivity(gap: np.ndarray, max_k: int) -> np.ndarray:
    """Build local sensitivity array for sqrt utility u = -sqrt(1 + gap).

    local_sensitivity[r, k] = LS(max(0, g - k)), where LS(g') is the local
    sensitivity at gap g': the max utility change from one database modification.

    LS(g') = sqrt(1 + g') - sqrt(g')   for g' >= 1
    LS(0)  = sqrt(2) - 1               (= LS(1), only upward change possible)

    Computes a compact (max_gap+1, K+1) table indexed by integer gap,
    then maps each candidate to its row — avoids redundant sqrt for
    candidates that share the same gap value.  K is capped at _MAX_DISTANCE.
    """
    max_k = min(max_k, _MAX_DISTANCE)

    k_vals = np.arange(max_k + 1, dtype=np.float32)
    g_vals = np.arange(int(np.max(gap)) + 1, dtype=np.float32)

    # LS(g') = sqrt(1+g') - sqrt(g'), with g' clamped to >= 1 so LS(0) = LS(1)
    g_at_k = np.maximum(1.0, g_vals[:, None] - k_vals[None, :])  # (G, K)
    table = np.sqrt(1.0 + g_at_k) - np.sqrt(g_at_k)             # (G, K)

    # Map each candidate to its row in the cached table
    gap_int = np.clip(gap.astype(np.intp), 0, len(g_vals) - 1)
    return table[gap_int]


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def _run_esnm_joint(
    counts: np.ndarray,
    epsilon: float,
    k: int,
    noise: str,
    sensitivity: int,
    degree_freedom: float,
) -> np.ndarray:
    """eSNM joint top-k with sqrt utility u(a) = -sqrt(1 + gap).

    Uses the C++ eSNM functions (single noise draw, pairwise calibration).
    No tau estimation needed — smooth sensitivity decays naturally with gap.
    """
    _validate_inputs(counts, epsilon, k, sensitivity)
    counts = np.asarray(counts, dtype=np.float64)

    # 1. Build joint candidates
    sort_indices = np.argsort(counts)[::-1]
    sorted_counts = counts[sort_indices]
    diff_matrix = make_diff_matrix(sorted_counts, k)
    diffs_to_positions = get_diffs_to_positions(diff_matrix)
    log_diff_counts = compute_log_diff_counts(diff_matrix, diffs_to_positions)
    sorted_diffs = diff_matrix[diffs_to_positions]
    valid_idx = np.flatnonzero(np.isfinite(log_diff_counts))

    # 2. Compute gap and utility for valid candidates
    gap = np.floor(sorted_diffs[valid_idx])
    utility = -np.sqrt(1.0 + gap)

    # 3. Build local sensitivity and optimize parameters
    max_gap = max(1, int(np.max(gap)))
    local_sensitivity = _build_local_sensitivity(gap, max_gap)

    if noise == "t":
        # Student-t: t must be < eps/(d+1) for positive s (Theorem 31).
        # Logspace grid per Bun & Steinke §7.1.
        t_upper = epsilon / (degree_freedom + 1.0) - 1e-9
        t_candidates = np.logspace(-9, np.log10(max(t_upper, 1e-9)), 150)
        best_t, best_s, _, best_ss = optimize_params_tdist(
            epsilon, degree_freedom, t_candidates, local_sensitivity
        )
        winner_local = _esnm_t_cpp(
            np.ascontiguousarray(utility),
            np.ascontiguousarray(best_ss * sensitivity),
            np.ascontiguousarray(best_s),
            degree_freedom,
        )
    elif noise == "lln":
        # LLN: t/σ < ε, so t can range up to ε (Proposition 3).
        # Logspace grid per Bun & Steinke §7.1.
        t_upper = max(epsilon, 1e-8)
        t_candidates = np.logspace(-9, np.log10(t_upper), 150)
        best_t, best_sigma, best_s, _, best_ss = optimize_params_lln(
            epsilon, t_candidates, local_sensitivity
        )
        winner_local = _esnm_lln_cpp(
            np.ascontiguousarray(utility),
            np.ascontiguousarray(best_ss * sensitivity),
            np.ascontiguousarray(best_s),
            np.ascontiguousarray(best_sigma),
        )
    elif noise == "gcp":
        # GCP: t < eps/gamma for positive s (Corollary cor:gcp-adm; sigma=1 WLOG).
        # `degree_freedom` carries gamma here. Logspace grid per Bun & Steinke §7.1.
        t_upper = epsilon / degree_freedom - 1e-9
        t_candidates = np.logspace(-9, np.log10(max(t_upper, 1e-9)), 150)
        best_t, best_s, _, best_ss = optimize_params_gcp(
            epsilon, degree_freedom, t_candidates, local_sensitivity
        )
        winner_local = _esnm_gcp_cpp(
            np.ascontiguousarray(utility),
            np.ascontiguousarray(best_ss * sensitivity),
            np.ascontiguousarray(best_s),
            degree_freedom,
        )
    else:
        raise ValueError("noise must be 't', 'lln', or 'gcp'.")

    # 4. Reconstruct sequence from winner
    winner_idx = int(valid_idx[int(winner_local)])
    diff_row = int(diffs_to_positions[0][winner_idx])
    diff_col = int(diffs_to_positions[1][winner_idx])
    diff_value = float(sorted_diffs[winner_idx])
    sequence = sequence_from_diff(diff_value, diff_row, diff_col, diff_matrix)
    return sort_indices[sequence]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def esnm_joint_t(
    counts: np.ndarray,
    epsilon: float,
    k: int,
    sensitivity: int = 1,
    degree_freedom: float = 3.0,
) -> np.ndarray:
    """One-shot eSNM joint top-k with Student's t noise.

    Uses u(a) = -sqrt(1 + gap(a)) with naturally decaying smooth sensitivity.
    No tau needed.  Single noise draw per candidate (no MaxZ).
    """
    return _run_esnm_joint(counts, epsilon, k, "t", sensitivity, degree_freedom)


def esnm_joint_lln(
    counts: np.ndarray,
    epsilon: float,
    k: int,
    sensitivity: int = 1,
) -> np.ndarray:
    """One-shot eSNM joint top-k with LLN noise.

    Uses u(a) = -sqrt(1 + gap(a)) with naturally decaying smooth sensitivity.
    No tau needed.  Single noise draw per candidate (no MaxZ).
    """
    return _run_esnm_joint(counts, epsilon, k, "lln", sensitivity, 3.0)


def esnm_joint_gcp(
    counts: np.ndarray,
    epsilon: float,
    k: int,
    sensitivity: int = 1,
    gamma: float = 5.0,
) -> np.ndarray:
    """One-shot eSNM joint top-k with GCP (Gaussian-core Pareto-tail) noise.

    Uses u(a) = -sqrt(1 + gap(a)) with naturally decaying smooth sensitivity.
    `gamma` (> 2) is the tail exponent; sigma is fixed to 1 (WLOG). No tau needed.
    """
    return _run_esnm_joint(counts, epsilon, k, "gcp", sensitivity, gamma)
