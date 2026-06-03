"""eSNM joint top-k via report-noisy-max over the space of all k-subsets.

The paper's mechanism is report-noisy-max over the space of ALL k-subsets with
subset utility u = -sqrt(1 + gap), gap = floor(largest count difference).  Many
subsets share the same canonical diff value, hence the same utility; they form a
"cell".  Drawing one independent noise variable on each of the C(d, k) subsets
and taking the argmax is distributionally identical (verified, TV ~ MC floor for
every family) to the following efficient construction over the canonical diff
CELLS produced by the joint exponential machinery in `topk.joint`:

    per cell c: draw M_c = max of m_c i.i.d. noise draws from the admissible
                family; cell score = u_c + N_c * M_c; pick argmax cell c*;
                then output ONE uniform random subset within c*.

The cells are exactly the entries of the canonical diff matrix.  For sorted
counts, make_diff_matrix builds the k x d matrix D[r, c] = count_r - count_c plus
a uniquifying term; get_diffs_to_positions orders all entries; sorted_diffs[c] is
the c-th smallest canonical diff and (row, col) = its position.  A cell is VALID
when it can be the largest diff of at least one k-subset; compute_log_diff_counts
returns log m_c, the log NUMBER OF SEQUENCES whose largest count difference is
sorted_diffs[c], and is -inf for the (few) leading positions that no subset can
realise.  We keep only the finite-log_m cells.  gap_c = floor(sorted_diffs[c])
and u_c = -sqrt(1 + gap_c).

The max-of-m draw uses the exact inverse-CDF transform M = F^{-1}(U^{1/m}),
where F is the CDF of the family's base noise Z (the SAME Z the C++ sampler
draws).  m is kept in log space because it can be astronomically large:
U^{1/m} = exp(log(U) * exp(-log_m)).  Per family:

  * t   : Z ~ Student-t(df).             F^{-1} = scipy.stats.t.ppf (closed form).
  * gcp : Gaussian-core Pareto-tail.     F^{-1} closed form (invert the survival).
  * lcp : Laplace-core Pareto-tail.      F^{-1} closed form (invert the survival).
  * lln : Z = Laplace(0,1) * e^{sigma N}. No closed-form ppf, so F^{-1} is built
          by numerical inversion of a cached, monotone CDF grid (interpolated).

The numpy quantile functions are validated against the C++ samplers (empirical
quantiles of many sample_*_noise draws); the whole construction is validated
against the brute-force per-subset noisy-max on tiny instances.
"""

import numpy as np
from scipy import stats

from optimize_params import (
    optimize_params_gcp,
    optimize_params_lcp,
    optimize_params_lln,
    optimize_params_tdist,
    solve_sigma_lln,
)
from topk.joint import (
    compute_log_diff_counts,
    get_diffs_to_positions,
    make_diff_matrix,
    sequence_from_diff,
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


def _build_local_sensitivity(
    gap: np.ndarray, max_k: int, cap: float | None = None
) -> np.ndarray:
    """Build local sensitivity array for utility u = -sqrt(1 + min(gap, cap)).

    local_sensitivity[r, k] = LS(max(0, g - k)), where LS(g') is the local
    sensitivity at gap g': the max utility change from one database modification.

    With F(g) = sqrt(1 + min(g, cap)) the utility is u = -F(g) and the local
    sensitivity is the (downward-dominant) increment LS(g') = F(g') - F(g'-1):

        LS(g') = sqrt(1 + g') - sqrt(g')   for 1 <= g' <= cap
        LS(g') = 0                         for g' > cap   (flat region)
        LS(0)  = sqrt(2) - 1               (= LS(1), only upward change possible)

    `cap=None` recovers the plain sqrt utility (no saturation).  The flat region
    has LS = 0, so capping can only LOWER the smooth sensitivity, never raise it
    — the capped mechanism is iso-private with (never less private than) the
    uncapped one at the same epsilon.

    Computes a compact (max_gap+1, K+1) table indexed by integer gap,
    then maps each candidate to its row — avoids redundant sqrt for
    candidates that share the same gap value.  K is capped at _MAX_DISTANCE.
    """
    max_k = min(max_k, _MAX_DISTANCE)

    k_vals = np.arange(max_k + 1, dtype=np.float32)
    g_vals = np.arange(int(np.max(gap)) + 1, dtype=np.float32)

    # g' clamped to >= 1 so LS(0) = LS(1) (only an upward change at gap 0).
    g_at_k = np.maximum(1.0, g_vals[:, None] - k_vals[None, :])  # (G, K)
    if cap is None:
        # LS(g') = sqrt(1+g') - sqrt(g')
        table = np.sqrt(1.0 + g_at_k) - np.sqrt(g_at_k)  # (G, K)
    else:
        # LS(g') = F(g') - F(g'-1) with F(g) = sqrt(1 + min(g, cap)); the
        # min collapses both terms to sqrt(1+cap) once g' > cap, giving LS = 0.
        gc = np.minimum(g_at_k, float(cap))
        gc_prev = np.minimum(g_at_k - 1.0, float(cap))
        table = np.sqrt(1.0 + gc) - np.sqrt(1.0 + gc_prev)  # (G, K)

    # Map each candidate to its row in the cached table
    gap_int = np.clip(gap.astype(np.intp), 0, len(g_vals) - 1)
    return table[gap_int]


# ---------------------------------------------------------------------------
# Canonical diff cells (positions, log multiplicities, gaps)
# ---------------------------------------------------------------------------

# Cache of the data-deterministic cell structure, keyed by (counts bytes, k).
# Every trial in an experiment reuses the same counts, so this turns the
# canonical-cell construction into a one-shot cost rather than a per-trial one.
_CELL_CACHE: dict[tuple[bytes, int], tuple] = {}
_CELL_CACHE_MAX = 8


def _cell_structure(counts: np.ndarray, k: int) -> tuple:
    """Return the canonical diff-cell structure, cached on (counts, k).

    Builds the canonical diff matrix and keeps only the VALID cells — those a
    k-subset can actually realise as its largest diff (finite log multiplicity).

    Returns
    -------
    (sort_indices, diff_matrix, rows, cols, sorted_diffs, gap, log_m)
        sort_indices : argsort of counts (descending); ranks -> original ids.
        diff_matrix  : the k x d canonical diff matrix on the sorted counts.
        rows, cols   : the (row, col) position of each valid cell in diff_matrix.
        sorted_diffs : the canonical diff value of each valid cell.
        gap          : floor(sorted_diffs), the integer gap of each cell.
        log_m        : log number of SEQUENCES whose largest diff is the cell's
                       canonical diff (the cell multiplicity m_c, always finite).
    """
    key = (counts.tobytes(), k)
    hit = _CELL_CACHE.get(key)
    if hit is not None:
        return hit

    sort_indices = np.argsort(counts)[::-1]
    diff_matrix = make_diff_matrix(counts[sort_indices], k)

    positions = get_diffs_to_positions(diff_matrix)
    log_diff_counts = compute_log_diff_counts(diff_matrix, positions)
    sorted_diffs = diff_matrix[positions]

    # Valid cells: finite log multiplicity (a subset can pin its max here).
    valid = np.isfinite(log_diff_counts)
    rows = positions[0][valid]
    cols = positions[1][valid]
    sorted_diffs = sorted_diffs[valid]
    log_m = log_diff_counts[valid]
    gap = np.floor(sorted_diffs).astype(np.int64)

    result = (sort_indices, diff_matrix, rows, cols, sorted_diffs, gap, log_m)

    if len(_CELL_CACHE) >= _CELL_CACHE_MAX:
        _CELL_CACHE.clear()
    _CELL_CACHE[key] = result
    return result


# ---------------------------------------------------------------------------
# Per-family inverse-CDF (quantile) of the base noise Z
# ---------------------------------------------------------------------------
#
# Each quantile function takes an array p in (0, 1) and returns Q(p) such that
# P(Z <= Q(p)) = p, for the SAME Z drawn by the matching C++ sampler in
# mechanism.cpp / distributions/*.h.  Only p >= 0.5 is ever passed here
# (U^{1/m} >= 0.5 once m is large), but the functions handle the full range.


def _quantile_tdist(p: np.ndarray, df: float) -> np.ndarray:
    """Quantile of standard Student-t(df).

    The C++ sampler draws Z = gaussian / sqrt(chisq(df)/df), i.e. standard
    Student-t with `df` degrees of freedom and unit scale.
    """
    return stats.t.ppf(p, df)


def _quantile_gcp(p: np.ndarray, gamma: float) -> np.ndarray:
    """Quantile of GCP(sigma=1, gamma): Gaussian core, Pareto tail.

    Inverts the survival in gcp.h.  With z0 = sqrt(gamma+1) and normaliser
    kappa, the upper half-mass is split into a Gaussian-core part on [0, z0]
    and a Pareto tail beyond z0:

        P(Z > x) = (core(x) + tail0) / kappa            for 0 <= x <= z0
        P(Z > x) = (tail0 * (z0/x)^gamma) / kappa       for x > z0

    where core(x) = sqrt(2 pi) (Phi(z0) - Phi(x)) and tail0 = z0 e^{-g1/2}/gamma.
    Solved by symmetry about 0 (Q(p) = -Q(1-p)).
    """
    p = np.asarray(p, dtype=np.float64)
    g1 = gamma + 1.0
    z0 = np.sqrt(g1)
    kappa = np.sqrt(2.0 * np.pi) * (2.0 * stats.norm.cdf(z0) - 1.0) + (
        2.0 / gamma
    ) * z0 * np.exp(-0.5 * g1)
    tail0 = z0 * np.exp(-0.5 * g1) / gamma  # one-sided tail mass (unnormalised)

    # Work with the survival s = P(Z > x) of the absolute value via symmetry.
    upper = p >= 0.5
    sgn = np.where(upper, 1.0, -1.0)
    surv = np.where(upper, 1.0 - p, p)  # in (0, 0.5]; = P(|Z| > |x|)/2 region

    surv_norm = surv * kappa  # unnormalised survival mass
    in_tail = surv_norm < tail0

    # Tail region: tail0 * (z0/x)^gamma = surv_norm  ->  x = z0 (tail0/surv_norm)^{1/gamma}
    with np.errstate(divide="ignore", invalid="ignore"):
        x_tail = z0 * np.power(tail0 / np.maximum(surv_norm, 1e-300), 1.0 / gamma)

    # Core region: surv_norm = sqrt(2 pi)(Phi(z0) - Phi(x)) + tail0
    #   -> Phi(x) = Phi(z0) - (surv_norm - tail0)/sqrt(2 pi)
    phi_z0 = stats.norm.cdf(z0)
    phi_x = phi_z0 - (surv_norm - tail0) / np.sqrt(2.0 * np.pi)
    phi_x = np.clip(phi_x, 1e-300, 1.0 - 1e-16)
    x_core = stats.norm.ppf(phi_x)

    out = np.where(in_tail, x_tail, x_core)
    return sgn * out


def _quantile_lcp(p: np.ndarray, gamma: float) -> np.ndarray:
    """Quantile of LCP(sigma=1, gamma): Laplace core, Pareto tail.

    Inverts the survival in lcp.h exactly.  There z0 = gamma+1 and the
    normaliser is C = 2*kappa.  Working in UNNORMALISED survival mass
    m = P(Z > x) * C (so m runs over (0, kappa] on the upper half):

        m = e^{-x} - e^{-g1} + tailb        for 0 <= x <= z0   (Laplace core)
        m = tailb * (z0/x)^gamma            for x > z0         (Pareto tail)

    where tailb = e^{-g1} * z0 / gamma is the common boundary mass at x = z0
    (the core term e^{-z0} - e^{-g1} vanishes there since z0 = g1).  Solved by
    symmetry about 0 (Q(p) = -Q(1-p)).
    """
    p = np.asarray(p, dtype=np.float64)
    g1 = gamma + 1.0
    z0 = g1
    kappa = (1.0 - np.exp(-g1)) + np.exp(-g1) * g1 / gamma
    C = 2.0 * kappa
    tailb = np.exp(-g1) * z0 / gamma  # survival mass at the core/tail boundary

    upper = p >= 0.5
    sgn = np.where(upper, 1.0, -1.0)
    surv = np.where(upper, 1.0 - p, p)

    m = surv * C  # unnormalised survival mass, in (0, kappa]
    in_tail = m < tailb

    # Tail region: tailb * (z0/x)^gamma = m  ->  x = z0 (tailb/m)^{1/gamma}
    with np.errstate(divide="ignore", invalid="ignore"):
        x_tail = z0 * np.power(tailb / np.maximum(m, 1e-300), 1.0 / gamma)

    # Core region: e^{-x} - e^{-g1} + tailb = m  ->  e^{-x} = e^{-g1} + m - tailb
    e_neg_x = np.exp(-g1) + (m - tailb)
    e_neg_x = np.clip(e_neg_x, 1e-300, 1.0)
    x_core = -np.log(e_neg_x)

    out = np.where(in_tail, x_tail, x_core)
    return sgn * out


# Cache of numerically-inverted LLN CDF grids, keyed by sigma (rounded).
_LLN_GRID_CACHE: dict[float, tuple[np.ndarray, np.ndarray]] = {}


def _lln_cdf_grid(sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Build (xs, cdf) tables for Z = Laplace(0,1) * e^{sigma N(0,1)}.

    LLN has no closed-form quantile, so we tabulate the CDF on a dense,
    log-spaced grid (LLN tails are heavy) and interpolate to invert it.

    The CDF is computed exactly the way lln.h / mechanism.cpp define Z:
    Z | N=y is Laplace(0, e^{sigma y}), so F(x) = E_y[ Laplace_cdf(x; e^{sigma y}) ]
    over y ~ N(0,1), evaluated by Gauss-Hermite quadrature.
    """
    key = round(float(sigma), 6)
    cached = _LLN_GRID_CACHE.get(key)
    if cached is not None:
        return cached

    # Gauss-Hermite nodes/weights for E_{N(0,1)}[.] (probabilists' variant).
    nodes, weights = np.polynomial.hermite_e.hermegauss(64)
    w = weights / np.sqrt(2.0 * np.pi)

    # Symmetric, heavy-tailed support; geometric spacing to cover Pareto-like tails.
    pos = np.concatenate(
        [
            np.linspace(0.0, 5.0, 4000),
            np.geomspace(5.0, 1e12, 6000),
        ]
    )
    xs = np.unique(np.concatenate([-pos[::-1], pos]))

    # Laplace(0, s).cdf(x): scale s = e^{sigma y} per node.
    s = np.exp(sigma * nodes)  # (G,)
    xg = xs[:, None] / s[None, :]
    # Laplace CDF in a form that avoids overflow on the large-|xg| tails.
    lap_cdf = np.where(
        xg < 0.0,
        0.5 * np.exp(np.minimum(xg, 0.0)),
        1.0 - 0.5 * np.exp(-np.maximum(xg, 0.0)),
    )
    cdf = lap_cdf @ w
    cdf = np.clip(cdf, 0.0, 1.0)
    # Enforce strict monotonicity for stable interpolation.
    cdf = np.maximum.accumulate(cdf)

    _LLN_GRID_CACHE[key] = (xs, cdf)
    return xs, cdf


def _quantile_lln(p: np.ndarray, sigma: float) -> np.ndarray:
    """Quantile of LLN(sigma) via interpolation of the tabulated CDF."""
    xs, cdf = _lln_cdf_grid(sigma)
    return np.interp(np.asarray(p, dtype=np.float64), cdf, xs)


# ---------------------------------------------------------------------------
# Max-of-m draw via exact inverse-CDF
# ---------------------------------------------------------------------------


def _max_of_m(quantile_fn, log_m: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw, per cell, the max of m_c i.i.d. base-noise variables.

    For X_1..X_m i.i.d. with CDF F, max has CDF F(x)^m, so
        max =d F^{-1}( U^{1/m} ),  U ~ Uniform(0,1).
    With m in log space, U^{1/m} = exp(log(U) * exp(-log_m)); for astronomically
    large m this tends to 1 and the max tends to the upper quantile, handled
    smoothly by the quantile function.
    """
    u = rng.random(log_m.shape)
    # log(U)/m in a numerically safe way; exp(-log_m) underflows to 0 for huge m,
    # giving u1m -> 1 (max pushed to the extreme upper quantile).
    log_u = np.log(u)
    u1m = np.exp(log_u * np.exp(-log_m))
    # Guard the endpoints so the quantile functions stay finite.
    u1m = np.clip(u1m, 1e-300, 1.0 - 1e-15)
    return quantile_fn(u1m)


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
    cap: float | None = None,
) -> np.ndarray:
    """eSNM joint top-k as report-noisy-max over all k-subsets.

    Efficient canonical-cell construction: per valid diff cell draw the max of
    its m_c member sequences' noise, score the cell by u_c + N_c * M_c, take the
    argmax cell, then reconstruct one uniform subset at its canonical position.

    `cap` saturates the utility at gap = cap (u_c = -sqrt(1 + min(gap_c, cap))):
    cells whose gap exceeds the cap become indistinguishable and incur zero
    smooth sensitivity, so the privacy budget is spent only near the decision
    boundary.  `cap=None` is the plain sqrt utility.
    """
    _validate_inputs(counts, epsilon, k, sensitivity)
    counts = np.asarray(counts, dtype=np.float64)

    # 1. Canonical cells (cached across trials on the same data): valid cell
    #    positions, canonical diffs, integer gaps and log multiplicities m_c.
    sort_indices, diff_matrix, rows, cols, sorted_diffs, gap, log_m = _cell_structure(
        counts, k
    )

    # 2. Utility per cell: u_c = -sqrt(1 + min(gap_c, cap)).
    gap_f = gap.astype(np.float64)
    if cap is not None:
        gap_f = np.minimum(gap_f, float(cap))
    utility = -np.sqrt(1.0 + gap_f)

    # 3. Smooth sensitivity per cell at a single FIXED, data-independent t.
    #    ss_c depends only on the cell's gap; derive (s, sigma) at this t via the
    #    existing optimizers called with a length-1 t array.
    max_gap = max(1, int(np.max(gap)))
    local_sensitivity = _build_local_sensitivity(
        gap.astype(np.float64), max_gap, cap=cap
    )

    if noise == "t":
        # Student-t: cap = eps/(df+1); balanced split sets t to half the cap.
        fixed_t = 0.5 * epsilon / (degree_freedom + 1.0)
        _, best_s, _, best_ss = optimize_params_tdist(
            epsilon, degree_freedom, np.array([fixed_t]), local_sensitivity
        )

        def quantile_fn(p):
            return _quantile_tdist(p, degree_freedom)
    elif noise == "lln":
        # LLN: cap = eps; t = eps/2.  Sigma is solved at this single t.
        fixed_t = 0.5 * epsilon
        _, _, best_s, _, best_ss = optimize_params_lln(
            epsilon, np.array([fixed_t]), local_sensitivity
        )
        sigma = float(solve_sigma_lln(epsilon, fixed_t))

        def quantile_fn(p):
            return _quantile_lln(p, sigma)
    elif noise == "gcp":
        # GCP: cap = eps/gamma; t = eps/(2 gamma).  `degree_freedom` carries gamma.
        fixed_t = 0.5 * epsilon / degree_freedom
        _, best_s, _, best_ss = optimize_params_gcp(
            epsilon, degree_freedom, np.array([fixed_t]), local_sensitivity
        )

        def quantile_fn(p):
            return _quantile_gcp(p, degree_freedom)
    elif noise == "lcp":
        # LCP: cap = eps/gamma; t = eps/(2 gamma).  `degree_freedom` carries gamma.
        fixed_t = 0.5 * epsilon / degree_freedom
        _, best_s, _, best_ss = optimize_params_lcp(
            epsilon, degree_freedom, np.array([fixed_t]), local_sensitivity
        )

        def quantile_fn(p):
            return _quantile_lcp(p, degree_freedom)
    else:
        raise ValueError("noise must be 't', 'lln', 'gcp', or 'lcp'.")

    # 4. Per-cell noise scale N_c = (ss_c + global_max_ss) / s, with the
    #    max_{r' != c} term taken as the global max smooth sensitivity over ALL
    #    cells (valid because ties exist: every cell has same-utility siblings).
    #    N_c is CONSTANT within a cell.  s is constant across cells at the fixed t.
    ss = best_ss * sensitivity
    global_max_ss = float(np.max(ss))
    s_val = float(best_s[0])
    n_scale = (ss + global_max_ss) / s_val

    # 5. Max-of-m noise per cell, then cell scores and argmax.
    rng = np.random.default_rng()
    cell_max = _max_of_m(quantile_fn, log_m, rng)
    scores = utility + n_scale * cell_max
    c_star = int(np.argmax(scores))

    # 6. Reconstruct ONE subset uniformly within the winning cell at its
    #    canonical (row, col).  sequence_from_diff samples a sequence whose
    #    largest diff is sorted_diffs[c_star], uniform among such sequences.
    sequence = sequence_from_diff(
        sorted_diffs[c_star], int(rows[c_star]), int(cols[c_star]), diff_matrix
    )
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
    cap: float | None = None,
) -> np.ndarray:
    """eSNM joint top-k with Student's t noise (report-noisy-max over k-subsets).

    `cap` saturates the utility at gap = cap (None = plain sqrt utility).
    """
    return _run_esnm_joint(counts, epsilon, k, "t", sensitivity, degree_freedom, cap)


def esnm_joint_lln(
    counts: np.ndarray,
    epsilon: float,
    k: int,
    sensitivity: int = 1,
    cap: float | None = None,
) -> np.ndarray:
    """eSNM joint top-k with LLN noise (report-noisy-max over k-subsets).

    `cap` saturates the utility at gap = cap (None = plain sqrt utility).
    """
    return _run_esnm_joint(counts, epsilon, k, "lln", sensitivity, 3.0, cap)


def esnm_joint_gcp(
    counts: np.ndarray,
    epsilon: float,
    k: int,
    sensitivity: int = 1,
    gamma: float = 5.0,
    cap: float | None = None,
) -> np.ndarray:
    """eSNM joint top-k with GCP noise (report-noisy-max over k-subsets).

    `gamma` (> 2) is the tail exponent; sigma is fixed to 1 (WLOG).
    `cap` saturates the utility at gap = cap (None = plain sqrt utility).
    """
    return _run_esnm_joint(counts, epsilon, k, "gcp", sensitivity, gamma, cap)


def esnm_joint_lcp(
    counts: np.ndarray,
    epsilon: float,
    k: int,
    sensitivity: int = 1,
    gamma: float = 5.0,
    cap: float | None = None,
) -> np.ndarray:
    """eSNM joint top-k with LCP noise (report-noisy-max over k-subsets).

    `gamma` (> 2) is the tail exponent; sigma is fixed to 1 (WLOG).
    `cap` saturates the utility at gap = cap (None = plain sqrt utility).
    """
    return _run_esnm_joint(counts, epsilon, k, "lcp", sensitivity, gamma, cap)
