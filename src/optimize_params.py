import math
from typing import Callable, Union

import numpy as np
from numba import njit, prange

from standard_selection import safe_exp

Array1DFloat = np.ndarray[tuple[int], np.dtype[np.float64]]
Array2DFloat = np.ndarray[tuple[int, int], np.dtype[np.float64]]
SmoothSensitivityFunc = Callable[[int, float], float]
LocalSensitivityType = Union[Array2DFloat, SmoothSensitivityFunc]


# =============================================================================
# Student's T Distribution Optimization
# =============================================================================
# From Theorem 31 (Section 3.4):
#   Privacy: ε = |t|·(d+1) + |s|·(d+1)/(2√d)
#   Optimal s = 2√d · (ε/(d+1) - t)
#   Variance of T(d) = d/(d-2) for d > 2
#   Noise variance = ss² / (4·(d-2)·(ε/(d+1) - t)²)
# =============================================================================


@njit(cache=True)
def optimal_s_tdist(eps_div: float, d: float, t: float) -> float:
    """Compute optimal scale parameter s for t-distribution.

    Args:
        eps_div: Pre-computed ε/(d+1)
        d: Degrees of freedom
        t: Smoothing parameter

    Returns:
        Optimal scale s = 2√d · (ε/(d+1) - t), or 0 if infeasible
    """
    if t >= eps_div:
        return 0.0
    return 2.0 * np.sqrt(d) * (eps_div - t)


@njit(cache=True, parallel=True)
def _optimize_params_tdist_array(
    eps: float,
    d: float,
    t_candidates: Array1DFloat,
    local_sensitivity: Array2DFloat,
) -> tuple[Array1DFloat, Array1DFloat, Array1DFloat, Array1DFloat]:
    """Optimize t-distribution parameters using array-based local sensitivity.

    Finds optimal (t, s) that minimize variance for each element r.
    Variance formula: ss² / (4·(d-2)·(ε/(d+1) - t)²)

    Performance
    -----------
    * Serial pre-pass: per-t scalars (B, s) and decay vectors are computed
      once and stored, so each parallel thread avoids redundant work.
    * Incremental decay: exp(-t·k) is built as 1 exp() + k multiplications
      instead of k exp() calls (exp is ~20× more expensive than multiply).
    * Parallel over R: elements are independent; the O(R·T·K) inner work
      runs across all available cores.
    """
    R, K = local_sensitivity.shape
    T    = t_candidates.size

    best_var = np.full(R, 1e308)
    best_t   = np.zeros(R)
    best_s   = np.zeros(R)
    best_ss  = np.zeros(R)

    eps_div         = eps / (d + 1.0)
    inv_4_d_minus_2 = 1.0 / (4.0 * (d - 2.0))
    sqrt_d          = np.sqrt(d)

    # ------------------------------------------------------------------
    # Serial pre-pass: for each valid t candidate, compute the variance
    # coefficient B, the optimal s, and the length-K decay vector.
    # Upper-bound allocation (T rows); only n_valid rows are filled.
    # ------------------------------------------------------------------
    pre_t     = np.empty(T)
    pre_B     = np.empty(T)
    pre_s     = np.empty(T)
    pre_decay = np.empty((T, K))
    n_valid   = 0

    for ti in range(T):
        t = t_candidates[ti]
        if t < 0.0 or t >= eps_div:
            continue

        denom = eps_div - t
        s     = 2.0 * sqrt_d * denom
        if s <= 0.0:
            continue

        pre_t[n_valid] = t
        pre_B[n_valid] = inv_4_d_minus_2 / (denom * denom)
        pre_s[n_valid] = s

        # Incremental: exp(-t·k) = exp(-t)^k via one exp + k multiplies.
        exp_neg_t = np.exp(-t)
        decay     = 1.0
        for k in range(K):
            pre_decay[n_valid, k] = decay
            decay *= exp_neg_t

        n_valid += 1

    # ------------------------------------------------------------------
    # Parallel main pass: each element r independently finds the t that
    # minimises its variance. No shared writes — r indexes are disjoint.
    # ------------------------------------------------------------------
    for r in prange(R):
        for vi in range(n_valid):
            # Smooth sensitivity: max_k( LS[r,k] · e^{-t·k} )
            mx = 0.0
            for k in range(K):
                v = local_sensitivity[r, k] * pre_decay[vi, k]
                if v > mx:
                    mx = v

            var_r = pre_B[vi] * mx * mx
            if var_r < best_var[r]:
                best_var[r] = var_r
                best_t[r]   = pre_t[vi]
                best_s[r]   = pre_s[vi]
                best_ss[r]  = mx

    return best_t, best_s, best_var, best_ss


def _optimize_params_tdist_func(
    eps: float,
    d: float,
    t_candidates: Array1DFloat,
    smooth_sensitivity_func: SmoothSensitivityFunc,
    R: int,
) -> tuple[Array1DFloat, Array1DFloat, Array1DFloat, Array1DFloat]:
    """Optimize t-distribution parameters using function-based smooth sensitivity."""
    best_var = np.full(R, 1e308)
    best_t = np.zeros(R)
    best_s = np.zeros(R)
    best_ss = np.zeros(R)

    eps_div = eps / (d + 1.0)
    inv_4_d_minus_2 = 1.0 / (4.0 * (d - 2.0))

    for ti in range(t_candidates.size):
        t = t_candidates[ti]

        if t < 0.0 or t >= eps_div:
            continue

        denom = eps_div - t
        s = 2.0 * np.sqrt(d) * denom

        if s <= 0.0:
            continue

        B = inv_4_d_minus_2 / (denom * denom)

        for r in range(R):
            ss = smooth_sensitivity_func(r, t)
            var_r = B * ss * ss

            if var_r < best_var[r]:
                best_var[r] = var_r
                best_t[r] = t
                best_s[r] = s
                best_ss[r] = ss

    return best_t, best_s, best_var, best_ss


def optimize_params_tdist(
    eps: float,
    d: float,
    t_candidates: Array1DFloat,
    local_sensitivity: LocalSensitivityType,
    R: int | None = None,
) -> tuple[Array1DFloat, Array1DFloat, Array1DFloat, Array1DFloat]:
    """
    Optimize parameters for t-distribution based mechanism.

    Based on Theorem 31 from Bun & Steinke (2019).

    Parameters
    ----------
    eps : float
        Privacy parameter epsilon.
    d : float
        Degrees of freedom parameter (must be > 2 for finite variance).
    t_candidates : Array1DFloat
        Array of candidate t values to search over.
    local_sensitivity : Array2DFloat | Callable[[int, float], float]
        Either a 2D numpy array of shape (R, K) where local_sensitivity[r, k]
        gives the sensitivity for element r at distance k, or a callable
        function(r, t) -> float that returns the smooth sensitivity directly.
    R : int | None
        Number of elements. Required if local_sensitivity is a function.

    Returns
    -------
    tuple[Array1DFloat, Array1DFloat, Array1DFloat, Array1DFloat]
        (best_t, best_s, best_var, best_ss) arrays of length R.
    """
    if d <= 2.0:
        raise ValueError(f"d must be > 2 for finite variance, got {d}")

    if isinstance(local_sensitivity, np.ndarray):
        return _optimize_params_tdist_array(eps, d, t_candidates, local_sensitivity)
    elif callable(local_sensitivity):
        if R is None:
            raise ValueError("R must be provided when local_sensitivity is a function.")
        return _optimize_params_tdist_func(eps, d, t_candidates, local_sensitivity, R)
    else:
        raise TypeError(
            f"local_sensitivity must be a numpy array or callable, got {type(local_sensitivity)}"
        )


# =============================================================================
# Gaussian-core Pareto-tail (GCP) Distribution Optimization
# =============================================================================
# From Corollary cor:gcp-adm (with sigma fixed to 1, WLOG):
#   Privacy: ε = √(γ+1)·s + γ·t
#   Optimal s = (ε - γ·t)/√(γ+1)
#   Variance of GCP(1,γ) = V(γ) = M₂(γ)/κ(γ) for γ > 2
#   Noise variance = ss² · V(γ)·(γ+1) / (ε - γ·t)²
# =============================================================================


def gcp_variance_coeff(gamma: float) -> float:
    """Variance constant V(γ) = M₂(γ)/κ(γ) of GCP(σ=1, γ).

    Requires γ > 2 for finite variance. Φ is the standard normal CDF.
    Computed once, in pure Python (outside numba).
    """
    g1 = gamma + 1.0
    sg1 = math.sqrt(g1)
    phi_sg1 = 0.5 * (1.0 + math.erf(sg1 / math.sqrt(2.0)))
    exp_term = math.exp(-0.5 * g1)

    kappa = math.sqrt(2.0 * math.pi) * (2.0 * phi_sg1 - 1.0) + (2.0 / gamma) * sg1 * exp_term
    m2 = (
        math.sqrt(2.0 * math.pi) * (2.0 * phi_sg1 - 1.0)
        - 2.0 * sg1 * exp_term
        + (2.0 / (gamma - 2.0)) * g1**1.5 * exp_term
    )
    return m2 / kappa


@njit(cache=True)
def optimal_s_gcp(eps: float, gamma: float, t: float) -> float:
    """Compute optimal scale parameter s for GCP.

    s = (ε - γ·t)/√(γ+1), or 0 if infeasible.
    """
    if t >= eps / gamma:
        return 0.0
    return (eps - gamma * t) / np.sqrt(gamma + 1.0)


@njit(cache=True, parallel=True)
def _optimize_params_gcp_array(
    eps: float,
    gamma: float,
    V: float,
    t_candidates: Array1DFloat,
    local_sensitivity: Array2DFloat,
) -> tuple[Array1DFloat, Array1DFloat, Array1DFloat, Array1DFloat]:
    """Optimize GCP parameters using array-based local sensitivity.

    Finds optimal (t, s) that minimize variance for each element r.
    Variance formula: ss² · V·(γ+1) / (ε - γ·t)²

    Performance characteristics mirror _optimize_params_tdist_array:
    a serial pre-pass computes per-t scalars (B, s) and decay vectors once,
    then a prange(R) main pass minimises variance per element.
    """
    R, K = local_sensitivity.shape
    T    = t_candidates.size

    best_var = np.full(R, 1e308)
    best_t   = np.zeros(R)
    best_s   = np.zeros(R)
    best_ss  = np.zeros(R)

    eps_div = eps / gamma
    sqrt_g1 = np.sqrt(gamma + 1.0)

    # ------------------------------------------------------------------
    # Serial pre-pass: for each valid t candidate, compute the variance
    # coefficient B, the optimal s, and the length-K decay vector.
    # ------------------------------------------------------------------
    pre_t     = np.empty(T)
    pre_B     = np.empty(T)
    pre_s     = np.empty(T)
    pre_decay = np.empty((T, K))
    n_valid   = 0

    for ti in range(T):
        t = t_candidates[ti]
        if t < 0.0 or t >= eps_div:
            continue

        denom = eps - gamma * t
        s     = denom / sqrt_g1
        if s <= 0.0:
            continue

        pre_t[n_valid] = t
        pre_B[n_valid] = V * (gamma + 1.0) / (denom * denom)
        pre_s[n_valid] = s

        # Incremental: exp(-t·k) = exp(-t)^k via one exp + k multiplies.
        exp_neg_t = np.exp(-t)
        decay     = 1.0
        for k in range(K):
            pre_decay[n_valid, k] = decay
            decay *= exp_neg_t

        n_valid += 1

    # ------------------------------------------------------------------
    # Parallel main pass: each element r independently finds the t that
    # minimises its variance. No shared writes — r indexes are disjoint.
    # ------------------------------------------------------------------
    for r in prange(R):
        for vi in range(n_valid):
            # Smooth sensitivity: max_k( LS[r,k] · e^{-t·k} )
            mx = 0.0
            for k in range(K):
                v = local_sensitivity[r, k] * pre_decay[vi, k]
                if v > mx:
                    mx = v

            var_r = pre_B[vi] * mx * mx
            if var_r < best_var[r]:
                best_var[r] = var_r
                best_t[r]   = pre_t[vi]
                best_s[r]   = pre_s[vi]
                best_ss[r]  = mx

    return best_t, best_s, best_var, best_ss


def _optimize_params_gcp_func(
    eps: float,
    gamma: float,
    V: float,
    t_candidates: Array1DFloat,
    smooth_sensitivity_func: SmoothSensitivityFunc,
    R: int,
) -> tuple[Array1DFloat, Array1DFloat, Array1DFloat, Array1DFloat]:
    """Optimize GCP parameters using function-based smooth sensitivity."""
    best_var = np.full(R, 1e308)
    best_t = np.zeros(R)
    best_s = np.zeros(R)
    best_ss = np.zeros(R)

    eps_div = eps / gamma
    sqrt_g1 = np.sqrt(gamma + 1.0)

    for ti in range(t_candidates.size):
        t = t_candidates[ti]

        if t < 0.0 or t >= eps_div:
            continue

        denom = eps - gamma * t
        s = denom / sqrt_g1

        if s <= 0.0:
            continue

        B = V * (gamma + 1.0) / (denom * denom)

        for r in range(R):
            ss = smooth_sensitivity_func(r, t)
            var_r = B * ss * ss

            if var_r < best_var[r]:
                best_var[r] = var_r
                best_t[r] = t
                best_s[r] = s
                best_ss[r] = ss

    return best_t, best_s, best_var, best_ss


def optimize_params_gcp(
    eps: float,
    gamma: float,
    t_candidates: Array1DFloat,
    local_sensitivity: LocalSensitivityType,
    R: int | None = None,
) -> tuple[Array1DFloat, Array1DFloat, Array1DFloat, Array1DFloat]:
    """
    Optimize parameters for GCP (Gaussian-core Pareto-tail) based mechanism.

    Based on Corollary cor:gcp-adm. With sigma fixed to 1 (WLOG).

    Parameters
    ----------
    eps : float
        Privacy parameter epsilon.
    gamma : float
        Tail shape parameter (must be > 2 for finite variance).
    t_candidates : Array1DFloat
        Array of candidate t values to search over.
    local_sensitivity : Array2DFloat | Callable[[int, float], float]
        Either a 2D numpy array of shape (R, K) where local_sensitivity[r, k]
        gives the sensitivity for element r at distance k, or a callable
        function(r, t) -> float that returns the smooth sensitivity directly.
    R : int | None
        Number of elements. Required if local_sensitivity is a function.

    Returns
    -------
    tuple[Array1DFloat, Array1DFloat, Array1DFloat, Array1DFloat]
        (best_t, best_s, best_var, best_ss) arrays of length R.
    """
    if gamma <= 2.0:
        raise ValueError(f"gamma must be > 2 for finite variance, got {gamma}")

    V = gcp_variance_coeff(gamma)

    if isinstance(local_sensitivity, np.ndarray):
        return _optimize_params_gcp_array(eps, gamma, V, t_candidates, local_sensitivity)
    elif callable(local_sensitivity):
        if R is None:
            raise ValueError("R must be provided when local_sensitivity is a function.")
        return _optimize_params_gcp_func(eps, gamma, V, t_candidates, local_sensitivity, R)
    else:
        raise TypeError(
            f"local_sensitivity must be a numpy array or callable, got {type(local_sensitivity)}"
        )


# =============================================================================
# Laplace-core Pareto-tail (LCP) Distribution Optimization
# =============================================================================
# With sigma fixed to 1, WLOG:
#   Privacy: epsilon = s + gamma * t
#   Optimal s = epsilon - gamma * t
#   Variance of LCP(1,gamma) = V(gamma) for gamma > 2
#   Noise variance = ss^2 * V(gamma) / (epsilon - gamma * t)^2
# =============================================================================


def lcp_variance_coeff(gamma: float) -> float:
    """Variance constant V(gamma) of LCP(sigma=1, gamma).

    Requires gamma > 2 for finite variance. Computed once in pure Python
    outside numba.
    """
    g1 = gamma + 1.0
    exp_term = math.exp(-g1)

    kappa = 1.0 - exp_term + g1 * exp_term / gamma
    m2 = (
        2.0
        - exp_term * (g1 * g1 + 2.0 * g1 + 2.0)
        + exp_term * g1**3 / (gamma - 2.0)
    )
    return m2 / kappa


@njit(cache=True)
def optimal_s_lcp(eps: float, gamma: float, t: float) -> float:
    """Compute optimal scale parameter s for LCP.

    s = epsilon - gamma * t, or 0 if infeasible.
    """
    if t >= eps / gamma:
        return 0.0
    return eps - gamma * t


@njit(cache=True, parallel=True)
def _optimize_params_lcp_array(
    eps: float,
    gamma: float,
    V: float,
    t_candidates: Array1DFloat,
    local_sensitivity: Array2DFloat,
) -> tuple[Array1DFloat, Array1DFloat, Array1DFloat, Array1DFloat]:
    """Optimize fixed-gamma LCP parameters using array local sensitivity."""
    R, K = local_sensitivity.shape
    T = t_candidates.size

    best_var = np.full(R, 1e308)
    best_t = np.zeros(R)
    best_s = np.zeros(R)
    best_ss = np.zeros(R)

    eps_div = eps / gamma

    pre_t = np.empty(T)
    pre_B = np.empty(T)
    pre_s = np.empty(T)
    pre_decay = np.empty((T, K))
    n_valid = 0

    for ti in range(T):
        t = t_candidates[ti]
        if t < 0.0 or t >= eps_div:
            continue

        denom = eps - gamma * t
        if denom <= 0.0:
            continue

        pre_t[n_valid] = t
        pre_B[n_valid] = V / (denom * denom)
        pre_s[n_valid] = denom

        exp_neg_t = np.exp(-t)
        decay = 1.0
        for k in range(K):
            pre_decay[n_valid, k] = decay
            decay *= exp_neg_t

        n_valid += 1

    for r in prange(R):
        for vi in range(n_valid):
            mx = 0.0
            for k in range(K):
                v = local_sensitivity[r, k] * pre_decay[vi, k]
                if v > mx:
                    mx = v

            var_r = pre_B[vi] * mx * mx
            if var_r < best_var[r]:
                best_var[r] = var_r
                best_t[r] = pre_t[vi]
                best_s[r] = pre_s[vi]
                best_ss[r] = mx

    return best_t, best_s, best_var, best_ss


def _optimize_params_lcp_func(
    eps: float,
    gamma: float,
    V: float,
    t_candidates: Array1DFloat,
    smooth_sensitivity_func: SmoothSensitivityFunc,
    R: int,
) -> tuple[Array1DFloat, Array1DFloat, Array1DFloat, Array1DFloat]:
    """Optimize fixed-gamma LCP parameters using smooth sensitivity calls."""
    best_var = np.full(R, 1e308)
    best_t = np.zeros(R)
    best_s = np.zeros(R)
    best_ss = np.zeros(R)

    eps_div = eps / gamma

    for ti in range(t_candidates.size):
        t = t_candidates[ti]

        if t < 0.0 or t >= eps_div:
            continue

        denom = eps - gamma * t
        if denom <= 0.0:
            continue

        B = V / (denom * denom)

        for r in range(R):
            ss = smooth_sensitivity_func(r, t)
            var_r = B * ss * ss

            if var_r < best_var[r]:
                best_var[r] = var_r
                best_t[r] = t
                best_s[r] = denom
                best_ss[r] = ss

    return best_t, best_s, best_var, best_ss


def optimize_params_lcp(
    eps: float,
    gamma: float,
    t_candidates: Array1DFloat,
    local_sensitivity: LocalSensitivityType,
    R: int | None = None,
) -> tuple[Array1DFloat, Array1DFloat, Array1DFloat, Array1DFloat]:
    """
    Optimize fixed-gamma LCP (Laplace-core Pareto-tail) mechanism parameters.

    With sigma fixed to 1, the pure-DP calibration is
    epsilon = s + gamma * t.
    """
    if gamma <= 2.0:
        raise ValueError(f"gamma must be > 2 for finite variance, got {gamma}")

    V = lcp_variance_coeff(gamma)

    if isinstance(local_sensitivity, np.ndarray):
        return _optimize_params_lcp_array(eps, gamma, V, t_candidates, local_sensitivity)
    elif callable(local_sensitivity):
        if R is None:
            raise ValueError("R must be provided when local_sensitivity is a function.")
        return _optimize_params_lcp_func(eps, gamma, V, t_candidates, local_sensitivity, R)
    else:
        raise TypeError(
            f"local_sensitivity must be a numpy array or callable, got {type(local_sensitivity)}"
        )


# =============================================================================
# Laplace Log-Normal (LLN) Distribution Optimization
# =============================================================================
# From Section 3.1.1:
#   Privacy: ε = t/σ + e^(3σ²/2) · s
#   Optimal σ solves cubic: 5(ε/t)σ³ - 5σ² - 1 = 0
#   Optimal s = e^(-3σ²/2) · (ε - t/σ)
#   Variance of LLN(σ) = 2e^(2σ²)
#   Noise variance = ss² · 2e^(5σ²) / (ε - t/σ)²
# =============================================================================


@njit(cache=True)
def sigma_function_lln(sigma: float, eps: float, t: float) -> float:
    """Cubic function for optimal σ: 5(ε/t)σ³ - 5σ² - 1 = 0"""
    return 5.0 * (eps / t) * sigma**3 - 5.0 * sigma**2 - 1.0


@njit(cache=True)
def solve_sigma_lln(
    eps: float, t: float, max_iter: int = 100, tol: float = 1e-12
) -> float:
    """Solve for optimal σ using bisection.

    The paper shows the solution is in (t/ε, max{2t/ε, 1/2}].
    """
    # Lower bound: σ > t/ε (required for εσ > t)
    tol_err = 1.0 + tol
    low = (t / eps) * tol_err
    f_low = sigma_function_lln(low, eps, t)

    # Upper bound: use max{2t/ε, 1/2} as starting point
    high = max(2.0 * t / eps, 0.5)
    f_high = sigma_function_lln(high, eps, t)

    # Expand upper bound if needed
    while f_high < 0.0 and high < 1e6:
        high *= 2.0
        f_high = sigma_function_lln(high, eps, t)

    # Bisection search
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = sigma_function_lln(mid, eps, t)

        if abs(f_mid) < tol:
            return mid

        if f_mid * f_low < 0.0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid

    return 0.5 * (low + high)


@njit(cache=True)
def optimal_s_lln(eps: float, t: float, sigma: float) -> float:
    """Compute optimal scale parameter s for LLN.

    s = e^(-3σ²/2) · (ε - t/σ)
    """
    return np.exp(-1.5 * sigma * sigma) * (eps - t / sigma)


@njit(cache=True, parallel=True)
def _optimize_params_lln_array(
    eps: float, t_candidates: Array1DFloat, local_sensitivity: Array2DFloat
) -> tuple[Array1DFloat, Array1DFloat, Array1DFloat, Array1DFloat, Array1DFloat]:
    """Optimize LLN parameters using array-based local sensitivity.

    Variance formula: ss² · 2e^(5σ²) / (ε - t/σ)²

    Performance
    -----------
    * solve_sigma_lln (100-iteration bisection) and the variance coefficient A
      depend only on t, not on r.  A serial pre-pass computes them once per
      valid t, avoiding R-fold redundant solver calls inside the parallel loop.
    * Incremental decay and prange(R) as in _optimize_params_tdist_array.
    """
    R, K = local_sensitivity.shape
    T    = t_candidates.size

    best_var   = np.full(R, 1e308)
    best_t     = np.zeros(R)
    best_sigma = np.zeros(R)
    best_s     = np.zeros(R)
    best_ss    = np.zeros(R)

    # ------------------------------------------------------------------
    # Serial pre-pass: sigma and A depend only on t, not on r.
    # ------------------------------------------------------------------
    pre_t     = np.empty(T)
    pre_A     = np.empty(T)
    pre_s     = np.empty(T)
    pre_sigma = np.empty(T)
    pre_decay = np.empty((T, K))
    n_valid   = 0

    for ti in range(T):
        t = t_candidates[ti]
        if t <= 0.0:
            continue

        sigma = solve_sigma_lln(eps, t)
        s     = optimal_s_lln(eps, t, sigma)
        if s <= 0.0:
            continue

        denom = eps - t / sigma
        if denom <= 0.0:
            continue

        exp_term = 5.0 * sigma * sigma
        if exp_term > 700.0:  # Prevent overflow
            continue

        pre_t[n_valid]     = t
        pre_A[n_valid]     = 2.0 * np.exp(exp_term) / (denom * denom)
        pre_s[n_valid]     = s
        pre_sigma[n_valid] = sigma

        exp_neg_t = np.exp(-t)
        decay     = 1.0
        for k in range(K):
            pre_decay[n_valid, k] = decay
            decay *= exp_neg_t

        n_valid += 1

    # ------------------------------------------------------------------
    # Parallel main pass.
    # ------------------------------------------------------------------
    for r in prange(R):
        for vi in range(n_valid):
            mx = 0.0
            for k in range(K):
                v = local_sensitivity[r, k] * pre_decay[vi, k]
                if v > mx:
                    mx = v

            var_r = pre_A[vi] * mx * mx
            if var_r < best_var[r]:
                best_var[r]   = var_r
                best_t[r]     = pre_t[vi]
                best_sigma[r] = pre_sigma[vi]
                best_s[r]     = pre_s[vi]
                best_ss[r]    = mx

    return best_t, best_sigma, best_s, best_var, best_ss


def _optimize_params_lln_func(
    eps: float,
    t_candidates: Array1DFloat,
    smooth_sensitivity_func: SmoothSensitivityFunc,
    R: int,
) -> tuple[Array1DFloat, Array1DFloat, Array1DFloat, Array1DFloat, Array1DFloat]:
    """Optimize LLN parameters using function-based smooth sensitivity."""
    best_var = np.full(R, 1e308)
    best_t = np.zeros(R)
    best_sigma = np.zeros(R)
    best_s = np.zeros(R)
    best_ss = np.zeros(R)

    for ti in range(t_candidates.size):
        t = t_candidates[ti]
        if t <= 0.0:
            continue

        sigma = solve_sigma_lln(eps, t)
        s = optimal_s_lln(eps, t, sigma)

        if s <= 0.0:
            continue

        denom = eps - t / sigma
        if denom <= 0.0:
            continue

        # Variance coefficient with overflow protection
        exp_term = 5.0 * sigma * sigma
        A = 2.0 * safe_exp(exp_term) / (denom * denom)

        for r in range(R):
            ss = smooth_sensitivity_func(r, t)
            var_r = A * ss * ss

            if var_r < best_var[r]:
                best_var[r] = var_r
                best_t[r] = t
                best_sigma[r] = sigma
                best_s[r] = s
                best_ss[r] = ss

    return best_t, best_sigma, best_s, best_var, best_ss


def optimize_params_lln(
    eps: float,
    t_candidates: Array1DFloat,
    local_sensitivity: LocalSensitivityType,
    R: int | None = None,
) -> tuple[Array1DFloat, Array1DFloat, Array1DFloat, Array1DFloat, Array1DFloat]:
    """
    Optimize parameters for Log-Laplace-Normal (LLN) based mechanism.

    Based on Section 3.1.1 from Bun & Steinke (2019).

    Parameters
    ----------
    eps : float
        Privacy parameter epsilon.
    t_candidates : Array1DFloat
        Array of candidate t values to search over.
    local_sensitivity : Array2DFloat | Callable[[int, float], float]
        Either a 2D numpy array of shape (R, K) where local_sensitivity[r, k]
        gives the sensitivity for element r at distance k, or a callable
        function(r, t) -> float that returns the smooth sensitivity directly.
    R : int | None
        Number of elements. Required if local_sensitivity is a function.

    Returns
    -------
    tuple[Array1DFloat, Array1DFloat, Array1DFloat, Array1DFloat, Array1DFloat]
        (best_t, best_sigma, best_s, best_var, best_ss) arrays of length R.
    """
    if isinstance(local_sensitivity, np.ndarray):
        return _optimize_params_lln_array(eps, t_candidates, local_sensitivity)
    elif callable(local_sensitivity):
        if R is None:
            raise ValueError("R must be provided when local_sensitivity is a function.")
        return _optimize_params_lln_func(eps, t_candidates, local_sensitivity, R)
    else:
        raise TypeError(
            f"local_sensitivity must be a numpy array or callable, got {type(local_sensitivity)}"
        )
