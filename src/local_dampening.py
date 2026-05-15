from typing import Callable

import numpy as np
from numba import njit, prange

from standard_selection import em


@njit(cache=True, nogil=True, fastmath=True)
def b(t: int, r: int, ls: np.ndarray, gs: float) -> float:
    """Accumulation function for local dampening.

    Computes b(t) = sum_{k=0}^{t-1} delta(k) for t > 0, where delta(k) is
    ls[r, k] for k < ls.shape[1] and gs (global sensitivity) otherwise.
    For t < 0, returns -b(-t).
    For t = 0, returns 0.

    According to Definition 7 (Boundedness) in the paper, bounded sensitivity
    functions satisfy delta(x, t, r) = global_sensitivity for t >= n.
    """
    if t == 0:
        return 0.0

    if t < 0:
        total = 0.0
        for k in range(-t):
            if k < ls.shape[1]:
                total += ls[r, k]
            else:
                total += gs
        return -total

    total = 0.0
    for k in range(t):
        if k < ls.shape[1]:
            total += ls[r, k]
        else:
            total += gs
    return total


@njit(cache=True, nogil=True, fastmath=True)
def dampening_func(u: np.ndarray, r: int, gs: float, ls: np.ndarray) -> float:
    """Compute the dampening function for element r.

    Maps utility u[r] to a dampened value based on local sensitivity.
    Uses incremental computation of b(t) for O(t) instead of O(t²) complexity.
    """
    err = 1e-12
    sign = 1.0
    u_r = u[r]

    if u_r < 0:
        sign = -1.0
        u_r = -u_r

    K = ls.shape[1]
    t = 0
    bt = 0.0  # b(0) = 0

    while True:
        # Compute delta(t) = ls[r, t] if t < K else gs
        delta_t = ls[r, t] if t < K else gs
        btp = bt + delta_t  # b(t+1) = b(t) + delta(t)

        # Check if sensitivity has converged to global sensitivity.
        # From here on delta = gs, so b(i) = bt + (i-t)*gs for i >= t,
        # and per Definition 12, D = (u_r - bt)/gs + t.
        if delta_t >= gs - err or delta_t == 0:
            return ((u_r - bt) / gs + t) * sign

        # Check if u_r is in interval [b(t), b(t+1))
        if u_r >= bt and u_r < btp:
            break

        t += 1
        bt = btp  # Move to next interval

        # Safety bound
        if t > K:
            return ((u_r - bt) / gs + t) * sign

    return (((u_r - bt) / (btp - bt)) + t) * sign


@njit(cache=True, nogil=True, fastmath=True)
def shifted_dampening_func(
    u: np.ndarray, r: int, gs: float, ls: np.ndarray, shift: float
) -> float:
    """Compute the shifted dampening function for element r.

    For monotonically non-decreasing sensitivity functions, shifts utilities
    left so all values are negative, then applies dampening.

    Uses incremental computation of b(t) for O(|t|) instead of O(t²) complexity.

    Args:
        u: Utility scores
        r: Element index
        gs: Global sensitivity
        ls: Local sensitivity matrix [r, k]
        shift: Shift amount (typically n * gs + max(u) per Lemma 5)

    Returns:
        Shifted dampened utility
    """
    err = 1e-12
    K = ls.shape[1]

    # Shifted utility (now negative)
    u_shifted = u[r] - shift

    # Since u_shifted is negative, we work with negative t values
    # b(t) for negative t: b(-1) = -delta(0), b(-2) = -delta(0) - delta(1), etc.
    t = 0
    bt = 0.0  # b(0) = 0

    # Go negative until we find the interval containing u_shifted
    while u_shifted < bt:
        # For negative t, we need delta(-t-1) = ls[r, -t-1] if -t-1 < K else gs
        # When going from t to t-1: b(t-1) = b(t) - delta(-t)
        neg_t = -t  # This is the index we need: 0, 1, 2, ...
        delta = ls[r, neg_t] if neg_t < K else gs

        t -= 1
        bt_prev = bt
        bt = bt - delta  # b(t-1) = b(t) - delta(-t)

        # Check if sensitivity has converged to global sensitivity
        interval_width = bt_prev - bt  # This equals delta
        if interval_width >= gs - err:
            # Sensitivity converged - use gs for remaining computation
            remaining = bt - u_shifted
            additional_steps = remaining / gs
            return float(t) - additional_steps

    # Now u_shifted is in [b(t), b(t+1))
    # Compute b(t+1) incrementally
    if t < 0:
        # For t < 0, b(t+1) = b(t) + delta(-t-1)
        neg_t_minus_1 = -t - 1
        delta = ls[r, neg_t_minus_1] if neg_t_minus_1 < K else gs
        btp = bt + delta
    else:
        # t == 0, so b(1) = b(0) + delta(0)
        delta = ls[r, 0] if K > 0 else gs
        btp = bt + delta

    interval_width = btp - bt

    # If interval width is 0 or has converged to gs
    if abs(interval_width) < err:
        return float(t)

    if interval_width >= gs - err:
        return ((u_shifted - bt) / gs) + t

    return ((u_shifted - bt) / interval_width) + t


@njit(cache=True, parallel=True, fastmath=True)
def _compute_dampened_utilities(u: np.ndarray, gs: float, ls: np.ndarray) -> np.ndarray:
    """Compute dampened utilities for all elements in parallel."""
    n = len(u)
    dampened = np.empty(n, dtype=np.float64)
    for r in prange(n):
        dampened[r] = dampening_func(u, r, gs, ls)
    return dampened


@njit(cache=True, parallel=True, fastmath=True)
def _compute_shifted_dampened_utilities(
    u: np.ndarray, gs: float, ls: np.ndarray, shift: float
) -> np.ndarray:
    """Compute shifted dampened utilities for all elements in parallel."""
    n = len(u)
    dampened = np.empty(n, dtype=np.float64)
    for r in prange(n):
        dampened[r] = shifted_dampening_func(u, r, gs, ls, shift)
    return dampened


# ---------------------------------------------------------------------------
# Shared-LS variants
# ---------------------------------------------------------------------------
# When every outcome shares the same LS curve (e.g. percentile / quantile
# selection where LS depends on x but not on the output value), materialising
# an (R, K) matrix costs O(R·K) memory for no information gain. The kernels
# below accept a length-K 1D curve and read it directly per outcome.


@njit(cache=True, nogil=True, fastmath=True)
def _dampening_func_shared(u_r: float, gs: float, ls_1d: np.ndarray) -> float:
    err = 1e-12
    sign = 1.0
    if u_r < 0:
        sign = -1.0
        u_r = -u_r

    K = ls_1d.shape[0]
    t = 0
    bt = 0.0

    while True:
        delta_t = ls_1d[t] if t < K else gs
        btp = bt + delta_t

        if delta_t >= gs - err or delta_t == 0:
            return ((u_r - bt) / gs + t) * sign

        if u_r >= bt and u_r < btp:
            break

        t += 1
        bt = btp

        if t > K:
            return ((u_r - bt) / gs + t) * sign

    return (((u_r - bt) / (btp - bt)) + t) * sign


@njit(cache=True, nogil=True, fastmath=True)
def _shifted_dampening_func_shared(
    u_r: float, gs: float, ls_1d: np.ndarray, shift: float
) -> float:
    err = 1e-12
    K = ls_1d.shape[0]

    u_shifted = u_r - shift

    t = 0
    bt = 0.0

    while u_shifted < bt:
        neg_t = -t
        delta = ls_1d[neg_t] if neg_t < K else gs

        t -= 1
        bt_prev = bt
        bt = bt - delta

        interval_width = bt_prev - bt
        if interval_width >= gs - err:
            remaining = bt - u_shifted
            additional_steps = remaining / gs
            return float(t) - additional_steps

    if t < 0:
        neg_t_minus_1 = -t - 1
        delta = ls_1d[neg_t_minus_1] if neg_t_minus_1 < K else gs
        btp = bt + delta
    else:
        delta = ls_1d[0] if K > 0 else gs
        btp = bt + delta

    interval_width = btp - bt

    if abs(interval_width) < err:
        return float(t)

    if interval_width >= gs - err:
        return ((u_shifted - bt) / gs) + t

    return ((u_shifted - bt) / interval_width) + t


@njit(cache=True, parallel=True, fastmath=True)
def _compute_dampened_utilities_shared(
    u: np.ndarray, gs: float, ls_1d: np.ndarray
) -> np.ndarray:
    n = len(u)
    dampened = np.empty(n, dtype=np.float64)
    for r in prange(n):
        dampened[r] = _dampening_func_shared(u[r], gs, ls_1d)
    return dampened


@njit(cache=True, parallel=True, fastmath=True)
def _compute_shifted_dampened_utilities_shared(
    u: np.ndarray, gs: float, ls_1d: np.ndarray, shift: float
) -> np.ndarray:
    n = len(u)
    dampened = np.empty(n, dtype=np.float64)
    for r in prange(n):
        dampened[r] = _shifted_dampening_func_shared(u[r], gs, ls_1d, shift)
    return dampened


def ld_pmf(u: np.ndarray, gs: float, eps: float, ls: np.ndarray):
    """Compute probability mass function using local dampening.

    Args:
        u: Utility scores
        gs: Global sensitivity
        eps: Privacy budget
        ls: Local sensitivity. Either a 2D matrix of shape (R, K) where
            ls[r, k] is per-outcome, or a 1D curve of length K shared
            across all outcomes.

    Returns:
        PMF over elements
    """
    if not u.flags["C_CONTIGUOUS"]:
        u = np.ascontiguousarray(u, dtype=np.float64)
    if not ls.flags["C_CONTIGUOUS"]:
        ls = np.ascontiguousarray(ls, dtype=np.float64)
    if ls.ndim == 1:
        dampened_u = _compute_dampened_utilities_shared(u, gs, ls)
    else:
        dampened_u = _compute_dampened_utilities(u, gs, ls)
    # Use numpy for PMF computation (fast)
    dampened_u = dampened_u - dampened_u.max()
    scores = np.exp((dampened_u * eps) / 2.0)
    return scores / scores.sum()


def ld(u: np.ndarray, gs: float, eps: float, ls: np.ndarray) -> int:
    """Select element using local dampening mechanism.

    Args:
        u: Utility scores
        gs: Global sensitivity
        eps: Privacy budget
        ls: Local sensitivity matrix [r, k]

    Returns:
        Selected element index
    """
    # Ensure contiguous arrays only if needed
    if not u.flags["C_CONTIGUOUS"]:
        u = np.ascontiguousarray(u, dtype=np.float64)
    if not ls.flags["C_CONTIGUOUS"]:
        ls = np.ascontiguousarray(ls, dtype=np.float64)
    dampened_u = _compute_dampened_utilities(u, gs, ls)
    # Use Gumbel Max Trick for fast selection (sensitivity=1 for dampened utilities)
    return em(dampened_u, 1.0, eps)


def shifted_ld_pmf(
    u: np.ndarray, gs: float, eps: float, ls: np.ndarray, shift: float | None = None
):
    """Compute PMF using shifted local dampening.

    For monotonically non-decreasing sensitivity functions.

    Args:
        u: Utility scores
        gs: Global sensitivity
        eps: Privacy budget
        ls: Local sensitivity. Either a 2D matrix of shape (R, K) where
            ls[r, k] is per-outcome, or a 1D curve of length K shared
            across all outcomes.
        shift: Shift amount. If None, computed as n * gs + max(u)

    Returns:
        PMF over elements
    """
    if not u.flags["C_CONTIGUOUS"]:
        u = np.ascontiguousarray(u, dtype=np.float64)
    if not ls.flags["C_CONTIGUOUS"]:
        ls = np.ascontiguousarray(ls, dtype=np.float64)
    if shift is None:
        shift = len(u) * gs + u.max()

    if ls.ndim == 1:
        dampened_u = _compute_shifted_dampened_utilities_shared(u, gs, ls, shift)
    else:
        dampened_u = _compute_shifted_dampened_utilities(u, gs, ls, shift)
    # Use numpy for PMF computation (fast)
    dampened_u = dampened_u - dampened_u.max()
    scores = np.exp((dampened_u * eps) / 2.0)
    return scores / scores.sum()


def shifted_ld(
    u: np.ndarray, gs: float, eps: float, ls: np.ndarray, shift: float | None = None
) -> int:
    """Select element using shifted local dampening mechanism.

    For monotonically non-decreasing sensitivity functions.

    Args:
        u: Utility scores
        gs: Global sensitivity
        eps: Privacy budget
        ls: Local sensitivity matrix [r, k]
        shift: Shift amount. If None, computed as n * gs + max(u)

    Returns:
        Selected element index
    """
    # Ensure contiguous arrays only if needed
    if not u.flags["C_CONTIGUOUS"]:
        u = np.ascontiguousarray(u, dtype=np.float64)
    if not ls.flags["C_CONTIGUOUS"]:
        ls = np.ascontiguousarray(ls, dtype=np.float64)
    if shift is None:
        shift = len(u) * gs + u.max()

    dampened_u = _compute_shifted_dampened_utilities(u, gs, ls, shift)
    # Use Gumbel Max Trick for fast selection (sensitivity=1 for dampened utilities)
    return em(dampened_u, 1.0, eps)


@njit
def dampening_func_from_ls_func(
    u: np.ndarray, r: int, gs: float, ls_func: Callable[[int, int], float]
) -> float:
    """Compute dampened utility using a local sensitivity function.

    Args:
        u: Utility array.
        r: Index of the element to compute dampened utility for.
        gs: Global sensitivity.
        ls_func: Function that receives element index r and distance t,
                 and returns the local sensitivity value.

    Returns:
        Dampened utility value for element r.
    """
    err = 1e-12
    sign = 1
    u_r = u[r]
    if u_r < 0:
        sign = -1
        u_r *= sign

    t = 0
    bt = b_from_ls_func(u_r, t, r, ls_func)
    while True:
        btp = b_from_ls_func(u_r, t + 1, r, ls_func)

        if btp - bt >= gs - err or btp - bt == 0:
            return ((u_r - bt) / gs + t) * sign

        if u_r >= bt and u_r < btp:
            break

        t += 1

    return (((u_r - bt) / (btp - bt)) + t) * sign


@njit
def b_from_ls_func(
    u_r: float, t: int, r: int, ls_func: Callable[[int, int], float]
) -> float:
    """Compute the b function using a local sensitivity function.

    Args:
        u_r: Utility value for element r.
        t: Distance parameter.
        r: Element index.
        ls_func: Function that receives element index r and distance t,
                 and returns the local sensitivity value.

    Returns:
        The b value for the given parameters.
    """
    if t == 0:
        return 0.0

    if t < 0:
        return b_from_ls_func(u_r, -t, r, ls_func)

    # Sum ls_func(r, dist) for dist from t-1 to infinity (or until convergence)
    total = 0.0
    for dist in range(t - 1, t + 1000):  # Reasonable upper bound
        val = ls_func(r, dist)
        if val < 1e-15:  # Converged to zero
            break
        total += val

    return total


def ld_func(
    u: np.ndarray, gs: float, eps: float, ls_func: Callable[[int, int], float]
) -> int:
    """Local Dampening selection using a local sensitivity function.

    This variant receives the local sensitivity as a function instead of a
    pre-computed numpy array, allowing for dynamic or lazy computation of
    local sensitivity values.

    Args:
        u: Utility array.
        gs: Global sensitivity.
        eps: Privacy budget (epsilon).
        ls_func: Function that receives element index r and distance t,
                 and returns the local sensitivity value.

    Returns:
        Index of the selected element.
    """
    u = np.array(u, dtype=np.float64)
    ls_func = njit(ls_func)
    dampened_u = np.array(
        [dampening_func_from_ls_func(u, el, gs, ls_func) for el in range(u.size)]
    )
    return em(dampened_u, 1, eps)
