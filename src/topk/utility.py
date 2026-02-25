import numpy as np
import numba as nb


@nb.njit(cache=True)
def utility(counts: np.ndarray, tau: float) -> np.ndarray:
    """Compute the bounded gap utility for every item.

    u_tau(x, r) = -min(tau, c_{(1)} - c_r)

    Parameters
    ----------
    counts : 1-D array of item counts (integers or floats).
    tau    : clipping threshold (> 0).

    Returns
    -------
    utilities : 1-D array of the same length, values in [-tau, 0].
    """
    c_max = counts.max()
    n = counts.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        gap = c_max - counts[i]
        out[i] = -min(tau, gap)
    return out


@nb.njit(cache=True)
def local_sensitivity(counts: np.ndarray, tau: float, m: int) -> np.ndarray:
    """Element-wise local sensitivity of u_tau at distance m.

    LS^{(m)}(x, r) = 1  if  c_{(1)} - c_r <= tau + m
                      0  otherwise

    Parameters
    ----------
    counts : 1-D array of item counts.
    tau    : clipping threshold (> 0).
    m      : Hamming distance budget (non-negative integer).

    Returns
    -------
    ls : 1-D array of local sensitivities (0.0 or 1.0).
    """
    c_max = counts.max()
    threshold = tau + m
    n = counts.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        gap = c_max - counts[i]
        out[i] = 1.0 if gap <= threshold else 0.0
    return out
