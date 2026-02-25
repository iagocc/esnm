"""Sensitivity functions for EBC metric.

Based on Lemma 12 and Definition 20 from the Local Dampening paper.
"""

from typing import Callable

import numpy as np
from numba import njit


def global_sensitivity_ebc(max_degree: int) -> float:
    """Compute global sensitivity for EBC.

    From Lemma 12:
    ΔEBC = max(Δ(G)*(Δ(G)-1)/4, Δ(G))

    Args:
        max_degree: Maximum degree in the graph (Δ(G)).

    Returns:
        Global sensitivity for EBC.
    """
    return max(max_degree * (max_degree - 1) / 4, max_degree)


@njit(cache=True)
def element_local_sensitivity_ebc(degree: int, t: int) -> float:
    """Compute element local sensitivity for EBC at distance t.

    From Definition 20:
    δ^EBC(G, t, v) = max((d(v)+t)*(d(v)+t-1)/4, d(v)+t)

    Args:
        degree: Degree of the node (d(v)).
        t: Distance parameter.

    Returns:
        Element local sensitivity for the node at distance t.
    """
    d_t = degree + t
    return max(d_t * (d_t - 1) / 4, d_t)


@njit(cache=True)
def build_ls_matrix(degrees: np.ndarray, max_distance: int) -> np.ndarray:
    """Build element local sensitivity matrix.

    Args:
        degrees: Array of node degrees.
        max_distance: Maximum distance to compute.

    Returns:
        Matrix of shape (n_nodes, max_distance+1) where entry [i, t]
        is the element local sensitivity for node i at distance t.
    """
    n_nodes = len(degrees)
    ls_matrix = np.zeros((n_nodes, max_distance + 1), dtype=np.float64)

    for i in range(n_nodes):
        for t in range(max_distance + 1):
            ls_matrix[i, t] = element_local_sensitivity_ebc(degrees[i], t)

    return ls_matrix


def smooth_sensitivity_ebc(degree: int, t: float) -> float:
    """Compute smooth sensitivity for EBC.

    S*(f, t, v) = max_k (e^{-t*k} * δ^EBC(G, k, v))

    For EBC, the local sensitivity is monotonically non-decreasing in k,
    so the smooth sensitivity is achieved at k=0 or computed as:
    S*(f, t, v) = max_k (e^{-t*k} * element_local_sensitivity_ebc(degree, k))

    Args:
        degree: Degree of the node.
        t: Smoothing parameter.

    Returns:
        Smooth sensitivity for the node.
    """
    if t <= 0:
        # Return the local sensitivity at distance 0 (max possible)
        return element_local_sensitivity_ebc(degree, 1000)

    # For non-decreasing local sensitivity, smooth sensitivity is bounded
    # Search for the maximum
    max_ss = 0.0
    for k in range(1001):
        decay = np.exp(-t * k)
        ls = element_local_sensitivity_ebc(degree, k)
        ss = decay * ls
        if ss > max_ss:
            max_ss = ss
        # Early termination if decay is too small
        if decay < 1e-15:
            break

    return max_ss


def create_smooth_sensitivity_func(
    degrees: np.ndarray,
) -> Callable[[int, float], float]:
    """Create a smooth sensitivity function for use with ESNM.

    Args:
        degrees: Array of node degrees.

    Returns:
        Function (r, t) -> smooth sensitivity for element r at parameter t.
    """

    def ss_func(r: int, t: float) -> float:
        return smooth_sensitivity_ebc(degrees[r], t)

    return ss_func
