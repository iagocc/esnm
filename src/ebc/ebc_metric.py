"""Egocentric Betweenness Centrality (EBC) computation.

Based on Definition 19 from the Local Dampening paper (VLDB Journal).

EBC(c) = Σ_{u,v∈N_c|u≠v} p_uv(c) / q_uv(c)

where:
- N_c = neighbors of c
- q_uv = number of geodesic paths between u and v in G[N_c ∪ {c}]
- p_uv = number of those paths containing c
"""

import numpy as np
from numba import njit, prange
from numba.typed import List


@njit(cache=True)
def _compute_ebc_single(center: int, neighbors: np.ndarray, all_neighbors: List) -> float:
    """Compute EBC for a single node.

    For each pair of neighbors (u, v) of the center node c:
    - If u and v are directly connected: contribution = 0
      (geodesic path u-v doesn't go through c)
    - If u and v are not connected: contribution = 1
      (only path u-c-v, and c is on it)

    Args:
        center: The center node index.
        neighbors: Array of neighbor indices for the center node.
        all_neighbors: Typed list where all_neighbors[i] is array of neighbors of node i.

    Returns:
        EBC score for the center node.
    """
    n_neighbors = len(neighbors)
    if n_neighbors <= 1:
        return 0.0

    ebc = 0.0

    # For each pair of neighbors
    for i in range(n_neighbors):
        u = neighbors[i]
        u_neighbors = all_neighbors[u]

        for j in range(i + 1, n_neighbors):
            v = neighbors[j]

            # Check if u and v are directly connected
            # Binary search since neighbors are sorted
            connected = False
            lo, hi = 0, len(u_neighbors)
            while lo < hi:
                mid = (lo + hi) // 2
                if u_neighbors[mid] == v:
                    connected = True
                    break
                elif u_neighbors[mid] < v:
                    lo = mid + 1
                else:
                    hi = mid

            if not connected:
                # u and v only connected through c, so p_uv/q_uv = 1/1 = 1
                ebc += 1.0
            # If connected, there's a direct path u-v, so contribution = 0

    return ebc


@njit(cache=True, parallel=True)
def compute_ebc_all(neighbors_list: List, n_nodes: int) -> np.ndarray:
    """Compute EBC for all nodes using numba parallelization.

    Args:
        neighbors_list: Typed list where neighbors_list[i] is array of neighbors of node i.
        n_nodes: Total number of nodes.

    Returns:
        Array of EBC scores for all nodes.
    """
    ebc_scores = np.zeros(n_nodes, dtype=np.float64)

    for c in prange(n_nodes):
        neighbors = neighbors_list[c]
        ebc_scores[c] = _compute_ebc_single(c, neighbors, neighbors_list)

    return ebc_scores


@njit(cache=True)
def compute_ebc_single(center: int, neighbors_list: List) -> float:
    """Compute EBC for a single node (non-parallel version).

    Args:
        center: The center node index.
        neighbors_list: Typed list where neighbors_list[i] is array of neighbors of node i.

    Returns:
        EBC score for the center node.
    """
    neighbors = neighbors_list[center]
    return _compute_ebc_single(center, neighbors, neighbors_list)
