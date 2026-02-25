"""Graph loading and utility functions for EBC computation."""

from pathlib import Path

import numpy as np
from numba.typed import List


def load_edge_list(filepath: Path | str) -> tuple[dict[int, set], int, int]:
    """Load graph from edge list file.

    Args:
        filepath: Path to edge list file (tab-separated: src dst).

    Returns:
        Tuple of (adjacency_list, n_nodes, n_edges) where adjacency_list
        maps node IDs to sets of neighbor IDs.
    """
    filepath = Path(filepath)

    adj_list: dict[int, set] = {}
    n_edges = 0

    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            src, dst = int(parts[0]), int(parts[1])

            if src not in adj_list:
                adj_list[src] = set()
            if dst not in adj_list:
                adj_list[dst] = set()

            # Add edges in both directions (undirected graph)
            if dst not in adj_list[src]:
                adj_list[src].add(dst)
                adj_list[dst].add(src)
                n_edges += 1

    n_nodes = len(adj_list)
    return adj_list, n_nodes, n_edges


def get_degrees(adj_list: dict[int, set]) -> np.ndarray:
    """Get degree array for all nodes.

    Args:
        adj_list: Adjacency list mapping node IDs to neighbor sets.

    Returns:
        Array of degrees indexed by node ID.
    """
    if not adj_list:
        return np.array([], dtype=np.int64)

    max_node = max(adj_list.keys())
    degrees = np.zeros(max_node + 1, dtype=np.int64)

    for node, neighbors in adj_list.items():
        degrees[node] = len(neighbors)

    return degrees


def get_max_degree(adj_list: dict[int, set]) -> int:
    """Get maximum degree in the graph.

    Args:
        adj_list: Adjacency list mapping node IDs to neighbor sets.

    Returns:
        Maximum degree (Δ(G)).
    """
    if not adj_list:
        return 0
    return max(len(neighbors) for neighbors in adj_list.values())


def get_neighbors(adj_list: dict[int, set], node: int) -> set:
    """Get neighbors of a node.

    Args:
        adj_list: Adjacency list mapping node IDs to neighbor sets.
        node: Node ID.

    Returns:
        Set of neighbor node IDs.
    """
    return adj_list.get(node, set())


def adj_list_to_numba(adj_list: dict[int, set]) -> tuple[List, int]:
    """Convert adjacency list to numba-compatible format.

    Args:
        adj_list: Adjacency list mapping node IDs to neighbor sets.

    Returns:
        Tuple of (numba typed list of neighbor arrays, number of nodes).
    """
    if not adj_list:
        return List(), 0

    max_node = max(adj_list.keys())
    n_nodes = max_node + 1

    # Create numba typed list of numpy arrays
    neighbors_list = List()
    for i in range(n_nodes):
        if i in adj_list:
            neighbors_list.append(np.array(sorted(adj_list[i]), dtype=np.int64))
        else:
            neighbors_list.append(np.array([], dtype=np.int64))

    return neighbors_list, n_nodes
