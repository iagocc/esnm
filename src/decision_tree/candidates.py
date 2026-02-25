from typing import Dict, Optional, Tuple, Union

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _candidates_step_numba(prev_candidates: np.ndarray, tau: int) -> np.ndarray:
    """
    Compute candidates at distance t given candidates at distance t-1.

    This implements lines 6-13 of Algorithm 5:
    - For each (a, b) pair from distance t-1:
      - If a > 0 and b > 0: add (a-1, b-1) [removal of tuple where r_A=j, r_C=c]
      - If a < τ: add (a+1, b) [insertion of tuple where r_A=j, r_C≠c]

    Parameters
    ----------
    prev_candidates : np.ndarray
        Array of shape (n, 2) containing (a, b) pairs from distance t-1
        where a = τ_j^A and b = τ_{j,c}^A
    tau : int
        Total cardinality of the dataset (τ^T)

    Returns
    -------
    np.ndarray
        Array of shape (m, 2) containing unique (a, b) pairs at distance t
    """
    # Use set for deduplication (encode pairs as integers)
    # Encoding: key = a * multiplier + b
    # We use tau + 2 to ensure unique encoding since 0 <= b <= a <= tau
    seen = set()
    multiplier = tau + 2

    for i in range(prev_candidates.shape[0]):
        a = prev_candidates[i, 0]
        b = prev_candidates[i, 1]

        # Operation type 3 (from proof): Remove tuple where r_A = j and r_C = c
        # This decreases both τ_j and τ_{j,c} by 1
        if a > 0 and b > 0:
            key = (a - 1) * multiplier + (b - 1)
            seen.add(key)

        # Operation type 2 (from proof): Insert tuple where r_A = j and r_C ≠ c
        # This increases τ_j by 1, τ_{j,c} stays same
        if a < tau:
            key = (a + 1) * multiplier + b
            seen.add(key)

    # Convert back to array
    n_candidates = len(seen)
    result = np.empty((n_candidates, 2), dtype=np.int64)
    idx = 0
    for key in seen:
        result[idx, 0] = key // multiplier
        result[idx, 1] = key % multiplier
        idx += 1

    return result


@njit(cache=True)
def _compute_f(x: int) -> float:
    """
    Compute f(x) = x * log2((x+1)/x) + log2(x+1)

    f(x) measures the impact of adding a tuple.
    f(0) = 0 by definition.
    """
    if x <= 0:
        return 0.0
    return x * np.log2((x + 1.0) / x) + np.log2(x + 1.0)


@njit(cache=True)
def _compute_g(x: int) -> float:
    """
    Compute g(x) = x * log2((x-1)/x) - log2(x-1)

    g(x) measures the impact of removing a tuple.
    g(0) = g(1) = 0 by definition.
    """
    if x <= 1:
        return 0.0
    return x * np.log2((x - 1.0) / x) - np.log2(x - 1.0)


@njit(cache=True)
def _compute_h(a: int, b: int) -> float:
    """
    Compute h(a, b) = max(f(a) - f(b), g(b) - g(a))

    This function computes the element local sensitivity contribution
    for a candidate pair (a, b) where:
    - a = τ_j^A (count of records where attribute A = j)
    - b = τ_{j,c}^A (count of records where A = j and class C = c)

    Parameters
    ----------
    a : int
        Count τ_j^A
    b : int
        Count τ_{j,c}^A

    Returns
    -------
    float
        The h(a, b) value representing sensitivity contribution
    """
    f_diff = _compute_f(a) - _compute_f(b)
    g_diff = _compute_g(b) - _compute_g(a)
    return max(f_diff, g_diff)


@njit(cache=True)
def compute_h_for_candidates(candidates: np.ndarray) -> np.ndarray:
    """
    Compute h(a, b) for all (a, b) pairs in candidates array.

    Parameters
    ----------
    candidates : np.ndarray
        Array of shape (n, 2) containing (a, b) pairs

    Returns
    -------
    np.ndarray
        Array of shape (n,) containing h values
    """
    n = candidates.shape[0]
    result = np.empty(n, dtype=np.float64)

    for i in range(n):
        result[i] = _compute_h(candidates[i, 0], candidates[i, 1])

    return result


@njit(cache=True)
def compute_max_h_for_candidates(candidates: np.ndarray) -> float:
    """
    Compute max h(a, b) over all (a, b) pairs in candidates array.

    Parameters
    ----------
    candidates : np.ndarray
        Array of shape (n, 2) containing (a, b) pairs

    Returns
    -------
    float
        Maximum h value
    """
    if candidates.shape[0] == 0:
        return 0.0

    max_h = -np.inf
    for i in range(candidates.shape[0]):
        h_val = _compute_h(candidates[i, 0], candidates[i, 1])
        if h_val > max_h:
            max_h = h_val

    return max_h


@njit(cache=True)
def _compute_all_candidates_up_to_t(tau_j: int, tau_jc: int, tau: int, t: int) -> float:
    """
    Compute max h(a,b) over all candidates from distance 0 to t.

    This is a fully JIT-compiled version that computes everything in one go
    without Python-level caching.

    Parameters
    ----------
    tau_j : int
        Initial count τ_j^A
    tau_jc : int
        Initial count τ_{j,c}^A
    tau : int
        Total dataset cardinality
    t : int
        Maximum distance

    Returns
    -------
    float
        Maximum h(a, b) value over all candidates up to distance t
    """
    # Initialize with distance 0
    current = np.array([[tau_j, tau_jc]], dtype=np.int64)

    # Track max h
    max_h = _compute_h(tau_j, tau_jc)

    # Iterate through distances
    for dist in range(1, t + 1):
        # Compute next candidates
        next_candidates = _candidates_step_numba(current, tau)

        # Update max h
        for i in range(next_candidates.shape[0]):
            h_val = _compute_h(next_candidates[i, 0], next_candidates[i, 1])
            if h_val > max_h:
                max_h = h_val

        current = next_candidates

    return max_h


# =============================================================================
# Main Class with Memoization
# =============================================================================


class CandidatesAlgorithm:
    """
    Implementation of Algorithm 5: Candidates Algorithm.

    This class computes candidate pairs (τ_j^A, τ_{j,c}^A) at distance t
    from the original dataset using memoization to efficiently compute
    candidates at increasing distances.

    The algorithm is used for computing element local sensitivity for
    Information Gain in private decision tree induction.

    Attributes
    ----------
    tau_j : int
        Count of records where attribute A takes value j (τ_j^A)
    tau_jc : int
        Count of records where attribute A = j AND class C = c (τ_{j,c}^A)
    tau : int
        Total cardinality of the dataset (τ^T)

    Examples
    --------
    >>> # Initialize with counts
    >>> algo = CandidatesAlgorithm(tau_j=100, tau_jc=30, tau=500)
    >>>
    >>> # Get candidates at distance 2
    >>> candidates = algo.get_candidates(t=2)
    >>> print(f"Number of candidates at distance 2: {len(candidates)}")
    >>>
    >>> # Compute element local sensitivity at distance 5
    >>> sensitivity = algo.compute_local_sensitivity_at_t(t=5)
    >>> print(f"Element local sensitivity: {sensitivity:.4f}")
    """

    def __init__(self, tau_j: int, tau_jc: int, tau: int):
        """
        Initialize the Candidates algorithm.

        Parameters
        ----------
        tau_j : int
            Count of records where attribute A takes value j (τ_j^A)
        tau_jc : int
            Count of records where attribute A = j AND class C = c (τ_{j,c}^A)
        tau : int
            Total cardinality of the dataset (τ^T)

        Raises
        ------
        ValueError
            If tau_jc > tau_j or tau_j > tau
        """
        # Validate inputs
        if tau_jc > tau_j:
            raise ValueError(
                f"tau_jc ({tau_jc}) cannot be greater than tau_j ({tau_j})"
            )
        if tau_j > tau:
            raise ValueError(f"tau_j ({tau_j}) cannot be greater than tau ({tau})")
        if tau_j < 0 or tau_jc < 0 or tau < 0:
            raise ValueError("All counts must be non-negative")

        self.tau_j = int(tau_j)
        self.tau_jc = int(tau_jc)
        self.tau = int(tau)

        # Cache for memoization: maps distance t to candidates array
        self._cache: Dict[int, np.ndarray] = {}

        # Initialize distance 0
        self._cache[0] = np.array([[self.tau_j, self.tau_jc]], dtype=np.int64)

        # Cache for h values at each distance
        self._h_cache: Dict[int, float] = {}
        self._h_cache[0] = _compute_h(self.tau_j, self.tau_jc)

        # Track maximum cached distance
        self._max_cached_t = 0

    def get_candidates(self, t: int) -> np.ndarray:
        """
        Get candidates at distance t using memoization.

        The algorithm caches results for distance t-1 to efficiently
        compute distance t, as specified in the paper.

        Parameters
        ----------
        t : int
            Distance from original dataset (must be >= 0)

        Returns
        -------
        np.ndarray
            Array of shape (n, 2) containing (a, b) pairs where:
            - a = τ_j^{A,T'} (count of records where A = j in T')
            - b = τ_{j,c}^{A,T'} (count where A = j and C = c in T')
            for datasets T' at distance t from original T

        Raises
        ------
        ValueError
            If t is negative
        """
        if t < 0:
            raise ValueError("Distance t must be non-negative")

        # Return cached result if available
        if t in self._cache:
            return self._cache[t].copy()

        # Compute iteratively from last cached distance
        for dist in range(self._max_cached_t + 1, t + 1):
            prev_candidates = self._cache[dist - 1]
            new_candidates = _candidates_step_numba(prev_candidates, self.tau)
            self._cache[dist] = new_candidates

            # Also compute and cache max h for this distance
            self._h_cache[dist] = compute_max_h_for_candidates(new_candidates)

        self._max_cached_t = max(self._max_cached_t, t)

        return self._cache[t].copy()

    def compute_local_sensitivity_at_t(self, t: int) -> float:
        """
        Compute element local sensitivity at distance t.

        This computes:
        max_{0 <= t' <= t} max_{(a,b) in Candidates(T, t', j, c)} h(a, b)

        Parameters
        ----------
        t : int
            Maximum distance for local sensitivity computation

        Returns
        -------
        float
            Element local sensitivity at distance t
        """
        if t < 0:
            raise ValueError("Distance t must be non-negative")

        # Ensure all candidates up to t are computed
        if t > self._max_cached_t:
            self.get_candidates(t)

        # Find maximum h across all distances 0 to t
        max_h = 0.0
        for t_prime in range(t + 1):
            if t_prime in self._h_cache:
                h_val = self._h_cache[t_prime]
            else:
                candidates = self._cache[t_prime]
                h_val = compute_max_h_for_candidates(candidates)
                self._h_cache[t_prime] = h_val

            if h_val > max_h:
                max_h = h_val

        return max_h

    def get_all_h_values(self, t: int) -> np.ndarray:
        """
        Get h(a, b) values for all candidates at distance t.

        Parameters
        ----------
        t : int
            Distance

        Returns
        -------
        np.ndarray
            Array of h values corresponding to each candidate pair
        """
        candidates = self.get_candidates(t)
        return compute_h_for_candidates(candidates)

    def clear_cache(self, keep_base: bool = True):
        """
        Clear the memoization cache.

        Parameters
        ----------
        keep_base : bool, default=True
            If True, keep distance 0 in cache
        """
        if keep_base:
            base = self._cache[0].copy()
            base_h = self._h_cache[0]
            self._cache = {0: base}
            self._h_cache = {0: base_h}
            self._max_cached_t = 0
        else:
            self._cache = {}
            self._h_cache = {}
            self._max_cached_t = -1

    @property
    def cache_size(self) -> int:
        """Number of cached distance levels."""
        return len(self._cache)

    def __repr__(self) -> str:
        return (
            f"CandidatesAlgorithm(tau_j={self.tau_j}, tau_jc={self.tau_jc}, "
            f"tau={self.tau}, cached_distances={self.cache_size})"
        )


# =============================================================================
# High-level functions for computing element local sensitivity
# =============================================================================


def compute_element_local_sensitivity_IG(
    dataset: np.ndarray,
    attribute_col: int,
    class_col: int,
    t: int,
    attribute_value: Optional[Union[int, np.integer]] = None,
    class_value: Optional[Union[int, np.integer]] = None,
    return_details: bool = False,
) -> Union[float, Tuple[float, Dict]]:
    """
    Compute element local sensitivity at distance t for Information Gain.

    This implements Lemma 15 from the paper:
    LS^{IG}(T, t, A) = max_{j∈A, c∈C, 0≤t'≤t} max_{(a,b)∈Candidates(T,t',j,c)} h(a,b)

    Parameters
    ----------
    dataset : np.ndarray
        Dataset with shape (n_samples, n_features)
    attribute_col : int
        Index of the attribute column A
    class_col : int
        Index of the class column C
    t : int
        Distance for local sensitivity computation
    attribute_value : int, optional
        If provided, compute only for this attribute value j
    class_value : int, optional
        If provided, compute only for this class value c
    return_details : bool, default=False
        If True, return detailed information about the computation

    Returns
    -------
    float or Tuple[float, Dict]
        Element local sensitivity at distance t.
        If return_details=True, also returns a dict with:
        - 'max_j': attribute value achieving maximum
        - 'max_c': class value achieving maximum
        - 'sensitivities': dict of (j, c) -> sensitivity

    Examples
    --------
    >>> # Create sample dataset
    >>> dataset = np.array([
    ...     [0, 1, 0],  # attribute, feature, class
    ...     [0, 0, 0],
    ...     [1, 1, 1],
    ...     [1, 0, 1],
    ...     [0, 1, 1],
    ... ])
    >>>
    >>> # Compute element local sensitivity for attribute column 0
    >>> sensitivity = compute_element_local_sensitivity_IG(
    ...     dataset, attribute_col=0, class_col=2, t=3
    ... )
    >>> print(f"Element local sensitivity: {sensitivity:.4f}")
    """
    A = dataset[:, attribute_col]
    C = dataset[:, class_col]
    tau = len(dataset)

    # Get unique values
    if attribute_value is not None:
        unique_j = [attribute_value]
    else:
        unique_j = np.unique(A)

    if class_value is not None:
        unique_c = [class_value]
    else:
        unique_c = np.unique(C)

    max_sensitivity = 0.0
    max_j = None
    max_c = None
    sensitivities = {}

    for j in unique_j:
        tau_j = int(np.sum(A == j))

        for c in unique_c:
            tau_jc = int(np.sum((A == j) & (C == c)))

            # Create candidates algorithm instance
            candidates_algo = CandidatesAlgorithm(tau_j, tau_jc, tau)

            # Compute local sensitivity for this (j, c) pair
            sensitivity = candidates_algo.compute_local_sensitivity_at_t(t)
            sensitivities[(j, c)] = sensitivity

            if sensitivity > max_sensitivity:
                max_sensitivity = sensitivity
                max_j = j
                max_c = c

    if return_details:
        details = {
            "max_j": max_j,
            "max_c": max_c,
            "sensitivities": sensitivities,
            "tau": tau,
            "unique_attributes": list(unique_j),
            "unique_classes": list(unique_c),
        }
        return max_sensitivity, details

    return max_sensitivity


def compute_element_local_sensitivity_IG_fast(
    tau_j: int, tau_jc: int, tau: int, t: int
) -> float:
    """
    Fast computation of element local sensitivity without caching overhead.

    This is a fully JIT-compiled version optimized for single computations
    where memoization across multiple t values is not needed.

    Parameters
    ----------
    tau_j : int
        Count τ_j^A (records where attribute A = j)
    tau_jc : int
        Count τ_{j,c}^A (records where A = j and C = c)
    tau : int
        Total dataset cardinality
    t : int
        Distance for local sensitivity computation

    Returns
    -------
    float
        Element local sensitivity at distance t
    """
    return _compute_all_candidates_up_to_t(tau_j, tau_jc, tau, t)


# =============================================================================
# Element Local Sensitivity Matrix Computation
# =============================================================================


@njit(cache=True)
def _candidates_step_fast(current_a: np.ndarray, current_b: np.ndarray, tau: int):
    """
    Generate next candidates from current candidates using numpy operations.
    Returns deduplicated (a, b) pairs as separate arrays.
    """
    n = len(current_a)
    multiplier = tau + 2

    # Pre-allocate arrays for potential candidates (max 2 per existing)
    max_new = 2 * n
    new_a = np.empty(max_new, dtype=np.int64)
    new_b = np.empty(max_new, dtype=np.int64)

    # Generate all potential candidates
    idx = 0
    for i in range(n):
        a = current_a[i]
        b = current_b[i]

        # Operation: remove tuple (a-1, b-1)
        if a > 0 and b > 0:
            new_a[idx] = a - 1
            new_b[idx] = b - 1
            idx += 1

        # Operation: add tuple with different class (a+1, b)
        if a < tau:
            new_a[idx] = a + 1
            new_b[idx] = b
            idx += 1

    if idx == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    # Trim to actual size
    new_a = new_a[:idx]
    new_b = new_b[:idx]

    # Deduplicate using encoding and sorting
    keys = new_a * multiplier + new_b
    sorted_indices = np.argsort(keys)
    keys_sorted = keys[sorted_indices]

    # Count unique elements
    n_unique = 1
    for i in range(1, len(keys_sorted)):
        if keys_sorted[i] != keys_sorted[i - 1]:
            n_unique += 1

    # Extract unique values
    unique_a = np.empty(n_unique, dtype=np.int64)
    unique_b = np.empty(n_unique, dtype=np.int64)

    unique_a[0] = new_a[sorted_indices[0]]
    unique_b[0] = new_b[sorted_indices[0]]

    unique_idx = 1
    for i in range(1, len(keys_sorted)):
        if keys_sorted[i] != keys_sorted[i - 1]:
            orig_idx = sorted_indices[i]
            unique_a[unique_idx] = new_a[orig_idx]
            unique_b[unique_idx] = new_b[orig_idx]
            unique_idx += 1

    return unique_a, unique_b


@njit(cache=True)
def _compute_sensitivity_for_single_pair(
    tau_j: int, tau_jc: int, tau: int, T: int
) -> np.ndarray:
    """
    Compute sensitivity trajectory for a single (tau_j, tau_jc) pair.
    Returns array of shape (T,) with running max sensitivity at each distance.
    """
    result = np.zeros(T, dtype=np.float64)

    # Initialize candidates at distance 0
    current_a = np.array([tau_j], dtype=np.int64)
    current_b = np.array([tau_jc], dtype=np.int64)

    # Distance 0
    running_max = _compute_h(tau_j, tau_jc)
    result[0] = running_max

    # Compute for distances 1 to T-1
    for t in range(1, T):
        # Generate new candidates using optimized function
        current_a, current_b = _candidates_step_fast(current_a, current_b, tau)

        if len(current_a) == 0:
            # No more candidates, fill rest with running max
            for t2 in range(t, T):
                result[t2] = running_max
            break

        # Compute max h for new candidates
        for i in range(len(current_a)):
            h_val = _compute_h(current_a[i], current_b[i])
            if h_val > running_max:
                running_max = h_val

        result[t] = running_max

    return result


@njit(cache=True)
def _compute_sensitivity_matrix_for_attribute(
    tau_j_values: np.ndarray, tau_jc_values: np.ndarray, tau: int, T: int
) -> np.ndarray:
    """
    Compute element local sensitivity for all distances 0 to T-1
    for a single attribute with multiple (j, c) pairs.

    Parameters
    ----------
    tau_j_values : np.ndarray
        Array of tau_j values for each (j, c) pair
    tau_jc_values : np.ndarray
        Array of tau_jc values for each (j, c) pair
    tau : int
        Total dataset size
    T : int
        Maximum distance (typically dataset size)

    Returns
    -------
    np.ndarray
        Array of shape (T,) with max sensitivity at each distance
    """
    n_pairs = len(tau_j_values)

    if n_pairs == 0:
        return np.zeros(T, dtype=np.float64)

    # Compute sensitivity for each pair sequentially
    # Store results in a 2D array (n_pairs x T)
    all_results = np.zeros((n_pairs, T), dtype=np.float64)

    for pair_idx in range(n_pairs):
        tau_j = tau_j_values[pair_idx]
        tau_jc = tau_jc_values[pair_idx]
        all_results[pair_idx, :] = _compute_sensitivity_for_single_pair(
            tau_j, tau_jc, tau, T
        )

    # Take element-wise max across all pairs
    result = np.zeros(T, dtype=np.float64)
    for t in range(T):
        max_val = all_results[0, t]
        for pair_idx in range(1, n_pairs):
            if all_results[pair_idx, t] > max_val:
                max_val = all_results[pair_idx, t]
        result[t] = max_val

    return result


@njit(cache=True)
def _compute_tau_counts(
    attr_col_data: np.ndarray,
    class_col_data: np.ndarray,
    unique_attrs: np.ndarray,
    unique_classes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute tau_j and tau_jc counts for all (j, c) pairs.

    Parameters
    ----------
    attr_col_data : np.ndarray
        Attribute column values
    class_col_data : np.ndarray
        Class column values
    unique_attrs : np.ndarray
        Unique attribute values
    unique_classes : np.ndarray
        Unique class values

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        tau_j_values and tau_jc_values arrays
    """
    n_attrs = len(unique_attrs)
    n_classes = len(unique_classes)
    n_pairs = n_attrs * n_classes
    n_samples = len(attr_col_data)

    tau_j_values = np.zeros(n_pairs, dtype=np.int64)
    tau_jc_values = np.zeros(n_pairs, dtype=np.int64)

    # Count occurrences
    pair_idx = 0
    for j_idx in range(n_attrs):
        j = unique_attrs[j_idx]
        tau_j = 0

        # Count tau_j
        for i in range(n_samples):
            if attr_col_data[i] == j:
                tau_j += 1

        for c_idx in range(n_classes):
            c = unique_classes[c_idx]
            tau_jc = 0

            # Count tau_jc
            for i in range(n_samples):
                if attr_col_data[i] == j and class_col_data[i] == c:
                    tau_jc += 1

            tau_j_values[pair_idx] = tau_j
            tau_jc_values[pair_idx] = tau_jc
            pair_idx += 1

    return tau_j_values, tau_jc_values


@njit(cache=True, parallel=True)
def _element_local_sensitivity_core(
    x: np.ndarray,
    R: np.ndarray,
    class_col: int,
    T: int,
) -> np.ndarray:
    """
    Core numba-compiled function for computing element local sensitivity matrix.

    Parameters
    ----------
    x : np.ndarray
        Dataset with shape (n_samples, n_features)
    R : np.ndarray
        Array of attribute column indices
    class_col : int
        Index of the class column (must be non-negative)
    T : int
        Maximum distance to compute

    Returns
    -------
    np.ndarray
        Matrix of shape (|R|, T) with sensitivity values
    """
    n_samples = x.shape[0]
    tau = n_samples
    n_R = len(R)

    # Get class column values
    C = x[:, class_col].copy()

    # Get unique classes (need to do this outside prange)
    unique_classes = np.unique(C)

    # Initialize result matrix
    result = np.zeros((n_R, T), dtype=np.float64)

    # Process each attribute in R in parallel
    for r_idx in prange(n_R):
        attr_col = R[r_idx]
        A = x[:, attr_col].copy()
        unique_attrs = np.unique(A)

        # Compute tau counts
        tau_j_values, tau_jc_values = _compute_tau_counts(
            A, C, unique_attrs, unique_classes
        )

        # Compute sensitivity for all distances (non-parallel inner function)
        result[r_idx, :] = _compute_sensitivity_matrix_for_attribute(
            tau_j_values, tau_jc_values, tau, T
        )

    return result


def element_local_sensitivity_at(
    x: np.ndarray,
    R: np.ndarray,
    class_col: int = -1,
    max_distance: Optional[int] = None,
) -> np.ndarray:
    """
    Compute element local sensitivity matrix for all elements in domain R
    and all distances t from 0 to T.

    This implements the element local sensitivity computation from the paper:
    LS^u(x, t, r) for all r ∈ R and t ∈ {0, 1, ..., T-1}

    For the Information Gain application in decision tree induction,
    R represents the set of attribute column indices.

    Parameters
    ----------
    x : np.ndarray
        Dataset with shape (n_samples, n_features).
        Each row is a record, each column is an attribute.
    R : np.ndarray
        Domain of possible outputs. For decision tree induction,
        this is an array of attribute column indices to evaluate.
        Example: np.array([0, 1, 2]) for first 3 attribute columns.
    class_col : int, default=-1
        Index of the class column. Default is -1 (last column).
    max_distance : int, optional
        Maximum distance to compute. If None, uses len(x).

    Returns
    -------
    np.ndarray
        Matrix of shape (|R|, T) where:
        - |R| is the number of elements in the domain R
        - T is max_distance (or len(x) if not specified)
        - Entry [r_idx, t] contains LS^{IG}(x, t, R[r_idx])

    Examples
    --------
    >>> # Create sample dataset with 3 attributes and 1 class column
    >>> np.random.seed(42)
    >>> n_samples = 100
    >>> dataset = np.column_stack([
    ...     np.random.randint(0, 3, n_samples),  # Attribute 0
    ...     np.random.randint(0, 4, n_samples),  # Attribute 1
    ...     np.random.randint(0, 2, n_samples),  # Attribute 2
    ...     np.random.randint(0, 2, n_samples),  # Class column
    ... ])
    >>>
    >>> # Define domain R as attribute indices [0, 1, 2]
    >>> R = np.array([0, 1, 2])
    >>>
    >>> # Compute element local sensitivity matrix
    >>> sensitivity_matrix = element_local_sensitivity_at(dataset, R, class_col=3)
    >>> print(f"Shape: {sensitivity_matrix.shape}")  # (3, 100)
    >>> print(f"LS for attribute 0 at t=5: {sensitivity_matrix[0, 5]:.4f}")

    Notes
    -----
    The element local sensitivity at distance t for attribute A is:

    LS^{IG}(x, t, A) = max_{j∈A, c∈C, 0≤t'≤t} max_{(a,b)∈Candidates(x,t',j,c)} h(a,b)

    where:
    - j iterates over unique values of attribute A
    - c iterates over unique class values
    - h(a, b) = max(f(a) - f(b), g(b) - g(a))
    - f(x) = x·log₂((x+1)/x) + log₂(x+1)
    - g(x) = x·log₂((x-1)/x) - log₂(x-1)
    """
    # Validate inputs
    if x.ndim != 2:
        raise ValueError(f"Dataset x must be 2D, got shape {x.shape}")

    n_samples, n_features = x.shape
    tau = n_samples

    # Handle class column index
    if class_col < 0:
        class_col = n_features + class_col

    # Validate R
    R = np.asarray(R, dtype=np.int64)
    if R.ndim != 1:
        raise ValueError(f"Domain R must be 1D array, got shape {R.shape}")

    for r in R:
        if r < 0 or r >= n_features:
            raise ValueError(
                f"Invalid attribute index {r} in R. Must be in [0, {n_features - 1}]"
            )
        if r == class_col:
            raise ValueError(f"Domain R contains class column index {class_col}")

    # Determine T (max distance)
    T = max_distance if max_distance is not None else tau
    T = min(T, tau)  # Can't have distance > dataset size

    # Ensure x is contiguous and proper dtype for numba
    x_arr = np.ascontiguousarray(x, dtype=np.int64)

    # Call optimized core function
    return _element_local_sensitivity_core(x_arr, R, class_col, T)
