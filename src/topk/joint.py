import numba as nb
import numpy as np


def make_diff_matrix(item_counts, k):
    """Makes diff matrix where diff_matrix[i,j] = c_i - c_j + uniquifying term.

    Args:
        item_counts: Array of item counts, sorted in decreasing order.
        k: Number of top counts desired.

    Returns:
        k x d matrix diff_matrix where diff_matrix[i,j] = c_i - c_j + (d(k-i-1) + j
        + 1) / (2dk). diff_matrix is strictly increasing along rows and strictly
        decreasing down columns.
    """
    d = len(item_counts)
    base_along_row = np.arange(1, d + 1)
    base_down_col = np.arange(k - 1, -1, -1) * d
    uniquifying_terms = (
        base_along_row[np.newaxis, :] + base_down_col[:, np.newaxis]
    ) / (2 * d * k)
    return (
        item_counts[:k, np.newaxis] - item_counts[np.newaxis, :]
    ) + uniquifying_terms


def get_diffs_to_positions(diff_matrix):
    """Computes array a where diff_matrix[a[0][i], a[1][i]] = sorted_diffs[i].

    Args:
        diff_matrix: Matrix of distinct count differences.

    Returns:
        Array a where diff_matrix[a[0][i], a[1][i]] = sorted_diffs[i], where
        sorted_diffs contains all entries of diff_matrix in increasing order.
    """
    return np.unravel_index(
        np.argsort(diff_matrix, axis=None), diff_matrix.shape
    )


def compute_log_diff_counts(diff_matrix, diffs_to_positions):
    """Computes array of log(sequence count) for each diff in diff_matrix.

    Uses Lemma 3.7 from the paper to compute the number of sequences where
    the largest count difference equals each possible diff value.

    Args:
        diff_matrix: Matrix of distinct count differences.
        diffs_to_positions: Tuple of (row_indices, col_indices) mapping sorted
            diffs to their positions in diff_matrix.

    Returns:
        Array log_counts where log_counts[i] = log(# of sequences where largest
        count difference is sorted_diffs[i]).

    Raises:
        RuntimeError: ns vector never filled.
    """
    k, d = diff_matrix.shape
    num_diffs = d * k
    log_diff_counts = np.empty(num_diffs)
    log_ns = np.empty(k)
    indices_filled = set()
    last_diff_idx_processed = -1

    # For diff at position (i, j): update = log(j + 1 - i)
    # This represents the number of valid candidates for row i at this diff level
    with np.errstate(divide="ignore", invalid="ignore"):
        updates = np.log(
            (diffs_to_positions[1] + 1).astype(np.float64)
            - diffs_to_positions[0].astype(np.float64)
        )

    # Phase 1: Find the first diff where all rows have at least one candidate
    for diff_idx, (i, u) in enumerate(zip(diffs_to_positions[0], updates)):
        if np.isnan(u):
            continue
        log_ns[i] = u
        indices_filled.add(i)
        if len(indices_filled) == k:
            last_diff_idx_processed = diff_idx
            break

    if last_diff_idx_processed == -1:
        raise RuntimeError("ns vector never filled")

    # All diffs before this point have zero valid sequences
    log_diff_counts[:last_diff_idx_processed] = -np.inf

    # Phase 2: Compute counts for remaining diffs using Lemma 3.7
    log_ns_sum = np.sum(log_ns)
    for diff_idx in range(last_diff_idx_processed, num_diffs):
        i = diffs_to_positions[0][diff_idx]
        u = updates[diff_idx]

        # The count for this diff excludes the current row's contribution
        log_ns_sum -= log_ns[i]
        log_diff_counts[diff_idx] = log_ns_sum

        # Update the count for this row
        log_ns[i] = u
        log_ns_sum += log_ns[i]

    return log_diff_counts


@nb.njit
def _racing_sample(log_terms):
    """Numerically stable sampling from exponential mechanism.

    Uses the "racing" algorithm from "Duff: A Dataset-Distance-Based
    Utility Function Family for the Exponential Mechanism".

    Args:
        log_terms: Array of terms of form log(coefficient) + (eps * utility / 2).

    Returns:
        Index of the sampled element.
    """
    n = len(log_terms)
    race_times = np.empty(n, dtype=np.float64)
    for i in range(n):
        u = np.random.random()
        race_times[i] = np.log(np.log(1.0 / u)) - log_terms[i]
    return np.argmin(race_times)


def sample_diff_idx(log_diff_counts, sorted_diffs, epsilon, sensitivity=1):
    """Samples a diff index from the exponential mechanism.

    Args:
        log_diff_counts: Array of log(# sequences with diff) for each diff.
        sorted_diffs: Increasing array of possible diffs.
        epsilon: Privacy parameter epsilon.
        sensitivity: Sensitivity of the query (1 for add/remove, 2 for swap).

    Returns:
        Index idx sampled from distribution P(diff) ~ count[diff] * exp(-epsilon * floor(diff) / 2).
    """
    log_weights = log_diff_counts - (
        epsilon * np.floor(sorted_diffs) / (2 * sensitivity)
    )
    return _racing_sample(log_weights)


def sequence_from_diff(diff, diff_row, diff_col, diff_matrix):
    """Samples a sequence with given diff uniformly at random.

    Args:
        diff: Diff (negative utility) of sequence to sample.
        diff_row: Row index where diff occurs in diff_matrix.
        diff_col: Column index where diff occurs in diff_matrix.
        diff_matrix: Matrix of distinct count differences.

    Returns:
        Array of item indices forming a sequence with the specified diff.
    """
    k, d = diff_matrix.shape
    sequence = np.full(k, diff_col, dtype=np.int64)

    # For each row, find threshold t: number of columns with diff <= selected diff
    ts = np.array(
        [
            np.searchsorted(diff_matrix[row, :], diff, side="right")
            for row in range(k)
        ]
    )

    # Assign items to each row (except the pivot row)
    for row in range(k):
        if row != diff_row:
            # Find candidates: columns < t[row] that aren't already used
            candidates = [i for i in range(ts[row]) if i not in sequence]
            if candidates:
                sequence[row] = np.random.choice(candidates)

    return sequence


def joint_mechanism(counts, epsilon, k, sensitivity=1):
    """Main Entry Point: Applies joint exponential mechanism for top-k selection.

    Args:
        counts: Array of item counts (does not need to be sorted).
        epsilon: Privacy budget.
        k: Number of items to select.
        sensitivity: Sensitivity of the query (1 for add/remove, 2 for swap).

    Returns:
        Array of k item indices as estimated by the joint exponential mechanism.
    """
    counts = np.asarray(counts, dtype=np.float64)

    # Sort counts in non-increasing order
    sort_indices = np.argsort(counts)[::-1]
    sorted_counts = counts[sort_indices]

    # Build the diff matrix (negative utility matrix)
    diff_matrix = make_diff_matrix(sorted_counts, k)

    # Get mapping from sorted diffs to matrix positions
    diffs_to_positions = get_diffs_to_positions(diff_matrix)

    # Compute log counts using Lemma 3.7
    log_diff_counts = compute_log_diff_counts(diff_matrix, diffs_to_positions)

    # Get sorted diffs
    sorted_diffs = diff_matrix[diffs_to_positions]

    # Sample a diff index using the exponential mechanism
    diff_idx = sample_diff_idx(log_diff_counts, sorted_diffs, epsilon, sensitivity)

    # Get the position of the sampled diff
    diff_row = diffs_to_positions[0][diff_idx]
    diff_col = diffs_to_positions[1][diff_idx]

    # Sample a sequence achieving this diff
    sequence = sequence_from_diff(
        sorted_diffs[diff_idx], diff_row, diff_col, diff_matrix
    )

    # Map back to original indices
    return sort_indices[sequence]


# Alias for backward compatibility
def get_utilities(counts, k):
    """Helper: Returns diff matrix and metadata for inspection."""
    counts = np.asarray(counts, dtype=np.float64)
    sort_order = np.argsort(counts)[::-1]
    sorted_counts = counts[sort_order]

    diff_matrix = make_diff_matrix(sorted_counts, k)
    diffs_to_positions = get_diffs_to_positions(diff_matrix)
    log_diff_counts = compute_log_diff_counts(diff_matrix, diffs_to_positions)
    sorted_diffs = diff_matrix[diffs_to_positions]

    return {
        "diff_matrix": diff_matrix,
        "diffs_to_positions": diffs_to_positions,
        "log_diff_counts": log_diff_counts,
        "sorted_diffs": sorted_diffs,
        "sort_order": sort_order,
        "d": len(counts),
    }
