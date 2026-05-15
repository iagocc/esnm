"""ShiftedInverse percentile sampler — pure-function copy for in-process use.

These functions are lifted verbatim from
`ShiftedInverse/Script/Collect1DPercentileShiftedInverse.py` so `exp_percentile.py`
can run the SI baseline alongside eSNM/LD methods without invoking the
standalone CLI script. The canonical implementation remains under
`ShiftedInverse/Script/`.
"""

import bisect
import math

import numpy as np


def compute_selection_values(
    desc_values, index, epsilon, beta, upper_bound, error_level
):
    tau = int(math.ceil(2.0 / epsilon * math.log((upper_bound / error_level + 1.0) / beta)))
    query_result = 0.0 if index > len(desc_values) else float(desc_values[index - 1])
    check_fs = [0.0] * (2 * tau + 2)

    if index > len(desc_values):
        return tau, query_result, check_fs

    # The 1D benchmark models one tuple per user. In that special case,
    # ComputeR_Selection's cover sequence is exactly the next descending
    # order statistics after k.
    max_j = min(2 * tau + 1, len(desc_values) - index)
    for j in range(1, max_j + 1):
        check_fs[j] = float(desc_values[index + j - 1])

    return tau, query_result, check_fs


def rounded_sampler_inputs(tau, query_result, check_fs, upper_bound, error_level):
    values = [float(upper_bound), float(query_result)]
    values.extend(float(check_fs[j]) for j in range(1, 2 * tau + 1))
    values.append(0.0)
    return [math.ceil(value / error_level) * error_level for value in values]


def sampler_components(rounded_values, tau, epsilon):
    upper_check_fs = rounded_values[: tau + 2]
    mid_check_f = rounded_values[tau + 1]
    lower_check_fs = rounded_values[tau + 1 :]

    upper_differences = [
        upper_check_fs[i] - upper_check_fs[i + 1] for i in range(len(upper_check_fs) - 1)
    ]
    lower_differences = [
        lower_check_fs[i] - lower_check_fs[i + 1] for i in range(len(lower_check_fs) - 1)
    ]

    pdf = [
        math.exp(epsilon / 2.0 * (-tau + i - 1)) * upper_differences[i]
        for i in range(len(upper_differences))
    ]
    pdf.append(1.0)
    pdf.extend(
        math.exp(epsilon / 2.0 * (-i - 1)) * lower_differences[i]
        for i in range(len(lower_differences))
    )

    cdf = []
    running = 0.0
    for probability in pdf:
        running += probability
        cdf.append(running)

    return (
        upper_check_fs,
        mid_check_f,
        lower_check_fs,
        upper_differences,
        lower_differences,
        pdf,
        cdf,
    )


def sample_output(components, tau, error_level, rng):
    (
        upper_check_fs,
        mid_check_f,
        lower_check_fs,
        upper_differences,
        lower_differences,
        _,
        cdf,
    ) = components
    total_mass = cdf[-1]
    sample_1 = rng.random() * total_mass
    index = bisect.bisect_right(cdf, sample_1)

    if index <= tau:
        difference = max(0.0, upper_differences[index])
        sample_2 = math.floor(rng.random() * difference / error_level) * error_level
        return upper_check_fs[index] - sample_2

    if index == tau + 1:
        return mid_check_f

    lower_index = index - tau - 2
    difference = max(0.0, lower_differences[lower_index])
    sample_2 = math.floor(rng.random() * difference / error_level) * error_level
    return lower_check_fs[lower_index] - sample_2 - error_level


def mean_abs_error_on_grid(start, difference, error_level, offset, query_result):
    steps = int(round(difference / error_level))
    if steps <= 0:
        return 0.0

    x = (start + offset - query_result) / error_level
    last_step = steps - 1

    if x <= 0:
        total = steps * (steps - 1) / 2.0 - steps * x
    elif x >= last_step:
        total = steps * x - steps * (steps - 1) / 2.0
    else:
        split = int(math.floor(x))
        left_count = split + 1
        left_sum = left_count * x - split * (split + 1) / 2.0

        right_count = steps - left_count
        right_index_sum = steps * (steps - 1) / 2.0 - split * (split + 1) / 2.0
        right_sum = right_index_sum - right_count * x
        total = left_sum + right_sum

    return error_level * total / steps


def expected_absolute_error(components, tau, error_level, query_result):
    (
        upper_check_fs,
        mid_check_f,
        lower_check_fs,
        upper_differences,
        lower_differences,
        pdf,
        cdf,
    ) = components
    total_mass = cdf[-1]
    expected = 0.0

    for i, difference in enumerate(upper_differences):
        if pdf[i] <= 0:
            continue
        expected += pdf[i] / total_mass * mean_abs_error_on_grid(
            upper_check_fs[i], difference, error_level, 0.0, query_result
        )

    mid_index = len(upper_differences)
    expected += pdf[mid_index] / total_mass * abs(mid_check_f - query_result)

    lower_offset = mid_index + 1
    for i, difference in enumerate(lower_differences):
        probability = pdf[lower_offset + i]
        if probability <= 0:
            continue
        expected += probability / total_mass * mean_abs_error_on_grid(
            lower_check_fs[i], difference, error_level, -error_level, query_result
        )

    return expected


def expected_rank_error(
    components, tau, error_level, sorted_values_arr, target_rank, min_value
):
    (
        upper_check_fs,
        mid_check_f,
        lower_check_fs,
        upper_differences,
        lower_differences,
        pdf,
        cdf,
    ) = components
    total_mass = cdf[-1]
    if total_mass <= 0:
        return 0.0
    n = sorted_values_arr.shape[0]

    target_rank_0 = target_rank - 1

    def accumulate(values_shifted, region_prob):
        if values_shifted.size == 0 or region_prob <= 0:
            return 0.0
        actual = values_shifted + min_value
        pos = np.searchsorted(sorted_values_arr, actual, side="left")
        pos = np.clip(pos, 0, n - 1)
        d = np.abs(pos - target_rank_0).astype(np.float64)
        per_point_prob = region_prob / values_shifted.size
        return float(per_point_prob * d.sum())

    expected = 0.0

    for i, difference in enumerate(upper_differences):
        steps = int(round(difference / error_level))
        if steps <= 0 or pdf[i] <= 0:
            continue
        vals = upper_check_fs[i] - np.arange(steps) * error_level
        expected += accumulate(vals, pdf[i] / total_mass)

    mid_index = len(upper_differences)
    if pdf[mid_index] > 0:
        vals = np.array([mid_check_f], dtype=np.float64)
        expected += accumulate(vals, pdf[mid_index] / total_mass)

    lower_offset = mid_index + 1
    for i, difference in enumerate(lower_differences):
        steps = int(round(difference / error_level))
        if steps <= 0 or pdf[lower_offset + i] <= 0:
            continue
        vals = lower_check_fs[i] - (np.arange(steps) + 1) * error_level
        expected += accumulate(vals, pdf[lower_offset + i] / total_mass)

    return expected
