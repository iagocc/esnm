import os
import time
from typing import NamedTuple

import numpy as np
from esnm.mechanism import esnm_lln_pmf, esnm_t_pmf
from numba import njit

from src.optimize_params import (
    Array1DFloat,
    Array2DFloat,
    optimize_params_lln,
    optimize_params_tdist,
)

Params = NamedTuple(
    "Params",
    [
        ("t", Array1DFloat),
        ("s", Array1DFloat | None),
        ("ss", Array1DFloat),
        ("sigma", Array1DFloat | None),
    ],
)


def get_params_tdist(eps: float, ls: Array2DFloat, d: float = 3) -> Params:
    t_candidates = np.linspace(0, eps / (d + 1), 150)
    t, s, _, ss = optimize_params_tdist(eps, d, t_candidates, ls)
    return Params(t, s, ss, None)


def get_params_lln(eps: float, ls: Array2DFloat) -> Params:
    t_candidates = np.logspace(-9, 10, 150)
    t, sigmas, s, _, ss = optimize_params_lln(eps, t_candidates, ls)
    return Params(t, s, ss, sigmas)


def broadcast_params(params: Params, size: int) -> Params:
    s = None if params.s is None else np.full(size, float(params.s[0]), dtype=np.float64)
    sigma = (
        None
        if params.sigma is None
        else np.full(size, float(params.sigma[0]), dtype=np.float64)
    )
    return Params(
        np.full(size, float(params.t[0]), dtype=np.float64),
        s,
        np.full(size, float(params.ss[0]), dtype=np.float64),
        sigma,
    )


# ---------------------------------------------------------------------------
# Tie-aware rank-distance utility (Lemma 1/2: element-wise smooth sensitivity ≤ 1)
# ---------------------------------------------------------------------------
# For grid point r ∈ R, define
#
#   u(x, r) = -dist(k_p, [a_r(x), b_r(x)]),
#
# where [a_r(x), b_r(x)] is the tie block of r in sorted x:
#   - if r appears in x: a = |{i : x[i] < r}|, b = |{i : x[i] ≤ r}| - 1.
#   - if r is between two distinct x values (or out of range): a = b =
#     |{i : x[i] < r}|, i.e. the rank-position immediately above the
#     largest x ≤ r.
#
# `dist(k, [a, b]) = max(0, a − k, k − b)` is the unsigned distance from
# integer k to the closed interval [a, b]. The utility is 0 precisely
# when k_p ∈ [a, b], i.e. when r's tie block in x contains the target
# rank — which is exactly the set of values that are "correct" answers
# for the percentile.
#
# Why this and not -|rank_x_right(r) − (k_p + 1)|: a single-sided rank
# function (left or right) "lags" through tie blocks. If x[k_p] sits
# inside a long tie block that extends well above or below k_p, the
# single-sided argmax skews to the value immediately outside the block
# (e.g. HEPTH p=25 plateaus at MAE = 1 with a monotone increase of
# rank_err in ε because the mechanism concentrates more strongly on the
# wrong winner as noise shrinks). The tie-aware utility makes every
# value in the correct tie block a perfect argmax.
#
# Privacy proof (Lemma 1, distance 0): for neighbours x, y differing in
# one tuple, both endpoints shift by at most 1: |a_r(x) − a_r(y)| ≤ 1
# and |b_r(x) − b_r(y)| ≤ 1. The interval-distance function is
# 1-Lipschitz in each endpoint, so |u(x, r) − u(y, r)| ≤ 1.
#
# Lemma 2 (distance k): max over y, z within k of x of |u(y, r) −
# u(z, r)| is bounded by Lemma 1 applied to y — so LS_u(x, k, r) ≤ 1 for
# every k. Hence S_t(x, r) = max_k e^{−kt} · LS_u(x, k, r) = 1 at k = 0.
#
# Noise scale: N(x, r) = (S_t(x, r) + max_{r' ≠ r} S_t(x, r'))/s = 2/s,
# constant across all outcomes — no tail contamination.


def build_grid(
    x_sorted: np.ndarray,
    error_level: float = 1.0,
    max_grid_size: int | None = None,
) -> np.ndarray:
    """Public value grid over [x.min(), x.max()] mirroring SI's ladder.

    `error_level` is SI's δ. Passing `max_grid_size` coarsens the grid
    for smoke tests; production comparisons leave it unset so the grid is
    exactly SI's integer ladder.
    """
    a = float(x_sorted[0])
    b = float(x_sorted[-1])
    span = b - a
    if span <= 0:
        return np.array([a], dtype=np.float64)
    step = error_level
    natural_size = int(np.ceil(span / step)) + 1
    if max_grid_size is not None and natural_size > max_grid_size:
        step = span / (max_grid_size - 1)
    grid = np.arange(a, b + step * 0.5, step, dtype=np.float64)
    return grid


def utility_smooth_value(
    x_sorted: np.ndarray,
    k_p: int,
    grid: np.ndarray,
) -> np.ndarray:
    """Value utility with quantile-gap smooth sensitivity.

    u(x, r) = -|x_(k_p) - r| over the public value grid. Its global
    sensitivity is the public range, but its local sensitivity is bounded
    by nearby order-statistic gaps; see
    `quantile_gap_local_sensitivity`.
    """
    return -np.abs(grid - x_sorted[k_p]).astype(np.float64)


def utility_rank(x_sorted: np.ndarray, k_p: int, grid: np.ndarray) -> np.ndarray:
    """Tie-aware rank utility: u(x, r) = -dist(k_p, [a_r(x), b_r(x)])."""
    n = x_sorted.shape[0]
    left = np.searchsorted(x_sorted, grid, side="left")
    right = np.searchsorted(x_sorted, grid, side="right")
    in_data = left < right
    # In-data: tie block [left, right-1]; distance to k_p.
    a = left
    b_in = right - 1
    dist_in = np.maximum(np.maximum(a - k_p, k_p - b_in), 0)
    # Out-of-data: the grid value sits between sorted x positions left-1
    # and left; treat as a degenerate "block" {left}.
    pos = np.clip(left, 0, n - 1)
    dist_out = np.abs(pos - k_p)
    dist = np.where(in_data, dist_in, dist_out)
    return -dist.astype(np.float64)


def quantile_gap_local_sensitivity(
    x_sorted: np.ndarray,
    k_p: int,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> np.ndarray:
    """Upper-bound LS_d for u(x, r) = -|x_(k_p) - r|.

    For fixed public output value r,

        |u(y, r) - u(z, r)| <= |y_(k_p) - z_(k_p)|.

    If y is within d substitutions of x and z is a neighbour of y, the
    two k_p-th order statistics can be bracketed by a window of d + 1
    adjacent order-statistic moves in x. With 1-indexed order statistic
    m = k_p + 1 and public sentinels x_0 = lower_bound,
    x_{n+1} = upper_bound, the standard smooth-quantile bound is

        LS_d(x) <= max_{0 <= t <= d+1} x_{m+t} - x_{m+t-d-1}.

    This equals the larger adjacent gap at d = 0 and is always <= the
    public global sensitivity upper_bound - lower_bound.
    """
    n = x_sorted.shape[0]
    if n == 0:
        return np.zeros(1, dtype=np.float64)

    lo_public = float(x_sorted[0] if lower_bound is None else lower_bound)
    hi_public = float(x_sorted[-1] if upper_bound is None else upper_bound)
    ext = np.empty(n + 2, dtype=np.float64)
    ext[0] = lo_public
    ext[1 : n + 1] = x_sorted
    ext[n + 1] = hi_public

    m = k_p + 1  # 1-indexed position in `ext`.
    ls = np.empty(n + 1, dtype=np.float64)
    global_sensitivity = max(0.0, hi_public - lo_public)
    for d in range(n + 1):
        t = np.arange(d + 2)
        hi_idx = np.clip(m + t, 0, n + 1)
        lo_idx = np.clip(m + t - d - 1, 0, n + 1)
        ls[d] = min(global_sensitivity, float(np.max(ext[hi_idx] - ext[lo_idx])))
    return ls


@njit
def utility(x: np.ndarray, p: float) -> np.ndarray:
    """Legacy value-based utility, kept for reference / smoke tests."""
    n = x.shape[0]
    k = int(np.ceil(p * (n + 1))) - 1
    if k < 0:
        k = 0
    elif k > n - 1:
        k = n - 1
    return -1 * np.abs(x[k] - x)


def tie_intervals(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    diff = np.r_[True, x[1:] != x[:-1]]
    group_id = np.cumsum(diff) - 1
    group_lo = np.flatnonzero(diff)
    group_hi = np.r_[group_lo[1:] - 1, n - 1]
    lo = group_lo[group_id]
    hi = group_hi[group_id]
    return lo, hi


def interval_gap(
    lo_a: np.ndarray, hi_a: np.ndarray, lo_b: int, hi_b: int
) -> np.ndarray:
    return np.maximum(0, np.maximum(lo_a - hi_b, lo_b - hi_a))


# ---------------------------------------------------------------------------
# Per-grid-point metric arrays (analytical metric = probs · array).sum())
# ---------------------------------------------------------------------------


def precompute_grid_metrics(
    grid: np.ndarray,
    x_sorted: np.ndarray,
    k_p: int,
) -> dict[str, np.ndarray]:
    """For each grid point compute the four per-outcome metric values
    (rank distance, tie-aware rank distance, |value error|, value error²).

    Rank semantics follow SI's `rank_difference`:
      - left = bisect_left(x, r), right = bisect_right(x, r)
      - if left < right (r is a value in x): r belongs to tie block
        [left, right-1] (0-indexed). tie-aware distance to k_p is 0
        if k_p in that block, else min distance to either end.
      - else (r is between data points or out of range): use the closer
        of indices left-1 (if ≥ 0) and left (if ≤ n-1) for rank distance.
    """
    n = x_sorted.shape[0]
    left = np.searchsorted(x_sorted, grid, side="left")
    right = np.searchsorted(x_sorted, grid, side="right")
    in_data = left < right
    first0 = left
    last0 = np.where(in_data, right - 1, left)

    # For grid points outside any tie block, the two flanking x positions
    # are `left - 1` and `left`. Clipping into [0, n-1] makes both flanks
    # collapse to the boundary at the edges (left == 0 or left == n),
    # which is the correct nearest position there.
    flank_lo = np.maximum(left - 1, 0)
    flank_hi = np.minimum(left, n - 1)
    out_nearest = np.minimum(np.abs(flank_lo - k_p), np.abs(flank_hi - k_p))

    # Rank distance (tie-unaware): leftmost rank for in-data, nearest flank otherwise.
    rank_err = np.where(in_data, np.abs(left - k_p), out_nearest).astype(np.float64)

    # Tie-aware rank distance: 0 if k_p lies inside the tie block, else
    # distance to the closer endpoint; nearest flank when not in any block.
    inside_block = in_data & (first0 <= k_p) & (k_p <= last0)
    rd_in = np.where(
        inside_block, 0, np.minimum(np.abs(k_p - first0), np.abs(k_p - last0))
    )
    rank_diff = np.where(in_data, rd_in, out_nearest).astype(np.float64)

    diff = grid - x_sorted[k_p]
    val_abs = np.abs(diff)
    val_sq = diff * diff

    return {
        "rank_err": rank_err,
        "rank_diff": rank_diff,
        "val_abs": val_abs,
        "val_sq": val_sq,
    }


def compute_metrics(
    probs: np.ndarray,
    grid_metrics: dict[str, np.ndarray],
    n_mc: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    rank_err = grid_metrics["rank_err"]
    rank_diff = grid_metrics["rank_diff"]
    val_abs = grid_metrics["val_abs"]
    val_sq = grid_metrics["val_sq"]

    metrics = {
        "exp_err": float(np.sum(probs * rank_err)),
        "mae": float(np.sum(probs * val_abs)),
        "mse": float(np.sum(probs * val_sq)),
        "rank_diff": float(np.sum(probs * rank_diff)),
    }

    rng = np.random.default_rng(seed)
    samples = rng.choice(probs.shape[0], size=n_mc, p=probs)
    metrics["exp_err_mc"] = float(np.mean(rank_err[samples]))
    metrics["mae_mc"] = float(np.mean(val_abs[samples]))
    metrics["mse_mc"] = float(np.mean(val_sq[samples]))
    metrics["rank_diff_mc"] = float(np.mean(rank_diff[samples]))
    return metrics


if __name__ == "__main__":
    utility_name = os.environ.get("ESNM_PERCENTILE_UTILITY", "smooth_value")
    methods = os.environ.get("ESNM_PERCENTILE_METHODS", "lln").split(",")
    datasets = os.environ.get("ESNM_PERCENTILE_DATASETS", "hepth,income,patent").split(",")
    percentiles = [
        float(v) / 100.0
        for v in os.environ.get("ESNM_PERCENTILE_PCTS", "25,50,75").split(",")
    ]
    eps_count = int(os.environ.get("ESNM_PERCENTILE_EPS_COUNT", "50"))

    pipe = {
        "get_params": {
            "lln": get_params_lln,
            "tdist": get_params_tdist,
        },
        "pmf": {"lln": esnm_lln_pmf, "tdist": esnm_t_pmf},
    }

    columns = [
        "p",
        "eps",
        "exp_err",
        "mae",
        "mse",
        "rank_diff",
        "exp_err_mc",
        "mae_mc",
        "mse_mc",
        "rank_diff_mc",
        "m",
        "global_sensitivity",
        "max_smooth_sensitivity",
        "smooth_over_global",
        "time",
    ]
    header = "\t".join(columns)

    result_dir = {
        "smooth_value": "results/percentile_smooth_value",
        "rank": "results/percentile_rank",
    }.get(utility_name)
    if result_dir is None:
        raise ValueError("ESNM_PERCENTILE_UTILITY must be one of: smooth_value, rank")
    os.makedirs(result_dir, exist_ok=True)

    for m_name in methods:
        print(f"Method: {m_name} ({utility_name} utility)")
        for ds in datasets:
            print(f"Dataset: {ds}")
            x = np.load(f"data/1D/{ds.upper()}.n4096.npy").astype(np.float64)
            x.sort()
            n = x.shape[0]
            grid = build_grid(x, error_level=1.0)
            m_size = grid.shape[0]
            print(f"  grid size m = {m_size}")
            value_range = float(grid[-1] - grid[0])

            for p in percentiles:
                k = max(0, min(n - 1, int(np.ceil(p * (n + 1))) - 1))
                utility_global_sensitivity = {
                    "smooth_value": value_range,
                    "rank": 1.0,
                }[utility_name]
                if utility_name == "smooth_value":
                    u = utility_smooth_value(x, k, grid)
                    ls_curve = quantile_gap_local_sensitivity(
                        x, k, float(grid[0]), float(grid[-1])
                    )
                    ls_for_optimizer = ls_curve
                else:
                    u = utility_rank(x, k, grid)
                    # LS is identically 1 (Lemma 2). K=1 is sufficient
                    # since the max of e^{-kt}·1 is at k=0.
                    ls_for_optimizer = np.ones((m_size, 1), dtype=np.float64)
                gm = precompute_grid_metrics(grid, x, k)

                out_path = f"{result_dir}/{ds}_esnm_{m_name}_{int(p * 100)}.txt"
                with open(out_path, "w+") as f:
                    print(header, file=f)
                    print(header)
                    eps_arr = np.linspace(0.1, 10, eps_count)
                    for eps in eps_arr:
                        start_time = time.perf_counter()
                        if utility_name == "smooth_value":
                            scalar_params = pipe["get_params"][m_name](
                                eps, ls_for_optimizer.reshape(1, -1)
                            )
                            params = broadcast_params(scalar_params, m_size)
                        else:
                            params = pipe["get_params"][m_name](eps, ls_for_optimizer)

                        if m_name == "tdist":
                            probs = pipe["pmf"][m_name](u, params.ss, params.s, 3.0)
                        else:
                            probs = pipe["pmf"][m_name](
                                u, params.ss, params.s, params.sigma
                            )
                        probs = np.asarray(probs, dtype=np.float64)
                        s = probs.sum()
                        if s <= 0 or not np.isfinite(s):
                            probs = np.full_like(probs, 1.0 / probs.shape[0])
                        else:
                            probs /= s
                        elapsed = time.perf_counter() - start_time
                        max_smooth = float(np.max(params.ss))
                        smooth_over_global = (
                            max_smooth / utility_global_sensitivity
                            if utility_global_sensitivity > 0
                            else 0.0
                        )

                        mvals = compute_metrics(probs, gm)
                        row = (
                            f"{p}\t{eps:.4f}\t"
                            f"{mvals['exp_err']:.12f}\t{mvals['mae']:.12f}\t"
                            f"{mvals['mse']:.12f}\t{mvals['rank_diff']:.12f}\t"
                            f"{mvals['exp_err_mc']:.12f}\t{mvals['mae_mc']:.12f}\t"
                            f"{mvals['mse_mc']:.12f}\t{mvals['rank_diff_mc']:.12f}\t"
                            f"{m_size}\t"
                            f"{utility_global_sensitivity:.12f}\t"
                            f"{max_smooth:.12f}\t{smooth_over_global:.12f}\t"
                            f"{elapsed:.4f}"
                        )
                        print(row, file=f)
                        print(row)
