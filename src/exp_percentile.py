"""Compare DP percentile-selection mechanisms over a privacy-budget sweep.

Three utility specifications and five mechanism methods can be combined:

  --utility {smooth_value, rank, paper}
  --methods {lln, tdist, ld, shifted_ld, si}   (comma-separated)

For every (dataset, percentile, mechanism) combination the script writes a
single TSV with one row per budget point to `results/percentile/`. Each
row reports the analytical mean-absolute-value error and the analytical
expected rank error at a given **rho-zCDP** budget (the `eps` column is
the rho axis, kept under that name for backward compatibility).

Budget alignment across methods. The loop value (rho) is converted to a per-call
epsilon = sqrt(2 * rho) for every method, so each one-shot call is rho-zCDP
(rho = eps^2 / 2):
  * tdist, ld, shifted_ld, si: pure-eps-DP mechanisms (Student's-T is pure-DP by
    Bun & Steinke 2019, Thm 31), rho-zCDP via eps-DP => (1/2 eps^2)-zCDP
    (`rho_zcdp_to_eps_for_pure_dp`).
  * lln: Laplace-log-normal noise is NOT pure-DP; it is directly (1/2 eps^2)-CDP
    (Prop. 3), which equals rho-zCDP at the same epsilon (`rho_zcdp_to_cdp_eps`).

The module is organised as five layers:

  1.  Utility / grid construction          (build_grid, utility_*, *_metrics)
  2.  Local-sensitivity computation        (quantile_gap_*, load_or_build_ls)
  3.  Setup builder                        (Setup, build_setup)
  4.  Mechanism runners                    (run_esnm, run_ld, run_si)
  5.  CLI + orchestration                  (parse_args, main)
"""

import argparse
import os
import time
from typing import Callable, NamedTuple

import numpy as np
from esnm.mechanism import esnm_lln_pmf, esnm_t_pmf
from esnm.percentile import get_ls

from src.dp_conv import rho_zcdp_to_cdp_eps, rho_zcdp_to_eps_for_pure_dp
from src.local_dampening import ld_pmf, shifted_ld_pmf
from src.optimize_params import (
    Array1DFloat,
    Array2DFloat,
    optimize_params_lln,
    optimize_params_tdist,
)
from src.shifted_inverse import (
    compute_selection_values,
    expected_absolute_error,
    expected_rank_error,
    rounded_sampler_inputs,
    sampler_components,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_UTILITIES = ("smooth_value", "rank", "paper")
_VALID_METHODS = ("lln", "tdist", "ld", "shifted_ld", "si")
_RESULTS_DIR = "results/percentile"
_LS_CACHE_DIR = "states"
_EPS_RANGE = (0.1, 10.0)
_COLUMNS = ("p", "eps", "mae", "rank_err", "time")


# ---------------------------------------------------------------------------
# eSNM noise-parameter optimisation
# ---------------------------------------------------------------------------


class Params(NamedTuple):
    t: Array1DFloat
    s: Array1DFloat | None
    ss: Array1DFloat
    sigma: Array1DFloat | None


def get_params_tdist(eps: float, ls: Array2DFloat, d: float = 3) -> Params:
    t_candidates = np.linspace(0, eps / (d + 1), 150)
    t, s, _, ss = optimize_params_tdist(eps, d, t_candidates, ls)
    return Params(t, s, ss, None)


def get_params_lln(eps: float, ls: Array2DFloat) -> Params:
    t_candidates = np.logspace(-9, 10, 150)
    t, sigmas, s, _, ss = optimize_params_lln(eps, t_candidates, ls)
    return Params(t, s, ss, sigmas)


def broadcast_params(params: Params, size: int) -> Params:
    def _fill(arr: Array1DFloat | None) -> Array1DFloat | None:
        if arr is None:
            return None
        return np.full(size, float(arr[0]), dtype=np.float64)

    return Params(
        np.full(size, float(params.t[0]), dtype=np.float64),
        _fill(params.s),
        np.full(size, float(params.ss[0]), dtype=np.float64),
        _fill(params.sigma),
    )


# ---------------------------------------------------------------------------
# Utilities and per-utility output grids
# ---------------------------------------------------------------------------


def build_grid(x_sorted: np.ndarray, error_level: float = 1.0) -> np.ndarray:
    """Integer-ladder grid covering `[x.min(), x.max()]` at step `error_level`."""
    a = float(x_sorted[0])
    b = float(x_sorted[-1])
    if b <= a:
        return np.array([a], dtype=np.float64)
    return np.arange(a, b + error_level * 0.5, error_level, dtype=np.float64)


def utility_smooth_value(
    x_sorted: np.ndarray, k_p: int, grid: np.ndarray
) -> np.ndarray:
    """u(x, r) = -|x_{k_p} - r| over a public value grid."""
    return -np.abs(grid - x_sorted[k_p]).astype(np.float64)


def utility_rank(x_sorted: np.ndarray, k_p: int, grid: np.ndarray) -> np.ndarray:
    """Tie-aware rank utility u(x, r) = -dist(k_p, tie_block_of_r_in_x).

    Element-wise smooth sensitivity is bounded by 1 (Lemma 1/2 in the
    accompanying note): for neighbours x, y differing in one tuple, both
    endpoints a_r and b_r of r's tie block shift by at most 1, so the
    1-Lipschitz interval-distance function changes by at most 1.
    """
    n = x_sorted.shape[0]
    left = np.searchsorted(x_sorted, grid, side="left")
    right = np.searchsorted(x_sorted, grid, side="right")
    in_data = left < right
    dist_in = np.maximum(np.maximum(left - k_p, k_p - (right - 1)), 0)
    dist_out = np.abs(np.clip(left, 0, n - 1) - k_p)
    return -np.where(in_data, dist_in, dist_out).astype(np.float64)


def utility_paper(x_sorted: np.ndarray, k_p: int) -> np.ndarray:
    """u_p(x, i) = -|x[k_p] - x[i]| over indices i = 0..n-1 (Farias et al.)."""
    return -np.abs(x_sorted[k_p] - x_sorted).astype(np.float64)


# ---------------------------------------------------------------------------
# Local-sensitivity computation
# ---------------------------------------------------------------------------


def quantile_gap_local_sensitivity(
    x_sorted: np.ndarray,
    k_p: int,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    """NRS smooth-quantile bound: `LS_d <= max_{0<=t<=d+1} x_{m+t} - x_{m+t-d-1}`.

    Public sentinels `x_0 = lower_bound`, `x_{n+1} = upper_bound` handle
    boundaries. Returned array has length n+1 (entries for d = 0..n) and
    is clipped to the public global sensitivity `upper_bound - lower_bound`.
    """
    n = x_sorted.shape[0]
    if n == 0:
        return np.zeros(1, dtype=np.float64)

    ext = np.empty(n + 2, dtype=np.float64)
    ext[0] = lower_bound
    ext[1 : n + 1] = x_sorted
    ext[n + 1] = upper_bound

    m = k_p + 1
    ls = np.empty(n + 1, dtype=np.float64)
    global_sensitivity = max(0.0, upper_bound - lower_bound)
    for d in range(n + 1):
        t = np.arange(d + 2)
        hi_idx = np.clip(m + t, 0, n + 1)
        lo_idx = np.clip(m + t - d - 1, 0, n + 1)
        ls[d] = min(
            global_sensitivity, float(np.max(ext[hi_idx] - ext[lo_idx]))
        )
    return ls


def load_or_build_ls(
    path: str, x: np.ndarray, p: float, cap_lambda: float
) -> np.ndarray:
    """Load the paper's (n, n) per-index LS matrix from disk, or build it.

    Uses `esnm.percentile.get_ls` (Lemma 10 + Algorithm 2). Building from
    scratch is `O(n^3 log n)` and slow; results are persisted as compressed
    `.npz` files keyed by dataset and integer percentile.
    """
    if os.path.exists(path):
        return np.load(path)["ls"]
    print(f"  {path} not found — building (n={x.shape[0]}, this is slow)...")
    ls_full = np.asarray(get_ls(x.tolist(), p, cap_lambda), dtype=np.float64)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, ls=ls_full)
    return ls_full


# ---------------------------------------------------------------------------
# Per-output metric tables
# ---------------------------------------------------------------------------


def grid_rank_err(
    x_sorted: np.ndarray, k_p: int, grid: np.ndarray
) -> np.ndarray:
    """Rank distance per grid point under bisect-left + clip projection.

    Matches the convention used by SI's `expected_rank_error` so all
    methods report comparable rank distances.
    """
    n = x_sorted.shape[0]
    pos = np.clip(np.searchsorted(x_sorted, grid, side="left"), 0, n - 1)
    return np.abs(pos - k_p).astype(np.float64)


def grid_metrics(
    x_sorted: np.ndarray, k_p: int, grid: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "val_abs": np.abs(grid - x_sorted[k_p]),
        "rank_err": grid_rank_err(x_sorted, k_p, grid),
    }


def index_metrics(x_sorted: np.ndarray, k_p: int) -> dict[str, np.ndarray]:
    n = x_sorted.shape[0]
    return {
        "val_abs": np.abs(x_sorted - x_sorted[k_p]),
        "rank_err": np.abs(np.arange(n) - k_p).astype(np.float64),
    }


def metrics_from_pmf(
    probs: np.ndarray, gm: dict[str, np.ndarray]
) -> dict[str, float]:
    return {
        "mae": float(np.sum(probs * gm["val_abs"])),
        "rank_err": float(np.sum(probs * gm["rank_err"])),
    }


# ---------------------------------------------------------------------------
# Setup: utility + grid + LS + metric tables for one (utility, x, p) instance
# ---------------------------------------------------------------------------


class Setup(NamedTuple):
    u: np.ndarray
    ls: np.ndarray
    gm: dict[str, np.ndarray]
    m_size: int
    gs: float


def build_setup(
    utility: str,
    x_sorted: np.ndarray,
    k: int,
    p: float,
    ds: str,
    cap_lambda_override: float | None,
) -> Setup:
    if utility == "smooth_value":
        grid = build_grid(x_sorted)
        return Setup(
            u=utility_smooth_value(x_sorted, k, grid),
            ls=quantile_gap_local_sensitivity(
                x_sorted, k, float(grid[0]), float(grid[-1])
            ),
            gm=grid_metrics(x_sorted, k, grid),
            m_size=grid.shape[0],
            gs=float(grid[-1] - grid[0]),
        )
    if utility == "rank":
        grid = build_grid(x_sorted)
        m_size = grid.shape[0]
        return Setup(
            u=utility_rank(x_sorted, k, grid),
            # LS_u <= 1 everywhere, so K=1 suffices.
            ls=np.ones((m_size, 1), dtype=np.float64),
            gm=grid_metrics(x_sorted, k, grid),
            m_size=m_size,
            gs=1.0,
        )
    if utility == "paper":
        cap = (
            float(cap_lambda_override)
            if cap_lambda_override is not None
            else float(x_sorted[-1])
        )
        cache_path = (
            f"{_LS_CACHE_DIR}/ls_perc_{ds}_{int(p * 100)}.npz"
        )
        return Setup(
            u=utility_paper(x_sorted, k),
            ls=load_or_build_ls(cache_path, x_sorted, p, cap),
            gm=index_metrics(x_sorted, k),
            m_size=x_sorted.shape[0],
            gs=cap,
        )
    raise ValueError(f"unknown utility {utility!r}")


# ---------------------------------------------------------------------------
# Mechanism runners — each takes a Setup and returns a metric dict
# ---------------------------------------------------------------------------


def _normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    total = probs.sum()
    if total <= 0 or not np.isfinite(total):
        return np.full_like(probs, 1.0 / probs.shape[0])
    return probs / total


def _esnm_params(m_name: str, setup: Setup, eps: float) -> Params:
    optimizer = {"lln": get_params_lln, "tdist": get_params_tdist}[m_name]
    if setup.ls.ndim == 1:
        scalar = optimizer(eps, setup.ls.reshape(1, -1))
        return broadcast_params(scalar, setup.m_size)
    return optimizer(eps, setup.ls)


def _esnm_pmf(m_name: str, u: np.ndarray, params: Params) -> np.ndarray:
    if m_name == "tdist":
        return esnm_t_pmf(u, params.ss, params.s, 3.0)
    return esnm_lln_pmf(u, params.ss, params.s, params.sigma)


def run_esnm(
    m_name: str, setup: Setup, eps: float, **_: object
) -> dict[str, float]:
    # tdist (Student's-T) is pure eps-DP (Thm 31); lln is (1/2 eps^2)-CDP (Prop 3).
    # Both reach rho-zCDP at the same eps = sqrt(2*rho), via different proofs.
    convert = rho_zcdp_to_eps_for_pure_dp if m_name == "tdist" else rho_zcdp_to_cdp_eps
    cdp_eps = convert(eps)
    params = _esnm_params(m_name, setup, cdp_eps)
    probs = _normalize_probs(_esnm_pmf(m_name, setup.u, params))
    return metrics_from_pmf(probs, setup.gm)


def _ld_ls_for(setup: Setup) -> np.ndarray:
    """Pick the LD-friendly LS view that avoids materialising redundant rows.

    `ld_pmf` / `shifted_ld_pmf` accept either a (R, K) matrix or a length-K
    1D curve shared across outcomes. The 1D path is used when the LS is
    actually shared (smooth_value: per-distance curve; rank: scalar).
    """
    if setup.ls.ndim == 1:
        return setup.ls
    if setup.ls.shape[1] == 1:
        # rank utility — every row is [1.0]; collapse.
        return setup.ls[0]
    return setup.ls


def run_ld(
    m_name: str, setup: Setup, eps: float, **_: object
) -> dict[str, float]:
    pmf = {"ld": ld_pmf, "shifted_ld": shifted_ld_pmf}[m_name]
    pure_eps = rho_zcdp_to_eps_for_pure_dp(eps)
    probs = _normalize_probs(
        pmf(setup.u, setup.gs, pure_eps, _ld_ls_for(setup))
    )
    return metrics_from_pmf(probs, setup.gm)


def run_si(
    _m_name: str,
    _setup: Setup,
    eps: float,
    *,
    x: np.ndarray,
    k: int,
    **_: object,
) -> dict[str, float]:
    return _si_metrics(x, k, rho_zcdp_to_eps_for_pure_dp(eps))


_RUNNERS: dict[str, Callable[..., dict[str, float]]] = {
    "lln": run_esnm,
    "tdist": run_esnm,
    "ld": run_ld,
    "shifted_ld": run_ld,
    "si": run_si,
}


# ---------------------------------------------------------------------------
# ShiftedInverse: build sampler components and read off analytical metrics
# ---------------------------------------------------------------------------


def _si_metrics(
    x_sorted: np.ndarray, k: int, eps: float
) -> dict[str, float]:
    """Analytical SI metrics at this (dataset, percentile-index k, eps)."""
    n = x_sorted.shape[0]
    min_value = float(x_sorted[0])
    shifted = x_sorted - min_value
    desc_shifted = shifted[::-1].tolist()
    target_rank = k + 1  # 1-based, matches SI's convention
    selection_index = n - target_rank + 1
    upper_bound = max(1.0, float(np.ceil(desc_shifted[0])))
    beta = 0.1
    error_level = 1.0

    tau, query_result, check_fs = compute_selection_values(
        desc_shifted, selection_index, eps, beta, upper_bound, error_level
    )
    rounded = rounded_sampler_inputs(
        tau, query_result, check_fs, upper_bound, error_level
    )
    components = sampler_components(rounded, tau, eps)

    return {
        "mae": float(
            expected_absolute_error(
                components, tau, error_level, query_result
            )
        ),
        "rank_err": float(
            expected_rank_error(
                components,
                tau,
                error_level,
                x_sorted,
                target_rank,
                min_value,
            )
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _csv(raw: str) -> list[str]:
    parts = [v.strip() for v in raw.split(",") if v.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("value cannot be empty")
    return parts


def _csv_methods(raw: str) -> list[str]:
    parts = _csv(raw)
    bad = [v for v in parts if v not in _VALID_METHODS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown method(s) {bad}; expected any of {list(_VALID_METHODS)}"
        )
    return parts


def _csv_percentiles(raw: str) -> list[float]:
    out: list[float] = []
    for tok in _csv(raw):
        try:
            iv = int(tok)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"percentile {tok!r} is not an integer"
            ) from e
        if not 1 <= iv <= 99:
            raise argparse.ArgumentTypeError(
                f"percentile {iv} out of range [1, 99]"
            )
        out.append(iv / 100.0)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare DP percentile-selection mechanisms over a rho-zCDP "
            "budget sweep (the `eps` column / flag is the rho axis). "
            "Every method receives epsilon = sqrt(2*rho) so each call is "
            "rho-zCDP: pure-DP methods (tdist, ld, shifted_ld, si) via "
            "rho_zcdp_to_eps_for_pure_dp, and lln (which is (1/2 eps^2)-CDP, "
            "not pure-DP) via rho_zcdp_to_cdp_eps. "
            f"One TSV per (dataset, method, percentile) under {_RESULTS_DIR}/."
        )
    )
    parser.add_argument(
        "--utility",
        choices=_VALID_UTILITIES,
        default="smooth_value",
        help=(
            "Utility function (default: %(default)s). 'paper' uses the "
            "index-output utility u_p(x,i) = -|x[k] - x[i]| with per-index "
            "LS from esnm.percentile.get_ls (Lemma 10 + Algorithm 2)."
        ),
    )
    parser.add_argument(
        "--methods",
        type=_csv_methods,
        default="lln,tdist",
        help=(
            "Comma-separated mechanism methods from "
            f"{list(_VALID_METHODS)} (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--datasets",
        type=_csv,
        default="hepth,income,patent",
        help="Comma-separated dataset names (default: %(default)s).",
    )
    parser.add_argument(
        "--percentiles",
        type=_csv_percentiles,
        default="25,50,75",
        help=(
            "Comma-separated integer percentiles in [1, 99] "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--eps-count",
        type=int,
        default=50,
        help=(
            f"Number of epsilons in linspace{_EPS_RANGE} "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--cap-lambda",
        type=float,
        default=None,
        help=(
            "Public upper bound on data values, used for --utility paper. "
            "Defaults to x.max() (matches the on-disk LS cache convention)."
        ),
    )
    args = parser.parse_args(argv)

    # argparse only runs the type hook when the user provides a value;
    # defaults are passed through unchanged, so coerce them once here.
    if isinstance(args.methods, str):
        args.methods = _csv_methods(args.methods)
    if isinstance(args.datasets, str):
        args.datasets = _csv(args.datasets)
    if isinstance(args.percentiles, str):
        args.percentiles = _csv_percentiles(args.percentiles)
    return args


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _percentile_index(p: float, n: int) -> int:
    return max(0, min(n - 1, int(np.ceil(p * (n + 1))) - 1))


def _load_dataset(ds: str) -> np.ndarray:
    x = np.load(f"data/1D/{ds.upper()}.n4096.npy").astype(np.float64)
    x.sort()
    return x


def _run_one_file(
    method: str,
    utility: str,
    ds: str,
    x: np.ndarray,
    p: float,
    eps_arr: np.ndarray,
    cap_lambda_override: float | None,
) -> None:
    k = _percentile_index(p, x.shape[0])
    setup = build_setup(utility, x, k, p, ds, cap_lambda_override)
    runner = _RUNNERS[method]
    header = "\t".join(_COLUMNS)
    out_path = f"{_RESULTS_DIR}/{ds}_{utility}_{method}_{int(p * 100)}.txt"

    with open(out_path, "w") as f:
        print(header, file=f)
        print(header)
        for eps in eps_arr:
            t0 = time.perf_counter()
            m = runner(method, setup, eps, x=x, k=k, p=p, ds=ds)
            elapsed = time.perf_counter() - t0
            row = (
                f"{p}\t{eps:.4f}\t"
                f"{m['mae']:.12f}\t{m['rank_err']:.12f}\t{elapsed:.4f}"
            )
            print(row, file=f)
            print(row)


def main(args: argparse.Namespace) -> None:
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    eps_arr = np.linspace(*_EPS_RANGE, args.eps_count)

    for method in args.methods:
        for ds in args.datasets:
            x = _load_dataset(ds)
            print(
                f"=== {args.utility} / {method} / {ds} "
                f"(n={x.shape[0]}) ==="
            )
            for p in args.percentiles:
                _run_one_file(
                    method=method,
                    utility=args.utility,
                    ds=ds,
                    x=x,
                    p=p,
                    eps_arr=eps_arr,
                    cap_lambda_override=args.cap_lambda,
                )


if __name__ == "__main__":
    main(parse_args())
