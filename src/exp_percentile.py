import os
import time
from typing import NamedTuple

import numpy as np
from esnm.mechanism import esnm_lln_pmf, esnm_t_pmf
from esnm.percentile import get_ls
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


@njit
def utility(x: np.ndarray, p: float) -> np.ndarray:
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


def load_or_build_ls(path: str, x: np.ndarray, p: float) -> np.ndarray:
    if os.path.exists(path):
        return np.load(path)["ls"]
    print(f"  {path} not found — building (n={x.shape[0]}, this is slow)...")
    cap_lambda = float(x.max())
    ls_list = get_ls(x.tolist(), p, cap_lambda)
    ls_full = np.asarray(ls_list, dtype=np.float64)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, ls=ls_full)
    return ls_full


def compute_metrics(
    probs: np.ndarray,
    x: np.ndarray,
    k: int,
    lo: np.ndarray,
    hi: np.ndarray,
    n_mc: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    n = x.shape[0]
    idx = np.arange(n)

    rank_err_i = np.abs(idx - k).astype(np.float64)
    val_abs_i = np.abs(x - x[k])
    val_sq_i = (x - x[k]) ** 2
    rank_diff_i = interval_gap(lo, hi, int(lo[k]), int(hi[k])).astype(np.float64)

    metrics = {
        "exp_err": float(np.sum(probs * rank_err_i)),
        "mae": float(np.sum(probs * val_abs_i)),
        "mse": float(np.sum(probs * val_sq_i)),
        "rank_diff": float(np.sum(probs * rank_diff_i)),
    }

    rng = np.random.default_rng(seed)
    samples = rng.choice(n, size=n_mc, p=probs)
    metrics["exp_err_mc"] = float(np.mean(rank_err_i[samples]))
    metrics["mae_mc"] = float(np.mean(val_abs_i[samples]))
    metrics["mse_mc"] = float(np.mean(val_sq_i[samples]))
    metrics["rank_diff_mc"] = float(np.mean(rank_diff_i[samples]))
    return metrics


if __name__ == "__main__":
    methods = ["lln"]  # ["tdist", "lln"]
    datasets = ["hepth", "income", "patent"]
    percentiles = [0.25, 0.5, 0.75, 0.9, 0.99]

    pipe = {
        "get_params": {
            "tdist": get_params_tdist,
            "lln": get_params_lln,
        },
        "pmf": {"tdist": esnm_t_pmf, "lln": esnm_lln_pmf},
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
        "time",
    ]
    header = "\t".join(columns)

    for m in methods:
        print(f"Method: {m}")
        for ds in datasets:
            print(f"Dataset: {ds}")
            for p in percentiles:
                with open(
                    f"results/percentile/{ds}_esnm_{m}_{int(p * 100)}.txt", "w+"
                ) as f:
                    x = np.load(f"data/1D/{ds.upper()}.n4096.npy").astype(np.float64)
                    x.sort()
                    u = utility(x, p)
                    lo, hi = tie_intervals(x)
                    ls = load_or_build_ls(
                        f"states/ls_perc_{ds}_{int(p * 100)}.npz", x, p
                    )

                    eps_arr = np.linspace(0.1, 10, 50)
                    print(header, file=f)
                    print(header)
                    n = x.shape[0]
                    k = max(0, min(n - 1, int(np.ceil(p * (n + 1))) - 1))
                    for eps in eps_arr:
                        start_time = time.perf_counter()

                        params = pipe["get_params"][m](eps, ls)
                        ss = np.clip(params.ss, a_min=None, a_max=x.max())

                        if m == "tdist":
                            probs = pipe["pmf"][m](u, ss, params.s, 3.0)
                        else:
                            probs = pipe["pmf"][m](u, ss, params.s, params.sigma)
                        probs = np.asarray(probs, dtype=np.float64)
                        probs /= probs.sum()
                        elapsed = time.perf_counter() - start_time

                        mvals = compute_metrics(probs, x, k, lo, hi)
                        row = (
                            f"{p}\t{eps:.4f}\t"
                            f"{mvals['exp_err']:.12f}\t{mvals['mae']:.12f}\t"
                            f"{mvals['mse']:.12f}\t{mvals['rank_diff']:.12f}\t"
                            f"{mvals['exp_err_mc']:.12f}\t{mvals['mae_mc']:.12f}\t"
                            f"{mvals['mse_mc']:.12f}\t{mvals['rank_diff_mc']:.12f}\t"
                            f"{elapsed:.4f}"
                        )
                        print(row, file=f)
                        print(row)
