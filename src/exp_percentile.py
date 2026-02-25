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


@njit
def utility(x: np.ndarray, p: float) -> np.ndarray:
    n = x.shape[0]
    k = int(np.ceil(p * (n + 1)))
    return -1 * np.abs(x[k] - x)


def expected_error(u: np.ndarray, pmf: np.ndarray):
    return abs(u.max() - np.sum(u @ pmf))


if __name__ == "__main__":
    methods = ["lln"]  # ["tdist", "lln"]
    datasets = ["hepth"]  # ["hepth", "income", "patent"]
    percentiles = [0.99]  # [0.5, 0.9, 0.99]

    pipe = {
        "get_params": {
            "tdist": get_params_tdist,
            "lln": get_params_lln,
        },
        "pmf": {"tdist": esnm_t_pmf, "lln": esnm_lln_pmf},
    }

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
                    ls = np.load(f"states/ls_perc_{ds}_{int(p * 100)}.npz")["ls"]

                    eps_arr = [
                        4.7469,
                        4.9490,
                        5.1510,
                        5.3531,
                    ]  # np.linspace(0.1, 10, 50)
                    print("p\teps\terror\ttime", file=f)
                    print("p\teps\terror\ttime")
                    k = int(np.ceil(p * (x.shape[0] + 1)))
                    for eps in eps_arr:
                        start_time = time.perf_counter()

                        params = pipe["get_params"][m](eps, ls)
                        ss = params.ss
                        s = params.s

                        # GS Upper bound
                        ss = np.clip(ss, a_min=None, a_max=x.max())

                        if m == "tdist":
                            probs = pipe["pmf"][m](u, ss, s, 3.0)
                        else:
                            probs = pipe["pmf"][m](u, ss, s, params.sigma)
                        probs = np.array(probs)
                        probs /= probs.sum()

                    # end_time = time.perf_counter()
                    # elapsed_time = end_time - start_time
                    # print(
                    #     f"{p}\t{eps:.4f}\t{expected_error(u, probs):.12f}\t{elapsed_time:.4f}",
                    #     file=f,
                    # )
                    # print(
                    #     f"{p}\t{eps:.4f}\t{expected_error(u, probs):.12f}\t{elapsed_time:.4f}"
                    # )
