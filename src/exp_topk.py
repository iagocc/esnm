"""Top-k mechanisms compared at a common rho-zCDP budget.

The sweep variable `rho` is the rho-zCDP budget. Conversions per method:

  * `joint_mechanism` (pure eps-DP, one-shot): receives
    eps = sqrt(2*rho) via `src/dp_conv.py::rho_zcdp_to_eps_for_pure_dp`
    (Lemma 9 of Bun & Steinke: pure-eps-DP implies (1/2)*eps^2-CDP).
  * `esnm_joint_t` (one-shot): Student's-T noise is pure eps-DP (Bun & Steinke
    2019, Thm 31), so it receives eps = sqrt(2*rho) via
    `rho_zcdp_to_eps_for_pure_dp`; the call is rho-zCDP by pure-DP =>
    (1/2 eps^2)-zCDP.
  * `esnm_joint_lln` (one-shot): LLN noise is NOT pure-DP -- it is directly
    (1/2 eps^2)-CDP (Prop. 3), which equals rho-zCDP at the same
    eps = sqrt(2*rho) via `rho_zcdp_to_cdp_eps`.

The `rho` column in the TSV is the rho-zCDP axis.
"""

import multiprocessing as mp
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from dp_conv import rho_zcdp_to_cdp_eps, rho_zcdp_to_eps_for_pure_dp
from topk.esnm_joint import esnm_joint_lln, esnm_joint_t
from topk.joint import joint_mechanism


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_dataset(ds_name: str, count_col: str) -> np.ndarray:
    ds_dir = Path.cwd() / "data/topk/"
    filename = ds_name + ".parquet"

    if not (ds_dir / filename).exists():
        filename = ds_name + ".csv"
        if not (ds_dir / filename).exists():
            raise FileNotFoundError(f"Dataset {ds_name} not found.")
        df = pd.read_csv((ds_dir / filename).absolute(), on_bad_lines="skip")
    else:
        df = pd.read_parquet((ds_dir / filename).absolute())

    counts = df.loc[:, count_col].to_numpy(dtype=np.int64)
    return counts


def compute_errors(
    true_topk: np.ndarray, predicted_topk: np.ndarray
) -> tuple[float, float, float]:
    """Compute L1, L-infinity errors, and NDCG."""
    l1_error = np.linalg.norm(true_topk - predicted_topk, ord=1)
    l_inf_error = np.linalg.norm(true_topk - predicted_topk, ord=np.inf)
    ndcg = ndcg_score(true_topk.reshape(1, -1), predicted_topk.reshape(1, -1))
    return float(l1_error), float(l_inf_error), float(ndcg)


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------


def run_joint(counts, k, rho):
    selected = joint_mechanism(counts, rho_zcdp_to_eps_for_pure_dp(rho), k)
    return counts[selected]


def run_esnm_joint_t(counts, k, rho):
    # eSNM optimizer budget is a pure-DP epsilon; sqrt(2*rho) makes it rho-zCDP.
    eps = rho_zcdp_to_eps_for_pure_dp(rho)
    selected = esnm_joint_t(counts, eps, k, degree_freedom=5.0)
    return counts[selected]


def run_esnm_joint_lln(counts, k, rho):
    # LLN is (1/2 eps^2)-CDP (not pure-DP); eps = sqrt(2*rho) => rho-zCDP.
    eps = rho_zcdp_to_cdp_eps(rho)
    selected = esnm_joint_lln(counts, eps, k)
    return counts[selected]


RUNNERS = {
    "joint": run_joint,
    "esnm_joint_t": run_esnm_joint_t,
    "esnm_joint_lln": run_esnm_joint_lln,
}


# ---------------------------------------------------------------------------
# Worker functions for multiprocessing
# ---------------------------------------------------------------------------

_worker_state = {}


def _init_worker(counts):
    _worker_state["counts"] = counts


def _run_single_trial(args):
    seed, k, rho, method_name = args
    set_seeds(seed)
    counts = _worker_state["counts"]

    runner = RUNNERS[method_name]

    start = time.perf_counter()
    predicted = runner(counts, k, rho)
    elapsed = time.perf_counter() - start

    true_topk = np.sort(counts)[::-1][:k]
    return compute_errors(true_topk, predicted) + (elapsed,)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    set_seeds(42)
    n_trials = 50
    rho_values = [1.0]
    k_values = np.arange(5, 205, 10)
    n_workers = 2
    datasets = [
        ("games_cleaned", "purchase_count"),
        ("books", "text_reviews_count"),
        ("movies", "count"),
    ]
    method_names = [
        "joint",
        # "esnm_joint_t",
        "esnm_joint_lln",
    ]

    results_dir = Path.cwd() / "results/topk"
    results_dir.mkdir(parents=True, exist_ok=True)

    for ds_name, count_col in datasets:
        counts = load_dataset(ds_name, count_col)

        for method_name in method_names:
            pool = mp.Pool(
                processes=n_workers,
                initializer=_init_worker,
                initargs=(counts,),
            )

            result_file = results_dir / f"{ds_name}_{method_name}.txt"

            seed_rng = np.random.RandomState(42)
            total_trials = len(k_values) * len(rho_values) * n_trials
            all_seeds = seed_rng.randint(0, 2**31 - 1, size=total_trials)
            seed_idx = 0

            try:
                with open(result_file, "w+") as f:
                    print(
                        "method\tk\trho\tmean_l1\tmedian_l1\tp25_l1\tp75_l1\tmean_linf\tmedian_linf\tp25_linf\tp75_linf\tmean_ndcg\tmedian_ndcg\tp25_ndcg\tp75_ndcg\tmean_time\tmedian_time\tp25_time\tp75_time",
                        file=f,
                    )

                    for k in k_values:
                        for rho in rho_values:
                            trial_args = []
                            for run in range(n_trials):
                                trial_args.append(
                                    (
                                        int(all_seeds[seed_idx]),
                                        int(k),
                                        rho,
                                        method_name,
                                    )
                                )
                                seed_idx += 1

                            results = pool.map(_run_single_trial, trial_args)

                            l1_errors = [r[0] for r in results]
                            linf_errors = [r[1] for r in results]
                            ndcg_scores = [r[2] for r in results]
                            time_elapsed_arr = [r[3] for r in results]

                            mean_l1 = np.mean(l1_errors)
                            median_l1 = np.median(l1_errors)
                            p25_l1 = np.percentile(l1_errors, 25)
                            p75_l1 = np.percentile(l1_errors, 75)
                            mean_linf = np.mean(linf_errors)
                            median_linf = np.median(linf_errors)
                            p25_linf = np.percentile(linf_errors, 25)
                            p75_linf = np.percentile(linf_errors, 75)
                            mean_ndcg = np.mean(ndcg_scores)
                            median_ndcg = np.median(ndcg_scores)
                            p25_ndcg = np.percentile(ndcg_scores, 25)
                            p75_ndcg = np.percentile(ndcg_scores, 75)
                            mean_time = np.mean(time_elapsed_arr)
                            median_time = np.median(time_elapsed_arr)
                            p25_time = np.percentile(time_elapsed_arr, 25)
                            p75_time = np.percentile(time_elapsed_arr, 75)

                            print(
                                f"{method_name}\t{k}\t{rho:.4f}\t{mean_l1:.6f}\t{median_l1:.6f}\t{p25_l1:.6f}\t{p75_l1:.6f}\t{mean_linf:.6f}\t{median_linf:.6f}\t{p25_linf:.6f}\t{p75_linf:.6f}\t{mean_ndcg:.6f}\t{median_ndcg:.6f}\t{p25_ndcg:.6f}\t{p75_ndcg:.6f}\t{mean_time:.4f}\t{median_time:.4f}\t{p25_time:.4f}\t{p75_time:.4f}",
                                file=f,
                            )
                            print(
                                f"{method_name}\t{k}\t{rho:.4f}\t{mean_l1:.6f}\t{median_l1:.6f}\t{p25_l1:.6f}\t{p75_l1:.6f}\t{mean_linf:.6f}\t{median_linf:.6f}\t{p25_linf:.6f}\t{p75_linf:.6f}\t{mean_ndcg:.6f}\t{median_ndcg:.6f}\t{p25_ndcg:.6f}\t{p75_ndcg:.6f}\t{mean_time:.4f}\t{median_time:.4f}\t{p25_time:.4f}\t{p75_time:.4f}"
                            )

                pool.close()
                pool.join()
            except Exception:
                pool.terminate()
                pool.join()
                raise
