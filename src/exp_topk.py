"""Top-k mechanisms compared at a common pure eps-DP budget.

The sweep variable `eps` is the pure-(eps, 0)-DP budget. Every mechanism here is a
one-shot joint top-k selector that receives `eps` directly -- there is no zCDP
budget and no zCDP<->DP conversion anywhere:

  * `joint_mechanism`: the joint exponential mechanism, pure eps-DP by construction.
  * `esnm_joint_t`: Student's-T smooth-sensitivity noise, pure eps-DP (Bun &
    Steinke 2019, Thm 31; polynomial tails => finite shift and dilation).
  * `esnm_joint_gcp`: Gaussian-core Pareto-tail noise, pure eps-DP (polynomial tail).
  * `esnm_joint_lcp`: Laplace-core Pareto-tail noise, pure eps-DP (polynomial tail).

The `eps` column in the TSV is the pure-DP axis.
"""

import multiprocessing as mp
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from topk.esnm_joint import esnm_joint_gcp, esnm_joint_lcp, esnm_joint_t
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


def compute_rank_error(counts: np.ndarray, selected: np.ndarray, k: int) -> float:
    """Rank error: how far past rank k the returned set reaches.

    Each item has a true rank in descending-count order (1 = largest count).
    A perfect top-k return uses exactly ranks 1..k, so the worst (largest) rank
    among the k returned items is k.  The rank error is that worst rank minus k:

        rank_error = max_{i in selected} rank(i) - k

    It is 0 iff the returned set is exactly the true top-k, and equals the number
    of positions beyond k that the worst returned item sits at otherwise (>= 0,
    since k distinct items always span at least ranks 1..k).
    """
    order = np.argsort(counts)[::-1]  # original indices, largest count first
    rank = np.empty(counts.shape[0], dtype=np.int64)
    rank[order] = np.arange(1, counts.shape[0] + 1)  # 1-indexed true rank
    return float(int(rank[selected].max()) - k)


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------


def run_joint(counts, k, eps):
    selected = joint_mechanism(counts, eps, k)
    return selected


def run_esnm_joint_t(counts, k, eps):
    selected = esnm_joint_t(counts, eps, k, degree_freedom=5.0)
    return selected


def run_esnm_joint_gcp(counts, k, eps):
    selected = esnm_joint_gcp(counts, eps, k, gamma=5.0)
    return selected


def run_esnm_joint_lcp(counts, k, eps):
    selected = esnm_joint_lcp(counts, eps, k, gamma=5.0)
    return selected


# Saturating (capped) utility u = -sqrt(1 + min(gap, CAP)): full resolution
# only within CAP counts of the decision boundary, zero smooth sensitivity
# beyond.  CAP is a tolerance in count units (see esnm_joint cap parameter).
CAP = 8.0


def run_esnm_joint_t_cap(counts, k, eps):
    selected = esnm_joint_t(counts, eps, k, degree_freedom=5.0, cap=CAP)
    return selected


def run_esnm_joint_gcp_cap(counts, k, eps):
    selected = esnm_joint_gcp(counts, eps, k, gamma=5.0, cap=CAP)
    return selected


def run_esnm_joint_lcp_cap(counts, k, eps):
    selected = esnm_joint_lcp(counts, eps, k, gamma=5.0, cap=CAP)
    return selected


RUNNERS = {
    "joint": run_joint,
    "esnm_joint_t": run_esnm_joint_t,
    "esnm_joint_gcp": run_esnm_joint_gcp,
    "esnm_joint_lcp": run_esnm_joint_lcp,
    "esnm_joint_t_cap": run_esnm_joint_t_cap,
    "esnm_joint_gcp_cap": run_esnm_joint_gcp_cap,
    "esnm_joint_lcp_cap": run_esnm_joint_lcp_cap,
}


# ---------------------------------------------------------------------------
# Worker functions for multiprocessing
# ---------------------------------------------------------------------------

_worker_state = {}


def _init_worker(counts):
    _worker_state["counts"] = counts


def _run_single_trial(args):
    seed, k, eps, method_name = args
    set_seeds(seed)
    counts = _worker_state["counts"]

    runner = RUNNERS[method_name]

    start = time.perf_counter()
    selected = runner(counts, k, eps)
    elapsed = time.perf_counter() - start

    predicted = counts[selected]
    true_topk = np.sort(counts)[::-1][:k]
    rank_error = compute_rank_error(counts, selected, k)
    return compute_errors(true_topk, predicted) + (rank_error, elapsed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    set_seeds(42)
    n_trials = 50
    eps_values = [1.0]
    k_values = np.arange(5, 205, 10)
    n_workers = 2
    datasets = [
        ("games_cleaned", "purchase_count"),
        ("books", "text_reviews_count"),
        ("movies", "count"),
    ]
    method_names = [
        "joint",
        "esnm_joint_t",
        "esnm_joint_gcp",
        "esnm_joint_lcp",
        # "esnm_joint_t_cap",
        # "esnm_joint_gcp_cap",
        # "esnm_joint_lcp_cap",
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
            total_trials = len(k_values) * len(eps_values) * n_trials
            all_seeds = seed_rng.randint(0, 2**31 - 1, size=total_trials)
            seed_idx = 0

            try:
                with open(result_file, "w+") as f:
                    print(
                        "method\tk\teps\tmean_l1\tmedian_l1\tp25_l1\tp75_l1\tmean_linf\tmedian_linf\tp25_linf\tp75_linf\tmean_ndcg\tmedian_ndcg\tp25_ndcg\tp75_ndcg\tmean_rank\tmedian_rank\tp25_rank\tp75_rank\tmean_time\tmedian_time\tp25_time\tp75_time",
                        file=f,
                    )

                    for k in k_values:
                        for eps in eps_values:
                            trial_args = []
                            for run in range(n_trials):
                                trial_args.append(
                                    (
                                        int(all_seeds[seed_idx]),
                                        int(k),
                                        eps,
                                        method_name,
                                    )
                                )
                                seed_idx += 1

                            results = pool.map(_run_single_trial, trial_args)

                            l1_errors = [r[0] for r in results]
                            linf_errors = [r[1] for r in results]
                            ndcg_scores = [r[2] for r in results]
                            rank_errors = [r[3] for r in results]
                            time_elapsed_arr = [r[4] for r in results]

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
                            mean_rank = np.mean(rank_errors)
                            median_rank = np.median(rank_errors)
                            p25_rank = np.percentile(rank_errors, 25)
                            p75_rank = np.percentile(rank_errors, 75)
                            mean_time = np.mean(time_elapsed_arr)
                            median_time = np.median(time_elapsed_arr)
                            p25_time = np.percentile(time_elapsed_arr, 25)
                            p75_time = np.percentile(time_elapsed_arr, 75)

                            print(
                                f"{method_name}\t{k}\t{eps:.4f}\t{mean_l1:.6f}\t{median_l1:.6f}\t{p25_l1:.6f}\t{p75_l1:.6f}\t{mean_linf:.6f}\t{median_linf:.6f}\t{p25_linf:.6f}\t{p75_linf:.6f}\t{mean_ndcg:.6f}\t{median_ndcg:.6f}\t{p25_ndcg:.6f}\t{p75_ndcg:.6f}\t{mean_rank:.6f}\t{median_rank:.6f}\t{p25_rank:.6f}\t{p75_rank:.6f}\t{mean_time:.4f}\t{median_time:.4f}\t{p25_time:.4f}\t{p75_time:.4f}",
                                file=f,
                            )
                            print(
                                f"{method_name}\t{k}\t{eps:.4f}\t{mean_l1:.6f}\t{median_l1:.6f}\t{p25_l1:.6f}\t{p75_l1:.6f}\t{mean_linf:.6f}\t{median_linf:.6f}\t{p25_linf:.6f}\t{p75_linf:.6f}\t{mean_ndcg:.6f}\t{median_ndcg:.6f}\t{p25_ndcg:.6f}\t{p75_ndcg:.6f}\t{mean_rank:.6f}\t{median_rank:.6f}\t{p25_rank:.6f}\t{p75_rank:.6f}\t{mean_time:.4f}\t{median_time:.4f}\t{p25_time:.4f}\t{p75_time:.4f}"
                            )

                pool.close()
                pool.join()
            except Exception:
                pool.terminate()
                pool.join()
                raise
