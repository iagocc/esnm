import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from esnm.mechanism import esnm_lln, esnm_t
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

from decision_tree.candidates import element_local_sensitivity_at
from decision_tree.id3 import AttrType, DpID3, InfoGainFriedman, Selection
from local_dampening import shifted_ld
from src.optimize_params import optimize_params_lln, optimize_params_tdist


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


def global_sensitivity(n: int) -> float:
    return np.log(n + 1) + 1 / np.log(2)


def load_dataset(ds_name: str, label_name: str) -> tuple[np.ndarray, np.ndarray]:
    ds_dir = Path.cwd() / "data/tree/"
    filename = ds_name + ".parquet"

    if not (ds_dir / filename).exists():
        raise FileNotFoundError(f"The {filename} does not exists.")

    df = pd.read_parquet((ds_dir / filename).absolute())
    structure = _attr_structure(df.loc[:, df.columns != label_name])

    encoder = LabelEncoder()
    df[label_name] = encoder.fit_transform(df[label_name])

    # convert categorical columns to integer type
    for col, attr_type in zip(df.columns, structure):
        encoder = LabelEncoder()
        if attr_type == AttrType.D.value:
            df[col] = encoder.fit_transform(df[col])

    return df.to_numpy(), structure


def _attr_structure(data: pd.DataFrame) -> np.ndarray:
    types = np.repeat("", data.shape[1])
    c_attr = set(data._get_numeric_data().columns)  # type: ignore
    d_attr = set(data.columns) - c_attr

    cidx = data.columns.get_indexer(c_attr)
    didx = data.columns.get_indexer(d_attr)

    types[cidx] = AttrType.C.value
    types[didx] = AttrType.D.value
    return types


def dataset_domain(data: np.ndarray) -> dict[int, list]:
    domain = {}
    for col in range(data.shape[1]):
        domain[col] = list(set(data[:, col]))

    return domain


def discretize_dataset(data: np.ndarray, data_struc: np.ndarray, nbins=20):
    for i, a in enumerate(data_struc):
        if a == "c":
            att_d = data[:, i]
            buckets = np.linspace(np.min(att_d), np.max(att_d), num=nbins + 1)
            bucketized = np.searchsorted(buckets, att_d)
            data[:, i] = bucketized
            data_struc[i] = "d"

    return data, data_struc


def split_train_test(
    data: np.ndarray, train_ratio=0.8
) -> tuple[np.ndarray, np.ndarray]:
    np.random.shuffle(data)

    train_size: int = int(np.floor(data.shape[0] * train_ratio))
    return data[:train_size, :], data[train_size:, :]


def build_local_sensitivity_matrix(dataset: np.ndarray):
    _, m = dataset.shape
    return element_local_sensitivity_at(dataset, np.arange(m - 1))


@dataclass
class RNM(Selection):
    eps: float
    sens: float

    def __call__(self, u: np.ndarray) -> int:
        noise = np.random.default_rng().exponential(
            scale=((2 * self.sens) / self.eps), size=u.size
        )
        return int(np.argmax(u + noise))


@dataclass
class LocalDampening(Selection):
    eps: float
    gs: float
    ls: np.ndarray

    def __call__(self, u: np.ndarray) -> int:
        return shifted_ld(u, self.gs, self.eps, self.ls)


class Selection_esnm_lln(Selection):
    def __init__(self, ls: np.ndarray, eps: float, R: np.ndarray):
        t_candidates = np.logspace(-9, 10, 150)
        t, sigmas, s, _, ss = optimize_params_lln(eps, t_candidates, ls)

        self.ls = np.ascontiguousarray(ls)
        self.eps = eps
        self.t = np.ascontiguousarray(t)
        self.s = np.ascontiguousarray(s)
        self.ss = np.ascontiguousarray(ss)
        self.sigma = np.ascontiguousarray(sigmas)

    def __call__(self, u: np.ndarray) -> int:
        selected_idx = esnm_lln(
            np.ascontiguousarray(u),
            self.ss.copy(),
            self.s.copy(),
            self.sigma.copy(),
        )
        return selected_idx


class Selection_esnm_t(Selection):
    def __init__(self, ls: np.ndarray, eps: float, R: np.ndarray):
        degree_freedom = 3
        t_candidates = np.linspace(0, eps / (degree_freedom + 1), 150)
        d = len(R)
        t, s, _, ss = optimize_params_tdist(eps, d, t_candidates, ls)

        self.ls = np.ascontiguousarray(ls)
        self.eps = eps
        self.t = np.ascontiguousarray(t)
        self.s = np.ascontiguousarray(s)
        self.ss = np.ascontiguousarray(ss)

    def __call__(self, u: np.ndarray) -> int:
        selected_idx = esnm_t(
            np.ascontiguousarray(u),
            self.ss.copy(),
            self.s.copy(),
            3.0,
        )
        return selected_idx


if __name__ == "__main__":
    set_seeds(42)
    times = 10
    max_n = 50_000
    datasets = [("adult_clean", "income"), ("nltcs", "15"), ("acs", "22")]
    methods = {
        # "rnm": lambda eps, _: RNM(eps=eps, sens=global_sensitivity(max_n)),
        "ld": lambda eps, sens: LocalDampening(
            eps=eps, gs=global_sensitivity(max_n), ls=sens
        ),
        # "esnm_lln": lambda eps, sens: Selection_esnm_lln(sens, eps, R=R.copy()),
        # "esnm_t": lambda eps, sens: Selection_esnm_t(sens, eps, R=R.copy()),
    }
    eps = np.linspace(0.001, 1.0, num=5)
    depths = [2, 5]

    for d, label_col_name in datasets:
        ds, ds_struc = load_dataset(d, label_col_name)
        ds, ds_struc = discretize_dataset(ds, ds_struc, nbins=10)
        ds_dom = dataset_domain(ds)

        n = ds.shape[0]
        R = np.arange(ds.shape[1] - 1)

        ls = build_local_sensitivity_matrix(ds)

        for depth in depths:
            for method_name, method_fn in methods.items():
                with open(f"results/tree/{d}_d{depth}_{method_name}.txt", "w+") as f:
                    print(
                        "method\tdepth\teps\tmean_acc\tstd_acc\tmean_time\tstd_time",
                        file=f,
                    )
                    for e in eps:
                        accs = []
                        time_elapsed_arr = []
                        for _ in range(times):
                            kf = KFold(n_splits=10, shuffle=True)

                            for i, (train_index, test_index) in enumerate(kf.split(ds)):
                                train, test = (
                                    np.copy(ds[train_index, :]),
                                    np.copy(ds[test_index, :]),
                                )
                                start_time = time.perf_counter()

                                privacy_budget = e / (2 * (depth + 1))

                                tree = DpID3(
                                    root=None,
                                    dm=InfoGainFriedman(),
                                    selection=method_fn(privacy_budget, ls.copy()),
                                    dataset=train.copy(),
                                    dataset_struct=ds_struc.copy(),
                                    dataset_domain=ds_dom.copy(),
                                    max_depth=depth,
                                    min_sample=10,
                                    eps=privacy_budget,
                                )

                                tree.fit(train.copy())

                                y_true = test[:, -1].copy().astype(np.int64)
                                y_pred = tree.predict(test[:, :-1].copy())

                                end_time = time.perf_counter()
                                elapsed_time = end_time - start_time
                                time_elapsed_arr.append(elapsed_time)

                                acc = accuracy_score(y_true, y_pred)
                                accs.append(acc)

                        mean_acc = np.mean(accs)
                        std_acc = np.std(accs)

                        print(
                            f"{method_name}\t{depth}\t{e:.4f}\t{mean_acc:.6f}\t{std_acc:.6f}\t{np.mean(time_elapsed_arr):.4f}\t{np.std(time_elapsed_arr):.4f}",
                            file=f,
                        )
                        print(
                            f"{method_name}\t{depth}\t{e:.4f}\t{mean_acc:.6f}\t{std_acc:.6f}\t{np.mean(time_elapsed_arr):.4f}\t{np.std(time_elapsed_arr):.4f}"
                        )
