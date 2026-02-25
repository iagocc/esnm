from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from operator import itemgetter
from typing import Generic, TypeVar, Union, cast

import numpy as np

AttrVal = Union[str, int, float]


class AttrType(Enum):
    D = "d"
    C = "c"


T = TypeVar("T")


@dataclass
class Node(Generic[T]):
    attr: str | int | None
    split_point: T | None
    children: list["Node"] | None = field(default_factory=lambda: [])


@dataclass(kw_only=True)
class Leaf(Node):
    label_counts: Counter
    major: AttrVal | None

    def majority(self):
        return max(self.label_counts, key=itemgetter(1))


class InfoGainFriedman:
    def __init__(self) -> None:
        super().__init__()

    def entropy_(self, C):
        n = C.shape[0]
        unique, counts = np.unique(C, return_counts=True)
        props = counts / n
        return -np.sum(props * np.log2(props))

    def __call__(self, data: np.ndarray, attr: int) -> float:
        n = data.shape[0]
        X = data[:, :-1]
        y = data[:, -1]
        unique, counts = np.unique(X[:, attr], return_counts=True)
        sum_entropy = 0.0
        for j, count in zip(unique, counts):
            mask = X[:, attr] == j
            sum_entropy += (count / n) * self.entropy_(y[mask])
        return -n * sum_entropy


class Selection:
    sens: float

    def __call__(self, u: np.ndarray) -> int:
        raise Exception("The selection algorithm should be implemented.")


@dataclass
class DpID3:
    dm: InfoGainFriedman
    dataset: np.ndarray
    dataset_struct: np.ndarray
    dataset_domain: dict[int, list]
    max_depth: int
    min_sample: int
    eps: float
    root: Node | None
    selection: Selection
    _C: list = field(init=False)

    def __post_init__(self):
        self._C = list(set(self.dataset[:, -1]))

    def predict(self, dataset: np.ndarray) -> np.ndarray:
        predictions = []
        for r in dataset:
            node = self.root

            if not node:
                raise Exception("There is no root.")

            while node.children:
                for c in node.children:
                    if r[c.attr] == c.split_point:
                        node = c
                        break

            leaf: Leaf = cast(Leaf, node)
            predictions.append(leaf.major)

        return np.array(predictions)

    def fit(self, dataset: np.ndarray) -> None:
        node = Node(None, None)
        self.root = self._build_tree(
            node, dataset, set(np.arange(self.dataset_struct.shape[0])), self.max_depth
        )

    def _get_domain(self, dataset: np.ndarray, attr: int) -> np.ndarray:
        return np.array(list(set(dataset[:, attr])))

    def _build_tree(
        self, node: Node, dataset: np.ndarray, attr_set: set[int], depth: int
    ) -> Node:
        _, m = dataset.shape
        t = max([len(list(set(dataset[:, col]))) for col in range(m)]) / 5
        Nt = self._noisy_count(dataset.shape[0])

        if (
            (t == 0)
            or (m == 0)
            or (len(attr_set) == 0)
            or (depth == 0)
            or (Nt / (t * len(self._C)) <= np.sqrt(2) / self.eps)
        ):
            Tc = [self._partition(dataset, col_idx=-1, val=c) for c in self._C]
            Nc = [self._noisy_count(t.shape[0]) for t in Tc]
            return Leaf(
                attr=node.attr,
                split_point=node.split_point,
                children=None,
                label_counts=Counter(dataset[:, -1]),
                major=self._C[np.argmax(Nc)],
            )

        igs = np.array([self.dm(data=dataset, attr=col) for col in attr_set])
        selected_idx = self.selection(igs)
        selected = list(attr_set)[selected_idx]

        attr_set.remove(selected)

        for v in self.dataset_domain[selected]:
            node_v = Node(selected, v)
            T_i = self._partition(dataset=dataset, col_idx=selected, val=v)
            np.delete(dataset, selected)
            if node.children is not None:
                node.children.append(self._build_tree(node_v, T_i, attr_set, depth - 1))
            else:
                raise Exception("Node children is None.")

        return node

    def _noisy_count(self, count: int) -> float:
        return (
            count
            # + np.random.default_rng().laplace(loc=0, scale=(1 / self.eps), size=1)[0]
        )

    def _partition(self, dataset: np.ndarray, col_idx: int, val: AttrVal) -> np.ndarray:
        mask = dataset[:, col_idx] == val
        return dataset[mask, :]
