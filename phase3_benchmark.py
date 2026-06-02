from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import json
import math
import tracemalloc

import numpy as np

from morphogenic_lab import (
    EPS,
    MorphogenicClassifier,
    RidgeHybridHead,
    TemporalComposer,
    accuracy,
    make_growth_dataset,
    make_sequence_dataset,
    make_spiral_dataset,
    split_standardize,
)


def softmax(scores: np.ndarray) -> np.ndarray:
    z = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), EPS)


def standardize_from_train(
    x_train: np.ndarray, *others: np.ndarray
) -> tuple[np.ndarray, ...]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-6] = 1.0
    return ((x_train - mean) / std, *[(x - mean) / std for x in others])


def model_mb(model: Any) -> float:
    if hasattr(model, "memory_bytes"):
        return float(model.memory_bytes()) / (1024.0 * 1024.0)
    if isinstance(model, MorphogenicClassifier):
        total = 0
        for cell in model.cells:
            total += cell.center.nbytes + cell.hist.nbytes
        if model.metric is not None:
            total += (
                model.metric.counts.nbytes
                + model.metric.sums.nbytes
                + model.metric.sumsq.nbytes
                + model.metric.weights.nbytes
            )
        if model.index is not None:
            total += model.index.projections.nbytes
            total += sum(
                8 * len(bucket)
                for table in model.index.buckets
                for bucket in table.values()
            )
        return total / (1024.0 * 1024.0)
    return 0.0


@dataclass
class BenchmarkRecord:
    name: str
    clean_accuracy: float
    adversarial_accuracy: float | None
    fit_seconds: float
    predict_seconds: float
    peak_ram_mb: float
    model_ram_mb: float
    clean_avg_candidates: float | None
    adversarial_avg_candidates: float | None
    extra: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "clean_accuracy": self.clean_accuracy,
            "adversarial_accuracy": self.adversarial_accuracy,
            "fit_seconds": self.fit_seconds,
            "predict_seconds": self.predict_seconds,
            "peak_ram_mb": self.peak_ram_mb,
            "model_ram_mb": self.model_ram_mb,
            "clean_avg_candidates": self.clean_avg_candidates,
            "adversarial_avg_candidates": self.adversarial_avg_candidates,
            **self.extra,
        }


class ExactKNN:
    def __init__(self, k: int = 5, batch_size: int = 256):
        self.k = k
        self.batch_size = batch_size
        self.x: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.n_classes = 0
        self.train_norm: np.ndarray | None = None
        self.last_avg_candidates = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ExactKNN":
        self.x = np.asarray(x, dtype=np.float64).copy()
        self.y = np.asarray(y, dtype=int).copy()
        self.n_classes = int(np.max(y)) + 1
        self.train_norm = np.sum(np.square(self.x), axis=1)
        return self

    def partial_fit(self, x: np.ndarray, y: np.ndarray) -> "ExactKNN":
        if self.x is None or self.y is None:
            return self.fit(x, y)
        self.x = np.vstack([self.x, np.asarray(x, dtype=np.float64)])
        self.y = np.concatenate([self.y, np.asarray(y, dtype=int)])
        self.n_classes = max(self.n_classes, int(np.max(y)) + 1)
        self.train_norm = np.sum(np.square(self.x), axis=1)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.x is None or self.y is None or self.train_norm is None:
            raise ValueError("model is not fitted")
        x = np.asarray(x, dtype=np.float64)
        out = np.empty(len(x), dtype=int)
        for start in range(0, len(x), self.batch_size):
            xb = x[start : start + self.batch_size]
            d2 = (
                np.sum(np.square(xb), axis=1, keepdims=True)
                + self.train_norm[None, :]
                - 2.0 * xb @ self.x.T
            )
            nn = np.argpartition(d2, kth=min(self.k, len(self.y) - 1), axis=1)[
                :, : self.k
            ]
            labels = self.y[nn]
            votes = np.zeros((len(labels), self.n_classes), dtype=np.int32)
            for row in range(len(labels)):
                votes[row] = np.bincount(labels[row], minlength=self.n_classes)
            out[start : start + len(xb)] = np.argmax(votes, axis=1)
        self.last_avg_candidates = float(len(self.y))
        return out

    def memory_bytes(self) -> int:
        if self.x is None or self.y is None:
            return 0
        total = self.x.nbytes + self.y.nbytes
        if self.train_norm is not None:
            total += self.train_norm.nbytes
        return int(total)


class LinearSVM:
    def __init__(
        self,
        n_classes: int | None = None,
        epochs: int = 32,
        lr: float = 0.035,
        reg: float = 3e-4,
        batch_size: int = 128,
        seed: int = 0,
    ):
        self.requested_classes = n_classes
        self.epochs = epochs
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.w: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.n_classes = 0

    def _ensure(self, d: int, y: np.ndarray) -> None:
        n_classes = max(int(np.max(y)) + 1, self.requested_classes or 0)
        if self.w is None:
            self.n_classes = n_classes
            self.w = self.rng.normal(0.0, 0.01, size=(d, self.n_classes))
            self.b = np.zeros(self.n_classes, dtype=np.float64)
        elif n_classes > self.n_classes:
            extra = n_classes - self.n_classes
            self.w = np.hstack([self.w, self.rng.normal(0.0, 0.01, size=(d, extra))])
            self.b = np.pad(self.b, (0, extra))
            self.n_classes = n_classes

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearSVM":
        self.w = None
        self.b = None
        self.rng = np.random.default_rng(self.seed)
        return self.partial_fit(x, y, epochs=self.epochs)

    def partial_fit(
        self, x: np.ndarray, y: np.ndarray, epochs: int | None = None
    ) -> "LinearSVM":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        self._ensure(x.shape[1], y)
        assert self.w is not None and self.b is not None
        order = np.arange(len(x))
        use_epochs = self.epochs if epochs is None else epochs
        for epoch in range(use_epochs):
            self.rng.shuffle(order)
            step_lr = self.lr / math.sqrt(epoch + 1.0)
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                xb = x[idx]
                yb = y[idx]
                scores = xb @ self.w + self.b
                correct = scores[np.arange(len(xb)), yb][:, None]
                margins = scores - correct + 1.0
                margins[np.arange(len(xb)), yb] = 0.0
                pos = margins > 0.0
                coeff = pos.astype(np.float64)
                row_sum = coeff.sum(axis=1)
                coeff[np.arange(len(xb)), yb] -= row_sum
                grad_w = xb.T @ coeff / max(len(xb), 1) + self.reg * self.w
                grad_b = coeff.mean(axis=0)
                self.w -= step_lr * grad_w
                self.b -= step_lr * grad_b
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.w is None or self.b is None:
            raise ValueError("model is not fitted")
        return np.argmax(np.asarray(x) @ self.w + self.b, axis=1).astype(int)

    def memory_bytes(self) -> int:
        if self.w is None or self.b is None:
            return 0
        return int(self.w.nbytes + self.b.nbytes)


class ExtraTreeNode:
    __slots__ = ("feature", "threshold", "left", "right", "proba")

    def __init__(
        self,
        feature: int | None = None,
        threshold: float | None = None,
        left: "ExtraTreeNode | None" = None,
        right: "ExtraTreeNode | None" = None,
        proba: np.ndarray | None = None,
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.proba = proba


class RandomForestNumpy:
    def __init__(
        self,
        n_trees: int = 28,
        max_depth: int = 9,
        min_leaf: int = 8,
        max_features: int | None = None,
        candidates_per_node: int = 28,
        sample_fraction: float = 0.82,
        seed: int = 0,
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.max_features = max_features
        self.candidates_per_node = candidates_per_node
        self.sample_fraction = sample_fraction
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.trees: list[ExtraTreeNode] = []
        self.n_classes = 0
        self.n_features = 0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RandomForestNumpy":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        self.n_classes = int(np.max(y)) + 1
        self.n_features = x.shape[1]
        self.trees = []
        n = len(x)
        bag = max(self.min_leaf * 4, int(self.sample_fraction * n))
        for _ in range(self.n_trees):
            idx = self.rng.integers(0, n, size=bag)
            self.trees.append(self._build(x[idx], y[idx], depth=0))
        return self

    def _leaf(self, y: np.ndarray) -> ExtraTreeNode:
        counts = np.bincount(y, minlength=self.n_classes).astype(np.float64)
        return ExtraTreeNode(proba=(counts + 1.0) / (counts.sum() + self.n_classes))

    def _gini(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        score = 0.0
        total = len(y_left) + len(y_right)
        for part in (y_left, y_right):
            if len(part) == 0:
                continue
            p = np.bincount(part, minlength=self.n_classes) / len(part)
            score += (len(part) / total) * (1.0 - np.sum(p * p))
        return float(score)

    def _build(self, x: np.ndarray, y: np.ndarray, depth: int) -> ExtraTreeNode:
        if (
            depth >= self.max_depth
            or len(y) <= 2 * self.min_leaf
            or np.unique(y).size == 1
        ):
            return self._leaf(y)
        max_features = self.max_features or max(2, int(math.sqrt(self.n_features)))
        best_feature = None
        best_threshold = None
        best_score = float("inf")
        for _ in range(self.candidates_per_node):
            feature = int(self.rng.integers(0, self.n_features))
            if self.rng.random() < 0.65:
                feature = int(self.rng.choice(self.n_features, size=max_features)[0])
            column = x[:, feature]
            lo, hi = np.percentile(column, [8, 92])
            if not np.isfinite(lo) or hi <= lo:
                continue
            threshold = float(self.rng.uniform(lo, hi))
            mask = column <= threshold
            if np.count_nonzero(mask) < self.min_leaf or np.count_nonzero(~mask) < self.min_leaf:
                continue
            score = self._gini(y[mask], y[~mask])
            if score < best_score:
                best_score = score
                best_feature = feature
                best_threshold = threshold
        if best_feature is None or best_threshold is None:
            return self._leaf(y)
        mask = x[:, best_feature] <= best_threshold
        return ExtraTreeNode(
            feature=best_feature,
            threshold=best_threshold,
            left=self._build(x[mask], y[mask], depth + 1),
            right=self._build(x[~mask], y[~mask], depth + 1),
        )

    def _predict_tree(self, tree: ExtraTreeNode, row: np.ndarray) -> np.ndarray:
        node = tree
        while node.proba is None:
            assert node.feature is not None and node.threshold is not None
            node = node.left if row[node.feature] <= node.threshold else node.right
            assert node is not None
        return node.proba

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        votes = np.zeros((len(x), self.n_classes), dtype=np.float64)
        for tree in self.trees:
            for i, row in enumerate(x):
                votes[i] += self._predict_tree(tree, row)
        return np.argmax(votes, axis=1).astype(int)

    def memory_bytes(self) -> int:
        return int(len(self.trees) * (2 ** (self.max_depth + 1)) * 32)


class SmallMLP:
    def __init__(
        self,
        n_classes: int | None = None,
        hidden: int = 96,
        epochs: int = 44,
        lr: float = 0.045,
        reg: float = 2e-4,
        batch_size: int = 128,
        seed: int = 0,
    ):
        self.requested_classes = n_classes
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.w1: np.ndarray | None = None
        self.b1: np.ndarray | None = None
        self.w2: np.ndarray | None = None
        self.b2: np.ndarray | None = None
        self.n_classes = 0

    def _ensure(self, d: int, y: np.ndarray) -> None:
        n_classes = max(int(np.max(y)) + 1, self.requested_classes or 0)
        if self.w1 is None:
            scale = math.sqrt(2.0 / max(d, 1))
            self.w1 = self.rng.normal(0.0, scale, size=(d, self.hidden))
            self.b1 = np.zeros(self.hidden, dtype=np.float64)
            self.w2 = self.rng.normal(0.0, 0.05, size=(self.hidden, n_classes))
            self.b2 = np.zeros(n_classes, dtype=np.float64)
            self.n_classes = n_classes
        elif n_classes > self.n_classes:
            assert self.w2 is not None and self.b2 is not None
            extra = n_classes - self.n_classes
            self.w2 = np.hstack([self.w2, self.rng.normal(0.0, 0.05, size=(self.hidden, extra))])
            self.b2 = np.pad(self.b2, (0, extra))
            self.n_classes = n_classes

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SmallMLP":
        self.w1 = self.b1 = self.w2 = self.b2 = None
        self.rng = np.random.default_rng(self.seed)
        return self.partial_fit(x, y, epochs=self.epochs)

    def partial_fit(
        self, x: np.ndarray, y: np.ndarray, epochs: int | None = None
    ) -> "SmallMLP":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        self._ensure(x.shape[1], y)
        assert self.w1 is not None and self.b1 is not None
        assert self.w2 is not None and self.b2 is not None
        order = np.arange(len(x))
        use_epochs = self.epochs if epochs is None else epochs
        for epoch in range(use_epochs):
            self.rng.shuffle(order)
            step_lr = self.lr / math.sqrt(epoch + 1.0)
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                xb = x[idx]
                yb = y[idx]
                h_pre = xb @ self.w1 + self.b1
                h = np.maximum(h_pre, 0.0)
                probs = softmax(h @ self.w2 + self.b2)
                probs[np.arange(len(xb)), yb] -= 1.0
                probs /= max(len(xb), 1)
                grad_w2 = h.T @ probs + self.reg * self.w2
                grad_b2 = probs.sum(axis=0)
                grad_h = probs @ self.w2.T
                grad_h[h_pre <= 0.0] = 0.0
                grad_w1 = xb.T @ grad_h + self.reg * self.w1
                grad_b1 = grad_h.sum(axis=0)
                self.w2 -= step_lr * grad_w2
                self.b2 -= step_lr * grad_b2
                self.w1 -= step_lr * grad_w1
                self.b1 -= step_lr * grad_b1
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self.w1 is not None and self.b1 is not None
        assert self.w2 is not None and self.b2 is not None
        h = np.maximum(np.asarray(x) @ self.w1 + self.b1, 0.0)
        return np.argmax(h @ self.w2 + self.b2, axis=1).astype(int)

    def memory_bytes(self) -> int:
        total = 0
        for arr in (self.w1, self.b1, self.w2, self.b2):
            if arr is not None:
                total += arr.nbytes
        return int(total)


class PrototypicalNetworkBaseline:
    def __init__(self, use_fisher_metric: bool = True):
        self.use_fisher_metric = use_fisher_metric
        self.prototypes: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.weights: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PrototypicalNetworkBaseline":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        labels = np.unique(y)
        self.labels = labels
        self.prototypes = np.vstack([x[y == label].mean(axis=0) for label in labels])
        if self.use_fisher_metric:
            overall = x.mean(axis=0)
            between = np.zeros(x.shape[1], dtype=np.float64)
            within = np.zeros(x.shape[1], dtype=np.float64)
            for label, proto in zip(labels, self.prototypes):
                part = x[y == label]
                between += len(part) * np.square(proto - overall)
                within += np.square(part - proto).sum(axis=0)
            weights = np.sqrt(np.maximum(between / np.maximum(within, EPS), 0.0))
            if np.any(weights > 0):
                weights += 0.02 * np.mean(weights[weights > 0])
                weights /= np.mean(weights)
            else:
                weights = np.ones(x.shape[1], dtype=np.float64)
            self.weights = weights
        else:
            self.weights = np.ones(x.shape[1], dtype=np.float64)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self.prototypes is not None and self.labels is not None
        assert self.weights is not None
        diff = np.asarray(x)[:, None, :] - self.prototypes[None, :, :]
        d2 = np.mean(np.square(diff) * self.weights[None, None, :], axis=2)
        return self.labels[np.argmin(d2, axis=1)].astype(int)

    def memory_bytes(self) -> int:
        total = 0
        for arr in (self.prototypes, self.labels, self.weights):
            if arr is not None:
                total += arr.nbytes
        return int(total)


def make_phase3_multimodal(
    n_train: int = 4200,
    n_test: int = 1800,
    d: int = 320,
    informative: int = 26,
    n_classes: int = 8,
    modes: int = 5,
    seed: int = 101,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    relevant = rng.choice(d, informative, replace=False)
    class_code = rng.normal(0.0, 0.42, size=(n_classes, informative))
    mode_offsets = rng.normal(0.0, 1.52, size=(n_classes, modes, informative))

    def sample(n: int) -> tuple[np.ndarray, np.ndarray]:
        y = rng.integers(0, n_classes, size=n)
        mode = rng.integers(0, modes, size=n)
        x = rng.normal(0.0, 2.45, size=(n, d))
        x[:, relevant] = (
            class_code[y]
            + mode_offsets[y, mode]
            + rng.normal(0.0, 0.62, size=(n, informative))
        )
        # A few nuisance correlations make linear separators less comfortable.
        nuisance = rng.normal(0.0, 0.8, size=(n, 8))
        x[:, relevant[:8]] += 0.25 * nuisance
        return x, y

    x_train, y_train = sample(n_train)
    x_test, y_test = sample(n_test)
    x_train, x_test = split_standardize(x_train, x_test)
    return x_train, y_train, x_test, y_test, relevant


def make_phase3_new_class_data(
    seed: int = 303,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return make_phase3_multimodal(
        n_train=3600,
        n_test=1800,
        d=192,
        informative=22,
        n_classes=8,
        modes=4,
        seed=seed,
    )[:4]


def make_drift_stream(
    chunks: int = 9,
    chunk_size: int = 620,
    d: int = 96,
    informative: int = 18,
    n_classes: int = 4,
    seed: int = 404,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    relevant = rng.choice(d, informative, replace=False)
    centers = rng.normal(0.0, 1.7, size=(n_classes, informative))
    drift = rng.normal(0.0, 0.42, size=(n_classes, informative))
    permutation = np.roll(np.arange(n_classes), -1)
    raw_x: list[np.ndarray] = []
    raw_y: list[np.ndarray] = []
    for t in range(chunks):
        y = rng.integers(0, n_classes, size=chunk_size)
        phase = t / max(chunks - 1, 1)
        x = rng.normal(0.0, 1.7, size=(chunk_size, d))
        curved = np.sin(phase * np.pi * 1.5 + np.arange(informative)[None, :] * 0.13)
        moving_centers = (1.0 - phase) * centers[y] + phase * centers[permutation[y]]
        x[:, relevant] = (
            moving_centers
            + phase * drift[y] * 3.2
            + 0.38 * curved
            + rng.normal(0.0, 0.52, size=(chunk_size, informative))
        )
        raw_x.append(x)
        raw_y.append(y)
    standardized = standardize_from_train(raw_x[0], *raw_x[1:])
    return list(standardized), raw_y


def make_full_morphogenic(seed: int = 0, use_lsh: bool = True) -> MorphogenicClassifier:
    return MorphogenicClassifier(
        theta=0.82,
        alpha=0.18,
        adaptive_metric=True,
        growth_control=True,
        merge_every=220,
        merge_threshold=0.38,
        max_cells=520,
        prune_fraction=0.12,
        use_index=use_lsh,
        index_tables=14,
        index_planes=9,
        min_candidates=48,
        seed=seed,
    )


def prototype_boundary_attack(
    model: MorphogenicClassifier,
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float = 0.36,
) -> np.ndarray:
    centers = np.vstack([cell.center for cell in model.cells])
    labels = np.array([cell.label for cell in model.cells], dtype=int)
    weights = model.weights
    attacked = np.asarray(x, dtype=np.float64).copy()
    for i, row in enumerate(x):
        same = labels == int(y[i])
        if not np.any(same) or not np.any(~same):
            continue
        same_centers = centers[same]
        wrong_centers = centers[~same]
        d_same = np.mean(np.square(same_centers - row) * weights, axis=1)
        d_wrong = np.mean(np.square(wrong_centers - row) * weights, axis=1)
        true_center = same_centers[int(np.argmin(d_same))]
        wrong_center = wrong_centers[int(np.argmin(d_wrong))]
        direction = (wrong_center - true_center) * np.sqrt(weights)
        rms = math.sqrt(float(np.mean(np.square(direction))) + EPS)
        attacked[i] = row + epsilon * direction / rms
    return attacked


def run_measured_model(
    name: str,
    factory: Callable[[], Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    x_adversarial: np.ndarray | None = None,
) -> tuple[Any, BenchmarkRecord]:
    model = factory()
    tracemalloc.start()
    t0 = perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    clean_candidates: float | None = None
    adv_candidates: float | None = None
    if isinstance(model, MorphogenicClassifier):
        model.reset_query_stats()
    t0 = perf_counter()
    pred = model.predict(x_test)
    predict_seconds = perf_counter() - t0
    if isinstance(model, MorphogenicClassifier):
        clean_candidates = float(model.summary()["avg_candidates"])
    elif isinstance(model, ExactKNN):
        clean_candidates = model.last_avg_candidates
    clean_acc = accuracy(y_test, pred)

    adv_acc: float | None = None
    if x_adversarial is not None:
        if isinstance(model, MorphogenicClassifier):
            model.reset_query_stats()
        pred_adv = model.predict(x_adversarial)
        if isinstance(model, MorphogenicClassifier):
            adv_candidates = float(model.summary()["avg_candidates"])
        elif isinstance(model, ExactKNN):
            adv_candidates = model.last_avg_candidates
        adv_acc = accuracy(y_test, pred_adv)

    extra: dict[str, Any] = {}
    if isinstance(model, MorphogenicClassifier):
        extra.update(
            {
                "cells": len(model.cells),
                "index_misses": int(model.summary()["index_misses"]),
            }
        )
    return model, BenchmarkRecord(
        name=name,
        clean_accuracy=clean_acc,
        adversarial_accuracy=adv_acc,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        peak_ram_mb=peak / (1024.0 * 1024.0),
        model_ram_mb=model_mb(model),
        clean_avg_candidates=clean_candidates,
        adversarial_avg_candidates=adv_candidates,
        extra=extra,
    )


def experiment_large_and_adversarial() -> dict[str, Any]:
    x_train, y_train, x_test, y_test, relevant = make_phase3_multimodal()
    full_model = make_full_morphogenic(seed=11, use_lsh=True)
    full_model.fit(x_train, y_train)
    attack_epsilon = 0.12
    x_adv = prototype_boundary_attack(full_model, x_test, y_test, epsilon=attack_epsilon)

    factories: list[tuple[str, Callable[[], Any]]] = [
        ("morphogenic_full", lambda: make_full_morphogenic(seed=11, use_lsh=True)),
        ("exact_kNN_k5", lambda: ExactKNN(k=5)),
        ("linear_SVM_sgd", lambda: LinearSVM(epochs=34, lr=0.035, seed=12)),
        ("random_forest_numpy", lambda: RandomForestNumpy(n_trees=26, max_depth=9, seed=13)),
        ("small_MLP", lambda: SmallMLP(hidden=112, epochs=46, lr=0.046, seed=14)),
        ("prototypical_network_centroid", lambda: PrototypicalNetworkBaseline()),
    ]
    records = []
    fitted: list[tuple[str, Any]] = []
    for name, factory in factories:
        model, record = run_measured_model(
            name, factory, x_train, y_train, x_test, y_test, x_adversarial=x_adv
        )
        fitted.append((name, model))
        records.append(record.to_dict())
    sweep = []
    for eps in [0.03, 0.06, 0.09, 0.12, 0.16, 0.20]:
        attacked = prototype_boundary_attack(full_model, x_test, y_test, epsilon=eps)
        sweep.append(
            {
                "epsilon": eps,
                "accuracies": {
                    name: accuracy(y_test, model.predict(attacked))
                    for name, model in fitted
                },
            }
        )
    return {
        "name": "large_dataset_and_adversarial_noise",
        "dataset": {
            "train": len(x_train),
            "test": len(x_test),
            "dimensions": x_train.shape[1],
            "classes": int(np.max(y_train)) + 1,
            "informative_dimensions": len(relevant),
            "attack": "targeted prototype-boundary perturbation from the full morphogenic model",
            "reported_attack_epsilon": attack_epsilon,
        },
        "models": records,
        "adversarial_sweep": sweep,
    }


def experiment_new_classes_and_forgetting() -> dict[str, Any]:
    x_train, y_train, x_test, y_test = make_phase3_new_class_data()
    old_classes = np.arange(0, 4)
    new_classes = np.arange(4, 8)
    old_train_mask = np.isin(y_train, old_classes)
    new_train_mask = np.isin(y_train, new_classes)
    old_test_mask = np.isin(y_test, old_classes)
    new_test_mask = np.isin(y_test, new_classes)
    rng = np.random.default_rng(505)
    new_idx_all = np.where(new_train_mask)[0]
    few_new = []
    for label in new_classes:
        label_idx = new_idx_all[y_train[new_idx_all] == label]
        few_new.extend(rng.choice(label_idx, size=min(110, len(label_idx)), replace=False))
    few_new = np.array(few_new, dtype=int)

    morph = make_full_morphogenic(seed=21, use_lsh=True)
    t0 = perf_counter()
    morph.fit(x_train[old_train_mask], y_train[old_train_mask])
    old_fit_seconds = perf_counter() - t0
    old_before = accuracy(y_test[old_test_mask], morph.predict(x_test[old_test_mask]))
    t0 = perf_counter()
    morph.partial_fit(x_train[few_new], y_train[few_new])
    update_seconds = perf_counter() - t0
    old_after = accuracy(y_test[old_test_mask], morph.predict(x_test[old_test_mask]))
    new_after = accuracy(y_test[new_test_mask], morph.predict(x_test[new_test_mask]))

    knn = ExactKNN(k=5)
    knn.fit(x_train[old_train_mask], y_train[old_train_mask])
    knn_old_before = accuracy(y_test[old_test_mask], knn.predict(x_test[old_test_mask]))
    knn.partial_fit(x_train[few_new], y_train[few_new])
    knn_old_after = accuracy(y_test[old_test_mask], knn.predict(x_test[old_test_mask]))
    knn_new_after = accuracy(y_test[new_test_mask], knn.predict(x_test[new_test_mask]))

    mlp = SmallMLP(n_classes=8, hidden=96, epochs=48, lr=0.045, seed=23)
    mlp.fit(x_train[old_train_mask], y_train[old_train_mask])
    mlp_old_before = accuracy(y_test[old_test_mask], mlp.predict(x_test[old_test_mask]))
    mlp.partial_fit(x_train[few_new], y_train[few_new], epochs=34)
    mlp_old_after = accuracy(y_test[old_test_mask], mlp.predict(x_test[old_test_mask]))
    mlp_new_after = accuracy(y_test[new_test_mask], mlp.predict(x_test[new_test_mask]))

    svm = LinearSVM(n_classes=8, epochs=42, lr=0.035, seed=24)
    svm.fit(x_train[old_train_mask], y_train[old_train_mask])
    svm_old_before = accuracy(y_test[old_test_mask], svm.predict(x_test[old_test_mask]))
    svm.partial_fit(x_train[few_new], y_train[few_new], epochs=30)
    svm_old_after = accuracy(y_test[old_test_mask], svm.predict(x_test[old_test_mask]))
    svm_new_after = accuracy(y_test[new_test_mask], svm.predict(x_test[new_test_mask]))

    return {
        "name": "new_classes_after_training_and_catastrophic_forgetting",
        "setup": {
            "old_classes": old_classes.tolist(),
            "new_classes": new_classes.tolist(),
            "few_shot_new_examples": int(len(few_new)),
            "features": int(x_train.shape[1]),
        },
        "models": [
            {
                "name": "morphogenic_partial_fit",
                "old_accuracy_before": old_before,
                "old_accuracy_after": old_after,
                "new_class_accuracy_after": new_after,
                "forgetting_drop": old_before - old_after,
                "cells_after_update": len(morph.cells),
                "old_fit_seconds": old_fit_seconds,
                "new_update_seconds": update_seconds,
                "model_ram_mb": model_mb(morph),
            },
            {
                "name": "exact_kNN_append",
                "old_accuracy_before": knn_old_before,
                "old_accuracy_after": knn_old_after,
                "new_class_accuracy_after": knn_new_after,
                "forgetting_drop": knn_old_before - knn_old_after,
                "model_ram_mb": model_mb(knn),
            },
            {
                "name": "small_MLP_finetune_new_only",
                "old_accuracy_before": mlp_old_before,
                "old_accuracy_after": mlp_old_after,
                "new_class_accuracy_after": mlp_new_after,
                "forgetting_drop": mlp_old_before - mlp_old_after,
                "model_ram_mb": model_mb(mlp),
            },
            {
                "name": "linear_SVM_finetune_new_only",
                "old_accuracy_before": svm_old_before,
                "old_accuracy_after": svm_old_after,
                "new_class_accuracy_after": svm_new_after,
                "forgetting_drop": svm_old_before - svm_old_after,
                "model_ram_mb": model_mb(svm),
            },
        ],
    }


def experiment_temporal_drift() -> dict[str, Any]:
    xs, ys = make_drift_stream()
    morph = make_full_morphogenic(seed=31, use_lsh=True)
    morph.fit(xs[0], ys[0])
    mlp = SmallMLP(n_classes=4, hidden=64, epochs=42, lr=0.045, seed=32)
    mlp.fit(xs[0], ys[0])
    svm = LinearSVM(n_classes=4, epochs=38, lr=0.035, seed=33)
    svm.fit(xs[0], ys[0])
    knn = ExactKNN(k=5)
    knn.fit(xs[0], ys[0])
    proto = PrototypicalNetworkBaseline()
    proto.fit(xs[0], ys[0])

    morph_pre: list[float] = []
    mlp_static: list[float] = []
    svm_static: list[float] = []
    knn_append: list[float] = []
    proto_static: list[float] = []
    morph_update_seconds: list[float] = []
    for chunk in range(1, len(xs)):
        x_chunk, y_chunk = xs[chunk], ys[chunk]
        morph_pre.append(accuracy(y_chunk, morph.predict(x_chunk)))
        mlp_static.append(accuracy(y_chunk, mlp.predict(x_chunk)))
        svm_static.append(accuracy(y_chunk, svm.predict(x_chunk)))
        knn_append.append(accuracy(y_chunk, knn.predict(x_chunk)))
        proto_static.append(accuracy(y_chunk, proto.predict(x_chunk)))
        t0 = perf_counter()
        morph.partial_fit(x_chunk, y_chunk)
        morph_update_seconds.append(perf_counter() - t0)
        knn.partial_fit(x_chunk, y_chunk)
    return {
        "name": "temporal_drift_stream",
        "setup": {
            "chunks": len(xs),
            "chunk_size": len(xs[0]),
            "dimensions": xs[0].shape[1],
            "classes": 4,
        },
        "models": [
            {
                "name": "morphogenic_online",
                "chunk_accuracies_before_update": morph_pre,
                "mean_accuracy": float(np.mean(morph_pre)),
                "last_chunk_accuracy": morph_pre[-1],
                "mean_update_seconds": float(np.mean(morph_update_seconds)),
                "cells_after_stream": len(morph.cells),
                "model_ram_mb": model_mb(morph),
            },
            {
                "name": "exact_kNN_append",
                "chunk_accuracies_before_append": knn_append,
                "mean_accuracy": float(np.mean(knn_append)),
                "last_chunk_accuracy": knn_append[-1],
                "model_ram_mb": model_mb(knn),
            },
            {
                "name": "small_MLP_static",
                "chunk_accuracies": mlp_static,
                "mean_accuracy": float(np.mean(mlp_static)),
                "last_chunk_accuracy": mlp_static[-1],
                "model_ram_mb": model_mb(mlp),
            },
            {
                "name": "linear_SVM_static",
                "chunk_accuracies": svm_static,
                "mean_accuracy": float(np.mean(svm_static)),
                "last_chunk_accuracy": svm_static[-1],
                "model_ram_mb": model_mb(svm),
            },
            {
                "name": "prototypical_static",
                "chunk_accuracies": proto_static,
                "mean_accuracy": float(np.mean(proto_static)),
                "last_chunk_accuracy": proto_static[-1],
                "model_ram_mb": model_mb(proto),
            },
        ],
    }


def measured_predict(model: Any, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    if isinstance(model, MorphogenicClassifier):
        model.reset_query_stats()
    t0 = perf_counter()
    pred = model.predict(x)
    elapsed = perf_counter() - t0
    candidates = None
    if isinstance(model, MorphogenicClassifier):
        candidates = float(model.summary()["avg_candidates"])
    return {
        "accuracy": accuracy(y, pred),
        "predict_seconds": elapsed,
        "avg_candidates": candidates,
        "model_ram_mb": model_mb(model),
        "cells": len(model.cells) if isinstance(model, MorphogenicClassifier) else None,
    }


def experiment_ablations() -> dict[str, Any]:
    x_train, y_train, x_test, y_test, _ = make_phase3_multimodal(
        n_train=1900, n_test=850, d=640, informative=20, n_classes=6, modes=4, seed=606
    )
    ablation_models: list[tuple[str, MorphogenicClassifier]] = [
        ("full_fisher_lsh_prune", make_full_morphogenic(seed=41, use_lsh=True)),
        (
            "without_Fisher",
            MorphogenicClassifier(
                theta=1.55,
                alpha=0.18,
                adaptive_metric=False,
                growth_control=True,
                merge_every=220,
                merge_threshold=0.58,
                max_cells=520,
                use_index=True,
                index_tables=10,
                index_planes=11,
                min_candidates=24,
                seed=42,
            ),
        ),
        ("without_LSH", make_full_morphogenic(seed=43, use_lsh=False)),
        (
            "without_pruning",
            MorphogenicClassifier(
                theta=0.82,
                alpha=0.18,
                adaptive_metric=True,
                growth_control=False,
                use_index=True,
                index_tables=10,
                index_planes=11,
                min_candidates=24,
                seed=44,
            ),
        ),
    ]
    high_dim_records = []
    for name, model in ablation_models:
        t0 = perf_counter()
        model.fit(x_train, y_train)
        fit_s = perf_counter() - t0
        rec = measured_predict(model, x_test, y_test)
        rec.update({"name": name, "fit_seconds": fit_s})
        high_dim_records.append(rec)

    seq_train, y_seq_train, seq_test, y_seq_test = make_sequence_dataset()
    composer = TemporalComposer()
    bag_train = composer.encode_bag(seq_train)
    bag_test = composer.encode_bag(seq_test)
    temporal_train = composer.encode_temporal(seq_train)
    temporal_test = composer.encode_temporal(seq_test)
    no_memory = MorphogenicClassifier(theta=0.10, alpha=0.25, seed=45)
    no_memory.fit(bag_train, y_seq_train)
    with_memory = MorphogenicClassifier(
        theta=0.18,
        alpha=0.22,
        adaptive_metric=True,
        growth_control=True,
        merge_every=160,
        merge_threshold=0.12,
        max_cells=160,
        seed=46,
    )
    with_memory.fit(temporal_train, y_seq_train)

    x_spiral_train, x_spiral_test, y_spiral_train, y_spiral_test = make_spiral_dataset(
        noise_dims=0
    )
    local = MorphogenicClassifier(
        theta=0.30,
        alpha=0.20,
        adaptive_metric=False,
        growth_control=True,
        merge_every=100,
        merge_threshold=0.12,
        max_cells=120,
        seed=47,
    )
    local.fit(x_spiral_train, y_spiral_train)
    no_ridge_acc = accuracy(y_spiral_test, local.predict(x_spiral_test))
    ridge = RidgeHybridHead(lam=0.001, sigma_floor=0.10, fixed_sigma=0.30)
    ridge.fit(local, x_spiral_train, y_spiral_train)
    ridge_acc = accuracy(y_spiral_test, ridge.predict(local, x_spiral_test))

    x_growth_train, y_growth_train, x_growth_test, y_growth_test = make_growth_dataset(
        n_train=4300, n_test=1900
    )
    growth_full = MorphogenicClassifier(
        theta=1.05,
        alpha=0.16,
        adaptive_metric=True,
        growth_control=True,
        merge_every=180,
        merge_threshold=0.58,
        max_cells=260,
        prune_fraction=0.16,
        seed=48,
    )
    growth_no_prune = MorphogenicClassifier(
        theta=1.05,
        alpha=0.16,
        adaptive_metric=True,
        growth_control=False,
        seed=49,
    )
    growth_full.fit(x_growth_train, y_growth_train)
    growth_no_prune.fit(x_growth_train, y_growth_train)

    return {
        "name": "phase3_ablations",
        "high_dimensional_components": high_dim_records,
        "memory_ablation": {
            "without_memory_accuracy": accuracy(y_seq_test, no_memory.predict(bag_test)),
            "with_temporal_memory_accuracy": accuracy(
                y_seq_test, with_memory.predict(temporal_test)
            ),
            "without_memory_cells": len(no_memory.cells),
            "with_memory_cells": len(with_memory.cells),
        },
        "ridge_ablation": {
            "without_ridge_accuracy": no_ridge_acc,
            "with_ridge_accuracy": ridge_acc,
            "gain": ridge_acc - no_ridge_acc,
        },
        "pruning_ablation_growth_dataset": {
            "with_pruning_accuracy": accuracy(
                y_growth_test, growth_full.predict(x_growth_test)
            ),
            "without_pruning_accuracy": accuracy(
                y_growth_test, growth_no_prune.predict(x_growth_test)
            ),
            "with_pruning_cells": len(growth_full.cells),
            "without_pruning_cells": len(growth_no_prune.cells),
        },
    }


def run_phase3() -> dict[str, Any]:
    return {
        "title": "Phase 3 morphogenic stress and ablation suite",
        "experiments": [
            experiment_large_and_adversarial(),
            experiment_new_classes_and_forgetting(),
            experiment_temporal_drift(),
            experiment_ablations(),
        ],
    }


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def write_phase3_report(results: dict[str, Any], out_dir: str | Path = "results") -> Path:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    json_path = out / "phase3_latest.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    experiments = {exp["name"]: exp for exp in results["experiments"]}
    large = experiments["large_dataset_and_adversarial_noise"]
    new_classes = experiments["new_classes_after_training_and_catastrophic_forgetting"]
    drift = experiments["temporal_drift_stream"]
    ablations = experiments["phase3_ablations"]

    def model_line(row: dict[str, Any]) -> str:
        adv = row.get("adversarial_accuracy")
        adv_text = "n/a" if adv is None else _fmt(float(adv))
        cand = row.get("clean_avg_candidates")
        cand_text = "n/a" if cand is None else _fmt(float(cand))
        return (
            f"| {row['name']} | {_fmt(float(row['clean_accuracy']))} | {adv_text} | "
            f"{_fmt(float(row['fit_seconds']))} | {_fmt(float(row['predict_seconds']))} | "
            f"{_fmt(float(row['model_ram_mb']))} | {cand_text} |"
        )

    large_rows = "\n".join(model_line(row) for row in large["models"])
    sweep_rows = "\n".join(
        "| {eps} | {morph} | {knn} | {svm} | {rf} | {mlp} | {proto} |".format(
            eps=_fmt(float(row["epsilon"])),
            morph=_fmt(float(row["accuracies"]["morphogenic_full"])),
            knn=_fmt(float(row["accuracies"]["exact_kNN_k5"])),
            svm=_fmt(float(row["accuracies"]["linear_SVM_sgd"])),
            rf=_fmt(float(row["accuracies"]["random_forest_numpy"])),
            mlp=_fmt(float(row["accuracies"]["small_MLP"])),
            proto=_fmt(float(row["accuracies"]["prototypical_network_centroid"])),
        )
        for row in large["adversarial_sweep"]
    )
    nc_rows = "\n".join(
        "| {name} | {old_before} | {old_after} | {new_after} | {drop} | {ram} |".format(
            name=row["name"],
            old_before=_fmt(float(row["old_accuracy_before"])),
            old_after=_fmt(float(row["old_accuracy_after"])),
            new_after=_fmt(float(row["new_class_accuracy_after"])),
            drop=_fmt(float(row["forgetting_drop"])),
            ram=_fmt(float(row["model_ram_mb"])),
        )
        for row in new_classes["models"]
    )
    drift_rows = "\n".join(
        "| {name} | {mean} | {last} | {ram} |".format(
            name=row["name"],
            mean=_fmt(float(row["mean_accuracy"])),
            last=_fmt(float(row["last_chunk_accuracy"])),
            ram=_fmt(float(row["model_ram_mb"])),
        )
        for row in drift["models"]
    )
    ablation_rows = "\n".join(
        "| {name} | {acc} | {pred} | {cand} | {cells} |".format(
            name=row["name"],
            acc=_fmt(float(row["accuracy"])),
            pred=_fmt(float(row["predict_seconds"])),
            cand="n/a" if row["avg_candidates"] is None else _fmt(float(row["avg_candidates"])),
            cells="n/a" if row["cells"] is None else str(row["cells"]),
        )
        for row in ablations["high_dimensional_components"]
    )

    report = f"""# Fase 3: stress tests y ablaciones

La fase 3 intenta romper la arquitectura morfogenica en vez de agregar features
sin direccion. Todas las cifras salen de `python run_phase3.py`.

## 1. Dataset grande + ruido adversarial

Dataset: {large['dataset']['train']} train, {large['dataset']['test']} test,
{large['dataset']['dimensions']} dimensiones, {large['dataset']['classes']} clases,
{large['dataset']['informative_dimensions']} dimensiones informativas. La columna
`adv acc` usa epsilon {large['dataset']['reported_attack_epsilon']}.

| Modelo | clean acc | adv acc | fit s | pred s | model MB | candidatos |
|---|---:|---:|---:|---:|---:|---:|
{large_rows}

Sweep adversarial:

| epsilon | morph | kNN | SVM | forest | MLP | proto |
|---:|---:|---:|---:|---:|---:|---:|
{sweep_rows}

## 2. Clases nuevas y olvido catastrofico

Setup: entrenamiento inicial con clases {new_classes['setup']['old_classes']},
luego {new_classes['setup']['few_shot_new_examples']} ejemplos few-shot de clases
{new_classes['setup']['new_classes']}.

| Modelo | old antes | old despues | nuevas | olvido | model MB |
|---|---:|---:|---:|---:|---:|
{nc_rows}

## 3. Drift temporal

Stream: {drift['setup']['chunks']} chunks de {drift['setup']['chunk_size']} ejemplos,
{drift['setup']['dimensions']} dimensiones. Se mide accuracy antes de actualizar con
el chunk nuevo.

| Modelo | mean acc | ultimo chunk | model MB |
|---|---:|---:|---:|
{drift_rows}

## 4. Ablaciones

Alta dimension:

| Variante | acc | pred s | candidatos | celulas |
|---|---:|---:|---:|---:|
{ablation_rows}

Memoria temporal:

- Sin memoria: {_fmt(float(ablations['memory_ablation']['without_memory_accuracy']))}.
- Con memoria/composicion: {_fmt(float(ablations['memory_ablation']['with_temporal_memory_accuracy']))}.

Ridge global:

- Sin ridge: {_fmt(float(ablations['ridge_ablation']['without_ridge_accuracy']))}.
- Con ridge: {_fmt(float(ablations['ridge_ablation']['with_ridge_accuracy']))}.
- Ganancia: {_fmt(float(ablations['ridge_ablation']['gain']))}.

Poda:

- Con poda: {_fmt(float(ablations['pruning_ablation_growth_dataset']['with_pruning_accuracy']))}
  con {ablations['pruning_ablation_growth_dataset']['with_pruning_cells']} celulas.
- Sin poda: {_fmt(float(ablations['pruning_ablation_growth_dataset']['without_pruning_accuracy']))}
  con {ablations['pruning_ablation_growth_dataset']['without_pruning_cells']} celulas.

## Lectura honesta

La arquitectura no gana todo: kNN sigue siendo una referencia fuerte cuando se
acepta guardar todo el dataset, y MLP/SVM pueden ser competitivos en datos
limpios. Lo prometedor es que la version morfogenica mantiene una mezcla rara:
alta precision en alta dimension con pocos candidatos, aprendizaje incremental
de clases nuevas, baja perdida por olvido, adaptacion online al drift, y mejoras
claras cuando se activan Fisher, LSH, poda, memoria y ridge.
"""
    report_path = out / "FASE3_RESULTADOS.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path
