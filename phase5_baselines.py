from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from phase3_benchmark import ExactKNN, LinearSVM, RandomForestNumpy


EPS = 1e-9


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y), n_classes), dtype=np.float64)
    out[np.arange(len(y)), y.astype(int)] = 1.0
    return out


class WeightedKNN(ExactKNN):
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
            nn = np.argpartition(d2, kth=min(self.k, len(self.y) - 1), axis=1)[:, : self.k]
            labels = self.y[nn]
            weights = 1.0 / (np.sqrt(np.maximum(d2[np.arange(len(xb))[:, None], nn], 0.0)) + 1e-6)
            votes = np.zeros((len(labels), self.n_classes), dtype=np.float64)
            for row in range(len(labels)):
                np.add.at(votes[row], labels[row], weights[row])
            out[start : start + len(xb)] = np.argmax(votes, axis=1)
        self.last_avg_candidates = float(len(self.y))
        return out


class RadiusNeighbors:
    def __init__(self, radius_quantile: float = 0.08, fallback_k: int = 7, batch_size: int = 256):
        self.radius_quantile = radius_quantile
        self.fallback_k = fallback_k
        self.batch_size = batch_size
        self.x: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.train_norm: np.ndarray | None = None
        self.radius = 1.0
        self.n_classes = 0
        self.last_avg_candidates = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RadiusNeighbors":
        self.x = np.asarray(x, dtype=np.float64).copy()
        self.y = np.asarray(y, dtype=int).copy()
        self.train_norm = np.sum(np.square(self.x), axis=1)
        self.n_classes = int(np.max(y)) + 1
        rng = np.random.default_rng(123)
        sample = self.x[rng.choice(len(self.x), size=min(240, len(self.x)), replace=False)]
        d = np.sqrt(np.maximum(((sample[:, None, :] - sample[None, :, :]) ** 2).sum(axis=2), 0.0))
        d = d[d > 0]
        self.radius = float(np.quantile(d, self.radius_quantile)) if len(d) else 1.0
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self.x is not None and self.y is not None and self.train_norm is not None
        x = np.asarray(x, dtype=np.float64)
        out = np.empty(len(x), dtype=int)
        cand_total = 0
        for start in range(0, len(x), self.batch_size):
            xb = x[start : start + self.batch_size]
            d2 = (
                np.sum(np.square(xb), axis=1, keepdims=True)
                + self.train_norm[None, :]
                - 2.0 * xb @ self.x.T
            )
            for i in range(len(xb)):
                within = np.where(d2[i] <= self.radius * self.radius)[0]
                if len(within) == 0:
                    within = np.argpartition(d2[i], kth=min(self.fallback_k, len(self.y) - 1))[: self.fallback_k]
                cand_total += len(within)
                votes = np.bincount(self.y[within], minlength=self.n_classes)
                out[start + i] = int(np.argmax(votes))
        self.last_avg_candidates = cand_total / max(len(x), 1)
        return out

    def memory_bytes(self) -> int:
        if self.x is None or self.y is None:
            return 0
        return int(self.x.nbytes + self.y.nbytes + (self.train_norm.nbytes if self.train_norm is not None else 0))


class NearestCentroid:
    def __init__(self, shrink: float = 0.0):
        self.shrink = shrink
        self.centroids: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.weights: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NearestCentroid":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        labels = np.unique(y)
        self.labels = labels
        self.centroids = np.vstack([x[y == label].mean(axis=0) for label in labels])
        overall = x.mean(axis=0)
        self.centroids = (1.0 - self.shrink) * self.centroids + self.shrink * overall
        self.weights = np.ones(x.shape[1], dtype=np.float64)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self.centroids is not None and self.labels is not None and self.weights is not None
        d2 = np.mean(np.square(x[:, None, :] - self.centroids[None, :, :]) * self.weights, axis=2)
        return self.labels[np.argmin(d2, axis=1)].astype(int)

    def memory_bytes(self) -> int:
        total = 0
        for arr in (self.centroids, self.labels, self.weights):
            if arr is not None:
                total += arr.nbytes
        return int(total)


class GaussianNB:
    def fit(self, x: np.ndarray, y: np.ndarray) -> "GaussianNB":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        self.labels = np.unique(y)
        self.mean = np.vstack([x[y == label].mean(axis=0) for label in self.labels])
        self.var = np.vstack([x[y == label].var(axis=0) + 1e-6 for label in self.labels])
        counts = np.array([np.sum(y == label) for label in self.labels], dtype=np.float64)
        self.log_prior = np.log(counts / counts.sum())
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        logp = []
        for i in range(len(self.labels)):
            lp = -0.5 * np.sum(np.log(2.0 * np.pi * self.var[i]))
            lp += -0.5 * np.sum(np.square(x - self.mean[i]) / self.var[i], axis=1)
            logp.append(lp + self.log_prior[i])
        return self.labels[np.argmax(np.vstack(logp).T, axis=1)].astype(int)

    def memory_bytes(self) -> int:
        return int(self.mean.nbytes + self.var.nbytes + self.log_prior.nbytes + self.labels.nbytes)


class RBFKernelClassifier:
    def __init__(self, landmarks: int = 96, gamma: float | None = None, lam: float = 1e-2, seed: int = 0):
        self.landmarks = landmarks
        self.gamma = gamma
        self.lam = lam
        self.seed = seed

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RBFKernelClassifier":
        rng = np.random.default_rng(self.seed)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        self.n_classes = int(np.max(y)) + 1
        idx = rng.choice(len(x), size=min(self.landmarks, len(x)), replace=False)
        self.centers = x[idx].copy()
        if self.gamma is None:
            sample = x[rng.choice(len(x), size=min(260, len(x)), replace=False)]
            d2 = np.sum(np.square(sample[:, None, :] - sample[None, :, :]), axis=2)
            med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
            self.used_gamma = 1.0 / max(med, EPS)
        else:
            self.used_gamma = self.gamma
        phi = self._features(x)
        target = one_hot(y, self.n_classes)
        reg = self.lam * np.eye(phi.shape[1])
        reg[0, 0] = 0.0
        self.coef = np.linalg.solve(phi.T @ phi + reg, phi.T @ target)
        return self

    def _features(self, x: np.ndarray) -> np.ndarray:
        d2 = np.sum(np.square(x[:, None, :] - self.centers[None, :, :]), axis=2)
        phi = np.exp(-self.used_gamma * d2)
        return np.hstack([np.ones((len(x), 1)), phi])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self._features(np.asarray(x)) @ self.coef, axis=1).astype(int)

    def memory_bytes(self) -> int:
        return int(self.centers.nbytes + self.coef.nbytes)


class PassiveAggressiveClassifier:
    def __init__(self, epochs: int = 18, c: float = 0.8, seed: int = 0):
        self.epochs = epochs
        self.c = c
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PassiveAggressiveClassifier":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        self.n_classes = int(np.max(y)) + 1
        self.w = np.zeros((x.shape[1], self.n_classes), dtype=np.float64)
        order = np.arange(len(x))
        for _ in range(self.epochs):
            self.rng.shuffle(order)
            for idx in order:
                row = x[idx]
                true = int(y[idx])
                scores = row @ self.w
                pred = int(np.argmax(scores))
                if pred == true:
                    continue
                loss = max(0.0, 1.0 - scores[true] + scores[pred])
                tau = min(self.c, loss / (2.0 * np.dot(row, row) + EPS))
                self.w[:, true] += tau * row
                self.w[:, pred] -= tau * row
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(np.asarray(x) @ self.w, axis=1).astype(int)

    def memory_bytes(self) -> int:
        return int(self.w.nbytes)


class SGDClassifierNumpy(LinearSVM):
    pass


class ExtraTreesNumpy(RandomForestNumpy):
    def __init__(self, seed: int = 0):
        super().__init__(
            n_trees=18,
            max_depth=8,
            min_leaf=4,
            candidates_per_node=24,
            sample_fraction=0.9,
            seed=seed,
        )


class StumpBoosting:
    def __init__(self, rounds: int = 80, learning_rate: float = 0.18, bins: int | None = None, seed: int = 0):
        self.rounds = rounds
        self.learning_rate = learning_rate
        self.bins = bins
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "StumpBoosting":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        self.n_classes = int(np.max(y)) + 1
        self.stumps: list[tuple[int, float, np.ndarray, np.ndarray]] = []
        scores = np.zeros((len(x), self.n_classes), dtype=np.float64)
        features = np.arange(x.shape[1])
        for _ in range(self.rounds):
            probs = np.exp(scores - scores.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            residual = one_hot(y, self.n_classes) - probs
            best = None
            best_loss = float("inf")
            candidate_features = self.rng.choice(features, size=min(32, len(features)), replace=False)
            for feature in candidate_features:
                col = x[:, feature]
                thresholds = np.quantile(col, np.linspace(0.15, 0.85, 5 if self.bins is None else min(self.bins, 7)))
                for threshold in thresholds:
                    left = col <= threshold
                    if left.sum() < 3 or (~left).sum() < 3:
                        continue
                    lval = residual[left].mean(axis=0)
                    rval = residual[~left].mean(axis=0)
                    pred = np.where(left[:, None], lval[None, :], rval[None, :])
                    loss = float(np.mean(np.square(residual - pred)))
                    if loss < best_loss:
                        best_loss = loss
                        best = (int(feature), float(threshold), lval.copy(), rval.copy())
            if best is None:
                break
            self.stumps.append(best)
            feature, threshold, lval, rval = best
            left = x[:, feature] <= threshold
            scores += self.learning_rate * np.where(left[:, None], lval[None, :], rval[None, :])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        scores = np.zeros((len(x), self.n_classes), dtype=np.float64)
        for feature, threshold, lval, rval in self.stumps:
            left = x[:, feature] <= threshold
            scores += self.learning_rate * np.where(left[:, None], lval[None, :], rval[None, :])
        return np.argmax(scores, axis=1).astype(int)

    def memory_bytes(self) -> int:
        return int(len(self.stumps) * (16 + 16 * self.n_classes))


def baseline_factories(seed: int, include_slow: bool = True) -> dict[str, Any]:
    factories: dict[str, Any] = {
        "nearest_centroid": lambda: NearestCentroid(),
        "gaussian_nb": lambda: GaussianNB(),
        "weighted_kNN": lambda: WeightedKNN(k=7, batch_size=256),
        "radius_neighbors": lambda: RadiusNeighbors(radius_quantile=0.08, fallback_k=7),
        "rbf_svm_like": lambda: RBFKernelClassifier(landmarks=96, lam=1e-2, seed=seed),
        "online_passive_aggressive": lambda: PassiveAggressiveClassifier(epochs=14, seed=seed),
        "sgd_classifier": lambda: SGDClassifierNumpy(epochs=28, lr=0.035, seed=seed),
        "gradient_boosting_stumps": lambda: StumpBoosting(rounds=48, learning_rate=0.18, seed=seed),
        "hist_gradient_boosting_stumps": lambda: StumpBoosting(rounds=48, learning_rate=0.18, bins=5, seed=seed + 17),
    }
    if include_slow:
        factories["extra_trees"] = lambda: ExtraTreesNumpy(seed=seed)
    return factories
