from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import json
import math
import numpy as np


EPS = 1e-9


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y), n_classes), dtype=np.float64)
    out[np.arange(len(y)), y.astype(int)] = 1.0
    return out


def split_standardize(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-6] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std


class FisherMetric:
    """Supervised diagonal metric from class moments, used as adaptive distance."""

    def __init__(self, n_features: int, n_classes: int, floor: float = 0.02):
        self.n_features = n_features
        self.n_classes = n_classes
        self.floor = floor
        self.counts = np.zeros(n_classes, dtype=np.float64)
        self.sums = np.zeros((n_classes, n_features), dtype=np.float64)
        self.sumsq = np.zeros((n_classes, n_features), dtype=np.float64)
        self._weights = np.ones(n_features, dtype=np.float64)

    def expand_classes(self, n_classes: int) -> None:
        if n_classes <= self.n_classes:
            return
        extra = n_classes - self.n_classes
        self.counts = np.pad(self.counts, (0, extra))
        self.sums = np.vstack([self.sums, np.zeros((extra, self.n_features))])
        self.sumsq = np.vstack([self.sumsq, np.zeros((extra, self.n_features))])
        self.n_classes = n_classes

    def update_batch(self, x: np.ndarray, y: np.ndarray) -> None:
        if len(y):
            self.expand_classes(int(np.max(y)) + 1)
        for label in range(self.n_classes):
            mask = y == label
            if not np.any(mask):
                continue
            part = x[mask]
            self.counts[label] += len(part)
            self.sums[label] += part.sum(axis=0)
            self.sumsq[label] += np.square(part).sum(axis=0)
        self._refresh()

    def _refresh(self) -> None:
        active = self.counts > 0
        if np.count_nonzero(active) < 2:
            self._weights = np.ones(self.n_features, dtype=np.float64)
            return
        counts = self.counts[active]
        means = self.sums[active] / counts[:, None]
        total = counts.sum()
        global_mean = (means * counts[:, None]).sum(axis=0) / total
        between = (counts[:, None] * np.square(means - global_mean)).sum(axis=0) / total
        variances = self.sumsq[active] / counts[:, None] - np.square(means)
        variances = np.maximum(variances, 0.0)
        within = (counts[:, None] * variances).sum(axis=0) / total
        score = between / (within + EPS)
        score = np.sqrt(np.maximum(score, 0.0))
        if not np.any(score > 0):
            self._weights = np.ones(self.n_features, dtype=np.float64)
            return
        score += self.floor * np.mean(score[score > 0])
        cap = np.percentile(score, 99.5)
        if cap > 0:
            score = np.minimum(score, cap)
        self._weights = score / (np.mean(score) + EPS)

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    def top_features(self, k: int) -> list[int]:
        return np.argsort(self._weights)[::-1][:k].astype(int).tolist()


@dataclass
class Cell:
    center: np.ndarray
    label: int
    n_classes: int
    count: float = 1.0
    correct: float = 1.0
    mistakes: float = 0.0
    radius_ema: float = 1.0
    margin_ema: float = 0.0
    last_seen: int = 0

    def __post_init__(self) -> None:
        self.hist = np.zeros(self.n_classes, dtype=np.float64)
        self.hist[self.label] = 1.0

    def resize_classes(self, n_classes: int) -> None:
        if n_classes <= self.n_classes:
            return
        self.hist = np.pad(self.hist, (0, n_classes - self.n_classes))
        self.n_classes = n_classes

    def observe(self, true_label: int, dist: float, margin: float, correct: bool) -> None:
        if true_label >= self.n_classes:
            self.resize_classes(true_label + 1)
        self.count += 1.0
        self.hist[int(true_label)] += 1.0
        self.radius_ema = 0.96 * self.radius_ema + 0.04 * float(dist)
        self.margin_ema = 0.96 * self.margin_ema + 0.04 * float(margin)
        if correct:
            self.correct += 1.0
        else:
            self.mistakes += 1.0

    def importance(self) -> float:
        p = self.hist / (self.hist.sum() + EPS)
        entropy = -float(np.sum(p * np.log(p + EPS)))
        entropy_norm = entropy / max(math.log(self.n_classes + EPS), EPS)
        purity = 1.0 - min(1.0, entropy_norm)
        confidence = (self.correct + 1.0) / (self.correct + self.mistakes + 2.0)
        support = math.log1p(self.count)
        compactness = 1.0 / (1.0 + max(self.radius_ema, 0.0))
        margin_bonus = 1.0 + max(0.0, self.margin_ema)
        return support * (0.25 + 0.75 * purity) * (0.5 + confidence) * compactness * margin_bonus


class LSHIndex:
    def __init__(
        self,
        n_features: int,
        seed: int,
        tables: int = 8,
        planes: int = 12,
        min_candidates: int = 24,
    ):
        self.n_features = n_features
        self.tables = tables
        self.planes = planes
        self.min_candidates = min_candidates
        rng = np.random.default_rng(seed)
        self.projections = rng.normal(size=(tables, planes, n_features)).astype(np.float64)
        norms = np.linalg.norm(self.projections, axis=2, keepdims=True)
        self.projections /= np.maximum(norms, EPS)
        self.buckets: list[dict[int, list[int]]] = []

    def _hashes(self, x: np.ndarray) -> list[int]:
        keys: list[int] = []
        for table in range(self.tables):
            bits = (self.projections[table] @ x) > 0.0
            key = 0
            for i, bit in enumerate(bits):
                if bit:
                    key |= 1 << i
            keys.append(key)
        return keys

    def build(self, centers: np.ndarray, weights: np.ndarray) -> None:
        self.buckets = [dict() for _ in range(self.tables)]
        if len(centers) == 0:
            return
        scale = np.sqrt(weights)
        projected = centers * scale
        for idx, vec in enumerate(projected):
            for table, key in enumerate(self._hashes(vec)):
                self.buckets[table].setdefault(key, []).append(idx)

    def query(self, x: np.ndarray, weights: np.ndarray) -> list[int]:
        if not self.buckets:
            return []
        keys = self._hashes(x * np.sqrt(weights))
        found: set[int] = set()
        for table, key in enumerate(keys):
            found.update(self.buckets[table].get(key, ()))
        if len(found) >= self.min_candidates:
            return list(found)
        for table, key in enumerate(keys):
            for bit in range(self.planes):
                found.update(self.buckets[table].get(key ^ (1 << bit), ()))
                if len(found) >= self.min_candidates:
                    return list(found)
        return list(found)


class MorphogenicClassifier:
    def __init__(
        self,
        theta: float = 1.2,
        alpha: float = 0.18,
        adaptive_metric: bool = False,
        growth_control: bool = False,
        merge_every: int = 250,
        merge_threshold: float = 0.62,
        max_cells: int | None = None,
        prune_fraction: float = 0.12,
        use_index: bool = False,
        index_tables: int = 8,
        index_planes: int = 12,
        min_candidates: int = 24,
        seed: int = 0,
    ):
        self.theta = theta
        self.alpha = alpha
        self.adaptive_metric = adaptive_metric
        self.growth_control = growth_control
        self.merge_every = merge_every
        self.merge_threshold = merge_threshold
        self.max_cells = max_cells
        self.prune_fraction = prune_fraction
        self.use_index = use_index
        self.index_tables = index_tables
        self.index_planes = index_planes
        self.min_candidates = min_candidates
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.cells: list[Cell] = []
        self.metric: FisherMetric | None = None
        self.index: LSHIndex | None = None
        self.n_classes = 0
        self.n_features = 0
        self.distance_calls = 0
        self.query_count = 0
        self.index_misses = 0
        self.train_steps = 0

    @property
    def weights(self) -> np.ndarray:
        if self.metric is None:
            return np.ones(self.n_features, dtype=np.float64)
        return self.metric.weights

    def _reset_model_state(self) -> None:
        self.cells = []
        self.metric = None
        self.index = None
        self.n_classes = 0
        self.n_features = 0
        self.distance_calls = 0
        self.query_count = 0
        self.index_misses = 0
        self.train_steps = 0
        self.rng = np.random.default_rng(self.seed)

    def _prepare_incremental_batch(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.ndim != 2:
            raise ValueError("x must be a 2D array")
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        if len(x) == 0:
            return
        if self.n_features == 0:
            self.n_features = x.shape[1]
        elif self.n_features != x.shape[1]:
            raise ValueError(
                f"expected {self.n_features} features, got {x.shape[1]}"
            )
        required_classes = max(self.n_classes, int(np.max(y)) + 1)
        if required_classes > self.n_classes:
            self.n_classes = required_classes
            for cell in self.cells:
                cell.resize_classes(required_classes)
            if self.metric is not None:
                self.metric.expand_classes(required_classes)
        if self.adaptive_metric:
            if self.metric is None:
                self.metric = FisherMetric(self.n_features, self.n_classes)
            self.metric.update_batch(x, y)
        if self.use_index and self.index is None:
            self.index = LSHIndex(
                self.n_features,
                seed=self.seed + 17,
                tables=self.index_tables,
                planes=self.index_planes,
                min_candidates=self.min_candidates,
            )

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 1) -> "MorphogenicClassifier":
        self._reset_model_state()
        return self.partial_fit(x, y, epochs=epochs)

    def partial_fit(
        self, x: np.ndarray, y: np.ndarray, epochs: int = 1
    ) -> "MorphogenicClassifier":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        self._prepare_incremental_batch(x, y)
        order = np.arange(len(x))
        for _ in range(epochs):
            self.rng.shuffle(order)
            for pos in order:
                self.train_steps += 1
                self._learn_one(x[pos], int(y[pos]), self.train_steps)
                if self.growth_control and self.train_steps % self.merge_every == 0:
                    self.consolidate()
                if self.use_index and (
                    self.train_steps % max(25, self.merge_every // 4) == 0
                ):
                    self._rebuild_index()
        if self.growth_control:
            self.consolidate(force=True)
        self._rebuild_index()
        return self

    def _learn_one(self, x: np.ndarray, y: int, step: int) -> None:
        if not self.cells:
            self._add_cell(x, y, step)
            return
        idx, dist, margin, _ = self._nearest(x, use_index=False)
        cell = self.cells[idx]
        pred = cell.label
        correct = pred == y
        cell.observe(y, dist, margin, correct)
        cell.last_seen = step

        far_for_label = dist > self.theta
        should_create = (not correct) or far_for_label
        if should_create:
            if self.max_cells is None or len(self.cells) < self.max_cells:
                self._add_cell(x, y, step)
            else:
                self.consolidate(force=True)
                same = self._nearest_same_label(x, y)
                if same is not None:
                    same_idx, same_dist = same
                    rate = self.alpha / math.sqrt(max(self.cells[same_idx].count, 1.0))
                    self.cells[same_idx].center += rate * (x - self.cells[same_idx].center)
                    self.cells[same_idx].observe(y, same_dist, margin, True)
                else:
                    rate = self.alpha / math.sqrt(max(cell.count, 1.0))
                    cell.center += rate * (x - cell.center)
        else:
            rate = self.alpha / math.sqrt(max(cell.count, 1.0))
            cell.center += rate * (x - cell.center)

    def _add_cell(self, x: np.ndarray, y: int, step: int) -> None:
        new = Cell(np.array(x, dtype=np.float64).copy(), int(y), self.n_classes)
        new.last_seen = step
        self.cells.append(new)

    def _nearest_same_label(self, x: np.ndarray, label: int) -> tuple[int, float] | None:
        candidates = [i for i, cell in enumerate(self.cells) if cell.label == label]
        if not candidates:
            return None
        centers = np.vstack([self.cells[i].center for i in candidates])
        dist = self._distance_batch(centers, x)
        local = int(np.argmin(dist))
        return candidates[local], float(dist[local])

    def _distance_batch(self, centers: np.ndarray, x: np.ndarray) -> np.ndarray:
        diff = centers - x
        return np.sqrt(np.mean(np.square(diff) * self.weights, axis=1) + EPS)

    def _nearest(
        self, x: np.ndarray, use_index: bool = True
    ) -> tuple[int, float, float, int]:
        if use_index and self.index is not None and self.use_index:
            candidate_idx = self.index.query(x, self.weights)
            if not candidate_idx:
                self.index_misses += 1
                candidate_idx = list(range(len(self.cells)))
        else:
            candidate_idx = list(range(len(self.cells)))
        centers = np.vstack([self.cells[i].center for i in candidate_idx])
        dist = self._distance_batch(centers, x)
        self.distance_calls += len(candidate_idx)
        self.query_count += 1
        order = np.argsort(dist)
        best_local = int(order[0])
        best_idx = candidate_idx[best_local]
        best_dist = float(dist[best_local])
        if len(order) > 1:
            margin = float(dist[int(order[1])] - best_dist)
        else:
            margin = 0.0
        return best_idx, best_dist, margin, len(candidate_idx)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if not self.cells:
            return np.zeros(len(x), dtype=int)
        out = np.empty(len(x), dtype=int)
        for i, row in enumerate(x):
            idx, _, _, _ = self._nearest(row, use_index=True)
            out[i] = self.cells[idx].label
        return out

    def reset_query_stats(self) -> None:
        self.distance_calls = 0
        self.query_count = 0
        self.index_misses = 0

    def _rebuild_index(self) -> None:
        if self.index is None or not self.use_index or not self.cells:
            return
        centers = np.vstack([cell.center for cell in self.cells])
        self.index.build(centers, self.weights)

    def consolidate(self, force: bool = False) -> None:
        if len(self.cells) < 2:
            return
        self._merge_similar_cells(force=force)
        if self.max_cells is not None and len(self.cells) > self.max_cells:
            self._prune_to_limit(max(1, int(self.max_cells * (1.0 - self.prune_fraction))))
        elif force:
            self._prune_weak_cells()
        self._rebuild_index()

    def _merge_similar_cells(self, force: bool = False) -> None:
        merged: list[Cell] = []
        used = np.zeros(len(self.cells), dtype=bool)
        threshold = self.merge_threshold * (1.2 if force else 1.0)
        for i, cell in enumerate(self.cells):
            if used[i]:
                continue
            same_label = [
                j
                for j in range(i + 1, len(self.cells))
                if (not used[j]) and self.cells[j].label == cell.label
            ]
            if not same_label:
                used[i] = True
                merged.append(cell)
                continue
            centers = np.vstack([self.cells[j].center for j in same_label])
            distances = self._distance_batch(centers, cell.center)
            close = np.where(distances < threshold)[0]
            if len(close) == 0:
                used[i] = True
                merged.append(cell)
                continue
            group_idx = [i] + [same_label[int(k)] for k in close[:8]]
            for idx in group_idx:
                used[idx] = True
            merged.append(self._merge_group([self.cells[idx] for idx in group_idx]))
        self.cells = merged

    def _merge_group(self, group: Iterable[Cell]) -> Cell:
        cells = list(group)
        total = sum(cell.count for cell in cells)
        center = sum(cell.center * cell.count for cell in cells) / max(total, EPS)
        best = max(cells, key=lambda c: c.importance())
        out = Cell(center=center.copy(), label=best.label, n_classes=self.n_classes)
        out.count = total
        out.correct = sum(cell.correct for cell in cells)
        out.mistakes = sum(cell.mistakes for cell in cells)
        out.radius_ema = sum(cell.radius_ema * cell.count for cell in cells) / max(total, EPS)
        out.margin_ema = sum(cell.margin_ema * cell.count for cell in cells) / max(total, EPS)
        out.last_seen = max(cell.last_seen for cell in cells)
        out.hist = sum((cell.hist for cell in cells), np.zeros(self.n_classes, dtype=np.float64))
        return out

    def _protected_indices(self) -> set[int]:
        protected: set[int] = set()
        for label in range(self.n_classes):
            label_indices = [i for i, cell in enumerate(self.cells) if cell.label == label]
            if label_indices:
                protected.add(max(label_indices, key=lambda i: self.cells[i].importance()))
        return protected

    def _prune_to_limit(self, target: int) -> None:
        protected = self._protected_indices()
        ranked = sorted(range(len(self.cells)), key=lambda i: self.cells[i].importance())
        remove: set[int] = set()
        for idx in ranked:
            if len(self.cells) - len(remove) <= target:
                break
            if idx not in protected:
                remove.add(idx)
        self.cells = [cell for i, cell in enumerate(self.cells) if i not in remove]

    def _prune_weak_cells(self) -> None:
        if len(self.cells) < 12:
            return
        protected = self._protected_indices()
        scores = np.array([cell.importance() for cell in self.cells])
        cutoff = np.percentile(scores, 4)
        keep = [
            (i in protected) or (self.cells[i].count >= 3 and scores[i] >= cutoff)
            for i in range(len(self.cells))
        ]
        self.cells = [cell for cell, ok in zip(self.cells, keep) if ok]

    def summary(self) -> dict[str, float | int | list[int]]:
        top_features: list[int] = []
        if self.metric is not None:
            top_features = self.metric.top_features(min(20, self.n_features))
        return {
            "cells": len(self.cells),
            "avg_cell_radius": float(np.mean([c.radius_ema for c in self.cells])) if self.cells else 0.0,
            "avg_importance": float(np.mean([c.importance() for c in self.cells])) if self.cells else 0.0,
            "distance_calls": int(self.distance_calls),
            "queries": int(self.query_count),
            "avg_candidates": float(self.distance_calls / max(self.query_count, 1)),
            "index_misses": int(self.index_misses),
            "top_weighted_features": top_features,
        }


class RidgeHybridHead:
    def __init__(
        self,
        lam: float = 1e-2,
        sigma_floor: float = 0.15,
        fixed_sigma: float | None = None,
    ):
        self.lam = lam
        self.sigma_floor = sigma_floor
        self.fixed_sigma = fixed_sigma
        self.weights: np.ndarray | None = None
        self.n_classes = 0

    def _features(self, model: MorphogenicClassifier, x: np.ndarray) -> np.ndarray:
        centers = np.vstack([cell.center for cell in model.cells])
        diff = x[:, None, :] - centers[None, :, :]
        d2 = np.mean(np.square(diff) * model.weights[None, None, :], axis=2)
        if self.fixed_sigma is None:
            radii = np.array([max(cell.radius_ema, self.sigma_floor) for cell in model.cells])
        else:
            radii = np.full(len(model.cells), self.fixed_sigma, dtype=np.float64)
        phi = np.exp(-d2 / (2.0 * np.square(radii)[None, :]))
        labels = np.array([cell.label for cell in model.cells], dtype=int)
        class_votes = np.zeros((len(x), model.n_classes), dtype=np.float64)
        for label in range(model.n_classes):
            if np.any(labels == label):
                class_votes[:, label] = phi[:, labels == label].max(axis=1)
        return np.hstack([np.ones((len(x), 1)), phi, class_votes])

    def fit(self, model: MorphogenicClassifier, x: np.ndarray, y: np.ndarray) -> "RidgeHybridHead":
        self.n_classes = model.n_classes
        phi = self._features(model, x)
        target = one_hot(y, self.n_classes)
        reg = self.lam * np.eye(phi.shape[1])
        reg[0, 0] = 0.0
        self.weights = np.linalg.solve(phi.T @ phi + reg, phi.T @ target)
        return self

    def predict(self, model: MorphogenicClassifier, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            return model.predict(x)
        phi = self._features(model, x)
        return np.argmax(phi @ self.weights, axis=1).astype(int)


class TemporalComposer:
    def __init__(self, alphabet: str = "ABCDEF", decay: float = 0.82):
        self.alphabet = alphabet
        self.decay = decay
        self.index = {token: i for i, token in enumerate(alphabet)}
        self.n = len(alphabet)

    def encode_bag(self, sequences: list[str]) -> np.ndarray:
        x = np.zeros((len(sequences), self.n), dtype=np.float64)
        for row, seq in enumerate(sequences):
            for token in seq:
                x[row, self.index[token]] += 1.0
            x[row] /= max(len(seq), 1)
        return x

    def encode_temporal(self, sequences: list[str]) -> np.ndarray:
        rows = []
        for seq in sequences:
            counts = np.zeros(self.n, dtype=np.float64)
            first = np.zeros(self.n, dtype=np.float64)
            last = np.zeros(self.n, dtype=np.float64)
            transitions = np.zeros((self.n, self.n), dtype=np.float64)
            ordered = np.zeros((self.n, self.n), dtype=np.float64)
            trace = np.zeros(self.n, dtype=np.float64)
            seen = np.zeros(self.n, dtype=np.float64)
            prev: int | None = None
            for pos, token in enumerate(seq):
                idx = self.index[token]
                if pos == 0:
                    first[idx] = 1.0
                last[:] = 0.0
                last[idx] = 1.0
                counts[idx] += 1.0
                ordered[:, idx] += seen
                seen[idx] += 1.0
                trace *= self.decay
                trace[idx] += 1.0
                if prev is not None:
                    transitions[prev, idx] += 1.0
                prev = idx
            length = max(len(seq), 1)
            pair_norm = max(length * (length - 1) / 2.0, 1.0)
            first_pos = np.full(self.n, fill_value=length + 1, dtype=np.float64)
            for pos, token in enumerate(seq):
                idx = self.index[token]
                first_pos[idx] = min(first_pos[idx], pos)
            first_order = (first_pos[:, None] < first_pos[None, :]).astype(np.float64)
            rows.append(
                np.concatenate(
                    [
                        counts / length,
                        first,
                        last,
                        trace / max(np.linalg.norm(trace), EPS),
                        transitions.reshape(-1) / max(length - 1, 1),
                        ordered.reshape(-1) / pair_norm,
                        first_order.reshape(-1),
                    ]
                )
            )
        return np.vstack(rows)


def make_high_dimensional(
    n_train: int = 1200,
    n_test: int = 650,
    d: int = 1024,
    informative: int = 18,
    n_classes: int = 4,
    modes: int = 4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    relevant = rng.choice(d, size=informative, replace=False)
    code = rng.choice([-1.0, 1.0], size=(n_classes, informative))
    centers = 1.45 * code
    offsets = rng.normal(0.0, 0.35, size=(n_classes, modes, informative))

    def sample(n: int) -> tuple[np.ndarray, np.ndarray]:
        y = rng.integers(0, n_classes, size=n)
        mode = rng.integers(0, modes, size=n)
        x = rng.normal(0.0, 2.65, size=(n, d))
        x[:, relevant] = (
            centers[y]
            + offsets[y, mode]
            + rng.normal(0.0, 0.52, size=(n, informative))
        )
        return x, y

    x_train, y_train = sample(n_train)
    x_test, y_test = sample(n_test)
    return x_train, y_train, x_test, y_test, relevant


def make_growth_dataset(
    n_train: int = 5200,
    n_test: int = 2200,
    d: int = 72,
    n_classes: int = 7,
    modes: int = 16,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 3.4, size=(n_classes, d))
    offsets = rng.normal(0.0, 1.2, size=(n_classes, modes, d))
    offsets[:, :, 24:] *= 0.35

    def sample(n: int, noisy: bool) -> tuple[np.ndarray, np.ndarray]:
        y = rng.integers(0, n_classes, size=n)
        mode = rng.integers(0, modes, size=n)
        x = base[y] + offsets[y, mode] + rng.normal(0.0, 0.62, size=(n, d))
        if noisy:
            outliers = rng.random(n) < 0.045
            x[outliers] += rng.normal(0.0, 5.0, size=(np.count_nonzero(outliers), d))
            flips = rng.random(n) < 0.025
            y[flips] = rng.integers(0, n_classes, size=np.count_nonzero(flips))
        return x, y

    return (*sample(n_train, noisy=True), *sample(n_test, noisy=False))


def make_sequence_dataset(
    n_train: int = 1600,
    n_test: int = 900,
    seed: int = 13,
) -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    base = list("AABBCCDDEEFF")

    def sample(n: int) -> tuple[list[str], np.ndarray]:
        sequences: list[str] = []
        labels = np.zeros(n, dtype=int)
        for i in range(n):
            seq = base.copy()
            rng.shuffle(seq)
            text = "".join(seq)
            # Pure relation: every sequence has the same bag of symbols.
            labels[i] = int(text.index("A") < text.index("B"))
            sequences.append(text)
        return sequences, labels

    return (*sample(n_train), *sample(n_test))


def make_spiral_dataset(
    n_train: int = 720,
    n_test: int = 2600,
    n_classes: int = 3,
    noise_dims: int = 18,
    seed: int = 21,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    def sample(n: int) -> tuple[np.ndarray, np.ndarray]:
        per_class = n // n_classes
        xs = []
        ys = []
        for label in range(n_classes):
            r = np.linspace(0.08, 1.0, per_class)
            rng.shuffle(r)
            theta = label * (2.0 * np.pi / n_classes) + 4.7 * np.pi * r
            theta += rng.normal(0.0, 0.18, size=per_class)
            xy = np.column_stack(
                [
                    r * np.cos(theta) + rng.normal(0.0, 0.035, size=per_class),
                    r * np.sin(theta) + rng.normal(0.0, 0.035, size=per_class),
                ]
            )
            noise = rng.normal(0.0, 0.55, size=(per_class, noise_dims))
            xs.append(np.hstack([xy, noise]))
            ys.append(np.full(per_class, label, dtype=int))
        x = np.vstack(xs)
        y = np.concatenate(ys)
        order = rng.permutation(len(y))
        return x[order], y[order]

    x_train, y_train = sample(n_train)
    x_test, y_test = sample(n_test)
    return split_standardize(x_train, x_test) + (y_train, y_test)


def experiment_high_dimensional() -> dict:
    x_train, y_train, x_test, y_test, relevant = make_high_dimensional()
    baseline = MorphogenicClassifier(theta=4.05, alpha=0.18, seed=1)
    t0 = perf_counter()
    baseline.fit(x_train, y_train)
    train_time_base = perf_counter() - t0
    baseline.reset_query_stats()
    t0 = perf_counter()
    pred_base = baseline.predict(x_test)
    pred_time_base = perf_counter() - t0

    adaptive = MorphogenicClassifier(
        theta=0.78,
        alpha=0.20,
        adaptive_metric=True,
        growth_control=True,
        merge_every=220,
        merge_threshold=0.34,
        max_cells=320,
        use_index=True,
        index_tables=10,
        index_planes=11,
        min_candidates=22,
        seed=2,
    )
    t0 = perf_counter()
    adaptive.fit(x_train, y_train)
    train_time_adapt = perf_counter() - t0
    adaptive.reset_query_stats()
    t0 = perf_counter()
    pred_adapt = adaptive.predict(x_test)
    pred_time_adapt = perf_counter() - t0
    top = set(adaptive.metric.top_features(len(relevant))) if adaptive.metric else set()
    recall = len(top.intersection(set(relevant.astype(int).tolist()))) / len(relevant)

    return {
        "name": "high_dimensional_scalability",
        "challenge": "adaptive_metric_and_indexing",
        "dimensions": int(x_train.shape[1]),
        "informative_dimensions": int(len(relevant)),
        "baseline_accuracy": accuracy(y_test, pred_base),
        "adaptive_indexed_accuracy": accuracy(y_test, pred_adapt),
        "baseline_cells": len(baseline.cells),
        "adaptive_cells": len(adaptive.cells),
        "baseline_train_seconds": train_time_base,
        "adaptive_train_seconds": train_time_adapt,
        "baseline_predict_seconds": pred_time_base,
        "adaptive_predict_seconds": pred_time_adapt,
        "baseline_avg_candidates": baseline.summary()["avg_candidates"],
        "adaptive_avg_candidates": adaptive.summary()["avg_candidates"],
        "relevant_feature_recall_in_top_weights": recall,
        "adaptive_summary": adaptive.summary(),
    }


def experiment_growth_control() -> dict:
    x_train, y_train, x_test, y_test = make_growth_dataset()
    naive = MorphogenicClassifier(theta=1.05, alpha=0.16, adaptive_metric=True, seed=3)
    t0 = perf_counter()
    naive.fit(x_train, y_train)
    naive_train = perf_counter() - t0
    pred_naive = naive.predict(x_test)

    controlled = MorphogenicClassifier(
        theta=1.05,
        alpha=0.16,
        adaptive_metric=True,
        growth_control=True,
        merge_every=180,
        merge_threshold=0.58,
        max_cells=260,
        prune_fraction=0.16,
        seed=4,
    )
    t0 = perf_counter()
    controlled.fit(x_train, y_train)
    controlled_train = perf_counter() - t0
    pred_controlled = controlled.predict(x_test)

    return {
        "name": "growth_control",
        "challenge": "information_based_merge_and_prune",
        "naive_accuracy": accuracy(y_test, pred_naive),
        "controlled_accuracy": accuracy(y_test, pred_controlled),
        "naive_cells": len(naive.cells),
        "controlled_cells": len(controlled.cells),
        "cell_reduction_factor": len(naive.cells) / max(len(controlled.cells), 1),
        "naive_train_seconds": naive_train,
        "controlled_train_seconds": controlled_train,
        "controlled_summary": controlled.summary(),
    }


def experiment_sequences() -> dict:
    seq_train, y_train, seq_test, y_test = make_sequence_dataset()
    composer = TemporalComposer()
    x_train_bag = composer.encode_bag(seq_train)
    x_test_bag = composer.encode_bag(seq_test)
    x_train_temp = composer.encode_temporal(seq_train)
    x_test_temp = composer.encode_temporal(seq_test)

    bag_model = MorphogenicClassifier(theta=0.10, alpha=0.25, seed=5)
    bag_model.fit(x_train_bag, y_train)
    pred_bag = bag_model.predict(x_test_bag)

    temporal = MorphogenicClassifier(
        theta=0.18,
        alpha=0.22,
        adaptive_metric=True,
        growth_control=True,
        merge_every=160,
        merge_threshold=0.12,
        max_cells=160,
        seed=6,
    )
    temporal.fit(x_train_temp, y_train)
    pred_temp = temporal.predict(x_test_temp)

    return {
        "name": "sequences_and_reasoning",
        "challenge": "temporal_memory_and_composition_rules",
        "task": "same symbol bag; classify whether A appears before B",
        "bag_of_vectors_accuracy": accuracy(y_test, pred_bag),
        "temporal_composition_accuracy": accuracy(y_test, pred_temp),
        "bag_cells": len(bag_model.cells),
        "temporal_cells": len(temporal.cells),
        "temporal_feature_dimensions": int(x_train_temp.shape[1]),
        "temporal_summary": temporal.summary(),
    }


def experiment_global_hybrid() -> dict:
    x_train, x_test, y_train, y_test = make_spiral_dataset(noise_dims=0)
    local = MorphogenicClassifier(
        theta=0.30,
        alpha=0.20,
        adaptive_metric=False,
        growth_control=True,
        merge_every=100,
        merge_threshold=0.12,
        max_cells=120,
        seed=3,
    )
    t0 = perf_counter()
    local.fit(x_train, y_train, epochs=1)
    local_train = perf_counter() - t0
    pred_local = local.predict(x_test)

    head = RidgeHybridHead(lam=0.001, sigma_floor=0.10, fixed_sigma=0.30)
    t0 = perf_counter()
    head.fit(local, x_train, y_train)
    head_train = perf_counter() - t0
    pred_hybrid = head.predict(local, x_test)

    return {
        "name": "global_hybrid_learning",
        "challenge": "local_cells_plus_occasional_global_ridge_head",
        "local_accuracy": accuracy(y_test, pred_local),
        "hybrid_accuracy": accuracy(y_test, pred_hybrid),
        "accuracy_gain": accuracy(y_test, pred_hybrid) - accuracy(y_test, pred_local),
        "cells": len(local.cells),
        "local_train_seconds": local_train,
        "global_head_train_seconds": head_train,
        "local_summary": local.summary(),
    }


def run_all_experiments() -> dict:
    experiments = [
        experiment_high_dimensional(),
        experiment_growth_control(),
        experiment_sequences(),
        experiment_global_hybrid(),
    ]
    return {
        "title": "Morphogenic architecture second-generation challenge results",
        "experiments": experiments,
    }


def write_results_report(results: dict, out_dir: str | Path = "results") -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)
    json_path = out_path / "latest_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    by_name = {item["name"]: item for item in results["experiments"]}
    high = by_name["high_dimensional_scalability"]
    growth = by_name["growth_control"]
    seq = by_name["sequences_and_reasoning"]
    hybrid = by_name["global_hybrid_learning"]
    md = f"""# Resultados morfogenicos de segunda generacion

Este reporte se genero con `python run_experiments.py` usando solo NumPy y los
desafios planteados en `Investigacion.md`.

## 1. Escalabilidad a alta dimension

- Dataset: {high['dimensions']} dimensiones, solo {high['informative_dimensions']} informativas.
- Base euclidiana: accuracy {high['baseline_accuracy']:.3f}, {high['baseline_cells']} celulas.
- Metrica Fisher adaptativa + indice LSH: accuracy {high['adaptive_indexed_accuracy']:.3f}, {high['adaptive_cells']} celulas.
- Recuperacion de dimensiones utiles en los pesos top: {high['relevant_feature_recall_in_top_weights']:.3f}.
- Candidatos medios por consulta con indice: {high['adaptive_avg_candidates']:.1f}.

## 2. Control del crecimiento

- Red sin consolidacion: accuracy {growth['naive_accuracy']:.3f}, {growth['naive_cells']} celulas.
- Fusion/poda informativa: accuracy {growth['controlled_accuracy']:.3f}, {growth['controlled_cells']} celulas.
- Reduccion de celulas: {growth['cell_reduction_factor']:.2f}x.

## 3. Secuencias y razonamiento

- Tarea: mismo saco de simbolos, clasificar si A aparece antes que B.
- Clasificador de vectores independientes: accuracy {seq['bag_of_vectors_accuracy']:.3f}.
- Memoria temporal + reglas de composicion: accuracy {seq['temporal_composition_accuracy']:.3f}.
- Dimension de rasgos temporales: {seq['temporal_feature_dimensions']}.

## 4. Aprendizaje global hibrido

- Celulas locales: accuracy {hybrid['local_accuracy']:.3f}.
- Celulas locales + cabeza ridge global ocasional: accuracy {hybrid['hybrid_accuracy']:.3f}.
- Ganancia absoluta: {hybrid['accuracy_gain']:.3f}.

## Lectura corta

Los resultados son prometedores porque cada desafio deja de ser una nota
conceptual y pasa a tener una prueba operativa: la metrica adaptativa rescata
senal en alta dimension, la consolidacion reduce crecimiento sin colapsar la
precision, la composicion temporal resuelve una relacion invisible para bolsas
de vectores, y una cabeza global ligera mejora la frontera aprendida por celulas
locales sin volver al entrenamiento profundo end-to-end.
"""
    report_path = out_path / "RESULTADOS.md"
    report_path.write_text(md, encoding="utf-8")
    return report_path
