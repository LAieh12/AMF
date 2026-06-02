from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from morphogenic_lab import EPS, MorphogenicClassifier


@dataclass
class AMF5Config:
    top_features: int = 32
    vote_k: int = 8
    radius_scale: float = 0.25
    min_radius: float = 0.04
    importance_power: float = 0.5
    batch_size: int = 256
    use_fisher: bool = True
    use_distance: bool = True
    use_radius: bool = True
    use_importance: bool = True
    use_purity: bool = True
    uniform_vote: bool = False
    class_normalize: bool = False


class AMF5:
    """Configurable attentional morphogenic field used for Phase 5 anatomy."""

    def __init__(
        self,
        config: AMF5Config | None = None,
        seed: int = 0,
        base_model: MorphogenicClassifier | None = None,
        base_kwargs: dict[str, Any] | None = None,
    ):
        self.config = config or AMF5Config()
        self.seed = seed
        if base_model is not None:
            self.base = base_model
        else:
            kwargs = {
                "theta": 0.82 if self.config.use_fisher else 1.45,
                "alpha": 0.18,
                "adaptive_metric": self.config.use_fisher,
                "growth_control": True,
                "merge_every": 220,
                "merge_threshold": 0.38 if self.config.use_fisher else 0.58,
                "max_cells": 520,
                "prune_fraction": 0.12,
                "use_index": True,
                "index_tables": 14,
                "index_planes": 9,
                "min_candidates": 48,
                "seed": seed,
            }
            if base_kwargs:
                kwargs.update(base_kwargs)
            self.base = MorphogenicClassifier(**kwargs)
        self.n_classes = 0
        self._train_variance: np.ndarray | None = None
        self._features: np.ndarray | None = None
        self._weights: np.ndarray | None = None
        self._centers: np.ndarray | None = None
        self._histories: np.ndarray | None = None
        self._importance: np.ndarray | None = None
        self._radii: np.ndarray | None = None
        self._class_cell_counts: np.ndarray | None = None
        self._last_candidates = 0.0
        self._last_votes = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 1) -> "AMF5":
        x = np.asarray(x, dtype=np.float64)
        self._train_variance = np.var(x, axis=0)
        self.base.fit(x, y, epochs=epochs)
        self._refresh_cache()
        return self

    def partial_fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 1) -> "AMF5":
        x = np.asarray(x, dtype=np.float64)
        if self._train_variance is None:
            self._train_variance = np.var(x, axis=0)
        else:
            self._train_variance = 0.9 * self._train_variance + 0.1 * np.var(x, axis=0)
        self.base.partial_fit(x, y, epochs=epochs)
        self._refresh_cache()
        return self

    def with_config(self, config: AMF5Config) -> "AMF5":
        clone = AMF5(config=config, seed=self.seed, base_model=self.base)
        clone._train_variance = None if self._train_variance is None else self._train_variance.copy()
        clone._refresh_cache()
        return clone

    def _select_features(self) -> tuple[np.ndarray, np.ndarray]:
        d = self.base.n_features
        top = min(max(1, self.config.top_features), d)
        if self.config.use_fisher and self.base.metric is not None:
            weights = self.base.weights.copy()
            order = np.argsort(weights)[::-1]
        else:
            weights = np.ones(d, dtype=np.float64)
            if self._train_variance is not None:
                order = np.argsort(self._train_variance)[::-1]
            else:
                order = np.arange(d)
        features = order[:top]
        selected_weights = weights[features]
        if not self.config.use_fisher:
            selected_weights = np.ones_like(selected_weights)
        return features.astype(int), selected_weights.astype(np.float64)

    def _refresh_cache(self) -> None:
        self.n_classes = self.base.n_classes
        self._last_candidates = 0.0
        self._last_votes = 0.0
        if not self.base.cells:
            self._features = None
            self._weights = None
            self._centers = None
            self._histories = None
            self._importance = None
            self._radii = None
            self._class_cell_counts = None
            return
        features, weights = self._select_features()
        self._features = features
        self._weights = weights
        self._centers = np.vstack([cell.center[features] for cell in self.base.cells])
        histories = []
        class_counts = np.zeros(self.n_classes, dtype=np.float64)
        for cell in self.base.cells:
            class_counts[cell.label] += 1.0
            if self.config.use_purity:
                hist = cell.hist / (cell.hist.sum() + EPS)
            else:
                hist = np.zeros(self.n_classes, dtype=np.float64)
                hist[cell.label] = 1.0
            histories.append(hist)
        self._histories = np.vstack(histories)
        self._importance = np.array([max(cell.importance(), EPS) for cell in self.base.cells])
        self._radii = np.array([max(cell.radius_ema, 0.08) for cell in self.base.cells])
        self._class_cell_counts = np.maximum(class_counts, 1.0)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self._centers is None or self._histories is None:
            return np.zeros((len(x), max(self.n_classes, 1)), dtype=np.float64)
        assert self._features is not None and self._weights is not None
        assert self._importance is not None and self._radii is not None
        assert self._class_cell_counts is not None
        x_sub = np.asarray(x, dtype=np.float64)[:, self._features]
        centers = self._centers
        k = min(max(1, self.config.vote_k), len(centers))
        out: list[np.ndarray] = []
        candidates_total = 0
        votes_total = 0
        for start in range(0, len(x_sub), self.config.batch_size):
            xb = x_sub[start : start + self.config.batch_size]
            d2 = np.mean(
                np.square(xb[:, None, :] - centers[None, :, :])
                * self._weights[None, None, :],
                axis=2,
            )
            idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
            rows = np.arange(len(xb))[:, None]
            local_d2 = d2[rows, idx]
            if self.config.uniform_vote:
                weight = np.ones_like(local_d2)
            else:
                if self.config.use_distance:
                    if self.config.use_radius:
                        local_r = self._radii[idx] * self.config.radius_scale + self.config.min_radius
                    else:
                        local_r = np.full_like(local_d2, self.config.radius_scale + self.config.min_radius)
                    weight = np.exp(-local_d2 / (2.0 * np.square(local_r)))
                else:
                    weight = np.ones_like(local_d2)
                if self.config.use_importance:
                    weight *= np.power(self._importance[idx], self.config.importance_power)
            score = np.zeros((len(xb), self.n_classes), dtype=np.float64)
            for j in range(k):
                score += weight[:, j : j + 1] * self._histories[idx[:, j]]
            if self.config.class_normalize:
                score /= self._class_cell_counts[None, :]
            score /= score.sum(axis=1, keepdims=True) + EPS
            out.append(score)
            candidates_total += len(xb) * len(centers)
            votes_total += len(xb) * k
        self._last_candidates = candidates_total / max(len(x_sub), 1)
        self._last_votes = votes_total / max(len(x_sub), 1)
        return np.vstack(out)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1).astype(int)

    @property
    def selected_features(self) -> np.ndarray:
        return np.array([], dtype=int) if self._features is None else self._features.copy()

    def summary(self) -> dict[str, Any]:
        cells = len(self.base.cells)
        return {
            "cells": cells,
            "top_features": int(0 if self._features is None else len(self._features)),
            "vote_k": int(min(max(1, self.config.vote_k), max(1, cells))),
            "avg_candidates": float(self._last_candidates or cells),
            "avg_votes": float(self._last_votes or min(max(1, self.config.vote_k), max(1, cells))),
            "selected_features": self.selected_features.astype(int).tolist(),
        }

    def memory_bytes(self) -> int:
        total = 0
        for cell in self.base.cells:
            total += cell.center.nbytes + cell.hist.nbytes
        if self.base.metric is not None:
            total += (
                self.base.metric.counts.nbytes
                + self.base.metric.sums.nbytes
                + self.base.metric.sumsq.nbytes
                + self.base.metric.weights.nbytes
            )
        for arr in (
            self._features,
            self._weights,
            self._centers,
            self._histories,
            self._importance,
            self._radii,
            self._class_cell_counts,
        ):
            if arr is not None:
                total += arr.nbytes
        return int(total)


def make_amf5(seed: int = 0, config: AMF5Config | None = None) -> AMF5:
    return AMF5(config=config or AMF5Config(), seed=seed)
