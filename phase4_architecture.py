from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from morphogenic_lab import EPS, MorphogenicClassifier


@dataclass
class Phase4Config:
    top_features: int = 32
    vote_k: int = 8
    radius_scale: float = 0.25
    robust_radius_scale: float = 1.0
    min_radius: float = 0.04
    importance_power: float = 0.5
    batch_size: int = 256


class AttentionalMorphogenicClassifier:
    """Phase 4 architecture: Fisher-attentional subspace + soft cell fields.

    The phase 3 model made a hard nearest-cell decision. Phase 4 keeps the same
    morphogenic cell substrate, but inference happens in an attention-selected
    Fisher subspace and uses a small local field of reliable cells instead of a
    single winner. This is intentionally still local and interpretable: no deep
    end-to-end training is added.
    """

    def __init__(
        self,
        config: Phase4Config | None = None,
        base_kwargs: dict[str, Any] | None = None,
        seed: int = 0,
    ):
        self.config = config or Phase4Config()
        kwargs = {
            "theta": 0.82,
            "alpha": 0.18,
            "adaptive_metric": True,
            "growth_control": True,
            "merge_every": 220,
            "merge_threshold": 0.38,
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
        self._centers: np.ndarray | None = None
        self._histories: np.ndarray | None = None
        self._importance: np.ndarray | None = None
        self._radii: np.ndarray | None = None
        self._weights: np.ndarray | None = None
        self._features: np.ndarray | None = None
        self._last_candidates = 0.0
        self._last_votes = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 1) -> "AttentionalMorphogenicClassifier":
        self.base.fit(x, y, epochs=epochs)
        self._refresh_cache()
        return self

    def partial_fit(
        self, x: np.ndarray, y: np.ndarray, epochs: int = 1
    ) -> "AttentionalMorphogenicClassifier":
        self.base.partial_fit(x, y, epochs=epochs)
        self._refresh_cache()
        return self

    def _refresh_cache(self) -> None:
        self.n_classes = self.base.n_classes
        self._last_candidates = 0.0
        self._last_votes = 0.0
        if not self.base.cells:
            self._centers = None
            self._histories = None
            self._importance = None
            self._radii = None
            self._weights = None
            self._features = None
            return
        weights = self.base.weights
        top = min(self.config.top_features, len(weights))
        features = np.argsort(weights)[::-1][:top]
        self._features = features
        self._weights = weights[features]
        self._centers = np.vstack([cell.center[features] for cell in self.base.cells])
        histories = []
        for cell in self.base.cells:
            hist = cell.hist / (cell.hist.sum() + EPS)
            histories.append(hist)
        self._histories = np.vstack(histories)
        self._importance = np.array(
            [max(cell.importance(), EPS) for cell in self.base.cells], dtype=np.float64
        )
        self._radii = np.array(
            [max(cell.radius_ema, 0.08) for cell in self.base.cells], dtype=np.float64
        )

    def _score_with_scale(self, x: np.ndarray, radius_scale: float) -> np.ndarray:
        if self._centers is None or self._histories is None:
            return np.zeros((len(x), max(self.n_classes, 1)), dtype=np.float64)
        assert self._importance is not None and self._radii is not None
        assert self._weights is not None and self._features is not None

        x_sub = np.asarray(x, dtype=np.float64)[:, self._features]
        centers = self._centers
        k = min(self.config.vote_k, len(centers))
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
            local_r = self._radii[idx] * radius_scale + self.config.min_radius
            kernel = np.exp(-local_d2 / (2.0 * np.square(local_r)))
            weight = kernel * np.power(self._importance[idx], self.config.importance_power)
            score = np.zeros((len(xb), self.n_classes), dtype=np.float64)
            for j in range(k):
                score += weight[:, j : j + 1] * self._histories[idx[:, j]]
            score /= score.sum(axis=1, keepdims=True) + EPS
            out.append(score)
            candidates_total += len(xb) * len(centers)
            votes_total += len(xb) * k
        self._last_candidates = candidates_total / max(len(x_sub), 1)
        self._last_votes = votes_total / max(len(x_sub), 1)
        return np.vstack(out)

    def predict_proba(self, x: np.ndarray, robust: bool = False) -> np.ndarray:
        scale = self.config.robust_radius_scale if robust else self.config.radius_scale
        return self._score_with_scale(x, radius_scale=scale)

    def predict(self, x: np.ndarray, robust: bool = False) -> np.ndarray:
        return np.argmax(self.predict_proba(x, robust=robust), axis=1).astype(int)

    def summary(self) -> dict[str, float | int | list[int]]:
        cells = len(self.base.cells)
        top_features = [] if self._features is None else self._features.astype(int).tolist()
        return {
            "cells": cells,
            "top_features": len(top_features),
            "vote_k": int(min(self.config.vote_k, max(cells, 1))),
            "avg_candidates": float(self._last_candidates or cells),
            "avg_votes": float(self._last_votes or min(self.config.vote_k, max(cells, 1))),
            "effective_feature_ops": int(cells * max(len(top_features), 1)),
            "selected_features": top_features,
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
            self._centers,
            self._histories,
            self._importance,
            self._radii,
            self._weights,
            self._features,
        ):
            if arr is not None:
                total += arr.nbytes
        return int(total)
