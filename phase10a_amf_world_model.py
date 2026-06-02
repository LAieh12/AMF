from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from phase10a_toy_simulator import Transition


EPS = 1e-9


@dataclass
class AMFDynamicsExport:
    npz_path: str
    metadata_path: str
    cells: int
    memory_mb: float


class AMFDynamicsWorldModel:
    """Local morphogenic dynamics cells for predicting S_t+1 from S_t and action."""

    def __init__(
        self,
        cell_size: float = 0.135,
        activation_radius: float = 0.055,
        top_k: int = 12,
        max_cells: int = 9000,
        min_cell_usage: int = 2,
        explain_error_threshold: float = 0.0025,
        medium_error_threshold: float | None = None,
        novelty_confirmations: int = 3,
        fast_dynamics_lr: float = 0.35,
        identity_lr: float = 0.0,
        online_importance_boost: float = 5.0,
    ):
        self.cell_size = cell_size
        self.activation_radius = activation_radius
        self.top_k = top_k
        self.max_cells = max_cells
        self.min_cell_usage = min_cell_usage
        self.explain_error_threshold = explain_error_threshold
        self.medium_error_threshold = (
            float(medium_error_threshold)
            if medium_error_threshold is not None
            else float(explain_error_threshold * 5.0)
        )
        self.novelty_confirmations = novelty_confirmations
        self.fast_dynamics_lr = fast_dynamics_lr
        self.identity_lr = identity_lr
        self.online_importance_boost = float(online_importance_boost)
        self.identity_frozen = True
        self.identity_memory = {
            "state_dim": 4,
            "action_dim": 2,
            "style": "toy_gravity_bounce_delta_world_model",
        }
        self.novelty_buffer: dict[tuple[int, ...], int] = {}
        self.metaplasticity_stats: dict[str, Any] = {}
        self.centers = np.zeros((0, 15), dtype=np.float32)
        self.deltas = np.zeros((0, 4), dtype=np.float32)
        self.usage = np.zeros(0, dtype=np.float32)
        self.fit_cell_size = cell_size

    @staticmethod
    def encode(state: np.ndarray, action: np.ndarray) -> np.ndarray:
        x, y, vx, vy = [float(v) for v in state]
        ax, ay = [float(v) for v in action]
        speed = np.sqrt(vx * vx + vy * vy)
        near_left = max(0.0, 0.18 - (x + 1.0)) / 0.18
        near_right = max(0.0, 0.18 - (1.0 - x)) / 0.18
        near_floor = max(0.0, 0.18 - (y + 1.0)) / 0.18
        near_ceiling = max(0.0, 0.18 - (1.0 - y)) / 0.18
        return np.array(
            [
                x,
                y,
                0.55 * vx,
                0.55 * vy,
                ax,
                ay,
                x * y,
                0.30 * vx * vy,
                0.40 * ax * vx,
                0.40 * ay * vy,
                x * x,
                y * y,
                0.25 * speed,
                near_left + near_right,
                near_floor + near_ceiling,
            ],
            dtype=np.float32,
        )

    def _build_cells(self, transitions: list[Transition], cell_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        buckets: dict[tuple[int, ...], list[Any]] = {}
        for transition in transitions:
            feature = self.encode(transition.state, transition.action)
            key = tuple(np.round(feature / cell_size).astype(np.int32).tolist())
            delta = transition.next_state - transition.state
            if key not in buckets:
                buckets[key] = [1, feature.astype(np.float64), delta.astype(np.float64)]
            else:
                bucket = buckets[key]
                bucket[0] += 1
                bucket[1] += feature
                bucket[2] += delta
        centers = []
        deltas = []
        usage = []
        for count, feature_sum, delta_sum in buckets.values():
            centers.append(feature_sum / count)
            deltas.append(delta_sum / count)
            usage.append(count)
        return (
            np.vstack(centers).astype(np.float32),
            np.vstack(deltas).astype(np.float32),
            np.asarray(usage, dtype=np.float32),
        )

    def _prune_low_usage(self) -> int:
        if len(self.centers) == 0 or self.min_cell_usage <= 1:
            return 0
        keep = self.usage >= float(self.min_cell_usage)
        if not np.any(keep):
            return 0
        removed = int(len(self.usage) - int(np.sum(keep)))
        self.centers = self.centers[keep]
        self.deltas = self.deltas[keep]
        self.usage = self.usage[keep]
        return removed

    def _fuse_similar_cells(self) -> int:
        if len(self.centers) <= 1:
            return 0
        merge_center = max(self.fit_cell_size * 0.65, 0.055)
        merge_delta = 0.018
        buckets: dict[tuple[int, ...], list[Any]] = {}
        combined = np.hstack([self.centers / merge_center, self.deltas / merge_delta])
        for idx, row in enumerate(combined):
            key = tuple(np.round(row).astype(np.int32).tolist())
            weight = float(self.usage[idx])
            if key not in buckets:
                buckets[key] = [weight, self.centers[idx].astype(np.float64) * weight, self.deltas[idx].astype(np.float64) * weight]
            else:
                bucket = buckets[key]
                bucket[0] += weight
                bucket[1] += self.centers[idx] * weight
                bucket[2] += self.deltas[idx] * weight
        if len(buckets) == len(self.centers):
            return 0
        centers = []
        deltas = []
        usage = []
        for weight, center_sum, delta_sum in buckets.values():
            centers.append(center_sum / weight)
            deltas.append(delta_sum / weight)
            usage.append(weight)
        removed = len(self.centers) - len(centers)
        self.centers = np.vstack(centers).astype(np.float32)
        self.deltas = np.vstack(deltas).astype(np.float32)
        self.usage = np.asarray(usage, dtype=np.float32)
        return int(removed)

    def fit(self, transitions: list[Transition]) -> "AMFDynamicsWorldModel":
        cell_size = self.cell_size
        while True:
            centers, deltas, usage = self._build_cells(transitions, cell_size)
            if len(centers) <= self.max_cells or cell_size >= 0.40:
                break
            cell_size *= 1.18
        self.centers = centers
        self.deltas = deltas
        self.usage = usage
        self.fit_cell_size = cell_size
        raw_cells = int(len(self.centers))
        pruned = self._prune_low_usage()
        fused = self._fuse_similar_cells()
        if len(self.centers) > self.max_cells:
            order = np.argsort(self.usage)[-self.max_cells :]
            self.centers = self.centers[order]
            self.deltas = self.deltas[order]
            self.usage = self.usage[order]
        self.metaplasticity_stats = {
            "raw_cells": raw_cells,
            "pruned_low_usage": pruned,
            "fused_similar": fused,
            "final_cells": int(len(self.centers)),
            "min_cell_usage": int(self.min_cell_usage),
            "identity_frozen": bool(self.identity_frozen),
            "identity_lr": float(self.identity_lr),
            "fast_dynamics_lr": float(self.fast_dynamics_lr),
            "medium_error_threshold": float(self.medium_error_threshold),
            "online_importance_boost": float(self.online_importance_boost),
            "stores_delta": True,
        }
        return self

    def _nearest_cell_index(self, feature: np.ndarray) -> tuple[int, float]:
        if len(self.centers) == 0:
            return -1, float("inf")
        distances = np.mean(np.square(self.centers - feature), axis=1)
        idx = int(np.argmin(distances))
        return idx, float(distances[idx])

    def _adapt_existing_cell(
        self,
        cell_index: int,
        feature: np.ndarray,
        actual_delta: np.ndarray,
        lr_scale: float,
        usage_increment: float,
    ) -> None:
        lr = min(
            self.fast_dynamics_lr * lr_scale / np.sqrt(float(self.usage[cell_index]) + 1.0),
            0.25 * lr_scale,
        )
        self.centers[cell_index] = (1.0 - lr) * self.centers[cell_index] + lr * feature
        self.deltas[cell_index] = (1.0 - lr) * self.deltas[cell_index] + lr * actual_delta
        self.usage[cell_index] += usage_increment

    def learn_transition(self, transition: Transition) -> str:
        feature = self.encode(transition.state, transition.action)
        actual_delta = (transition.next_state - transition.state).astype(np.float32)
        if len(self.centers) == 0:
            self.centers = feature.reshape(1, -1).astype(np.float32)
            self.deltas = actual_delta.reshape(1, -1).astype(np.float32)
            self.usage = np.ones(1, dtype=np.float32)
            return "created_first_cell"
        predicted_delta = self.predict_delta(transition.state, transition.action)
        error = float(np.mean(np.square(predicted_delta - actual_delta)))
        nearest, distance = self._nearest_cell_index(feature)
        if error <= self.explain_error_threshold:
            self._adapt_existing_cell(nearest, feature, actual_delta, lr_scale=1.0, usage_increment=1.0)
            return "explained_by_existing_cell"
        medium_distance = max((self.fit_cell_size * 1.5) ** 2, (self.activation_radius * 2.5) ** 2)
        if error <= self.medium_error_threshold and distance <= medium_distance:
            self._adapt_existing_cell(nearest, feature, actual_delta, lr_scale=0.45, usage_increment=0.65)
            return "metaplasticity_adapted_cell"
        key = tuple(np.round(feature / self.fit_cell_size).astype(np.int32).tolist())
        self.novelty_buffer[key] = self.novelty_buffer.get(key, 0) + 1
        if self.novelty_buffer[key] < self.novelty_confirmations:
            return "buffered_possible_noise"
        self.centers = np.vstack([self.centers, feature.astype(np.float32)])
        self.deltas = np.vstack([self.deltas, actual_delta.astype(np.float32)])
        online_usage = max(
            float(self.min_cell_usage + self.novelty_confirmations),
            float(self.online_importance_boost),
        )
        self.usage = np.append(self.usage, online_usage).astype(np.float32)
        if len(self.centers) > self.max_cells:
            self._prune_low_usage()
            self._fuse_similar_cells()
        if len(self.centers) > self.max_cells:
            order = np.argsort(self.usage)[-self.max_cells :]
            self.centers = self.centers[order]
            self.deltas = self.deltas[order]
            self.usage = self.usage[order]
        return "created_confirmed_novelty"

    def predict_delta(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        if len(self.centers) == 0:
            return np.zeros(4, dtype=np.float32)
        feature = self.encode(state, action)
        distances = np.mean(np.square(self.centers - feature), axis=1)
        k = min(self.top_k, len(self.centers))
        if k == len(self.centers):
            order = np.argsort(distances)
        else:
            order = np.argpartition(distances, k - 1)[:k]
            order = order[np.argsort(distances[order])]
        local_distances = distances[order]
        exact_mask = local_distances <= max(EPS, (self.activation_radius * 0.025) ** 2)
        if np.any(exact_mask):
            exact_order = order[exact_mask]
            exact_weights = np.maximum(self.usage[exact_order], 1.0)
            exact_weights = exact_weights / (np.sum(exact_weights) + EPS)
            return np.sum(self.deltas[exact_order] * exact_weights[:, None], axis=0).astype(np.float32)
        weights = np.exp(-local_distances / (2.0 * self.activation_radius * self.activation_radius))
        weights = weights * np.sqrt(self.usage[order])
        if float(np.sum(weights)) <= EPS:
            weights = 1.0 / (local_distances + EPS)
        weights = weights / (np.sum(weights) + EPS)
        return np.sum(self.deltas[order] * weights[:, None], axis=0).astype(np.float32)

    def predict_next(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        predicted = state.astype(np.float32) + self.predict_delta(state, action)
        predicted[:2] = np.clip(predicted[:2], -1.05, 1.05)
        predicted[2:] = np.clip(predicted[2:], -3.0, 3.0)
        return predicted.astype(np.float32)

    def predict_batch(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return np.vstack([self.predict_next(state, action) for state, action in zip(states, actions)]).astype(np.float32)

    def memory_mb(self) -> float:
        total = self.centers.nbytes + self.deltas.nbytes + self.usage.nbytes
        return total / (1024.0 * 1024.0)

    def export(self, out_dir: str | Path = "data", name: str = "phase10a_warm_amf") -> AMFDynamicsExport:
        out = Path(out_dir)
        out.mkdir(exist_ok=True)
        npz_path = out / f"{name}.npz"
        metadata_path = out / f"{name}.json"
        np.savez_compressed(
            npz_path,
            centers=self.centers,
            deltas=self.deltas,
            usage=self.usage,
            cell_size=np.asarray([self.fit_cell_size], dtype=np.float32),
            activation_radius=np.asarray([self.activation_radius], dtype=np.float32),
            top_k=np.asarray([self.top_k], dtype=np.int32),
            min_cell_usage=np.asarray([self.min_cell_usage], dtype=np.int32),
            explain_error_threshold=np.asarray([self.explain_error_threshold], dtype=np.float32),
            medium_error_threshold=np.asarray([self.medium_error_threshold], dtype=np.float32),
            novelty_confirmations=np.asarray([self.novelty_confirmations], dtype=np.int32),
            online_importance_boost=np.asarray([self.online_importance_boost], dtype=np.float32),
        )
        metadata = {
            "model": "AMFDynamicsWorldModel",
            "cells": int(len(self.centers)),
            "feature_dim": int(self.centers.shape[1]) if len(self.centers) else 15,
            "state_dim": 4,
            "action_dim": 2,
            "cell_size": float(self.fit_cell_size),
            "activation_radius": float(self.activation_radius),
            "top_k": int(self.top_k),
            "min_cell_usage": int(self.min_cell_usage),
            "explain_error_threshold": float(self.explain_error_threshold),
            "medium_error_threshold": float(self.medium_error_threshold),
            "novelty_confirmations": int(self.novelty_confirmations),
            "online_importance_boost": float(self.online_importance_boost),
            "metaplasticity_stats": self.metaplasticity_stats,
            "identity_memory": self.identity_memory,
            "identity_frozen": self.identity_frozen,
            "memory_mb_arrays": self.memory_mb(),
            "rules": {
                "no_llm": True,
                "no_dense_decoder": True,
                "no_backprop": True,
                "pretraining": "synthetic toy dynamics transitions",
                "stores_delta": True,
                "no_cell_if_existing_explains": True,
                "fuses_similar_cells": True,
                "prunes_low_usage_cells": True,
                "freezes_identity_memory": True,
                "fast_dynamics_slow_identity": True,
                "filters_noise_before_novelty": True,
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return AMFDynamicsExport(
            npz_path=str(npz_path),
            metadata_path=str(metadata_path),
            cells=int(len(self.centers)),
            memory_mb=self.memory_mb(),
        )

    @classmethod
    def load(cls, npz_path: str | Path) -> "AMFDynamicsWorldModel":
        data = np.load(npz_path)
        model = cls(
            cell_size=float(data["cell_size"][0]),
            activation_radius=float(data["activation_radius"][0]),
            top_k=int(data["top_k"][0]),
            min_cell_usage=int(data["min_cell_usage"][0]) if "min_cell_usage" in data else 2,
            explain_error_threshold=float(data["explain_error_threshold"][0]) if "explain_error_threshold" in data else 0.0025,
            medium_error_threshold=(
                float(data["medium_error_threshold"][0])
                if "medium_error_threshold" in data
                else None
            ),
            novelty_confirmations=int(data["novelty_confirmations"][0]) if "novelty_confirmations" in data else 3,
            online_importance_boost=(
                float(data["online_importance_boost"][0])
                if "online_importance_boost" in data
                else 5.0
            ),
        )
        model.centers = data["centers"].astype(np.float32)
        model.deltas = data["deltas"].astype(np.float32)
        model.usage = data["usage"].astype(np.float32)
        model.fit_cell_size = float(data["cell_size"][0])
        return model
