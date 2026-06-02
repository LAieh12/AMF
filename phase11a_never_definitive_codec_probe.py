from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phase11a_frontier_slot_warp_probe import (
    HORIZONS,
    INPUT_FRAMES,
    Slot,
    _component_slots,
    _iou,
    _mae,
    _match_velocity,
    _mse,
    _read_region,
    _reflect,
    _render_slot,
    download_real_moving_mnist,
    load_sequences,
)


CANDIDATES = (
    "last",
    "pixel_linear",
    "slot_velocity",
    "amf_world",
    "slot_amf_mean",
    "slot_amf_max",
    "slot75_amf25",
)


@dataclass(frozen=True)
class CodecConfig:
    threshold: float = 0.10
    min_pixels: int = 8
    max_slots: int = 4
    max_shift: int = 12
    radius: float = 0.16
    top_k: int = 16
    max_cells: int = 30000
    ridge: float = 1e-3
    policy_tie_margin: float = 0.0025


class DenseAMFMemory:
    """Small vector AMF memory for transition residuals with bounded growth."""

    def __init__(self, feature_dim: int, target_dim: int, config: CodecConfig) -> None:
        self.feature_dim = feature_dim
        self.target_dim = target_dim
        self.config = config
        self.features = np.zeros((config.max_cells, feature_dim), dtype=np.float32)
        self.targets = np.zeros((config.max_cells, target_dim), dtype=np.float32)
        self.counts = np.zeros(config.max_cells, dtype=np.float32)
        self.size = 0

    def add(self, feature: np.ndarray, target: np.ndarray) -> None:
        feature = feature.astype(np.float32)
        target = target.astype(np.float32)
        if self.size == 0:
            self.features[0] = feature
            self.targets[0] = target
            self.counts[0] = 1.0
            self.size = 1
            return

        active = self.features[: self.size]
        dists = np.linalg.norm(active - feature[None, :], axis=1)
        idx = int(np.argmin(dists))
        if dists[idx] > self.config.radius and self.size < self.config.max_cells:
            idx = self.size
            self.size += 1
            self.features[idx] = feature
            self.targets[idx] = target
            self.counts[idx] = 1.0
            return

        count = self.counts[idx] + 1.0
        alpha = 1.0 / count
        self.features[idx] = (1.0 - alpha) * self.features[idx] + alpha * feature
        self.targets[idx] = (1.0 - alpha) * self.targets[idx] + alpha * target
        self.counts[idx] = count

    def predict(self, feature: np.ndarray) -> tuple[np.ndarray, float]:
        if self.size == 0:
            return np.zeros(self.target_dim, dtype=np.float32), 0.0
        feature = feature.astype(np.float32)
        active = self.features[: self.size]
        dists = np.linalg.norm(active - feature[None, :], axis=1)
        k = min(self.config.top_k, self.size)
        idxs = np.argpartition(dists, k - 1)[:k]
        local = dists[idxs]
        count_weight = np.sqrt(np.maximum(self.counts[idxs], 1.0))
        weights = np.exp(-local / max(self.config.radius, 1e-6)) * count_weight
        denom = float(weights.sum())
        if denom <= 1e-9:
            return np.zeros(self.target_dim, dtype=np.float32), 0.0
        pred = (self.targets[idxs] * weights[:, None]).sum(axis=0) / denom
        confidence = float(weights.max() / denom)
        return pred.astype(np.float32), confidence


def slot_feature(seq: np.ndarray, slot: Slot, horizon: int, config: CodecConfig) -> tuple[np.ndarray, tuple[float, float]]:
    frame0, frame1 = seq[0], seq[1]
    dy21, dx21 = _match_velocity(frame1, slot, config.max_shift)
    prior_top = int(round(slot.top - dy21))
    prior_left = int(round(slot.left - dx21))
    prior = Slot(
        top=prior_top,
        left=prior_left,
        crop=_read_region(frame1, prior_top, prior_left, *slot.crop.shape),
        mass=slot.mass,
        center_y=slot.center_y - dy21,
        center_x=slot.center_x - dx21,
    )
    dy10, dx10 = _match_velocity(frame0, prior, config.max_shift)
    acc_y = dy21 - dy10
    acc_x = dx21 - dx10
    feature = np.array(
        [
            slot.center_y / 64.0,
            slot.center_x / 64.0,
            slot.crop.shape[0] / 28.0,
            slot.crop.shape[1] / 28.0,
            math.log1p(slot.mass) / 8.0,
            dy21 / 12.0,
            dx21 / 12.0,
            acc_y / 12.0,
            acc_x / 12.0,
            horizon / 17.0,
        ],
        dtype=np.float32,
    )
    return feature, (dy21, dx21)


def match_target_slot(slot: Slot, targets: list[Slot]) -> Slot | None:
    if not targets:
        return None
    best = None
    best_score = math.inf
    for target in targets:
        center = math.hypot(slot.center_y - target.center_y, slot.center_x - target.center_x)
        mass = abs(math.log1p(slot.mass) - math.log1p(target.mass)) * 4.0
        shape = abs(slot.crop.shape[0] - target.crop.shape[0]) + abs(slot.crop.shape[1] - target.crop.shape[1])
        score = center + mass + 0.15 * shape
        if score < best_score:
            best_score = score
            best = target
    return best


def train_memory(seqs: np.ndarray, config: CodecConfig) -> DenseAMFMemory:
    memory = DenseAMFMemory(feature_dim=10, target_dim=2, config=config)
    for seq in seqs:
        current = _component_slots(seq[INPUT_FRAMES - 1], config.threshold, config.min_pixels, config.max_slots)
        if not current:
            continue
        for horizon in HORIZONS:
            future = _component_slots(seq[INPUT_FRAMES - 1 + horizon], config.threshold, config.min_pixels, config.max_slots)
            for slot in current:
                target = match_target_slot(slot, future)
                if target is None:
                    continue
                feature, (dy, dx) = slot_feature(seq, slot, horizon, config)
                velocity_delta = np.array([dy * horizon / 64.0, dx * horizon / 64.0], dtype=np.float32)
                target_delta = np.array([(target.top - slot.top) / 64.0, (target.left - slot.left) / 64.0], dtype=np.float32)
                memory.add(feature, target_delta - velocity_delta)
    return memory


def render_slots(seq: np.ndarray, horizon: int, memory: DenseAMFMemory | None, config: CodecConfig) -> tuple[np.ndarray, dict[str, float]]:
    frame2 = seq[INPUT_FRAMES - 1]
    h, w = frame2.shape
    canvas = np.zeros_like(frame2, dtype=np.float32)
    slots = _component_slots(frame2, config.threshold, config.min_pixels, config.max_slots)
    if not slots:
        return frame2.copy(), {"slot_count": 0.0, "mean_conf": 0.0, "mean_speed": 0.0, "total_mass": float(frame2.sum())}

    confidences: list[float] = []
    speeds: list[float] = []
    for slot in slots:
        feature, (dy, dx) = slot_feature(seq, slot, horizon, config)
        velocity_delta = np.array([dy * horizon / 64.0, dx * horizon / 64.0], dtype=np.float32)
        if memory is None:
            delta = velocity_delta
            confidence = 0.0
        else:
            residual, confidence = memory.predict(feature)
            delta = velocity_delta + residual * confidence
        pred_top = _reflect(slot.top + float(delta[0]) * 64.0, h - slot.crop.shape[0])
        pred_left = _reflect(slot.left + float(delta[1]) * 64.0, w - slot.crop.shape[1])
        _render_slot(canvas, slot, pred_top, pred_left, alpha=1.0)
        confidences.append(confidence)
        speeds.append(float(math.hypot(dy, dx)))

    return np.clip(canvas, 0.0, 1.0), {
        "slot_count": float(len(slots)),
        "mean_conf": float(np.mean(confidences)) if confidences else 0.0,
        "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
        "total_mass": float(sum(slot.mass for slot in slots)),
    }


def candidate_frames(seq: np.ndarray, horizon: int, memory: DenseAMFMemory, config: CodecConfig) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    frame1 = seq[INPUT_FRAMES - 2]
    frame2 = seq[INPUT_FRAMES - 1]
    slot_velocity, stats = render_slots(seq, horizon, None, config)
    amf_world, amf_stats = render_slots(seq, horizon, memory, config)
    stats = {**stats, "mean_conf": amf_stats["mean_conf"]}
    slot_amf_mean = np.clip(0.5 * slot_velocity + 0.5 * amf_world, 0.0, 1.0)
    slot_amf_max = np.maximum(slot_velocity, amf_world)
    slot75_amf25 = np.clip(0.75 * slot_velocity + 0.25 * amf_world, 0.0, 1.0)
    frames = {
        "last": frame2.copy(),
        "pixel_linear": np.clip(frame2 + horizon * (frame2 - frame1), 0.0, 1.0),
        "slot_velocity": slot_velocity,
        "amf_world": amf_world,
        "slot_amf_mean": slot_amf_mean,
        "slot_amf_max": slot_amf_max,
        "slot75_amf25": slot75_amf25,
    }
    return frames, stats


def selector_features(horizon: int, stats: dict[str, float], candidate_idx: int, candidate: np.ndarray) -> np.ndarray:
    return np.array(
        [
            1.0,
            horizon / 17.0,
            stats["slot_count"] / 4.0,
            math.log1p(stats["total_mass"]) / 8.0,
            stats["mean_speed"] / 12.0,
            stats["mean_conf"],
            candidate_idx / max(len(CANDIDATES) - 1, 1),
            float(candidate.sum()) / (64.0 * 64.0),
            float((candidate > 0.1).mean()),
        ],
        dtype=np.float32,
    )


def fit_selector(seqs: np.ndarray, memory: DenseAMFMemory, config: CodecConfig) -> np.ndarray:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for seq in seqs:
        for horizon in HORIZONS:
            target = seq[INPUT_FRAMES - 1 + horizon]
            frames, stats = candidate_frames(seq, horizon, memory, config)
            for idx, name in enumerate(CANDIDATES):
                xs.append(selector_features(horizon, stats, idx, frames[name]))
                ys.append(_mse(frames[name], target))
    x = np.stack(xs)
    y = np.array(ys, dtype=np.float32)
    reg = config.ridge * np.eye(x.shape[1], dtype=np.float32)
    return np.linalg.solve(x.T @ x + reg, x.T @ y).astype(np.float32)


def fit_horizon_policy(
    seqs: np.ndarray, memory: DenseAMFMemory, config: CodecConfig
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    totals: dict[str, dict[str, float]] = {f"h{h}": {name: 0.0 for name in CANDIDATES} for h in HORIZONS}
    for seq in seqs:
        for horizon in HORIZONS:
            key = f"h{horizon}"
            target = seq[INPUT_FRAMES - 1 + horizon]
            frames, _stats = candidate_frames(seq, horizon, memory, config)
            for name, frame in frames.items():
                totals[key][name] += _mse(frame, target)

    policy: dict[str, str] = {}
    averages: dict[str, dict[str, float]] = {}
    denom = float(max(len(seqs), 1))
    for key, values in totals.items():
        avg = {name: value / denom for name, value in values.items()}
        averages[key] = avg
        best_name, best_loss = min(avg.items(), key=lambda item: item[1])
        horizon = int(key[1:])
        if horizon >= 5 and best_name == "amf_world" and avg["slot_velocity"] <= best_loss + config.policy_tie_margin:
            policy[key] = "slot_velocity"
        else:
            policy[key] = best_name
    return policy, averages


def fit_knn_selector(seqs: np.ndarray, memory: DenseAMFMemory, config: CodecConfig) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for seq in seqs:
        for horizon in HORIZONS:
            target = seq[INPUT_FRAMES - 1 + horizon]
            frames, stats = candidate_frames(seq, horizon, memory, config)
            for idx, name in enumerate(CANDIDATES):
                xs.append(selector_features(horizon, stats, idx, frames[name]))
                ys.append(_mse(frames[name], target))
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.float32)


def choose_candidate(horizon: int, frames: dict[str, np.ndarray], stats: dict[str, float], weights: np.ndarray) -> str:
    predicted = []
    for idx, name in enumerate(CANDIDATES):
        feat = selector_features(horizon, stats, idx, frames[name])
        predicted.append((float(feat @ weights), name))
    predicted.sort(key=lambda item: item[0])
    return predicted[0][1]


def choose_candidate_knn(
    horizon: int,
    frames: dict[str, np.ndarray],
    stats: dict[str, float],
    selector_x: np.ndarray,
    selector_y: np.ndarray,
    k: int = 24,
) -> str:
    predicted = []
    for idx, name in enumerate(CANDIDATES):
        feat = selector_features(horizon, stats, idx, frames[name])
        dists = np.linalg.norm(selector_x - feat[None, :], axis=1)
        kk = min(k, len(dists))
        near = np.argpartition(dists, kk - 1)[:kk]
        weights = np.exp(-dists[near] / 0.35)
        denom = float(weights.sum())
        pred_loss = float((selector_y[near] * weights).sum() / max(denom, 1e-9))
        predicted.append((pred_loss, name))
    predicted.sort(key=lambda item: item[0])
    return predicted[0][1]


def evaluate(
    seqs: np.ndarray,
    memory: DenseAMFMemory,
    selector: np.ndarray,
    config: CodecConfig,
    horizon_policy: dict[str, str] | None = None,
    knn_selector: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = {}
    for horizon in HORIZONS:
        totals[f"h{horizon}"] = {
            "never_definitive_mse": 0.0,
            "never_definitive_mae": 0.0,
            "never_definitive_iou": 0.0,
            "candidate_oracle_mse": 0.0,
        }
        totals[f"h{horizon}"].update({f"{name}_mse": 0.0 for name in CANDIDATES})

    choices: dict[str, dict[str, int]] = {f"h{h}": {name: 0 for name in CANDIDATES} for h in HORIZONS}
    for seq in seqs:
        for horizon in HORIZONS:
            key = f"h{horizon}"
            target = seq[INPUT_FRAMES - 1 + horizon]
            frames, stats = candidate_frames(seq, horizon, memory, config)
            losses = {name: _mse(frame, target) for name, frame in frames.items()}
            if knn_selector is not None:
                choice = choose_candidate_knn(horizon, frames, stats, knn_selector[0], knn_selector[1])
            elif horizon_policy is None:
                choice = choose_candidate(horizon, frames, stats, selector)
            else:
                choice = horizon_policy[key]
            pred = frames[choice]
            choices[key][choice] += 1
            totals[key]["never_definitive_mse"] += losses[choice]
            totals[key]["never_definitive_mae"] += _mae(pred, target)
            totals[key]["never_definitive_iou"] += _iou(pred, target)
            totals[key]["candidate_oracle_mse"] += min(losses.values())
            for name in CANDIDATES:
                totals[key][f"{name}_mse"] += losses[name]

    out: dict[str, dict[str, float]] = {}
    denom = float(len(seqs))
    for key, values in totals.items():
        metrics = {name: value / denom for name, value in values.items()}
        last = max(metrics["last_mse"], 1e-9)
        metrics["mse_skill_vs_last"] = (metrics["last_mse"] - metrics["never_definitive_mse"]) / last
        metrics["oracle_gap_mse"] = metrics["never_definitive_mse"] - metrics["candidate_oracle_mse"]
        for name, count in choices[key].items():
            metrics[f"choice_{name}_rate"] = count / denom
        out[key] = metrics
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Definitive Never codec probe: encoder + AMF world memory + selector + copy-skip decoder.")
    parser.add_argument("--data-path", default="data/moving_mnist/mnist_test_seq.npy")
    parser.add_argument("--train-sequences", type=int, default=220)
    parser.add_argument("--test-sequences", type=int, default=40)
    parser.add_argument("--selector-sequences", type=int, default=60)
    parser.add_argument("--out", default="results/phase11a_never_definitive_codec_probe_220_40.json")
    args = parser.parse_args()

    started = time.time()
    data_path = Path(args.data_path)
    source = download_real_moving_mnist(data_path)
    seqs = load_sequences(data_path, args.train_sequences + args.test_sequences)
    train = seqs[: args.train_sequences]
    test = seqs[args.train_sequences : args.train_sequences + args.test_sequences]
    selector_count = min(args.selector_sequences, len(train))
    memory_train = train[selector_count:] if len(train) > selector_count else train
    selector_train = train[:selector_count]

    config = CodecConfig()
    memory = train_memory(memory_train, config)
    selector = fit_selector(selector_train, memory, config)
    horizon_policy, horizon_policy_validation = fit_horizon_policy(selector_train, memory, config)
    knn_selector = fit_knn_selector(selector_train, memory, config)
    ridge_metrics = evaluate(test, memory, selector, config)
    knn_metrics = evaluate(test, memory, selector, config, knn_selector=knn_selector)
    metrics = evaluate(test, memory, selector, config, horizon_policy=horizon_policy)

    result = {
        "probe": "phase11a_never_definitive_codec_probe",
        "real_dataset": True,
        "download_source": source,
        "data_path": str(data_path),
        "train_sequences": int(len(train)),
        "selector_sequences": int(len(selector_train)),
        "memory_sequences": int(len(memory_train)),
        "test_sequences": int(len(test)),
        "transition_cells": int(memory.size),
        "candidates": list(CANDIDATES),
        "config": {
            "threshold": config.threshold,
            "min_pixels": config.min_pixels,
            "max_slots": config.max_slots,
            "max_shift": config.max_shift,
            "radius": config.radius,
            "top_k": config.top_k,
            "max_cells": config.max_cells,
            "ridge": config.ridge,
            "policy_tie_margin": config.policy_tie_margin,
        },
        "selector_weights": selector.tolist(),
        "selector_strategy": "horizon_policy",
        "horizon_policy": horizon_policy,
        "horizon_policy_validation_mse": horizon_policy_validation,
        "knn_selector_examples": int(len(knn_selector[1])),
        "ridge_test_metrics": ridge_metrics,
        "knn_test_metrics": knn_metrics,
        "test_metrics": metrics,
        "elapsed_seconds": time.time() - started,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
