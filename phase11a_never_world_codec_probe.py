from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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


@dataclass
class TransitionCell:
    feature: np.ndarray
    target: np.ndarray
    count: int


@dataclass(frozen=True)
class CodecParams:
    threshold: float = 0.10
    min_pixels: int = 8
    max_slots: int = 4
    max_shift: int = 12
    radius: float = 0.18
    top_k: int = 16
    max_cells: int = 20000
    velocity_fallback: float = 0.35
    cell_blend: float = 0.75


class TransitionMemory:
    def __init__(self, radius: float, top_k: int, max_cells: int) -> None:
        self.radius = radius
        self.top_k = top_k
        self.max_cells = max_cells
        self.cells: list[TransitionCell] = []

    def add(self, feature: np.ndarray, target: np.ndarray) -> None:
        feature = feature.astype(np.float32)
        target = target.astype(np.float32)
        if not self.cells:
            self.cells.append(TransitionCell(feature=feature, target=target, count=1))
            return
        matrix = np.stack([cell.feature for cell in self.cells])
        dists = np.linalg.norm(matrix - feature[None, :], axis=1)
        idx = int(np.argmin(dists))
        if dists[idx] <= self.radius or len(self.cells) >= self.max_cells:
            cell = self.cells[idx]
            count = cell.count + 1
            alpha = 1.0 / float(count)
            cell.feature = (1.0 - alpha) * cell.feature + alpha * feature
            cell.target = (1.0 - alpha) * cell.target + alpha * target
            cell.count = count
        else:
            self.cells.append(TransitionCell(feature=feature, target=target, count=1))

    def predict(self, feature: np.ndarray) -> tuple[np.ndarray, float]:
        if not self.cells:
            return np.zeros(2, dtype=np.float32), 0.0
        feature = feature.astype(np.float32)
        matrix = np.stack([cell.feature for cell in self.cells])
        targets = np.stack([cell.target for cell in self.cells])
        counts = np.array([cell.count for cell in self.cells], dtype=np.float32)
        dists = np.linalg.norm(matrix - feature[None, :], axis=1)
        top_k = min(self.top_k, len(self.cells))
        idxs = np.argpartition(dists, top_k - 1)[:top_k]
        local_dists = dists[idxs]
        weights = np.exp(-local_dists / max(self.radius, 1e-6)) * np.sqrt(counts[idxs])
        denom = float(weights.sum())
        if denom <= 1e-9:
            return np.zeros(2, dtype=np.float32), 0.0
        pred = (targets[idxs] * weights[:, None]).sum(axis=0) / denom
        confidence = float(weights.max() / denom)
        return pred.astype(np.float32), confidence


def _slot_shape(slot: Slot) -> tuple[float, float]:
    return float(slot.crop.shape[0]), float(slot.crop.shape[1])


def _slot_feature(seq: np.ndarray, slot: Slot, horizon: int, params: CodecParams) -> tuple[np.ndarray, tuple[float, float]]:
    frame0, frame1 = seq[0], seq[1]
    dy21, dx21 = _match_velocity(frame1, slot, params.max_shift)
    prior_slot = Slot(
        top=int(round(slot.top - dy21)),
        left=int(round(slot.left - dx21)),
        crop=_read_region(frame1, int(round(slot.top - dy21)), int(round(slot.left - dx21)), *slot.crop.shape),
        mass=slot.mass,
        center_y=slot.center_y - dy21,
        center_x=slot.center_x - dx21,
    )
    dy10, dx10 = _match_velocity(frame0, prior_slot, params.max_shift)
    acc_y = dy21 - dy10
    acc_x = dx21 - dx10
    height, width = _slot_shape(slot)
    feature = np.array(
        [
            slot.center_y / 64.0,
            slot.center_x / 64.0,
            dy21 / 12.0,
            dx21 / 12.0,
            acc_y / 12.0,
            acc_x / 12.0,
            height / 28.0,
            width / 28.0,
            math.log1p(slot.mass) / 8.0,
            horizon / 17.0,
        ],
        dtype=np.float32,
    )
    return feature, (dy21, dx21)


def _match_target_slot(source: Slot, targets: list[Slot]) -> Slot | None:
    if not targets:
        return None
    best_slot = None
    best_score = math.inf
    for target in targets:
        dist = math.hypot(source.center_y - target.center_y, source.center_x - target.center_x)
        mass_penalty = abs(math.log1p(source.mass) - math.log1p(target.mass)) * 4.0
        shape_penalty = abs(source.crop.shape[0] - target.crop.shape[0]) + abs(source.crop.shape[1] - target.crop.shape[1])
        score = dist + mass_penalty + 0.15 * shape_penalty
        if score < best_score:
            best_score = score
            best_slot = target
    return best_slot


def train_memory(seqs: np.ndarray, params: CodecParams, horizons: Iterable[int] = HORIZONS) -> TransitionMemory:
    memory = TransitionMemory(radius=params.radius, top_k=params.top_k, max_cells=params.max_cells)
    for seq in seqs:
        current_slots = _component_slots(seq[INPUT_FRAMES - 1], params.threshold, params.min_pixels, params.max_slots)
        if not current_slots:
            continue
        for horizon in horizons:
            target_slots = _component_slots(seq[INPUT_FRAMES - 1 + horizon], params.threshold, params.min_pixels, params.max_slots)
            for slot in current_slots:
                target = _match_target_slot(slot, target_slots)
                if target is None:
                    continue
                feature, (dy, dx) = _slot_feature(seq, slot, horizon, params)
                raw_delta = np.array(
                    [
                        (target.top - slot.top) / 64.0,
                        (target.left - slot.left) / 64.0,
                    ],
                    dtype=np.float32,
                )
                velocity_delta = np.array([dy * horizon / 64.0, dx * horizon / 64.0], dtype=np.float32)
                residual = raw_delta - velocity_delta
                memory.add(feature, residual)
    return memory


def predict_frame(seq: np.ndarray, horizon: int, memory: TransitionMemory, params: CodecParams) -> np.ndarray:
    frame2 = seq[INPUT_FRAMES - 1]
    h, w = frame2.shape
    canvas = np.zeros_like(frame2, dtype=np.float32)
    slots = _component_slots(frame2, params.threshold, params.min_pixels, params.max_slots)
    if not slots:
        return frame2.copy()

    for slot in slots:
        feature, (dy, dx) = _slot_feature(seq, slot, horizon, params)
        residual, confidence = memory.predict(feature)
        velocity_delta = np.array([dy * horizon / 64.0, dx * horizon / 64.0], dtype=np.float32)
        learned_delta = velocity_delta + residual
        fallback_delta = velocity_delta
        blend = params.cell_blend * confidence
        delta = blend * learned_delta + (1.0 - blend) * (params.velocity_fallback * fallback_delta)
        pred_top = _reflect(slot.top + float(delta[0]) * 64.0, h - slot.crop.shape[0])
        pred_left = _reflect(slot.left + float(delta[1]) * 64.0, w - slot.crop.shape[1])
        _render_slot(canvas, slot, pred_top, pred_left, alpha=1.0)
    return np.clip(canvas, 0.0, 1.0)


def evaluate(seqs: np.ndarray, memory: TransitionMemory, params: CodecParams) -> dict[str, dict[str, float]]:
    sums: dict[str, dict[str, float]] = {}
    for horizon in HORIZONS:
        sums[f"h{horizon}"] = {
            "never_world_codec_mse": 0.0,
            "never_world_codec_mae": 0.0,
            "never_world_codec_iou": 0.0,
            "last_frame_mse": 0.0,
        }
    for seq in seqs:
        for horizon in HORIZONS:
            target = seq[INPUT_FRAMES - 1 + horizon]
            pred = predict_frame(seq, horizon, memory, params)
            last = seq[INPUT_FRAMES - 1]
            key = f"h{horizon}"
            sums[key]["never_world_codec_mse"] += _mse(pred, target)
            sums[key]["never_world_codec_mae"] += _mae(pred, target)
            sums[key]["never_world_codec_iou"] += _iou(pred, target)
            sums[key]["last_frame_mse"] += _mse(last, target)
    out: dict[str, dict[str, float]] = {}
    denom = float(len(seqs))
    for key, values in sums.items():
        metrics = {name: value / denom for name, value in values.items()}
        last_mse = max(metrics["last_frame_mse"], 1e-9)
        metrics["mse_skill_vs_last"] = (metrics["last_frame_mse"] - metrics["never_world_codec_mse"]) / last_mse
        out[key] = metrics
    return out


def tune_params(train: np.ndarray, tune_count: int) -> tuple[CodecParams, TransitionMemory, dict[str, dict[str, float]]]:
    candidates = [
        CodecParams(radius=0.12, top_k=8, max_cells=8000, cell_blend=0.65, velocity_fallback=0.50),
        CodecParams(radius=0.16, top_k=12, max_cells=16000, cell_blend=0.75, velocity_fallback=0.35),
        CodecParams(radius=0.20, top_k=16, max_cells=24000, cell_blend=0.85, velocity_fallback=0.25),
        CodecParams(threshold=0.06, radius=0.16, top_k=16, max_cells=24000, cell_blend=0.80, velocity_fallback=0.35),
        CodecParams(threshold=0.14, radius=0.14, top_k=12, max_cells=16000, cell_blend=0.75, velocity_fallback=0.35),
    ]
    tune = train[: min(tune_count, len(train))]
    fit = train[min(tune_count, len(train)) :] if len(train) > tune_count else train
    best_score = math.inf
    best_params = candidates[0]
    best_memory = TransitionMemory(best_params.radius, best_params.top_k, best_params.max_cells)
    best_metrics: dict[str, dict[str, float]] = {}

    for idx, params in enumerate(candidates, start=1):
        memory = train_memory(fit, params)
        metrics = evaluate(tune, memory, params)
        score = 0.20 * metrics["h5"]["never_world_codec_mse"] + 0.35 * metrics["h10"]["never_world_codec_mse"] + 0.45 * metrics["h17"]["never_world_codec_mse"]
        print(f"candidate {idx}: score={score:.6f} cells={len(memory.cells)} params={params}")
        if score < best_score:
            best_score = score
            best_params = params
            best_memory = memory
            best_metrics = metrics
    return best_params, best_memory, best_metrics


def params_to_json(params: CodecParams) -> dict[str, float | int]:
    return {
        "threshold": params.threshold,
        "min_pixels": params.min_pixels,
        "max_slots": params.max_slots,
        "max_shift": params.max_shift,
        "radius": params.radius,
        "top_k": params.top_k,
        "max_cells": params.max_cells,
        "velocity_fallback": params.velocity_fallback,
        "cell_blend": params.cell_blend,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Never-style AMF world codec probe on real MovingMNIST.")
    parser.add_argument("--data-path", default="data/moving_mnist/mnist_test_seq.npy")
    parser.add_argument("--train-sequences", type=int, default=220)
    parser.add_argument("--test-sequences", type=int, default=40)
    parser.add_argument("--tune-sequences", type=int, default=60)
    parser.add_argument("--out", default="results/phase11a_never_world_codec_probe_220_40.json")
    args = parser.parse_args()

    started = time.time()
    data_path = Path(args.data_path)
    source = download_real_moving_mnist(data_path)
    seqs = load_sequences(data_path, args.train_sequences + args.test_sequences)
    train = seqs[: args.train_sequences]
    test = seqs[args.train_sequences : args.train_sequences + args.test_sequences]

    best_params, best_memory, tune_metrics = tune_params(train, args.tune_sequences)
    if args.train_sequences > args.tune_sequences:
        best_memory = train_memory(train, best_params)
    test_metrics = evaluate(test, best_memory, best_params)

    result = {
        "probe": "phase11a_never_world_codec_probe",
        "data_path": str(data_path),
        "download_source": source,
        "real_dataset": True,
        "train_sequences": int(len(train)),
        "test_sequences": int(len(test)),
        "tune_sequences": int(min(args.tune_sequences, len(train))),
        "horizons": list(HORIZONS),
        "best_params": params_to_json(best_params),
        "transition_cells": len(best_memory.cells),
        "tune_metrics": tune_metrics,
        "test_metrics": test_metrics,
        "elapsed_seconds": time.time() - started,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
