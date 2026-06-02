from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DATA_URLS = [
    "https://github.com/tychovdo/MovingMNIST/raw/master/mnist_test_seq.npy",
    "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy",
]
HORIZONS = (1, 5, 10, 17)
INPUT_FRAMES = 3


@dataclass(frozen=True)
class Slot:
    top: int
    left: int
    crop: np.ndarray
    mass: float
    center_y: float
    center_x: float


@dataclass(frozen=True)
class WarpParams:
    threshold: float
    min_pixels: int
    max_slots: int
    max_shift: int
    momentum: float
    use_tiles: bool
    tile_size: int
    tile_stride: int
    tile_mass: float
    tile_alpha: float


def download_real_moving_mnist(data_path: Path) -> str:
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if data_path.exists():
        return "local"

    last_error: Exception | None = None
    tmp_path = data_path.with_suffix(data_path.suffix + ".tmp")
    for url in DATA_URLS:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
            urllib.request.urlretrieve(url, tmp_path)
            arr = np.load(tmp_path)
            if arr.ndim != 4:
                raise ValueError(f"Unexpected MovingMNIST ndim: {arr.ndim}")
            tmp_path.replace(data_path)
            return url
        except Exception as exc:  # pragma: no cover - runtime fallback path
            last_error = exc
            if tmp_path.exists():
                tmp_path.unlink()
    raise RuntimeError(f"Could not download real MovingMNIST: {last_error}")


def load_sequences(data_path: Path, count: int) -> np.ndarray:
    arr = np.load(data_path)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D MovingMNIST array, got {arr.shape}")
    if arr.shape[0] == 20:
        arr = np.transpose(arr, (1, 0, 2, 3))
    if arr.shape[1] < INPUT_FRAMES + max(HORIZONS):
        raise ValueError(f"Need at least {INPUT_FRAMES + max(HORIZONS)} frames, got {arr.shape}")
    arr = arr[:count].astype(np.float32)
    if arr.max() > 1.5:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)


def _component_slots(frame: np.ndarray, threshold: float, min_pixels: int, max_slots: int) -> list[Slot]:
    mask = frame > threshold
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    slots: list[Slot] = []

    for y0 in range(h):
        for x0 in range(w):
            if seen[y0, x0] or not mask[y0, x0]:
                continue
            stack = [(y0, x0)]
            seen[y0, x0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                coords.append((y, x))
                for yy in (y - 1, y, y + 1):
                    for xx in (x - 1, x, x + 1):
                        if yy == y and xx == x:
                            continue
                        if 0 <= yy < h and 0 <= xx < w and not seen[yy, xx] and mask[yy, xx]:
                            seen[yy, xx] = True
                            stack.append((yy, xx))

            if len(coords) < min_pixels:
                continue

            ys = np.array([c[0] for c in coords], dtype=np.int32)
            xs = np.array([c[1] for c in coords], dtype=np.int32)
            top = max(0, int(ys.min()) - 1)
            bottom = min(h, int(ys.max()) + 2)
            left = max(0, int(xs.min()) - 1)
            right = min(w, int(xs.max()) + 2)
            crop = frame[top:bottom, left:right].copy()
            crop_mask = mask[top:bottom, left:right]
            crop *= crop_mask
            mass = float(crop.sum())
            if mass <= 1e-6:
                continue
            local_y, local_x = np.indices(crop.shape, dtype=np.float32)
            center_y = float((crop * (local_y + top)).sum() / mass)
            center_x = float((crop * (local_x + left)).sum() / mass)
            slots.append(Slot(top=top, left=left, crop=crop, mass=mass, center_y=center_y, center_x=center_x))

    slots.sort(key=lambda slot: slot.mass, reverse=True)
    return slots[:max_slots]


def _patch_slots(frame: np.ndarray, tile_size: int, stride: int, min_mass: float, max_slots: int) -> list[Slot]:
    h, w = frame.shape
    slots: list[Slot] = []
    for top in range(0, h - tile_size + 1, stride):
        for left in range(0, w - tile_size + 1, stride):
            crop = frame[top : top + tile_size, left : left + tile_size].copy()
            mass = float(crop.sum())
            if mass < min_mass:
                continue
            yy, xx = np.indices(crop.shape, dtype=np.float32)
            center_y = float((crop * (yy + top)).sum() / max(mass, 1e-6))
            center_x = float((crop * (xx + left)).sum() / max(mass, 1e-6))
            slots.append(Slot(top=top, left=left, crop=crop, mass=mass, center_y=center_y, center_x=center_x))
    slots.sort(key=lambda slot: slot.mass, reverse=True)
    return slots[:max_slots]


def _read_region(frame: np.ndarray, top: int, left: int, height: int, width: int) -> np.ndarray:
    out = np.zeros((height, width), dtype=np.float32)
    h, w = frame.shape
    src_top = max(0, top)
    src_left = max(0, left)
    src_bottom = min(h, top + height)
    src_right = min(w, left + width)
    if src_bottom <= src_top or src_right <= src_left:
        return out
    dst_top = src_top - top
    dst_left = src_left - left
    out[dst_top : dst_top + (src_bottom - src_top), dst_left : dst_left + (src_right - src_left)] = frame[
        src_top:src_bottom, src_left:src_right
    ]
    return out


def _match_velocity(prev: np.ndarray, slot: Slot, max_shift: int) -> tuple[float, float]:
    crop = slot.crop
    norm_crop = float(np.sqrt(np.square(crop).sum()) + 1e-6)
    best_score = -1.0
    best_dy = 0
    best_dx = 0
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            prev_region = _read_region(prev, slot.top - dy, slot.left - dx, crop.shape[0], crop.shape[1])
            norm_prev = float(np.sqrt(np.square(prev_region).sum()) + 1e-6)
            score = float((crop * prev_region).sum() / (norm_crop * norm_prev))
            if score > best_score:
                best_score = score
                best_dy = dy
                best_dx = dx
    return float(best_dy), float(best_dx)


def _reflect(value: float, limit: int) -> int:
    if limit <= 0:
        return 0
    period = 2.0 * limit
    value = value % period
    if value > limit:
        value = period - value
    return int(round(value))


def _render_slot(canvas: np.ndarray, slot: Slot, top: int, left: int, alpha: float = 1.0) -> None:
    h, w = canvas.shape
    crop_h, crop_w = slot.crop.shape
    src = np.clip(slot.crop * alpha, 0.0, 1.0)
    dst_top = max(0, top)
    dst_left = max(0, left)
    dst_bottom = min(h, top + crop_h)
    dst_right = min(w, left + crop_w)
    if dst_bottom <= dst_top or dst_right <= dst_left:
        return
    src_top = dst_top - top
    src_left = dst_left - left
    patch = src[src_top : src_top + (dst_bottom - dst_top), src_left : src_left + (dst_right - dst_left)]
    canvas[dst_top:dst_bottom, dst_left:dst_right] = np.maximum(
        canvas[dst_top:dst_bottom, dst_left:dst_right], patch
    )


def _warp_slots(seq: np.ndarray, horizon: int, params: WarpParams) -> np.ndarray:
    frame0 = seq[0]
    frame1 = seq[1]
    frame2 = seq[2]
    h, w = frame2.shape
    canvas = np.zeros_like(frame2, dtype=np.float32)

    object_slots = _component_slots(frame2, params.threshold, params.min_pixels, params.max_slots)
    if not object_slots:
        return frame2.copy()

    for slot in object_slots:
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
        dy = params.momentum * dy21 + (1.0 - params.momentum) * dy10
        dx = params.momentum * dx21 + (1.0 - params.momentum) * dx10
        pred_top = _reflect(slot.top + dy * horizon, h - slot.crop.shape[0])
        pred_left = _reflect(slot.left + dx * horizon, w - slot.crop.shape[1])
        _render_slot(canvas, slot, pred_top, pred_left, alpha=1.0)

    if params.use_tiles:
        tile_slots = _patch_slots(
            frame2,
            tile_size=params.tile_size,
            stride=params.tile_stride,
            min_mass=params.tile_mass,
            max_slots=max(params.max_slots * 6, 12),
        )
        for slot in tile_slots:
            dy, dx = _match_velocity(frame1, slot, params.max_shift)
            pred_top = _reflect(slot.top + dy * horizon, h - slot.crop.shape[0])
            pred_left = _reflect(slot.left + dx * horizon, w - slot.crop.shape[1])
            _render_slot(canvas, slot, pred_top, pred_left, alpha=params.tile_alpha)

    return np.clip(canvas, 0.0, 1.0)


def _last_frame_baseline(seq: np.ndarray, horizon: int) -> np.ndarray:
    del horizon
    return seq[2].copy()


def _linear_frame_baseline(seq: np.ndarray, horizon: int) -> np.ndarray:
    pred = seq[2] + horizon * (seq[2] - seq[1])
    return np.clip(pred, 0.0, 1.0)


def _mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.square(pred - target)))


def _mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def _iou(pred: np.ndarray, target: np.ndarray, threshold: float = 0.1) -> float:
    pred_mask = pred > threshold
    target_mask = target > threshold
    union = np.logical_or(pred_mask, target_mask).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred_mask, target_mask).sum() / union)


def evaluate_params(seqs: np.ndarray, params: WarpParams, horizons: Iterable[int] = HORIZONS) -> dict[str, dict[str, float]]:
    sums: dict[str, dict[str, float]] = {}
    for horizon in horizons:
        sums[f"h{horizon}"] = {
            "frontier_slot_warp_mse": 0.0,
            "frontier_slot_warp_mae": 0.0,
            "frontier_slot_warp_iou": 0.0,
            "last_frame_mse": 0.0,
            "linear_frame_mse": 0.0,
        }

    for seq in seqs:
        for horizon in horizons:
            target = seq[INPUT_FRAMES - 1 + horizon]
            pred = _warp_slots(seq, horizon, params)
            last = _last_frame_baseline(seq, horizon)
            linear = _linear_frame_baseline(seq, horizon)
            key = f"h{horizon}"
            sums[key]["frontier_slot_warp_mse"] += _mse(pred, target)
            sums[key]["frontier_slot_warp_mae"] += _mae(pred, target)
            sums[key]["frontier_slot_warp_iou"] += _iou(pred, target)
            sums[key]["last_frame_mse"] += _mse(last, target)
            sums[key]["linear_frame_mse"] += _mse(linear, target)

    out: dict[str, dict[str, float]] = {}
    denom = float(len(seqs))
    for key, values in sums.items():
        metrics = {name: value / denom for name, value in values.items()}
        last_mse = max(metrics["last_frame_mse"], 1e-9)
        metrics["mse_skill_vs_last"] = (metrics["last_frame_mse"] - metrics["frontier_slot_warp_mse"]) / last_mse
        out[key] = metrics
    return out


def _weighted_score(metrics: dict[str, dict[str, float]]) -> float:
    weights = {"h1": 0.10, "h5": 0.20, "h10": 0.30, "h17": 0.40}
    return sum(weights[key] * metrics[key]["frontier_slot_warp_mse"] for key in weights)


def candidate_params() -> list[WarpParams]:
    candidates: list[WarpParams] = []
    for threshold in (0.06, 0.10, 0.14):
        for min_pixels in (8, 16):
            for max_shift in (8, 12):
                for momentum in (0.50, 0.75, 1.00):
                    candidates.append(
                        WarpParams(
                            threshold=threshold,
                            min_pixels=min_pixels,
                            max_slots=4,
                            max_shift=max_shift,
                            momentum=momentum,
                            use_tiles=False,
                            tile_size=16,
                            tile_stride=8,
                            tile_mass=8.0,
                            tile_alpha=0.0,
                        )
                    )
    for threshold in (0.08, 0.12):
        for max_shift in (8, 12):
            candidates.append(
                WarpParams(
                    threshold=threshold,
                    min_pixels=8,
                    max_slots=4,
                    max_shift=max_shift,
                    momentum=0.75,
                    use_tiles=True,
                    tile_size=16,
                    tile_stride=8,
                    tile_mass=5.0,
                    tile_alpha=0.35,
                )
            )
    return candidates


def tune_params(seqs: np.ndarray, limit: int) -> tuple[WarpParams, dict[str, dict[str, float]]]:
    tune = seqs[: min(limit, len(seqs))]
    best_params: WarpParams | None = None
    best_metrics: dict[str, dict[str, float]] | None = None
    best_score = math.inf
    for idx, params in enumerate(candidate_params(), start=1):
        metrics = evaluate_params(tune, params)
        score = _weighted_score(metrics)
        print(f"candidate {idx:02d}: weighted_mse={score:.6f} params={params}")
        if score < best_score:
            best_score = score
            best_params = params
            best_metrics = metrics
    if best_params is None or best_metrics is None:
        raise RuntimeError("No warp params evaluated")
    return best_params, best_metrics


def params_to_json(params: WarpParams) -> dict[str, float | int | bool]:
    return {
        "threshold": params.threshold,
        "min_pixels": params.min_pixels,
        "max_slots": params.max_slots,
        "max_shift": params.max_shift,
        "momentum": params.momentum,
        "use_tiles": params.use_tiles,
        "tile_size": params.tile_size,
        "tile_stride": params.tile_stride,
        "tile_mass": params.tile_mass,
        "tile_alpha": params.tile_alpha,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Frontier-inspired slot/tile warp decoder probe on real MovingMNIST.")
    parser.add_argument("--data-path", default="data/moving_mnist/mnist_test_seq.npy")
    parser.add_argument("--train-sequences", type=int, default=220)
    parser.add_argument("--test-sequences", type=int, default=40)
    parser.add_argument("--tune-sequences", type=int, default=80)
    parser.add_argument("--out", default="results/phase11a_frontier_slot_warp_probe.json")
    args = parser.parse_args()

    started = time.time()
    data_path = Path(args.data_path)
    source = download_real_moving_mnist(data_path)
    total = args.train_sequences + args.test_sequences
    seqs = load_sequences(data_path, total)
    train = seqs[: args.train_sequences]
    test = seqs[args.train_sequences : total]

    best_params, tune_metrics = tune_params(train, args.tune_sequences)
    test_metrics = evaluate_params(test, best_params)

    result = {
        "probe": "phase11a_frontier_slot_warp_probe",
        "data_path": str(data_path),
        "download_source": source,
        "real_dataset": True,
        "train_sequences": int(len(train)),
        "test_sequences": int(len(test)),
        "tune_sequences": int(min(args.tune_sequences, len(train))),
        "horizons": list(HORIZONS),
        "best_params": params_to_json(best_params),
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
