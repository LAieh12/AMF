from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from phase11a_confidence_selector_probe import train_amf
from phase11a_moving_mnist import (
    GT_HORIZONS,
    WARMUP_FRAMES,
    RealMovingMNISTCodec,
    build_transitions,
    causalize_sequences,
    load_real_moving_mnist,
    mask_iou,
    motion_token_sequences,
)
from phase11a_slot_ranker_probe import candidate_layers, compose_layers, rollout_pair_slots


def reflect_coord(value: float, limit: int) -> float:
    lo = 0.0
    hi = float(limit - 1)
    out = float(value)
    for _ in range(4):
        if out < lo:
            out = -out
        elif out > hi:
            out = 2.0 * hi - out
        else:
            break
    return float(np.clip(out, lo, hi))


def extract_patch(frame: np.ndarray, cy: float, cx: float, radius: int) -> np.ndarray:
    y = int(round(float(cy)))
    x = int(round(float(cx)))
    pad = radius + max(abs(y), abs(x), frame.shape[0], frame.shape[1])
    padded = np.pad(np.asarray(frame, dtype=np.float32), pad, mode="constant")
    y += pad
    x += pad
    return padded[y - radius : y + radius + 1, x - radius : x + radius + 1].astype(np.float32)


def place_patch(out: np.ndarray, patch: np.ndarray, cy: float, cx: float) -> None:
    radius = patch.shape[0] // 2
    y = int(round(float(cy)))
    x = int(round(float(cx)))
    h, w = out.shape
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    if y1 <= y0 or x1 <= x0:
        return
    py0 = y0 - (y - radius)
    py1 = py0 + (y1 - y0)
    px0 = x0 - (x - radius)
    px1 = px0 + (x1 - x0)
    out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], patch[py0:py1, px0:px1])


def patch_velocity(
    context: np.ndarray,
    cy: int,
    cx: int,
    patch_radius: int,
    search_radius: int,
    temperature: float,
) -> tuple[float, float]:
    current_index = context.shape[0] - 1
    current = context[current_index]
    query = extract_patch(current, cy, cx, patch_radius)
    if float(np.max(query)) < 0.05:
        return 0.0, 0.0
    values = []
    losses = []
    for past_index in range(current_index):
        dt = float(current_index - past_index)
        radius = int(round(search_radius * dt))
        past = context[past_index]
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                py = cy + dy
                px = cx + dx
                candidate = extract_patch(past, py, px, patch_radius)
                loss = float(np.mean(np.square(query - candidate)))
                vy = (float(cy) - float(py)) / dt
                vx = (float(cx) - float(px)) / dt
                values.append((vy, vx))
                losses.append(loss)
    loss_arr = np.asarray(losses, dtype=np.float32)
    value_arr = np.asarray(values, dtype=np.float32)
    weights = np.exp(-(loss_arr - float(np.min(loss_arr))) / max(1e-4, float(temperature)))
    weights = weights / (float(np.sum(weights)) + 1e-8)
    velocity = np.sum(value_arr * weights[:, None], axis=0)
    return float(velocity[0]), float(velocity[1])


def patch_attention_predict(
    context: np.ndarray,
    horizon: int,
    patch_size: int,
    stride: int,
    search_radius: int,
    temperature: float,
) -> np.ndarray:
    current = np.asarray(context[-1], dtype=np.float32)
    out = np.zeros_like(current, dtype=np.float32)
    radius = int(patch_size // 2)
    for cy in range(0, current.shape[0], stride):
        for cx in range(0, current.shape[1], stride):
            patch = extract_patch(current, cy, cx, radius)
            if float(np.max(patch)) < 0.05:
                continue
            vy, vx = patch_velocity(context, cy, cx, radius, search_radius, temperature)
            py = reflect_coord(float(cy) + float(horizon) * vy, current.shape[0])
            px = reflect_coord(float(cx) + float(horizon) * vx, current.shape[1])
            place_patch(out, patch, py, px)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def evaluate(
    codec,
    simple_model,
    token_model,
    simple_sequences,
    token_sequences,
    patch_size: int,
    stride: int,
    search_radius: int,
    temperature: float,
):
    rows = {
        str(h): {
            "simple": [],
            "token": [],
            "max_beta_1.00": [],
            "frame_oracle": [],
            "patch_attention": [],
        }
        for h in GT_HORIZONS
    }
    for simple_seq, token_seq in zip(simple_sequences, token_sequences):
        rolled = rollout_pair_slots(codec, simple_model, token_model, simple_seq, token_seq)
        context = simple_seq.frames[: WARMUP_FRAMES + 1]
        for horizon in GT_HORIZONS:
            key = str(horizon)
            item = rolled[key]
            candidates = candidate_layers(item["simple_layers"], item["token_layers"])
            actual = item["actual_frame"]
            candidate_frames = {name: compose_layers(candidates[name]) for name in candidates}
            frame_ious = [mask_iou(frame, actual) for frame in candidate_frames.values()]
            pred = patch_attention_predict(context, horizon, patch_size, stride, search_radius, temperature)
            rows[key]["simple"].append(mask_iou(candidate_frames["simple"], actual))
            rows[key]["token"].append(mask_iou(candidate_frames["token"], actual))
            rows[key]["max_beta_1.00"].append(mask_iou(candidate_frames["max_beta_1.00"], actual))
            rows[key]["frame_oracle"].append(float(np.max(frame_ious)))
            rows[key]["patch_attention"].append(mask_iou(pred, actual))
    return {key: {metric: float(np.mean(vals)) for metric, vals in metrics.items()} for key, metrics in rows.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 11A patch-attention temporal encoder probe.")
    parser.add_argument("--train-sequences", type=int, default=220)
    parser.add_argument("--test-sequences", type=int, default=20)
    parser.add_argument("--patch-size", type=int, default=11)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--search-radius", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=4107)
    parser.add_argument("--out", default="results/phase11a_patch_attention_probe.json")
    args = parser.parse_args()

    start = time.perf_counter()
    codec = RealMovingMNISTCodec()
    train, test, raw_shape = load_real_moving_mnist(
        "data/MovingMNIST/mnist_test_seq.npy",
        codec,
        args.train_sequences,
        args.test_sequences,
    )
    codec.fit_identity_bank(train)
    simple_train = causalize_sequences(train)
    simple_test = causalize_sequences(test)
    token_train = motion_token_sequences(train)
    token_test = motion_token_sequences(test)
    simple_model = train_amf(build_transitions(simple_train), args.seed)
    token_model = train_amf(build_transitions(token_train), args.seed)
    metrics = evaluate(
        codec,
        simple_model,
        token_model,
        simple_test,
        token_test,
        args.patch_size,
        args.stride,
        args.search_radius,
        args.temperature,
    )
    results = {
        "dataset_shape": raw_shape,
        "train_sequences": args.train_sequences,
        "test_sequences": args.test_sequences,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "search_radius": args.search_radius,
        "temperature": args.temperature,
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - start,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
