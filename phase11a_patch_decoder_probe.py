from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from phase11a_confidence_selector_probe import train_amf
from phase11a_moving_mnist import (
    GT_HORIZONS,
    build_transitions,
    causalize_sequences,
    feature_interaction,
    feature_wall,
    load_real_moving_mnist,
    mask_iou,
    motion_token_sequences,
    object_view,
)
from phase11a_slot_ranker_probe import candidate_layers, compose_layers, rollout_pair_slots


PATCH_CANDIDATES = (
    "simple",
    "token",
    "max_beta_0.50",
    "blend_beta_0.50",
    "max_beta_0.75",
    "blend_beta_0.75",
    "max_beta_1.00",
)


def selected_candidate_names(candidates: dict[str, np.ndarray]) -> list[str]:
    return [name for name in PATCH_CANDIDATES if name in candidates]


def frame_stack(candidates: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    return np.asarray([compose_layers(candidates[name]) for name in names], dtype=np.float32)


def decoder_features(item: dict, horizon: int, candidates: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    size = item["actual_frame"].shape[0]
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    x = xx / max(1.0, float(size - 1))
    y = yy / max(1.0, float(size - 1))
    frames = frame_stack(candidates, names)
    max_frame = np.max(frames, axis=0)
    min_frame = np.min(frames, axis=0)
    mean_frame = np.mean(frames, axis=0)
    std_frame = np.std(frames, axis=0)
    simple = compose_layers(candidates["simple"])
    token = compose_layers(candidates["token"])
    object_maps = []
    distance_maps = []
    for obj in range(2):
        simple_layer = candidates["simple"][obj]
        token_layer = candidates["token"][obj]
        max_layer = np.maximum(simple_layer, token_layer)
        object_maps.extend([simple_layer, token_layer, max_layer, np.abs(simple_layer - token_layer)])
        for dyn_key in ("simple_dyn", "token_dyn", "start_dyn"):
            v = object_view(item[dyn_key], obj)
            dx = x - float(v[0])
            dy = y - float(v[1])
            scale = max(0.04, 0.5 * float(v[4] + v[5]))
            distance_maps.append(np.exp(-((dx * dx + dy * dy) / (2.0 * scale * scale))).astype(np.float32))
    wall = feature_wall(item["simple_dyn"])
    inter = feature_interaction(item["simple_dyn"])
    global_maps = [np.full_like(x, float(v), dtype=np.float32) for v in np.concatenate([wall, inter])]
    maps = [
        x,
        y,
        np.full_like(x, float(horizon) / max(GT_HORIZONS), dtype=np.float32),
        simple,
        token,
        np.abs(simple - token),
        max_frame,
        min_frame,
        mean_frame,
        std_frame,
        *list(frames),
        *object_maps,
        *distance_maps,
        *global_maps,
    ]
    return np.stack(maps, axis=-1).reshape(size * size, -1).astype(np.float32)


def sample_pixels(
    rng: np.random.Generator,
    item: dict,
    candidates: dict[str, np.ndarray],
    names: list[str],
    per_frame: int,
) -> np.ndarray:
    actual = item["actual_frame"].reshape(-1)
    candidate_union = np.max(frame_stack(candidates, names), axis=0).reshape(-1)
    pos = np.flatnonzero(actual > 0.18)
    hard_neg = np.flatnonzero((actual <= 0.18) & (candidate_union > 0.05))
    easy_neg = np.flatnonzero((actual <= 0.18) & (candidate_union <= 0.05))
    half = max(8, per_frame // 2)
    pos_take = min(len(pos), half)
    hard_take = min(len(hard_neg), max(4, per_frame // 3))
    easy_take = max(0, per_frame - pos_take - hard_take)
    parts = []
    if pos_take:
        parts.append(rng.choice(pos, size=pos_take, replace=len(pos) < pos_take))
    if hard_take:
        parts.append(rng.choice(hard_neg, size=hard_take, replace=len(hard_neg) < hard_take))
    if easy_take and len(easy_neg):
        parts.append(rng.choice(easy_neg, size=easy_take, replace=len(easy_neg) < easy_take))
    if not parts:
        return rng.choice(actual.size, size=per_frame, replace=False)
    return np.concatenate(parts).astype(np.int32)


def collect_pixel_training(
    codec,
    simple_model,
    token_model,
    simple_sequences,
    token_sequences,
    seed: int,
    per_frame: int,
    max_pixels_per_horizon: int,
):
    rng = np.random.default_rng(seed)
    x_by_h = {str(h): [] for h in GT_HORIZONS}
    y_by_h = {str(h): [] for h in GT_HORIZONS}
    names_by_h = {}
    counts = {str(h): 0 for h in GT_HORIZONS}
    for simple_seq, token_seq in zip(simple_sequences, token_sequences):
        rolled = rollout_pair_slots(codec, simple_model, token_model, simple_seq, token_seq)
        for horizon in GT_HORIZONS:
            key = str(horizon)
            if counts[key] >= max_pixels_per_horizon:
                continue
            item = rolled[key]
            candidates = candidate_layers(item["simple_layers"], item["token_layers"])
            names = selected_candidate_names(candidates)
            names_by_h[key] = names
            features = decoder_features(item, horizon, candidates, names)
            idx = sample_pixels(rng, item, candidates, names, per_frame)
            remaining = max_pixels_per_horizon - counts[key]
            idx = idx[:remaining]
            x_by_h[key].append(features[idx])
            y_by_h[key].append((item["actual_frame"].reshape(-1)[idx] > 0.18).astype(np.int8))
            counts[key] += int(len(idx))
    x_by_h = {k: np.vstack(v) for k, v in x_by_h.items() if v}
    y_by_h = {k: np.concatenate(v) for k, v in y_by_h.items() if v}
    return x_by_h, y_by_h, names_by_h, counts


def train_patch_decoders(x_by_h, y_by_h, seed: int):
    models = {}
    stats = {}
    for horizon in GT_HORIZONS:
        key = str(horizon)
        x = x_by_h[key]
        y = y_by_h[key]
        pos_rate = float(np.mean(y))
        sample_weight = np.where(y > 0, 0.5 / max(pos_rate, 1e-4), 0.5 / max(1.0 - pos_rate, 1e-4))
        model = HistGradientBoostingClassifier(
            max_iter=120,
            max_leaf_nodes=31,
            learning_rate=0.06,
            l2_regularization=0.02,
            random_state=seed + horizon,
        )
        model.fit(x, y, sample_weight=sample_weight)
        models[key] = model
        stats[key] = {"n": int(len(y)), "positive_rate": pos_rate}
    return models, stats


def predict_patch_frame(model, features: np.ndarray, size: int) -> np.ndarray:
    probs = model.predict_proba(features)[:, 1]
    return probs.reshape(size, size).astype(np.float32)


def probability_iou(pred: np.ndarray, actual: np.ndarray, pred_threshold: float, actual_threshold: float = 0.18) -> float:
    pred_mask = np.asarray(pred, dtype=np.float32) > float(pred_threshold)
    actual_mask = np.asarray(actual, dtype=np.float32) > float(actual_threshold)
    union = np.logical_or(pred_mask, actual_mask).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred_mask, actual_mask).sum() / union)


def calibrate_thresholds(
    codec,
    simple_model,
    token_model,
    simple_sequences,
    token_sequences,
    models,
    threshold_grid: np.ndarray,
):
    scores = {str(h): {float(t): [] for t in threshold_grid} for h in GT_HORIZONS}
    for simple_seq, token_seq in zip(simple_sequences, token_sequences):
        rolled = rollout_pair_slots(codec, simple_model, token_model, simple_seq, token_seq)
        for horizon in GT_HORIZONS:
            key = str(horizon)
            item = rolled[key]
            candidates = candidate_layers(item["simple_layers"], item["token_layers"])
            names = selected_candidate_names(candidates)
            features = decoder_features(item, horizon, candidates, names)
            pred = predict_patch_frame(models[key], features, item["actual_frame"].shape[0])
            for threshold in threshold_grid:
                scores[key][float(threshold)].append(probability_iou(pred, item["actual_frame"], pred_threshold=float(threshold)))
    thresholds = {}
    summary = {}
    for horizon in GT_HORIZONS:
        key = str(horizon)
        means = {str(t): float(np.mean(vals)) for t, vals in scores[key].items() if vals}
        best = max(means.items(), key=lambda kv: kv[1])
        thresholds[key] = float(best[0])
        summary[key] = {"threshold": float(best[0]), "calibration_iou": float(best[1]), "grid": means}
    return thresholds, summary


def evaluate(codec, simple_model, token_model, simple_sequences, token_sequences, models, thresholds):
    rows = {
        str(h): {
            "simple": [],
            "token": [],
            "max_beta_1.00": [],
            "frame_oracle": [],
            "patch_decoder": [],
        }
        for h in GT_HORIZONS
    }
    for simple_seq, token_seq in zip(simple_sequences, token_sequences):
        rolled = rollout_pair_slots(codec, simple_model, token_model, simple_seq, token_seq)
        for horizon in GT_HORIZONS:
            key = str(horizon)
            item = rolled[key]
            candidates = candidate_layers(item["simple_layers"], item["token_layers"])
            names = selected_candidate_names(candidates)
            actual = item["actual_frame"]
            candidate_frames = {name: compose_layers(candidates[name]) for name in names}
            ious = [mask_iou(frame, actual) for frame in candidate_frames.values()]
            features = decoder_features(item, horizon, candidates, names)
            pred = predict_patch_frame(models[key], features, actual.shape[0])
            rows[key]["simple"].append(mask_iou(candidate_frames["simple"], actual))
            rows[key]["token"].append(mask_iou(candidate_frames["token"], actual))
            rows[key]["max_beta_1.00"].append(mask_iou(candidate_frames["max_beta_1.00"], actual))
            rows[key]["frame_oracle"].append(float(np.max(ious)))
            rows[key]["patch_decoder"].append(probability_iou(pred, actual, pred_threshold=thresholds[key]))
    return {key: {metric: float(np.mean(vals)) for metric, vals in metrics.items()} for key, metrics in rows.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 11A patch decoder probe for causal visual hypotheses.")
    parser.add_argument("--train-sequences", type=int, default=220)
    parser.add_argument("--test-sequences", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4107)
    parser.add_argument("--per-frame", type=int, default=384)
    parser.add_argument("--max-pixels-per-horizon", type=int, default=180000)
    parser.add_argument("--calibration-sequences", type=int, default=40)
    parser.add_argument("--out", default="results/phase11a_patch_decoder_probe.json")
    args = parser.parse_args()

    start = time.perf_counter()
    from phase11a_moving_mnist import RealMovingMNISTCodec

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
    split = max(1, min(len(simple_train) - 1, int(round(len(simple_train) * 0.82))))
    fit_simple = simple_train[:split]
    fit_token = token_train[:split]
    calib_simple = simple_train[split : split + args.calibration_sequences]
    calib_token = token_train[split : split + args.calibration_sequences]
    if not calib_simple:
        calib_simple = simple_train[: min(len(simple_train), args.calibration_sequences)]
        calib_token = token_train[: min(len(token_train), args.calibration_sequences)]

    x_by_h, y_by_h, names_by_h, counts = collect_pixel_training(
        codec,
        simple_model,
        token_model,
        fit_simple,
        fit_token,
        args.seed,
        args.per_frame,
        args.max_pixels_per_horizon,
    )
    models, train_stats = train_patch_decoders(x_by_h, y_by_h, args.seed)
    thresholds, calibration = calibrate_thresholds(
        codec,
        simple_model,
        token_model,
        calib_simple,
        calib_token,
        models,
        np.linspace(0.12, 0.72, 13, dtype=np.float32),
    )
    metrics = evaluate(codec, simple_model, token_model, simple_test, token_test, models, thresholds)
    results = {
        "dataset_shape": raw_shape,
        "train_sequences": args.train_sequences,
        "test_sequences": args.test_sequences,
        "fit_sequences": len(fit_simple),
        "calibration_sequences": len(calib_simple),
        "candidate_names_by_horizon": names_by_h,
        "pixel_counts": counts,
        "train_stats": train_stats,
        "thresholds": thresholds,
        "actual_mask_threshold": 0.18,
        "calibration": calibration,
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - start,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
