from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from phase11a_confidence_selector_probe import train_amf
from phase11a_moving_mnist import (
    GT_HORIZONS,
    RealMovingMNISTCodec,
    build_transitions,
    causalize_sequences,
    feature_interaction,
    feature_wall,
    load_real_moving_mnist,
    mask_iou,
    motion_token_sequences,
)
from phase11a_slot_ranker_probe import candidate_layers, compose_layers, rollout_pair_slots


def tile_bounds(size: int, tile_size: int) -> list[tuple[int, int, int, int]]:
    bounds = []
    for y0 in range(0, size, tile_size):
        for x0 in range(0, size, tile_size):
            bounds.append((y0, min(size, y0 + tile_size), x0, min(size, x0 + tile_size)))
    return bounds


def tile_iou(pred: np.ndarray, actual: np.ndarray, threshold: float = 0.18) -> float:
    return mask_iou(pred, actual, threshold=threshold)


def frame_stack(candidates: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    return np.asarray([compose_layers(candidates[name]) for name in names], dtype=np.float32)


def tile_feature(
    item: dict,
    horizon: int,
    frames: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> np.ndarray:
    size = float(item["actual_frame"].shape[0])
    tile = frames[:, y0:y1, x0:x1]
    union = np.max(tile, axis=0)
    simple_tile = frames[0, y0:y1, x0:x1]
    token_tile = frames[1, y0:y1, x0:x1]
    per_candidate = []
    for cand in tile:
        per_candidate.extend(
            [
                float(np.mean(cand > 0.18)),
                float(np.mean(cand)),
                float(np.max(cand)),
                float(np.std(cand)),
            ]
        )
    context = np.asarray(
        [
            horizon / max(GT_HORIZONS),
            (x0 + x1) * 0.5 / size,
            (y0 + y1) * 0.5 / size,
            (x1 - x0) * (y1 - y0) / (size * size),
            float(np.mean(union > 0.18)),
            float(np.mean(union)),
            float(np.max(union)),
            float(np.mean(np.abs(simple_tile - token_tile))),
            float(np.mean(np.minimum(simple_tile, token_tile) > 0.18)),
        ],
        dtype=np.float32,
    )
    dyn_features = np.concatenate(
        [
            feature_wall(item["simple_dyn"]),
            feature_interaction(item["simple_dyn"]),
            feature_wall(item["token_dyn"]),
            feature_interaction(item["token_dyn"]),
            feature_wall(item["start_dyn"]),
            feature_interaction(item["start_dyn"]),
        ]
    ).astype(np.float32)
    return np.concatenate([context, np.asarray(per_candidate, dtype=np.float32), dyn_features]).astype(np.float32)


def tile_label(frames: np.ndarray, actual: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> int:
    actual_tile = actual[y0:y1, x0:x1]
    scores = [tile_iou(frame[y0:y1, x0:x1], actual_tile) for frame in frames]
    return int(np.argmax(np.asarray(scores, dtype=np.float32)))


def tile_scores(frames: np.ndarray, actual: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    actual_tile = actual[y0:y1, x0:x1]
    return np.asarray([tile_iou(frame[y0:y1, x0:x1], actual_tile) for frame in frames], dtype=np.float32)


def collect_tile_training(codec, simple_model, token_model, simple_sequences, token_sequences, tile_size: int):
    x_by_h = {str(h): [] for h in GT_HORIZONS}
    y_by_h = {str(h): [] for h in GT_HORIZONS}
    score_y_by_h = {str(h): [] for h in GT_HORIZONS}
    weight_by_h = {str(h): [] for h in GT_HORIZONS}
    names_by_h: dict[str, list[str]] = {}
    for simple_seq, token_seq in zip(simple_sequences, token_sequences):
        rolled = rollout_pair_slots(codec, simple_model, token_model, simple_seq, token_seq)
        for horizon in GT_HORIZONS:
            key = str(horizon)
            item = rolled[key]
            candidates = candidate_layers(item["simple_layers"], item["token_layers"])
            names = list(candidates.keys())
            names_by_h[key] = names
            frames = frame_stack(candidates, names)
            actual = item["actual_frame"]
            for y0, y1, x0, x1 in tile_bounds(actual.shape[0], tile_size):
                x_by_h[key].append(tile_feature(item, horizon, frames, y0, y1, x0, x1))
                y_by_h[key].append(tile_label(frames, actual, y0, y1, x0, x1))
                score_y_by_h[key].append(tile_scores(frames, actual, y0, y1, x0, x1))
                actual_active = bool(np.any(actual[y0:y1, x0:x1] > 0.18))
                candidate_active = bool(np.any(np.max(frames[:, y0:y1, x0:x1], axis=0) > 0.18))
                weight_by_h[key].append(3.0 if (actual_active or candidate_active) else 0.25)
    x_by_h = {key: np.vstack(vals) for key, vals in x_by_h.items()}
    y_by_h = {key: np.asarray(vals, dtype=np.int32) for key, vals in y_by_h.items()}
    score_y_by_h = {key: np.vstack(vals) for key, vals in score_y_by_h.items()}
    weight_by_h = {key: np.asarray(vals, dtype=np.float32) for key, vals in weight_by_h.items()}
    return x_by_h, y_by_h, score_y_by_h, weight_by_h, names_by_h


def train_tile_routers(x_by_h, y_by_h, score_y_by_h, weight_by_h, seed: int):
    class_models = {}
    reg_models = {}
    stats = {}
    for horizon in GT_HORIZONS:
        key = str(horizon)
        x = x_by_h[key]
        y = y_by_h[key]
        yr = score_y_by_h[key]
        weights = weight_by_h[key]
        clf = ExtraTreesClassifier(
            n_estimators=240,
            max_depth=12,
            min_samples_leaf=3,
            random_state=seed + horizon,
            class_weight="balanced",
        )
        clf.fit(x, y, sample_weight=weights)
        reg = ExtraTreesRegressor(
            n_estimators=240,
            max_depth=12,
            min_samples_leaf=3,
            random_state=seed + 100 + horizon,
        )
        reg.fit(x, yr, sample_weight=weights)
        class_models[key] = clf
        reg_models[key] = reg
        stats[key] = {
            "n": int(len(y)),
            "class_hist": {str(int(c)): int(np.sum(y == c)) for c in np.unique(y)},
            "tile_oracle_train_mean": float(np.mean(np.max(yr, axis=1))),
            "active_weight_mean": float(np.mean(weights)),
        }
    return class_models, reg_models, stats


def route_frame(model, item: dict, horizon: int, candidates: dict[str, np.ndarray], names: list[str], tile_size: int, mode: str) -> np.ndarray:
    frames = frame_stack(candidates, names)
    out = np.zeros_like(frames[0])
    for y0, y1, x0, x1 in tile_bounds(out.shape[0], tile_size):
        feature = tile_feature(item, horizon, frames, y0, y1, x0, x1).reshape(1, -1)
        pred = model.predict(feature)[0]
        if mode == "reg":
            idx = int(np.argmax(pred))
        else:
            idx = int(pred)
        out[y0:y1, x0:x1] = frames[idx, y0:y1, x0:x1]
    return out.astype(np.float32)


def tile_oracle_frame(item: dict, candidates: dict[str, np.ndarray], names: list[str], tile_size: int) -> np.ndarray:
    frames = frame_stack(candidates, names)
    actual = item["actual_frame"]
    out = np.zeros_like(actual, dtype=np.float32)
    for y0, y1, x0, x1 in tile_bounds(actual.shape[0], tile_size):
        idx = tile_label(frames, actual, y0, y1, x0, x1)
        out[y0:y1, x0:x1] = frames[idx, y0:y1, x0:x1]
    return out


def evaluate(codec, simple_model, token_model, simple_sequences, token_sequences, class_models, reg_models, tile_size: int):
    rows = {
        str(h): {
            "simple": [],
            "token": [],
            "max_beta_1.00": [],
            "frame_oracle": [],
            "tile_oracle": [],
            "tile_router_class": [],
            "tile_router_reg": [],
        }
        for h in GT_HORIZONS
    }
    for simple_seq, token_seq in zip(simple_sequences, token_sequences):
        rolled = rollout_pair_slots(codec, simple_model, token_model, simple_seq, token_seq)
        for horizon in GT_HORIZONS:
            key = str(horizon)
            item = rolled[key]
            candidates = candidate_layers(item["simple_layers"], item["token_layers"])
            names = list(candidates.keys())
            actual = item["actual_frame"]
            candidate_frames = {name: compose_layers(candidates[name]) for name in names}
            ious = [mask_iou(frame, actual) for frame in candidate_frames.values()]
            class_router = route_frame(class_models[key], item, horizon, candidates, names, tile_size, "class")
            reg_router = route_frame(reg_models[key], item, horizon, candidates, names, tile_size, "reg")
            oracle = tile_oracle_frame(item, candidates, names, tile_size)
            rows[key]["simple"].append(mask_iou(candidate_frames["simple"], actual))
            rows[key]["token"].append(mask_iou(candidate_frames["token"], actual))
            rows[key]["max_beta_1.00"].append(mask_iou(candidate_frames["max_beta_1.00"], actual))
            rows[key]["frame_oracle"].append(float(np.max(ious)))
            rows[key]["tile_oracle"].append(mask_iou(oracle, actual))
            rows[key]["tile_router_class"].append(mask_iou(class_router, actual))
            rows[key]["tile_router_reg"].append(mask_iou(reg_router, actual))
    return {key: {metric: float(np.mean(vals)) for metric, vals in metrics.items()} for key, metrics in rows.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 11A tile/cell router decoder probe.")
    parser.add_argument("--train-sequences", type=int, default=220)
    parser.add_argument("--test-sequences", type=int, default=20)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4107)
    parser.add_argument("--out", default="results/phase11a_cell_router_probe.json")
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
    x_by_h, y_by_h, score_y_by_h, weight_by_h, names_by_h = collect_tile_training(
        codec,
        simple_model,
        token_model,
        simple_train,
        token_train,
        args.tile_size,
    )
    class_models, reg_models, train_stats = train_tile_routers(x_by_h, y_by_h, score_y_by_h, weight_by_h, args.seed)
    metrics = evaluate(codec, simple_model, token_model, simple_test, token_test, class_models, reg_models, args.tile_size)
    results = {
        "dataset_shape": raw_shape,
        "train_sequences": args.train_sequences,
        "test_sequences": args.test_sequences,
        "tile_size": args.tile_size,
        "candidate_names_by_horizon": names_by_h,
        "train_stats": train_stats,
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - start,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
