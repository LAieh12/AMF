from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from phase11a_confidence_selector_probe import train_amf
from phase11a_moving_mnist import (
    GT_HORIZONS,
    WARMUP_FRAMES,
    ConstantVelocityBaseline,
    LinearDeltaBaseline,
    MovingTransition,
    RealMovingMNISTCodec,
    RidgeLinearDynamicsBaseline,
    build_transitions,
    causalize_sequences,
    feature_interaction,
    feature_wall,
    load_real_moving_mnist,
    mask_iou,
    motion_token_sequences,
    transition_event,
)
from phase11a_slot_ranker_probe import compose_layers, render_rollout_layers


BRANCHES = (
    "simple_amf",
    "token_amf",
    "simple_ridge",
    "token_ridge",
    "simple_linear",
    "token_linear",
    "simple_cv",
    "token_cv",
)


def clone_or_self(model):
    return model.clone() if hasattr(model, "clone") else model


def adapt_if_amf(model, seq):
    if not hasattr(model, "learn_transition"):
        return model
    local = model.clone()
    for ctx in range(WARMUP_FRAMES):
        local.learn_transition(
            MovingTransition(
                state=seq.dyn[ctx],
                identity_features=seq.identity_features,
                next_state=seq.dyn[ctx + 1],
                sequence_id=seq.sequence_index,
                step=ctx,
                boundary_event=transition_event(seq.dyn[ctx]),
            )
        )
    return local


def roll_branch(codec, model, seq, horizon: int) -> np.ndarray:
    local = adapt_if_amf(model, seq)
    dyn = seq.dyn[WARMUP_FRAMES].copy()
    crops = codec.reference_crops(seq, WARMUP_FRAMES)
    for step in range(1, horizon + 1):
        dyn = local.predict_next(dyn, seq.identity_features)
    return render_rollout_layers(
        codec,
        dyn,
        seq.dyn[WARMUP_FRAMES],
        seq.frame_layers[WARMUP_FRAMES],
        horizon,
        crops,
    )


def rollout_multi(codec, models: dict[str, object], simple_seq, token_seq):
    out = {}
    seq_by_branch = {
        "simple_amf": simple_seq,
        "token_amf": token_seq,
        "simple_ridge": simple_seq,
        "token_ridge": token_seq,
        "simple_linear": simple_seq,
        "token_linear": token_seq,
        "simple_cv": simple_seq,
        "token_cv": token_seq,
    }
    for horizon in GT_HORIZONS:
        layers = {}
        for branch in BRANCHES:
            layers[branch] = roll_branch(codec, models[branch], seq_by_branch[branch], horizon)
        out[str(horizon)] = {
            "layers": layers,
            "actual_frame": simple_seq.frames[WARMUP_FRAMES + horizon],
            "simple_start_dyn": simple_seq.dyn[WARMUP_FRAMES],
            "token_start_dyn": token_seq.dyn[WARMUP_FRAMES],
        }
    return out


def branch_features(item: dict, horizon: int, names: list[str]) -> np.ndarray:
    frames = [compose_layers(item["layers"][name]) for name in names]
    frame_areas = [float(np.mean(frame > 0.18)) for frame in frames]
    frame_means = [float(np.mean(frame)) for frame in frames]
    pair_diffs = []
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            pair_diffs.append(float(np.mean(np.abs(frames[i] - frames[j]))))
    return np.concatenate(
        [
            np.asarray([horizon / max(GT_HORIZONS)], dtype=np.float32),
            np.asarray(frame_areas, dtype=np.float32),
            np.asarray(frame_means, dtype=np.float32),
            np.asarray(pair_diffs, dtype=np.float32),
            feature_wall(item["simple_start_dyn"]),
            feature_interaction(item["simple_start_dyn"]),
            feature_wall(item["token_start_dyn"]),
            feature_interaction(item["token_start_dyn"]),
        ]
    ).astype(np.float32)


def collect_training(codec, models, simple_sequences, token_sequences):
    x_by_h = {str(h): [] for h in GT_HORIZONS}
    y_by_h = {str(h): [] for h in GT_HORIZONS}
    for simple_seq, token_seq in zip(simple_sequences, token_sequences):
        rolled = rollout_multi(codec, models, simple_seq, token_seq)
        for horizon in GT_HORIZONS:
            key = str(horizon)
            item = rolled[key]
            names = list(BRANCHES)
            frames = [compose_layers(item["layers"][name]) for name in names]
            y = np.asarray([mask_iou(frame, item["actual_frame"]) for frame in frames], dtype=np.float32)
            x_by_h[key].append(branch_features(item, horizon, names))
            y_by_h[key].append(y)
    return {k: np.vstack(v) for k, v in x_by_h.items()}, {k: np.vstack(v) for k, v in y_by_h.items()}


def train_rankers(x_by_h, y_by_h, seed: int):
    rf = {}
    extra = {}
    stats = {}
    for horizon in GT_HORIZONS:
        key = str(horizon)
        x = x_by_h[key]
        y = y_by_h[key]
        rf_model = RandomForestRegressor(
            n_estimators=240,
            max_depth=8,
            min_samples_leaf=5,
            random_state=seed + horizon,
        )
        extra_model = ExtraTreesRegressor(
            n_estimators=260,
            max_depth=10,
            min_samples_leaf=4,
            random_state=seed + 100 + horizon,
        )
        rf_model.fit(x, y)
        extra_model.fit(x, y)
        rf[key] = rf_model
        extra[key] = extra_model
        stats[key] = {
            "n": int(len(y)),
            "branch_oracle_mean": float(np.mean(np.max(y, axis=1))),
        }
    return rf, extra, stats


def evaluate(codec, models, simple_sequences, token_sequences, rf, extra):
    rows = {
        str(h): {
            "simple_amf": [],
            "token_amf": [],
            "simple_ridge": [],
            "token_ridge": [],
            "branch_oracle": [],
            "rf_ranker": [],
            "extra_ranker": [],
            "ensemble_ranker": [],
        }
        for h in GT_HORIZONS
    }
    names = list(BRANCHES)
    for simple_seq, token_seq in zip(simple_sequences, token_sequences):
        rolled = rollout_multi(codec, models, simple_seq, token_seq)
        for horizon in GT_HORIZONS:
            key = str(horizon)
            item = rolled[key]
            frames = [compose_layers(item["layers"][name]) for name in names]
            actual = item["actual_frame"]
            ious = np.asarray([mask_iou(frame, actual) for frame in frames], dtype=np.float32)
            x = branch_features(item, horizon, names).reshape(1, -1)
            rf_scores = rf[key].predict(x)[0]
            extra_scores = extra[key].predict(x)[0]
            rf_idx = int(np.argmax(rf_scores))
            extra_idx = int(np.argmax(extra_scores))
            ensemble_idx = int(np.argmax(0.5 * rf_scores + 0.5 * extra_scores))
            rows[key]["simple_amf"].append(float(ious[names.index("simple_amf")]))
            rows[key]["token_amf"].append(float(ious[names.index("token_amf")]))
            rows[key]["simple_ridge"].append(float(ious[names.index("simple_ridge")]))
            rows[key]["token_ridge"].append(float(ious[names.index("token_ridge")]))
            rows[key]["branch_oracle"].append(float(np.max(ious)))
            rows[key]["rf_ranker"].append(mask_iou(frames[rf_idx], actual))
            rows[key]["extra_ranker"].append(mask_iou(frames[extra_idx], actual))
            rows[key]["ensemble_ranker"].append(mask_iou(frames[ensemble_idx], actual))
    return {key: {metric: float(np.mean(vals)) for metric, vals in metrics.items()} for key, metrics in rows.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 11A multi-encoder trajectory probe.")
    parser.add_argument("--train-sequences", type=int, default=220)
    parser.add_argument("--test-sequences", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4107)
    parser.add_argument("--out", default="results/phase11a_multi_encoder_probe.json")
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
    simple_transitions = build_transitions(simple_train)
    token_transitions = build_transitions(token_train)
    models = {
        "simple_amf": train_amf(simple_transitions, args.seed),
        "token_amf": train_amf(token_transitions, args.seed),
        "simple_ridge": RidgeLinearDynamicsBaseline().fit(simple_transitions),
        "token_ridge": RidgeLinearDynamicsBaseline().fit(token_transitions),
        "simple_linear": LinearDeltaBaseline().fit(simple_transitions),
        "token_linear": LinearDeltaBaseline().fit(token_transitions),
        "simple_cv": ConstantVelocityBaseline(),
        "token_cv": ConstantVelocityBaseline(),
    }
    x_by_h, y_by_h = collect_training(codec, models, simple_train, token_train)
    rf, extra, train_stats = train_rankers(x_by_h, y_by_h, args.seed)
    metrics = evaluate(codec, models, simple_test, token_test, rf, extra)
    results = {
        "dataset_shape": raw_shape,
        "train_sequences": args.train_sequences,
        "test_sequences": args.test_sequences,
        "branches": list(BRANCHES),
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
