from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from phase11a_moving_mnist import (
    AMFMovingMNISTWorldModel,
    GT_HORIZONS,
    WARMUP_FRAMES,
    MovingTransition,
    RealMovingMNISTCodec,
    build_transitions,
    causalize_sequences,
    center_error_px,
    feature_interaction,
    feature_wall,
    load_real_moving_mnist,
    mask_iou,
    motion_token_sequences,
    object_view,
    soft_iou,
    transition_event,
)


def train_amf(transitions, seed: int) -> AMFMovingMNISTWorldModel:
    model = AMFMovingMNISTWorldModel(
        metaplasticity=True,
        boundary_guard=True,
        residual_scale=0.0,
        collision_box=0.317,
    ).fit(transitions)
    rng = np.random.default_rng(seed)
    order = rng.choice(len(transitions), size=min(2500, len(transitions)), replace=False)
    for idx in order:
        model.learn_transition(transitions[int(idx)])
    return model


def rollout_pair(codec: RealMovingMNISTCodec, simple_model, token_model, simple_seq, token_seq):
    simple_local = simple_model.clone()
    token_local = token_model.clone()
    for ctx in range(WARMUP_FRAMES):
        simple_local.learn_transition(
            MovingTransition(
                state=simple_seq.dyn[ctx],
                identity_features=simple_seq.identity_features,
                next_state=simple_seq.dyn[ctx + 1],
                sequence_id=simple_seq.sequence_index,
                step=ctx,
                boundary_event=transition_event(simple_seq.dyn[ctx]),
            )
        )
        token_local.learn_transition(
            MovingTransition(
                state=token_seq.dyn[ctx],
                identity_features=token_seq.identity_features,
                next_state=token_seq.dyn[ctx + 1],
                sequence_id=token_seq.sequence_index,
                step=ctx,
                boundary_event=transition_event(token_seq.dyn[ctx]),
            )
        )
    simple_dyn = simple_seq.dyn[WARMUP_FRAMES].copy()
    token_dyn = token_seq.dyn[WARMUP_FRAMES].copy()
    crops = codec.reference_crops(simple_seq, WARMUP_FRAMES)
    out = {}
    for step in range(1, max(GT_HORIZONS) + 1):
        simple_dyn = simple_local.predict_next(simple_dyn, simple_seq.identity_features)
        token_dyn = token_local.predict_next(token_dyn, token_seq.identity_features)
        if step in GT_HORIZONS:
            simple_frame = codec.render_rollout_frame(
                simple_dyn,
                simple_seq.dyn[WARMUP_FRAMES],
                simple_seq.frame_layers[WARMUP_FRAMES],
                step,
                crops,
            )
            token_frame = codec.render_rollout_frame(
                token_dyn,
                token_seq.dyn[WARMUP_FRAMES],
                token_seq.frame_layers[WARMUP_FRAMES],
                step,
                crops,
            )
            out[str(step)] = {
                "simple_dyn": simple_dyn.copy(),
                "token_dyn": token_dyn.copy(),
                "simple_frame": simple_frame,
                "token_frame": token_frame,
                "actual_frame": simple_seq.frames[WARMUP_FRAMES + step],
                "actual_dyn": simple_seq.dyn[WARMUP_FRAMES + step],
                "start_dyn": simple_seq.dyn[WARMUP_FRAMES],
            }
    return out


def selector_features(item: dict, horizon: int) -> np.ndarray:
    simple_dyn = item["simple_dyn"]
    token_dyn = item["token_dyn"]
    start_dyn = item["start_dyn"]
    center_gap = []
    speed_gap = []
    simple_speed = []
    token_speed = []
    for obj in range(2):
        sv = object_view(simple_dyn, obj)
        tv = object_view(token_dyn, obj)
        center_gap.append(float(np.linalg.norm((sv[:2] - tv[:2]) * 64.0)))
        speed_gap.append(float(np.linalg.norm((sv[2:4] - tv[2:4]) * 64.0)))
        simple_speed.append(float(np.linalg.norm(sv[2:4]) * 64.0))
        token_speed.append(float(np.linalg.norm(tv[2:4]) * 64.0))
    frame_gap = float(np.mean(np.abs(item["simple_frame"] - item["token_frame"])))
    union_area = float(np.mean(np.maximum(item["simple_frame"], item["token_frame"]) > 0.18))
    simple_area = float(np.mean(item["simple_frame"] > 0.18))
    token_area = float(np.mean(item["token_frame"] > 0.18))
    return np.concatenate(
        [
            np.asarray([horizon / max(GT_HORIZONS), frame_gap, union_area, simple_area, token_area], dtype=np.float32),
            np.asarray(center_gap + speed_gap + simple_speed + token_speed, dtype=np.float32),
            feature_wall(simple_dyn),
            feature_interaction(simple_dyn),
            feature_wall(start_dyn),
            feature_interaction(start_dyn),
        ]
    ).astype(np.float32)


def collect_training(codec, simple_model, token_model, simple_sequences, token_sequences):
    features = {str(h): [] for h in GT_HORIZONS}
    labels = {str(h): [] for h in GT_HORIZONS}
    for simple_seq, token_seq in zip(simple_sequences, token_sequences):
        rolled = rollout_pair(codec, simple_model, token_model, simple_seq, token_seq)
        for horizon in GT_HORIZONS:
            key = str(horizon)
            item = rolled[key]
            simple_iou = mask_iou(item["simple_frame"], item["actual_frame"])
            token_iou = mask_iou(item["token_frame"], item["actual_frame"])
            features[key].append(selector_features(item, horizon))
            labels[key].append(1 if token_iou > simple_iou + 0.005 else 0)
    return features, labels


def train_selectors(features, labels, seed: int):
    selectors = {}
    stats = {}
    for horizon in GT_HORIZONS:
        key = str(horizon)
        x = np.vstack(features[key])
        y = np.asarray(labels[key], dtype=np.int32)
        stats[key] = {"positive_rate": float(np.mean(y)), "n": int(len(y))}
        if len(np.unique(y)) < 2:
            selectors[key] = None
        else:
            clf = RandomForestClassifier(
                n_estimators=160,
                max_depth=5,
                min_samples_leaf=6,
                random_state=seed + horizon,
                class_weight="balanced",
            )
            clf.fit(x, y)
            selectors[key] = clf
    return selectors, stats


def evaluate(codec, simple_model, token_model, simple_sequences, token_sequences, selectors):
    rows = {str(h): {"simple": [], "token": [], "max": [], "selected": [], "oracle": []} for h in GT_HORIZONS}
    center_rows = {str(h): [] for h in GT_HORIZONS}
    selector_use = {str(h): [] for h in GT_HORIZONS}
    for simple_seq, token_seq in zip(simple_sequences, token_sequences):
        rolled = rollout_pair(codec, simple_model, token_model, simple_seq, token_seq)
        for horizon in GT_HORIZONS:
            key = str(horizon)
            item = rolled[key]
            simple_frame = item["simple_frame"]
            token_frame = item["token_frame"]
            max_frame = np.maximum(simple_frame, token_frame)
            selector = selectors[key]
            if selector is None:
                choose_token = False
            else:
                choose_token = bool(selector.predict(selector_features(item, horizon).reshape(1, -1))[0])
            selected_frame = token_frame if choose_token else simple_frame
            actual = item["actual_frame"]
            rows[key]["simple"].append(mask_iou(simple_frame, actual))
            rows[key]["token"].append(mask_iou(token_frame, actual))
            rows[key]["max"].append(mask_iou(max_frame, actual))
            rows[key]["selected"].append(mask_iou(selected_frame, actual))
            rows[key]["oracle"].append(max(mask_iou(simple_frame, actual), mask_iou(token_frame, actual), mask_iou(max_frame, actual)))
            center_rows[key].append(center_error_px(item["simple_dyn"], item["actual_dyn"], codec.frame_size))
            selector_use[key].append(1.0 if choose_token else 0.0)
    summary = {
        key: {
            metric: float(np.mean(vals))
            for metric, vals in rows[key].items()
        }
        | {
            "center_error_px": float(np.mean(center_rows[key])),
            "token_selected_rate": float(np.mean(selector_use[key])),
        }
        for key in rows
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 11A confidence selector probe.")
    parser.add_argument("--train-sequences", type=int, default=220)
    parser.add_argument("--test-sequences", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4107)
    parser.add_argument("--out", default="results/phase11a_confidence_selector_probe.json")
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
    train_features, train_labels = collect_training(codec, simple_model, token_model, simple_train, token_train)
    selectors, selector_stats = train_selectors(train_features, train_labels, args.seed)
    results = {
        "dataset_shape": raw_shape,
        "train_sequences": args.train_sequences,
        "test_sequences": args.test_sequences,
        "selector_stats": selector_stats,
        "metrics": evaluate(codec, simple_model, token_model, simple_test, token_test, selectors),
        "elapsed_seconds": time.perf_counter() - start,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
