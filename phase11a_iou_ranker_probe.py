from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from phase11a_confidence_selector_probe import rollout_pair, selector_features, train_amf
from phase11a_moving_mnist import (
    GT_HORIZONS,
    RealMovingMNISTCodec,
    build_transitions,
    causalize_sequences,
    load_real_moving_mnist,
    mask_iou,
    motion_token_sequences,
    soft_iou,
)


BETAS = (0.0, 0.25, 0.50, 0.75, 1.0)


def candidate_frames(item: dict) -> dict[str, np.ndarray]:
    simple = item["simple_frame"]
    token = item["token_frame"]
    frames = {
        "simple": simple,
        "token": token,
    }
    for beta in BETAS:
        frames[f"max_beta_{beta:.2f}"] = np.maximum(simple, float(beta) * token)
        frames[f"blend_beta_{beta:.2f}"] = np.clip((1.0 - float(beta)) * simple + float(beta) * token, 0.0, 1.0)
    return frames


def collect_examples(codec, simple_model, token_model, simple_sequences, token_sequences):
    x_by_h = {str(h): [] for h in GT_HORIZONS}
    y_class_by_h = {str(h): [] for h in GT_HORIZONS}
    y_reg_by_h = {str(h): [] for h in GT_HORIZONS}
    names = None
    for simple_seq, token_seq in zip(simple_sequences, token_sequences):
        rolled = rollout_pair(codec, simple_model, token_model, simple_seq, token_seq)
        for horizon in GT_HORIZONS:
            key = str(horizon)
            item = rolled[key]
            frames = candidate_frames(item)
            if names is None:
                names = list(frames.keys())
            ious = np.asarray([mask_iou(frames[name], item["actual_frame"]) for name in names], dtype=np.float32)
            x_by_h[key].append(selector_features(item, horizon))
            y_class_by_h[key].append(int(np.argmax(ious)))
            y_reg_by_h[key].append(ious)
    return x_by_h, y_class_by_h, y_reg_by_h, names or []


def train_rankers(x_by_h, y_class_by_h, y_reg_by_h, seed: int):
    classifiers = {}
    regressors = {}
    stats = {}
    for horizon in GT_HORIZONS:
        key = str(horizon)
        x = np.vstack(x_by_h[key])
        yc = np.asarray(y_class_by_h[key], dtype=np.int32)
        yr = np.vstack(y_reg_by_h[key])
        stats[key] = {
            "n": int(len(yc)),
            "class_hist": {str(int(c)): int(np.sum(yc == c)) for c in np.unique(yc)},
            "oracle_mean": float(np.mean(np.max(yr, axis=1))),
        }
        if len(np.unique(yc)) < 2:
            classifiers[key] = None
        else:
            clf = RandomForestClassifier(
                n_estimators=220,
                max_depth=7,
                min_samples_leaf=5,
                random_state=seed + horizon,
                class_weight="balanced_subsample",
            )
            clf.fit(x, yc)
            classifiers[key] = clf
        reg = RandomForestRegressor(
            n_estimators=220,
            max_depth=7,
            min_samples_leaf=5,
            random_state=seed + 100 + horizon,
        )
        reg.fit(x, yr)
        regressors[key] = reg
    return classifiers, regressors, stats


def evaluate(codec, simple_model, token_model, simple_sequences, token_sequences, classifiers, regressors, names):
    rows = {
        str(h): {
            "simple": [],
            "token": [],
            "max_beta_1.00": [],
            "class_ranker": [],
            "reg_ranker": [],
            "oracle": [],
            "class_choice": [],
            "reg_choice": [],
        }
        for h in GT_HORIZONS
    }
    for simple_seq, token_seq in zip(simple_sequences, token_sequences):
        rolled = rollout_pair(codec, simple_model, token_model, simple_seq, token_seq)
        for horizon in GT_HORIZONS:
            key = str(horizon)
            item = rolled[key]
            frames = candidate_frames(item)
            ious = np.asarray([mask_iou(frames[name], item["actual_frame"]) for name in names], dtype=np.float32)
            x = selector_features(item, horizon).reshape(1, -1)
            if classifiers[key] is None:
                class_idx = 0
            else:
                class_idx = int(classifiers[key].predict(x)[0])
            reg_idx = int(np.argmax(regressors[key].predict(x)[0]))
            rows[key]["simple"].append(float(ious[names.index("simple")]))
            rows[key]["token"].append(float(ious[names.index("token")]))
            rows[key]["max_beta_1.00"].append(float(ious[names.index("max_beta_1.00")]))
            rows[key]["class_ranker"].append(float(ious[class_idx]))
            rows[key]["reg_ranker"].append(float(ious[reg_idx]))
            rows[key]["oracle"].append(float(np.max(ious)))
            rows[key]["class_choice"].append(float(class_idx))
            rows[key]["reg_choice"].append(float(reg_idx))
    summary = {}
    for key, metrics in rows.items():
        summary[key] = {}
        for metric, vals in metrics.items():
            if metric.endswith("_choice"):
                rounded = np.rint(vals).astype(np.int32)
                summary[key][metric] = {names[int(c)]: int(np.sum(rounded == c)) for c in np.unique(rounded)}
            else:
                summary[key][metric] = float(np.mean(vals))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 11A IoU ranker probe for causal encoder hypotheses.")
    parser.add_argument("--train-sequences", type=int, default=220)
    parser.add_argument("--test-sequences", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4107)
    parser.add_argument("--out", default="results/phase11a_iou_ranker_probe.json")
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
    x_by_h, y_class_by_h, y_reg_by_h, names = collect_examples(codec, simple_model, token_model, simple_train, token_train)
    classifiers, regressors, stats = train_rankers(x_by_h, y_class_by_h, y_reg_by_h, args.seed)
    metrics = evaluate(codec, simple_model, token_model, simple_test, token_test, classifiers, regressors, names)
    results = {
        "dataset_shape": raw_shape,
        "train_sequences": args.train_sequences,
        "test_sequences": args.test_sequences,
        "candidate_names": names,
        "ranker_stats": stats,
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - start,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
