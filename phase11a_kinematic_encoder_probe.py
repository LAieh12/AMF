from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from phase11a_confidence_selector_probe import train_amf
from phase11a_moving_mnist import (
    GT_HORIZONS,
    OBJECTS,
    WARMUP_FRAMES,
    MovingTransition,
    RealMovingMNISTCodec,
    build_transitions,
    causalize_sequences,
    feature_interaction,
    feature_wall,
    kinematic_token_sequences,
    load_real_moving_mnist,
    mask_iou,
    motion_token_sequences,
    object_view,
    transition_event,
)
from phase11a_slot_ranker_probe import compose_layers, render_rollout_layers


def learn_warmup(model, seq):
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


def rollout_pool(codec, models: dict[str, object], sequences: dict[str, object]):
    locals_ = {name: learn_warmup(models[name], sequences[name]) for name in models}
    dyns = {name: sequences[name].dyn[WARMUP_FRAMES].copy() for name in models}
    crops = codec.reference_crops(sequences["simple"], WARMUP_FRAMES)
    out = {}
    for step in range(1, max(GT_HORIZONS) + 1):
        for name in models:
            dyns[name] = locals_[name].predict_next(dyns[name], sequences[name].identity_features)
        if step in GT_HORIZONS:
            layers = {}
            for name in models:
                layers[name] = render_rollout_layers(
                    codec,
                    dyns[name],
                    sequences[name].dyn[WARMUP_FRAMES],
                    sequences[name].frame_layers[WARMUP_FRAMES],
                    step,
                    crops,
                )
            out[str(step)] = {
                "dyns": {name: dyn.copy() for name, dyn in dyns.items()},
                "layers": layers,
                "actual_frame": sequences["simple"].frames[WARMUP_FRAMES + step],
                "actual_dyn": sequences["simple"].dyn[WARMUP_FRAMES + step],
                "start_dyn": sequences["simple"].dyn[WARMUP_FRAMES],
            }
    return out


def candidate_layers_pool(base_layers: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    simple = base_layers["simple"]
    token = base_layers["token"]
    kin = base_layers["kinematic"]
    out = {
        "simple": simple,
        "token": token,
        "kinematic": kin,
        "max_simple_token": np.maximum(simple, token),
        "max_simple_kin": np.maximum(simple, kin),
        "max_token_kin": np.maximum(token, kin),
        "max_all": np.maximum(np.maximum(simple, token), kin),
        "blend_simple_token": np.clip(0.5 * simple + 0.5 * token, 0.0, 1.0),
        "blend_simple_kin": np.clip(0.5 * simple + 0.5 * kin, 0.0, 1.0),
        "blend_token_kin": np.clip(0.5 * token + 0.5 * kin, 0.0, 1.0),
        "blend_all": np.clip((simple + token + kin) / 3.0, 0.0, 1.0),
    }
    out["wide_simple"] = np.clip(0.92 * maximum_filter(simple, size=(1, 3, 3)), 0.0, 1.0).astype(np.float32)
    out["wide_token"] = np.clip(0.92 * maximum_filter(token, size=(1, 3, 3)), 0.0, 1.0).astype(np.float32)
    out["wide_kinematic"] = np.clip(0.92 * maximum_filter(kin, size=(1, 3, 3)), 0.0, 1.0).astype(np.float32)
    out["wide_max_all"] = np.clip(0.92 * maximum_filter(out["max_all"], size=(1, 3, 3)), 0.0, 1.0).astype(np.float32)
    out["very_wide_max_all"] = np.clip(0.78 * maximum_filter(out["max_all"], size=(1, 5, 5)), 0.0, 1.0).astype(np.float32)
    return out


def pool_features(item: dict, horizon: int) -> np.ndarray:
    vals = [horizon / max(GT_HORIZONS)]
    names = ("simple", "token", "kinematic")
    for obj in range(OBJECTS):
        views = [object_view(item["dyns"][name], obj) for name in names]
        for v in views:
            vals.extend(v.tolist())
            vals.append(float(np.linalg.norm(v[2:4]) * 64.0))
        vals.extend(
            [
                float(np.linalg.norm((views[0][:2] - views[1][:2]) * 64.0)),
                float(np.linalg.norm((views[0][:2] - views[2][:2]) * 64.0)),
                float(np.linalg.norm((views[1][:2] - views[2][:2]) * 64.0)),
                float(np.linalg.norm((views[0][2:4] - views[1][2:4]) * 64.0)),
                float(np.linalg.norm((views[0][2:4] - views[2][2:4]) * 64.0)),
                float(np.linalg.norm((views[1][2:4] - views[2][2:4]) * 64.0)),
            ]
        )
    vals.extend(feature_wall(item["dyns"]["simple"]).tolist())
    vals.extend(feature_interaction(item["dyns"]["simple"]).tolist())
    vals.extend(feature_wall(item["dyns"]["kinematic"]).tolist())
    vals.extend(feature_interaction(item["dyns"]["kinematic"]).tolist())
    return np.asarray(vals, dtype=np.float32)


def collect_training(codec, models, seqs_by_name):
    x_by_h = {str(h): [] for h in GT_HORIZONS}
    y_by_h = {str(h): [] for h in GT_HORIZONS}
    candidate_names = None
    for idx in range(len(seqs_by_name["simple"])):
        seqs = {name: seqs_by_name[name][idx] for name in seqs_by_name}
        rolled = rollout_pool(codec, models, seqs)
        for horizon in GT_HORIZONS:
            key = str(horizon)
            item = rolled[key]
            candidates = candidate_layers_pool(item["layers"])
            if candidate_names is None:
                candidate_names = list(candidates.keys())
            ious = np.asarray([mask_iou(compose_layers(candidates[name]), item["actual_frame"]) for name in candidate_names], dtype=np.float32)
            x_by_h[key].append(pool_features(item, horizon))
            y_by_h[key].append(ious)
    return x_by_h, y_by_h, candidate_names or []


def train_rankers(x_by_h, y_by_h, seed: int):
    rf = {}
    extra = {}
    stats = {}
    for horizon in GT_HORIZONS:
        key = str(horizon)
        x = np.vstack(x_by_h[key])
        y = np.vstack(y_by_h[key])
        rf_model = RandomForestRegressor(n_estimators=240, max_depth=8, min_samples_leaf=5, random_state=seed + horizon)
        extra_model = ExtraTreesRegressor(n_estimators=260, max_depth=10, min_samples_leaf=4, random_state=seed + 100 + horizon)
        rf_model.fit(x, y)
        extra_model.fit(x, y)
        rf[key] = rf_model
        extra[key] = extra_model
        stats[key] = {"n": int(len(y)), "oracle_mean": float(np.mean(np.max(y, axis=1)))}
    return rf, extra, stats


def evaluate(codec, models, seqs_by_name, rf, extra, candidate_names):
    rows = {
        str(h): {
            "simple": [],
            "token": [],
            "kinematic": [],
            "max_all": [],
            "wide_max_all": [],
            "rf_ranker": [],
            "extra_ranker": [],
            "ensemble_ranker": [],
            "oracle": [],
        }
        for h in GT_HORIZONS
    }
    for idx in range(len(seqs_by_name["simple"])):
        seqs = {name: seqs_by_name[name][idx] for name in seqs_by_name}
        rolled = rollout_pool(codec, models, seqs)
        for horizon in GT_HORIZONS:
            key = str(horizon)
            item = rolled[key]
            candidates = candidate_layers_pool(item["layers"])
            actual = item["actual_frame"]
            ious = np.asarray([mask_iou(compose_layers(candidates[name]), actual) for name in candidate_names], dtype=np.float32)
            x = pool_features(item, horizon).reshape(1, -1)
            rf_scores = rf[key].predict(x)[0]
            extra_scores = extra[key].predict(x)[0]
            rf_idx = int(np.argmax(rf_scores))
            extra_idx = int(np.argmax(extra_scores))
            ens_idx = int(np.argmax(0.5 * rf_scores + 0.5 * extra_scores))
            for name in ("simple", "token", "kinematic", "max_all", "wide_max_all"):
                rows[key][name].append(mask_iou(compose_layers(candidates[name]), actual))
            rows[key]["rf_ranker"].append(float(ious[rf_idx]))
            rows[key]["extra_ranker"].append(float(ious[extra_idx]))
            rows[key]["ensemble_ranker"].append(float(ious[ens_idx]))
            rows[key]["oracle"].append(float(np.max(ious)))
    return {key: {metric: float(np.mean(vals)) for metric, vals in metrics.items()} for key, metrics in rows.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 11A kinematic encoder pool probe.")
    parser.add_argument("--train-sequences", type=int, default=220)
    parser.add_argument("--test-sequences", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4107)
    parser.add_argument("--out", default="results/phase11a_kinematic_encoder_probe.json")
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
    train_seqs = {
        "simple": causalize_sequences(train),
        "token": motion_token_sequences(train),
        "kinematic": kinematic_token_sequences(train),
    }
    test_seqs = {
        "simple": causalize_sequences(test),
        "token": motion_token_sequences(test),
        "kinematic": kinematic_token_sequences(test),
    }
    models = {name: train_amf(build_transitions(seqs), args.seed) for name, seqs in train_seqs.items()}
    x_by_h, y_by_h, candidate_names = collect_training(codec, models, train_seqs)
    rf, extra, stats = train_rankers(x_by_h, y_by_h, args.seed)
    metrics = evaluate(codec, models, test_seqs, rf, extra, candidate_names)
    results = {
        "dataset_shape": raw_shape,
        "train_sequences": args.train_sequences,
        "test_sequences": args.test_sequences,
        "candidate_names": candidate_names,
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
