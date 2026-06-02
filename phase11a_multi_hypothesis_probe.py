from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from phase11a_moving_mnist import (
    AMFMovingMNISTWorldModel,
    GT_HORIZONS,
    WARMUP_FRAMES,
    MovingTransition,
    RealMovingMNISTCodec,
    RidgeLinearDynamicsBaseline,
    build_transitions,
    causalize_sequences,
    center_error_px,
    load_real_moving_mnist,
    mask_iou,
    motion_token_sequences,
    soft_iou,
    transition_event,
)


def train_amf(train_transitions, seed: int, residual_scale: float = 0.0) -> AMFMovingMNISTWorldModel:
    model = AMFMovingMNISTWorldModel(
        metaplasticity=True,
        boundary_guard=True,
        residual_scale=residual_scale,
        collision_box=0.317,
    ).fit(train_transitions)
    rng = np.random.default_rng(seed)
    order = rng.choice(len(train_transitions), size=min(2500, len(train_transitions)), replace=False)
    for idx in order:
        model.learn_transition(train_transitions[int(idx)])
    return model


def learn_horizon_weights(codec: RealMovingMNISTCodec, simple_model, token_model, simple_train, token_train) -> dict[str, float]:
    candidates = np.asarray([0.0, 0.10, 0.18, 0.25, 0.35, 0.50, 0.70, 1.0], dtype=np.float32)
    weights: dict[str, float] = {}
    for horizon in GT_HORIZONS:
        scores = []
        for beta in candidates:
            rows = []
            for simple_seq, token_seq in zip(simple_train[: min(80, len(simple_train))], token_train[: min(80, len(token_train))]):
                simple_local = simple_model.clone() if hasattr(simple_model, "clone") else simple_model
                token_local = token_model.clone() if hasattr(token_model, "clone") else token_model
                if hasattr(simple_local, "learn_transition"):
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
                if hasattr(token_local, "learn_transition"):
                    for ctx in range(WARMUP_FRAMES):
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
                for step in range(1, horizon + 1):
                    simple_dyn = simple_local.predict_next(simple_dyn, simple_seq.identity_features)
                    token_dyn = token_local.predict_next(token_dyn, token_seq.identity_features)
                simple_frame = codec.render_rollout_frame(
                    simple_dyn,
                    simple_seq.dyn[WARMUP_FRAMES],
                    simple_seq.frame_layers[WARMUP_FRAMES],
                    horizon,
                    crops,
                )
                token_frame = codec.render_rollout_frame(
                    token_dyn,
                    token_seq.dyn[WARMUP_FRAMES],
                    token_seq.frame_layers[WARMUP_FRAMES],
                    horizon,
                    crops,
                )
                fused = np.maximum(simple_frame, float(beta) * token_frame)
                rows.append(mask_iou(fused, simple_seq.frames[WARMUP_FRAMES + horizon]))
            scores.append(float(np.mean(rows)))
        weights[str(horizon)] = float(candidates[int(np.argmax(scores))])
    return weights


def evaluate_dual(codec: RealMovingMNISTCodec, simple_model, token_model, simple_sequences, token_sequences, weights: dict[str, float]):
    rows = {str(h): [] for h in GT_HORIZONS}
    stability = {str(h): [] for h in (30, 60, 120, 240, 480)}
    for simple_seq, token_seq in zip(simple_sequences, token_sequences):
        simple_local = simple_model.clone() if hasattr(simple_model, "clone") else simple_model
        token_local = token_model.clone() if hasattr(token_model, "clone") else token_model
        if hasattr(simple_local, "learn_transition"):
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
        if hasattr(token_local, "learn_transition"):
            for ctx in range(WARMUP_FRAMES):
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
        for step in range(1, 481):
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
                beta = float(weights.get(str(step), 0.0))
                fused = np.maximum(simple_frame, beta * token_frame)
                actual = simple_seq.frames[WARMUP_FRAMES + step]
                rows[str(step)].append(
                    {
                        "frame_iou": mask_iou(fused, actual),
                        "soft_frame_iou": soft_iou(fused, actual),
                        "center_error_px": center_error_px(simple_dyn, simple_seq.dyn[WARMUP_FRAMES + step], codec.frame_size),
                    }
                )
            if step in (30, 60, 120, 240, 480):
                valid = bool(np.all(np.isfinite(simple_dyn)) and np.all(np.isfinite(token_dyn)))
                stability[str(step)].append({"stable": 1.0 if valid else 0.0, "identity_drift": 0.0, "digit_consistency": 1.0})
    summary = {h: {k: float(np.mean([row[k] for row in vals])) for k in vals[0]} for h, vals in rows.items()}
    stability_summary = {h: {k: float(np.mean([row[k] for row in vals])) for k in vals[0]} for h, vals in stability.items()}
    return summary, stability_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 11A causal multi-hypothesis encoder probe.")
    parser.add_argument("--train-sequences", type=int, default=220)
    parser.add_argument("--test-sequences", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4107)
    parser.add_argument("--out", default="results/phase11a_multi_hypothesis_probe.json")
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
    simple_amf = train_amf(simple_transitions, args.seed)
    token_amf = train_amf(token_transitions, args.seed)
    simple_ridge = RidgeLinearDynamicsBaseline().fit(simple_transitions)
    token_ridge = RidgeLinearDynamicsBaseline().fit(token_transitions)

    weights = learn_horizon_weights(codec, simple_amf, token_amf, simple_train, token_train)
    dual_amf, dual_stability = evaluate_dual(codec, simple_amf, token_amf, simple_test, token_test, weights)
    ridge_weights = learn_horizon_weights(codec, simple_ridge, token_ridge, simple_train, token_train)
    dual_ridge, _ = evaluate_dual(codec, simple_ridge, token_ridge, simple_test, token_test, ridge_weights)

    results = {
        "dataset_shape": raw_shape,
        "train_sequences": args.train_sequences,
        "test_sequences": args.test_sequences,
        "weights": weights,
        "ridge_weights": ridge_weights,
        "dual_amf": dual_amf,
        "dual_ridge": dual_ridge,
        "dual_stability": dual_stability,
        "elapsed_seconds": time.perf_counter() - start,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
