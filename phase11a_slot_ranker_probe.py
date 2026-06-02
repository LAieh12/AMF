from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter, zoom
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor

from phase11a_confidence_selector_probe import train_amf
from phase11a_moving_mnist import (
    AMFMovingMNISTWorldModel,
    EPS,
    GT_HORIZONS,
    OBJECTS,
    WARMUP_FRAMES,
    MovingTransition,
    RealMovingMNISTCodec,
    build_transitions,
    causalize_sequences,
    feature_interaction,
    feature_wall,
    load_real_moving_mnist,
    mask_iou,
    motion_token_sequences,
    object_view,
    transition_event,
)


BETAS = (0.0, 0.25, 0.50, 0.75, 1.0)
PAIRWISE_TOP_K = 16
PAIRWISE_RANDOM_K = 40


def train_amf_scaled(transitions, seed: int, scale: str):
    if scale == "default":
        return train_amf(transitions, seed)
    if scale == "wide":
        model = AMFMovingMNISTWorldModel(
            metaplasticity=True,
            boundary_guard=True,
            residual_scale=0.0,
            collision_box=0.317,
            top_k=24,
            max_cells=24000,
            explain_error_threshold=0.000035,
            medium_error_threshold=0.00016,
            novelty_confirmations=2,
        ).fit(transitions)
        rng = np.random.default_rng(seed)
        order = rng.choice(len(transitions), size=min(5000, len(transitions)), replace=False)
        for idx in order:
            model.learn_transition(transitions[int(idx)])
        return model
    if scale == "xwide":
        model = AMFMovingMNISTWorldModel(
            metaplasticity=True,
            boundary_guard=True,
            residual_scale=0.0,
            collision_box=0.317,
            top_k=32,
            max_cells=48000,
            explain_error_threshold=0.000025,
            medium_error_threshold=0.00012,
            novelty_confirmations=2,
            cell_size=0.026,
            activation_radius=0.026,
        ).fit(transitions)
        rng = np.random.default_rng(seed)
        order = rng.choice(len(transitions), size=min(8000, len(transitions)), replace=False)
        for idx in order:
            model.learn_transition(transitions[int(idx)])
        return model
    raise ValueError(f"Unknown AMF scale: {scale}")


def render_crop_layer(codec: RealMovingMNISTCodec, dyn: np.ndarray, obj: int, crop: np.ndarray) -> np.ndarray:
    v = object_view(dyn, obj)
    crop = np.asarray(crop, dtype=np.float32)
    h = max(3, int(round(float(v[5]) * codec.frame_size)))
    w = max(3, int(round(float(v[4]) * codec.frame_size)))
    resized = zoom(crop, (h / crop.shape[0], w / crop.shape[1]), order=1)
    resized = resized / (float(np.max(resized)) + EPS)
    cx = float(v[0]) * (codec.frame_size - 1)
    cy = float(v[1]) * (codec.frame_size - 1)
    x0 = int(round(cx - w / 2.0))
    y0 = int(round(cy - h / 2.0))
    x1 = x0 + w
    y1 = y0 + h
    sx0 = max(0, -x0)
    sy0 = max(0, -y0)
    sx1 = w - max(0, x1 - codec.frame_size)
    sy1 = h - max(0, y1 - codec.frame_size)
    dx0 = max(0, x0)
    dy0 = max(0, y0)
    dx1 = dx0 + max(0, sx1 - sx0)
    dy1 = dy0 + max(0, sy1 - sy0)
    layer = np.zeros((codec.frame_size, codec.frame_size), dtype=np.float32)
    if dx1 > dx0 and dy1 > dy0:
        layer[dy0:dy1, dx0:dx1] = np.maximum(layer[dy0:dy1, dx0:dx1], resized[sy0:sy1, sx0:sx1])
    return layer.astype(np.float32)


def render_rollout_layers(
    codec: RealMovingMNISTCodec,
    dyn: np.ndarray,
    source_dyn: np.ndarray,
    layers: np.ndarray,
    horizon: int,
    reference_crops: np.ndarray | None,
) -> np.ndarray:
    rough = codec.warp_layers_to(layers, source_dyn, dyn)
    if reference_crops is None or len(codec.identity_bank_crops) == 0 or horizon <= 0:
        return rough
    alpha = min(0.65, max(0.0, float(horizon)) * 0.65 / max(1.0, float(max(GT_HORIZONS))))
    bank = np.asarray([render_crop_layer(codec, dyn, obj, reference_crops[obj]) for obj in range(OBJECTS)], dtype=np.float32)
    return np.maximum(rough, alpha * bank).astype(np.float32)


def compose_layers(layers: np.ndarray) -> np.ndarray:
    return np.max(np.asarray(layers, dtype=np.float32), axis=0).astype(np.float32)


def normalized_scores(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float32)
    span = float(np.max(arr) - np.min(arr))
    if span <= EPS:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - float(np.min(arr))) / (span + EPS)).astype(np.float32)


def layer_centroid(layer: np.ndarray) -> np.ndarray:
    arr = np.asarray(layer, dtype=np.float32)
    yy, xx = np.nonzero(arr > 0.18)
    if len(xx) == 0:
        return np.asarray([arr.shape[1] * 0.5, arr.shape[0] * 0.5], dtype=np.float32)
    weights = arr[yy, xx]
    return np.asarray(
        [
            float(np.sum(xx * weights) / (np.sum(weights) + EPS)),
            float(np.sum(yy * weights) / (np.sum(weights) + EPS)),
        ],
        dtype=np.float32,
    )


def shift_layer(layer: np.ndarray, dx: int, dy: int) -> np.ndarray:
    arr = np.asarray(layer, dtype=np.float32)
    out = np.zeros_like(arr)
    h, w = arr.shape
    sx0 = max(0, -int(dx))
    sy0 = max(0, -int(dy))
    sx1 = w - max(0, int(dx))
    sy1 = h - max(0, int(dy))
    dx0 = max(0, int(dx))
    dy0 = max(0, int(dy))
    dx1 = dx0 + max(0, sx1 - sx0)
    dy1 = dy0 + max(0, sy1 - sy0)
    if dx1 > dx0 and dy1 > dy0:
        out[dy0:dy1, dx0:dx1] = arr[sy0:sy1, sx0:sx1]
    return out.astype(np.float32)


def shift_layers_toward(source: np.ndarray, target: np.ndarray, scale: float) -> np.ndarray:
    shifted = np.zeros_like(source)
    for obj in range(OBJECTS):
        delta = layer_centroid(target[obj]) - layer_centroid(source[obj])
        delta = np.clip(delta, -5.0, 5.0)
        dx = int(round(float(delta[0]) * float(scale)))
        dy = int(round(float(delta[1]) * float(scale)))
        shifted[obj] = shift_layer(source[obj], dx, dy)
    return shifted.astype(np.float32)


def candidate_layers(simple_layers: np.ndarray, token_layers: np.ndarray) -> dict[str, np.ndarray]:
    candidates = {
        "simple": simple_layers,
        "token": token_layers,
    }
    for beta in BETAS:
        candidates[f"max_beta_{beta:.2f}"] = np.maximum(simple_layers, float(beta) * token_layers)
        candidates[f"blend_beta_{beta:.2f}"] = np.clip(
            (1.0 - float(beta)) * simple_layers + float(beta) * token_layers,
            0.0,
            1.0,
        )
    wide_simple = maximum_filter(candidates["simple"], size=(1, 3, 3))
    wide_token = maximum_filter(candidates["token"], size=(1, 3, 3))
    wide_max = maximum_filter(candidates["max_beta_1.00"], size=(1, 3, 3))
    wide_blend = maximum_filter(candidates["blend_beta_0.75"], size=(1, 3, 3))
    very_wide_max = maximum_filter(candidates["max_beta_1.00"], size=(1, 5, 5))
    very_wide_blend = maximum_filter(candidates["blend_beta_0.75"], size=(1, 5, 5))
    candidates["wide_simple"] = np.clip(0.92 * wide_simple, 0.0, 1.0).astype(np.float32)
    candidates["wide_token"] = np.clip(0.92 * wide_token, 0.0, 1.0).astype(np.float32)
    candidates["wide_max_beta_1.00"] = np.clip(0.92 * wide_max, 0.0, 1.0).astype(np.float32)
    candidates["wide_blend_beta_0.75"] = np.clip(0.92 * wide_blend, 0.0, 1.0).astype(np.float32)
    candidates["very_wide_max_beta_1.00"] = np.clip(0.78 * very_wide_max, 0.0, 1.0).astype(np.float32)
    candidates["very_wide_blend_beta_0.75"] = np.clip(0.78 * very_wide_blend, 0.0, 1.0).astype(np.float32)
    shift_simple_mid = shift_layers_toward(simple_layers, token_layers, 0.5)
    shift_token_mid = shift_layers_toward(token_layers, simple_layers, 0.5)
    shift_simple_anti = shift_layers_toward(simple_layers, token_layers, -0.5)
    shift_token_anti = shift_layers_toward(token_layers, simple_layers, -0.5)
    candidates["shift_simple_mid"] = shift_simple_mid
    candidates["shift_token_mid"] = shift_token_mid
    candidates["shift_simple_anti"] = shift_simple_anti
    candidates["shift_token_anti"] = shift_token_anti
    candidates["max_shift_mid"] = np.maximum(shift_simple_mid, shift_token_mid).astype(np.float32)
    return candidates


def rollout_pair_slots(codec: RealMovingMNISTCodec, simple_model, token_model, simple_seq, token_seq):
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
            simple_layers = render_rollout_layers(
                codec,
                simple_dyn,
                simple_seq.dyn[WARMUP_FRAMES],
                simple_seq.frame_layers[WARMUP_FRAMES],
                step,
                crops,
            )
            token_layers = render_rollout_layers(
                codec,
                token_dyn,
                token_seq.dyn[WARMUP_FRAMES],
                token_seq.frame_layers[WARMUP_FRAMES],
                step,
                crops,
            )
            out[str(step)] = {
                "simple_dyn": simple_dyn.copy(),
                "token_dyn": token_dyn.copy(),
                "simple_layers": simple_layers,
                "token_layers": token_layers,
                "actual_layers": simple_seq.frame_layers[WARMUP_FRAMES + step],
                "actual_frame": simple_seq.frames[WARMUP_FRAMES + step],
                "actual_dyn": simple_seq.dyn[WARMUP_FRAMES + step],
                "start_dyn": simple_seq.dyn[WARMUP_FRAMES],
            }
    return out


def slot_features(item: dict, horizon: int, obj: int) -> np.ndarray:
    simple_dyn = item["simple_dyn"]
    token_dyn = item["token_dyn"]
    start_dyn = item["start_dyn"]
    other = 1 - int(obj)
    sv = object_view(simple_dyn, obj)
    tv = object_view(token_dyn, obj)
    so = object_view(simple_dyn, other)
    to = object_view(token_dyn, other)
    st = object_view(start_dyn, obj)
    sto = object_view(start_dyn, other)
    simple_layer = item["simple_layers"][obj]
    token_layer = item["token_layers"][obj]
    max_layer = np.maximum(simple_layer, token_layer)
    blend_layer = 0.5 * simple_layer + 0.5 * token_layer
    center_gap = float(np.linalg.norm((sv[:2] - tv[:2]) * 64.0))
    speed_gap = float(np.linalg.norm((sv[2:4] - tv[2:4]) * 64.0))
    simple_other_gap = float(np.linalg.norm((sv[:2] - so[:2]) * 64.0))
    token_other_gap = float(np.linalg.norm((tv[:2] - to[:2]) * 64.0))
    start_other_gap = float(np.linalg.norm((st[:2] - sto[:2]) * 64.0))
    layer_stats = np.asarray(
        [
            float(np.mean(simple_layer > 0.18)),
            float(np.mean(token_layer > 0.18)),
            float(np.mean(max_layer > 0.18)),
            float(np.mean(blend_layer > 0.18)),
            float(np.mean(np.abs(simple_layer - token_layer))),
            float(np.mean(np.minimum(simple_layer, token_layer) > 0.18)),
        ],
        dtype=np.float32,
    )
    local = np.asarray(
        [
            horizon / max(GT_HORIZONS),
            float(obj),
            center_gap,
            speed_gap,
            float(np.linalg.norm(sv[2:4]) * 64.0),
            float(np.linalg.norm(tv[2:4]) * 64.0),
            simple_other_gap,
            token_other_gap,
            start_other_gap,
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            local,
            sv.astype(np.float32),
            tv.astype(np.float32),
            st.astype(np.float32),
            layer_stats,
            feature_wall(simple_dyn),
            feature_interaction(simple_dyn),
            feature_wall(token_dyn),
            feature_interaction(token_dyn),
            feature_wall(start_dyn),
            feature_interaction(start_dyn),
        ]
    ).astype(np.float32)


def pair_context_features(item: dict, horizon: int) -> np.ndarray:
    simple_frame = compose_layers(item["simple_layers"])
    token_frame = compose_layers(item["token_layers"])
    max_frame = np.maximum(simple_frame, token_frame)
    blend_frame = 0.5 * simple_frame + 0.5 * token_frame
    frame_stats = np.asarray(
        [
            horizon / max(GT_HORIZONS),
            float(np.mean(simple_frame > 0.18)),
            float(np.mean(token_frame > 0.18)),
            float(np.mean(max_frame > 0.18)),
            float(np.mean(blend_frame > 0.18)),
            float(np.mean(np.abs(simple_frame - token_frame))),
            float(np.mean(np.minimum(simple_frame, token_frame) > 0.18)),
        ],
        dtype=np.float32,
    )
    object_stats = []
    for obj in range(OBJECTS):
        sv = object_view(item["simple_dyn"], obj)
        tv = object_view(item["token_dyn"], obj)
        st = object_view(item["start_dyn"], obj)
        object_stats.extend(sv.tolist())
        object_stats.extend(tv.tolist())
        object_stats.extend(st.tolist())
        object_stats.extend(
            [
                float(np.linalg.norm((sv[:2] - tv[:2]) * 64.0)),
                float(np.linalg.norm((sv[2:4] - tv[2:4]) * 64.0)),
                float(np.mean(item["simple_layers"][obj] > 0.18)),
                float(np.mean(item["token_layers"][obj] > 0.18)),
                float(np.mean(np.abs(item["simple_layers"][obj] - item["token_layers"][obj]))),
            ]
        )
    return np.concatenate(
        [
            frame_stats,
            np.asarray(object_stats, dtype=np.float32),
            feature_wall(item["simple_dyn"]),
            feature_interaction(item["simple_dyn"]),
            feature_wall(item["token_dyn"]),
            feature_interaction(item["token_dyn"]),
            feature_wall(item["start_dyn"]),
            feature_interaction(item["start_dyn"]),
        ]
    ).astype(np.float32)


def layer_summary(layer: np.ndarray) -> np.ndarray:
    arr = np.asarray(layer, dtype=np.float32)
    centroid = layer_centroid(arr)
    yy, xx = np.nonzero(arr > 0.18)
    if len(xx) == 0:
        spread_x = 0.0
        spread_y = 0.0
    else:
        weights = arr[yy, xx]
        total = float(np.sum(weights) + EPS)
        cx = float(np.sum(xx * weights) / total)
        cy = float(np.sum(yy * weights) / total)
        spread_x = float(np.sqrt(np.sum(((xx - cx) ** 2) * weights) / total) / max(1, arr.shape[1]))
        spread_y = float(np.sqrt(np.sum(((yy - cy) ** 2) * weights) / total) / max(1, arr.shape[0]))
    return np.asarray(
        [
            float(np.mean(arr > 0.18)),
            float(np.mean(arr)),
            float(np.max(arr)),
            float(centroid[0] / max(1, arr.shape[1])),
            float(centroid[1] / max(1, arr.shape[0])),
            spread_x,
            spread_y,
        ],
        dtype=np.float32,
    )


def candidate_name_features(name: str) -> np.ndarray:
    return np.asarray(
        [
            float(name == "simple"),
            float(name == "token"),
            float(name.startswith("max")),
            float(name.startswith("blend")),
            float("wide" in name),
            float("very_wide" in name),
            float("shift" in name),
            float("anti" in name),
            float("mid" in name),
        ],
        dtype=np.float32,
    )


def pair_candidate_features(
    item: dict,
    horizon: int,
    candidates: dict[str, np.ndarray],
    left_name: str,
    right_name: str,
    context: np.ndarray | None = None,
) -> np.ndarray:
    left = candidates[left_name][0]
    right = candidates[right_name][1]
    pair_frame = np.maximum(left, right).astype(np.float32)
    simple_frame = compose_layers(item["simple_layers"])
    token_frame = compose_layers(item["token_layers"])
    max_frame = np.maximum(simple_frame, token_frame)
    left_centroid = layer_centroid(left)
    right_centroid = layer_centroid(right)
    pair_stats = np.asarray(
        [
            float(np.mean(pair_frame > 0.18)),
            float(np.mean(pair_frame)),
            float(np.max(pair_frame)),
            float(np.mean(np.minimum(left, right) > 0.18)),
            float(np.linalg.norm(left_centroid - right_centroid) / max(1, pair_frame.shape[0])),
            float(abs(np.mean(left > 0.18) - np.mean(right > 0.18))),
            mask_iou(pair_frame, simple_frame),
            mask_iou(pair_frame, token_frame),
            mask_iou(pair_frame, max_frame),
            float(np.mean(np.abs(pair_frame - simple_frame))),
            float(np.mean(np.abs(pair_frame - token_frame))),
            float(np.mean(np.abs(pair_frame - max_frame))),
            mask_iou(left, item["simple_layers"][0]),
            mask_iou(left, item["token_layers"][0]),
            mask_iou(right, item["simple_layers"][1]),
            mask_iou(right, item["token_layers"][1]),
        ],
        dtype=np.float32,
    )
    if context is None:
        context = pair_context_features(item, horizon)
    return np.concatenate(
        [
            context,
            candidate_name_features(left_name),
            candidate_name_features(right_name),
            layer_summary(left),
            layer_summary(right),
            layer_summary(pair_frame),
            pair_stats,
        ]
    ).astype(np.float32)


def compose_pair(candidates: dict[str, np.ndarray], left_name: str, right_name: str) -> np.ndarray:
    pair_layers = np.zeros_like(next(iter(candidates.values())))
    pair_layers[0] = candidates[left_name][0]
    pair_layers[1] = candidates[right_name][1]
    return compose_layers(pair_layers)


def pair_names(names: list[str]) -> list[tuple[str, str]]:
    return [(left_name, right_name) for left_name in names for right_name in names]


def pair_frame_ious(candidates: dict[str, np.ndarray], names: list[str], actual: np.ndarray) -> np.ndarray:
    return np.asarray(
        [mask_iou(compose_pair(candidates, left_name, right_name), actual) for left_name, right_name in pair_names(names)],
        dtype=np.float32,
    )


def collect_examples(codec, simple_model, token_model, simple_sequences, token_sequences, enable_pairwise: bool):
    x_by_h = {str(h): [] for h in GT_HORIZONS}
    y_class_by_h = {str(h): [] for h in GT_HORIZONS}
    y_reg_by_h = {str(h): [] for h in GT_HORIZONS}
    frame_x_by_h = {str(h): [] for h in GT_HORIZONS}
    frame_y_class_by_h = {str(h): [] for h in GT_HORIZONS}
    frame_y_reg_by_h = {str(h): [] for h in GT_HORIZONS}
    pair_x_by_h = {str(h): [] for h in GT_HORIZONS}
    pair_y_reg_by_h = {str(h): [] for h in GT_HORIZONS}
    pairwise_x_by_h = {str(h): [] for h in GT_HORIZONS}
    pairwise_y_by_h = {str(h): [] for h in GT_HORIZONS}
    names = None
    for seq_index, (simple_seq, token_seq) in enumerate(zip(simple_sequences, token_sequences)):
        rolled = rollout_pair_slots(codec, simple_model, token_model, simple_seq, token_seq)
        for horizon in GT_HORIZONS:
            key = str(horizon)
            item = rolled[key]
            candidates = candidate_layers(item["simple_layers"], item["token_layers"])
            if names is None:
                names = list(candidates.keys())
            pair_ious = pair_frame_ious(candidates, names, item["actual_frame"])
            pairs = pair_names(names)
            context = pair_context_features(item, horizon)
            pair_matrix = pair_ious.reshape(len(names), len(names))
            best_left, best_right = np.unravel_index(int(np.argmax(pair_matrix)), pair_matrix.shape)
            for obj in range(OBJECTS):
                ious = np.asarray(
                    [mask_iou(candidates[name][obj], item["actual_layers"][obj]) for name in names],
                    dtype=np.float32,
                )
                x_by_h[key].append(slot_features(item, horizon, obj))
                y_class_by_h[key].append(int(np.argmax(ious)))
                y_reg_by_h[key].append(ious)
                frame_x_by_h[key].append(slot_features(item, horizon, obj))
                if obj == 0:
                    frame_y_class_by_h[key].append(int(best_left))
                    frame_y_reg_by_h[key].append(np.max(pair_matrix, axis=1))
                else:
                    frame_y_class_by_h[key].append(int(best_right))
                    frame_y_reg_by_h[key].append(np.max(pair_matrix, axis=0))
            pair_x_by_h[key].append(context)
            pair_y_reg_by_h[key].append(pair_ious)
            if enable_pairwise:
                rng = np.random.default_rng(int(seq_index * 1009 + horizon * 9176))
                top_count = min(PAIRWISE_TOP_K, len(pair_ious))
                top_indices = np.argpartition(-pair_ious, top_count - 1)[:top_count]
                random_count = min(PAIRWISE_RANDOM_K, len(pair_ious))
                random_indices = rng.choice(len(pair_ious), size=random_count, replace=False)
                selected_indices = np.unique(np.concatenate([top_indices, random_indices]))
                for pair_idx in selected_indices:
                    left_name, right_name = pairs[int(pair_idx)]
                    pairwise_x_by_h[key].append(
                        pair_candidate_features(item, horizon, candidates, left_name, right_name, context)
                    )
                    pairwise_y_by_h[key].append(float(pair_ious[int(pair_idx)]))
    return (
        x_by_h,
        y_class_by_h,
        y_reg_by_h,
        frame_x_by_h,
        frame_y_class_by_h,
        frame_y_reg_by_h,
        pair_x_by_h,
        pair_y_reg_by_h,
        pairwise_x_by_h,
        pairwise_y_by_h,
        names or [],
    )


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
            "slot_oracle_mean": float(np.mean(np.max(yr, axis=1))),
        }
        if len(np.unique(yc)) < 2:
            classifiers[key] = None
        else:
            clf = RandomForestClassifier(
                n_estimators=220,
                max_depth=8,
                min_samples_leaf=5,
                random_state=seed + horizon,
                class_weight="balanced_subsample",
            )
            clf.fit(x, yc)
            classifiers[key] = clf
        reg = RandomForestRegressor(
            n_estimators=220,
            max_depth=8,
            min_samples_leaf=5,
            random_state=seed + 100 + horizon,
        )
        reg.fit(x, yr)
        regressors[key] = reg
    return classifiers, regressors, stats


def train_frame_slot_rankers(frame_x_by_h, frame_y_class_by_h, frame_y_reg_by_h, seed: int):
    classifiers = {}
    regressors = {}
    extra_regressors = {}
    stats = {}
    for horizon in GT_HORIZONS:
        key = str(horizon)
        x = np.vstack(frame_x_by_h[key])
        yc = np.asarray(frame_y_class_by_h[key], dtype=np.int32)
        yr = np.vstack(frame_y_reg_by_h[key])
        stats[key] = {
            "n": int(len(yc)),
            "class_hist": {str(int(c)): int(np.sum(yc == c)) for c in np.unique(yc)},
            "frame_marginal_oracle_mean": float(np.mean(np.max(yr, axis=1))),
        }
        if len(np.unique(yc)) < 2:
            classifiers[key] = None
        else:
            clf = RandomForestClassifier(
                n_estimators=260,
                max_depth=8,
                min_samples_leaf=5,
                random_state=seed + 300 + horizon,
                class_weight="balanced_subsample",
            )
            clf.fit(x, yc)
            classifiers[key] = clf
        reg = RandomForestRegressor(
            n_estimators=260,
            max_depth=8,
            min_samples_leaf=5,
            random_state=seed + 400 + horizon,
        )
        reg.fit(x, yr)
        regressors[key] = reg
        extra = ExtraTreesRegressor(
            n_estimators=260,
            max_depth=10,
            min_samples_leaf=4,
            random_state=seed + 500 + horizon,
        )
        extra.fit(x, yr)
        extra_regressors[key] = extra
    return classifiers, regressors, extra_regressors, stats


def train_pair_rankers(pair_x_by_h, pair_y_reg_by_h, pairwise_x_by_h, pairwise_y_by_h, seed: int, enable_pairwise: bool):
    pair_regressors = {}
    pairwise_regressors = {}
    pair_stats = {}
    for horizon in GT_HORIZONS:
        key = str(horizon)
        x = np.vstack(pair_x_by_h[key])
        yr = np.vstack(pair_y_reg_by_h[key])
        pair_stats[key] = {
            "n": int(len(yr)),
            "pair_oracle_mean": float(np.mean(np.max(yr, axis=1))),
        }
        reg = RandomForestRegressor(
            n_estimators=220,
            max_depth=8,
            min_samples_leaf=5,
            random_state=seed + 200 + horizon,
        )
        reg.fit(x, yr)
        pair_regressors[key] = reg
        if enable_pairwise:
            px = np.vstack(pairwise_x_by_h[key])
            py = np.asarray(pairwise_y_by_h[key], dtype=np.float32)
            pair_stats[key]["pairwise_n"] = int(len(py))
            pair_stats[key]["pairwise_target_mean"] = float(np.mean(py))
            pairwise = ExtraTreesRegressor(
                n_estimators=260,
                max_depth=10,
                min_samples_leaf=4,
                random_state=seed + 600 + horizon,
            )
            pairwise.fit(px, py)
            pairwise_regressors[key] = pairwise
        else:
            pairwise_regressors[key] = None
    return pair_regressors, pairwise_regressors, pair_stats


def evaluate(
    codec,
    simple_model,
    token_model,
    simple_sequences,
    token_sequences,
    classifiers,
    regressors,
    frame_classifiers,
    frame_regressors,
    frame_extra_regressors,
    pair_regressors,
    pairwise_regressors,
    names,
):
    pairs = pair_names(names)
    include_pairwise = bool(pairwise_regressors) and any(model is not None for model in pairwise_regressors.values())
    rows = {
        str(h): {
            "simple": [],
            "token": [],
            "max_beta_1.00": [],
            "slot_class_ranker": [],
            "slot_reg_ranker": [],
            "slot_frame_class_ranker": [],
            "slot_frame_reg_ranker": [],
            "slot_frame_extra_ranker": [],
            "slot_frame_ensemble_ranker": [],
            "slot_frame_ensemble_w25_ranker": [],
            "slot_frame_ensemble_w75_ranker": [],
            "slot_hybrid_ranker": [],
            "slot_hybrid_extra_ranker": [],
            "slot_pair_reg_ranker": [],
            "slot_object_oracle": [],
            "slot_frame_oracle": [],
            "frame_oracle": [],
            "slot_class_choice": [],
            "slot_reg_choice": [],
            "slot_frame_class_choice": [],
            "slot_frame_reg_choice": [],
            "slot_frame_extra_choice": [],
            "slot_frame_ensemble_choice": [],
            "slot_frame_ensemble_w25_choice": [],
            "slot_frame_ensemble_w75_choice": [],
            "slot_hybrid_choice": [],
            "slot_hybrid_extra_choice": [],
        }
        for h in GT_HORIZONS
    }
    if include_pairwise:
        for metrics in rows.values():
            metrics["slot_pairwise_ranker"] = []
            metrics["slot_pairwise_choice"] = []
    for simple_seq, token_seq in zip(simple_sequences, token_sequences):
        rolled = rollout_pair_slots(codec, simple_model, token_model, simple_seq, token_seq)
        for horizon in GT_HORIZONS:
            key = str(horizon)
            item = rolled[key]
            candidates = candidate_layers(item["simple_layers"], item["token_layers"])
            actual = item["actual_frame"]
            frame_ious = np.asarray([mask_iou(compose_layers(candidates[name]), actual) for name in names], dtype=np.float32)
            class_layers = np.zeros_like(item["simple_layers"])
            reg_layers = np.zeros_like(item["simple_layers"])
            frame_class_layers = np.zeros_like(item["simple_layers"])
            frame_reg_layers = np.zeros_like(item["simple_layers"])
            frame_extra_layers = np.zeros_like(item["simple_layers"])
            frame_ensemble_layers = np.zeros_like(item["simple_layers"])
            frame_ensemble_w25_layers = np.zeros_like(item["simple_layers"])
            frame_ensemble_w75_layers = np.zeros_like(item["simple_layers"])
            hybrid_layers = np.zeros_like(item["simple_layers"])
            hybrid_extra_layers = np.zeros_like(item["simple_layers"])
            object_oracle_layers = np.zeros_like(item["simple_layers"])
            class_indices = []
            reg_indices = []
            frame_class_indices = []
            frame_reg_indices = []
            frame_extra_indices = []
            frame_ensemble_indices = []
            frame_ensemble_w25_indices = []
            frame_ensemble_w75_indices = []
            hybrid_indices = []
            hybrid_extra_indices = []
            for obj in range(OBJECTS):
                x = slot_features(item, horizon, obj).reshape(1, -1)
                if classifiers[key] is None:
                    class_idx = 0
                else:
                    class_idx = int(classifiers[key].predict(x)[0])
                reg_scores = regressors[key].predict(x)[0]
                reg_idx = int(np.argmax(reg_scores))
                if frame_classifiers[key] is None:
                    frame_class_idx = 0
                else:
                    frame_class_idx = int(frame_classifiers[key].predict(x)[0])
                frame_reg_scores = frame_regressors[key].predict(x)[0]
                frame_extra_scores = frame_extra_regressors[key].predict(x)[0]
                frame_reg_idx = int(np.argmax(frame_reg_scores))
                frame_extra_idx = int(np.argmax(frame_extra_scores))
                frame_ensemble_idx = int(np.argmax(0.5 * frame_reg_scores + 0.5 * frame_extra_scores))
                frame_ensemble_w25_idx = int(np.argmax(0.25 * frame_reg_scores + 0.75 * frame_extra_scores))
                frame_ensemble_w75_idx = int(np.argmax(0.75 * frame_reg_scores + 0.25 * frame_extra_scores))
                reg_norm = normalized_scores(reg_scores)
                frame_reg_norm = normalized_scores(frame_reg_scores)
                frame_extra_norm = normalized_scores(frame_extra_scores)
                hybrid_idx = int(np.argmax(0.5 * reg_norm + 0.5 * frame_reg_norm))
                hybrid_extra_idx = int(np.argmax(0.34 * reg_norm + 0.33 * frame_reg_norm + 0.33 * frame_extra_norm))
                object_ious = np.asarray(
                    [mask_iou(candidates[name][obj], item["actual_layers"][obj]) for name in names],
                    dtype=np.float32,
                )
                oracle_idx = int(np.argmax(object_ious))
                class_layers[obj] = candidates[names[class_idx]][obj]
                reg_layers[obj] = candidates[names[reg_idx]][obj]
                frame_class_layers[obj] = candidates[names[frame_class_idx]][obj]
                frame_reg_layers[obj] = candidates[names[frame_reg_idx]][obj]
                frame_extra_layers[obj] = candidates[names[frame_extra_idx]][obj]
                frame_ensemble_layers[obj] = candidates[names[frame_ensemble_idx]][obj]
                frame_ensemble_w25_layers[obj] = candidates[names[frame_ensemble_w25_idx]][obj]
                frame_ensemble_w75_layers[obj] = candidates[names[frame_ensemble_w75_idx]][obj]
                hybrid_layers[obj] = candidates[names[hybrid_idx]][obj]
                hybrid_extra_layers[obj] = candidates[names[hybrid_extra_idx]][obj]
                object_oracle_layers[obj] = candidates[names[oracle_idx]][obj]
                class_indices.append(class_idx)
                reg_indices.append(reg_idx)
                frame_class_indices.append(frame_class_idx)
                frame_reg_indices.append(frame_reg_idx)
                frame_extra_indices.append(frame_extra_idx)
                frame_ensemble_indices.append(frame_ensemble_idx)
                frame_ensemble_w25_indices.append(frame_ensemble_w25_idx)
                frame_ensemble_w75_indices.append(frame_ensemble_w75_idx)
                hybrid_indices.append(hybrid_idx)
                hybrid_extra_indices.append(hybrid_extra_idx)
            pair_ious = pair_frame_ious(candidates, names, actual)
            slot_frame_oracle = float(np.max(pair_ious))
            pair_context = pair_context_features(item, horizon)
            pair_idx = int(np.argmax(pair_regressors[key].predict(pair_context.reshape(1, -1))[0]))
            pair_frame = compose_pair(candidates, pairs[pair_idx][0], pairs[pair_idx][1])
            if include_pairwise:
                pairwise_features = np.vstack(
                    [
                        pair_candidate_features(item, horizon, candidates, left_name, right_name, pair_context)
                        for left_name, right_name in pairs
                    ]
                )
                pairwise_idx = int(np.argmax(pairwise_regressors[key].predict(pairwise_features)))
                pairwise_frame = compose_pair(candidates, pairs[pairwise_idx][0], pairs[pairwise_idx][1])
            rows[key]["simple"].append(float(frame_ious[names.index("simple")]))
            rows[key]["token"].append(float(frame_ious[names.index("token")]))
            rows[key]["max_beta_1.00"].append(float(frame_ious[names.index("max_beta_1.00")]))
            rows[key]["slot_class_ranker"].append(mask_iou(compose_layers(class_layers), actual))
            rows[key]["slot_reg_ranker"].append(mask_iou(compose_layers(reg_layers), actual))
            rows[key]["slot_frame_class_ranker"].append(mask_iou(compose_layers(frame_class_layers), actual))
            rows[key]["slot_frame_reg_ranker"].append(mask_iou(compose_layers(frame_reg_layers), actual))
            rows[key]["slot_frame_extra_ranker"].append(mask_iou(compose_layers(frame_extra_layers), actual))
            rows[key]["slot_frame_ensemble_ranker"].append(mask_iou(compose_layers(frame_ensemble_layers), actual))
            rows[key]["slot_frame_ensemble_w25_ranker"].append(mask_iou(compose_layers(frame_ensemble_w25_layers), actual))
            rows[key]["slot_frame_ensemble_w75_ranker"].append(mask_iou(compose_layers(frame_ensemble_w75_layers), actual))
            rows[key]["slot_hybrid_ranker"].append(mask_iou(compose_layers(hybrid_layers), actual))
            rows[key]["slot_hybrid_extra_ranker"].append(mask_iou(compose_layers(hybrid_extra_layers), actual))
            rows[key]["slot_pair_reg_ranker"].append(mask_iou(pair_frame, actual))
            if include_pairwise:
                rows[key]["slot_pairwise_ranker"].append(mask_iou(pairwise_frame, actual))
            rows[key]["slot_object_oracle"].append(mask_iou(compose_layers(object_oracle_layers), actual))
            rows[key]["slot_frame_oracle"].append(slot_frame_oracle)
            rows[key]["frame_oracle"].append(float(np.max(frame_ious)))
            rows[key]["slot_class_choice"].extend(class_indices)
            rows[key]["slot_reg_choice"].extend(reg_indices)
            rows[key]["slot_frame_class_choice"].extend(frame_class_indices)
            rows[key]["slot_frame_reg_choice"].extend(frame_reg_indices)
            rows[key]["slot_frame_extra_choice"].extend(frame_extra_indices)
            rows[key]["slot_frame_ensemble_choice"].extend(frame_ensemble_indices)
            rows[key]["slot_frame_ensemble_w25_choice"].extend(frame_ensemble_w25_indices)
            rows[key]["slot_frame_ensemble_w75_choice"].extend(frame_ensemble_w75_indices)
            rows[key]["slot_hybrid_choice"].extend(hybrid_indices)
            rows[key]["slot_hybrid_extra_choice"].extend(hybrid_extra_indices)
            if include_pairwise:
                rows[key]["slot_pairwise_choice"].append(pairwise_idx)
    summary = {}
    for key, metrics in rows.items():
        summary[key] = {}
        for metric, vals in metrics.items():
            if metric.endswith("_choice"):
                rounded = np.rint(vals).astype(np.int32)
                if metric == "slot_pairwise_choice":
                    summary[key][metric] = {
                        f"{pairs[int(c)][0]}|{pairs[int(c)][1]}": int(np.sum(rounded == c))
                        for c in np.unique(rounded)
                    }
                else:
                    summary[key][metric] = {names[int(c)]: int(np.sum(rounded == c)) for c in np.unique(rounded)}
            else:
                summary[key][metric] = float(np.mean(vals))
    return summary


LEARNED_RANKER_METRICS = (
    "slot_class_ranker",
    "slot_reg_ranker",
    "slot_frame_class_ranker",
    "slot_frame_reg_ranker",
    "slot_frame_extra_ranker",
    "slot_frame_ensemble_ranker",
    "slot_frame_ensemble_w25_ranker",
    "slot_frame_ensemble_w75_ranker",
    "slot_hybrid_ranker",
    "slot_hybrid_extra_ranker",
    "slot_pair_reg_ranker",
    "slot_pairwise_ranker",
)

STABILITY_FALLBACKS = {
    "slot_frame_ensemble_w25_ranker": "slot_frame_ensemble_ranker",
    "slot_frame_ensemble_w75_ranker": "slot_frame_ensemble_ranker",
    "slot_hybrid_ranker": "slot_reg_ranker",
    "slot_hybrid_extra_ranker": "slot_reg_ranker",
}
MIN_CALIBRATION_GAIN = 0.003


def train_selector_suite(codec, simple_model, token_model, simple_sequences, token_sequences, seed: int, enable_pairwise: bool):
    (
        x_by_h,
        y_class_by_h,
        y_reg_by_h,
        frame_x_by_h,
        frame_y_class_by_h,
        frame_y_reg_by_h,
        pair_x_by_h,
        pair_y_reg_by_h,
        pairwise_x_by_h,
        pairwise_y_by_h,
        names,
    ) = collect_examples(
        codec,
        simple_model,
        token_model,
        simple_sequences,
        token_sequences,
        enable_pairwise,
    )
    classifiers, regressors, stats = train_rankers(x_by_h, y_class_by_h, y_reg_by_h, seed)
    frame_classifiers, frame_regressors, frame_extra_regressors, frame_stats = train_frame_slot_rankers(
        frame_x_by_h,
        frame_y_class_by_h,
        frame_y_reg_by_h,
        seed,
    )
    pair_regressors, pairwise_regressors, pair_stats = train_pair_rankers(
        pair_x_by_h,
        pair_y_reg_by_h,
        pairwise_x_by_h,
        pairwise_y_by_h,
        seed,
        enable_pairwise,
    )
    suite = {
        "classifiers": classifiers,
        "regressors": regressors,
        "frame_classifiers": frame_classifiers,
        "frame_regressors": frame_regressors,
        "frame_extra_regressors": frame_extra_regressors,
        "pair_regressors": pair_regressors,
        "pairwise_regressors": pairwise_regressors,
        "names": names,
    }
    stats_bundle = {
        "ranker_stats": stats,
        "frame_slot_ranker_stats": frame_stats,
        "pair_ranker_stats": pair_stats,
    }
    return suite, stats_bundle


def evaluate_suite(codec, simple_model, token_model, simple_sequences, token_sequences, suite):
    return evaluate(
        codec,
        simple_model,
        token_model,
        simple_sequences,
        token_sequences,
        suite["classifiers"],
        suite["regressors"],
        suite["frame_classifiers"],
        suite["frame_regressors"],
        suite["frame_extra_regressors"],
        suite["pair_regressors"],
        suite["pairwise_regressors"],
        suite["names"],
    )


def select_learned_rankers(calibration_metrics: dict) -> dict[str, str]:
    selection = {}
    for horizon in GT_HORIZONS:
        key = str(horizon)
        metric = max(LEARNED_RANKER_METRICS, key=lambda name: calibration_metrics[key].get(name, -1.0))
        fallback = STABILITY_FALLBACKS.get(metric)
        if fallback is not None:
            gain = calibration_metrics[key].get(metric, -1.0) - calibration_metrics[key].get(fallback, -1.0)
            if gain < MIN_CALIBRATION_GAIN:
                metric = fallback
        selection[key] = metric
    return selection


def attach_best_learned(metrics: dict, calibration_metrics: dict, selection: dict[str, str]) -> dict[str, dict[str, float | str]]:
    report = {}
    for horizon in GT_HORIZONS:
        key = str(horizon)
        metric = selection[key]
        metrics[key]["best_learned_ranker"] = float(metrics[key][metric])
        report[key] = {
            "selected_metric": metric,
            "calibration_iou": float(calibration_metrics[key][metric]),
            "test_iou": float(metrics[key][metric]),
        }
    return report


def main() -> None:
    if "--patch-decoder" in sys.argv:
        import phase11a_patch_decoder_probe

        sys.argv = [sys.argv[0]] + [arg for arg in sys.argv[1:] if arg != "--patch-decoder"]
        phase11a_patch_decoder_probe.main()
        return
    if "--kinematic-probe" in sys.argv:
        import phase11a_kinematic_encoder_probe

        sys.argv = [sys.argv[0]] + [arg for arg in sys.argv[1:] if arg != "--kinematic-probe"]
        phase11a_kinematic_encoder_probe.main()
        return
    if "--cell-router" in sys.argv:
        import phase11a_cell_router_probe

        sys.argv = [sys.argv[0]] + [arg for arg in sys.argv[1:] if arg != "--cell-router"]
        phase11a_cell_router_probe.main()
        return
    if "--multi-encoder" in sys.argv:
        import phase11a_multi_encoder_probe

        sys.argv = [sys.argv[0]] + [arg for arg in sys.argv[1:] if arg != "--multi-encoder"]
        phase11a_multi_encoder_probe.main()
        return
    if "--patch-attention" in sys.argv:
        import phase11a_patch_attention_probe

        sys.argv = [sys.argv[0]] + [arg for arg in sys.argv[1:] if arg != "--patch-attention"]
        phase11a_patch_attention_probe.main()
        return
    if "--simvp-probe" in sys.argv:
        import phase11a_simvp_probe

        sys.argv = [sys.argv[0]] + [arg for arg in sys.argv[1:] if arg != "--simvp-probe"]
        phase11a_simvp_probe.main()
        return

    parser = argparse.ArgumentParser(description="Fase 11A slot/object IoU ranker probe.")
    parser.add_argument("--train-sequences", type=int, default=220)
    parser.add_argument("--test-sequences", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4107)
    parser.add_argument("--selection-calibration-sequences", type=int, default=80)
    parser.add_argument("--enable-pairwise-ranker", action="store_true")
    parser.add_argument("--amf-scale", choices=("default", "wide", "xwide"), default="default")
    parser.add_argument("--out", default="results/phase11a_slot_ranker_probe.json")
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
    simple_model = train_amf_scaled(build_transitions(simple_train), args.seed, args.amf_scale)
    token_model = train_amf_scaled(build_transitions(token_train), args.seed, args.amf_scale)
    calibration_n = min(max(1, args.selection_calibration_sequences), max(1, len(simple_train) // 4))
    fit_n = max(1, len(simple_train) - calibration_n)
    fit_simple = simple_train[:fit_n]
    fit_token = token_train[:fit_n]
    calibration_simple = simple_train[fit_n:]
    calibration_token = token_train[fit_n:]
    calibration_suite, _ = train_selector_suite(
        codec,
        simple_model,
        token_model,
        fit_simple,
        fit_token,
        args.seed,
        args.enable_pairwise_ranker,
    )
    calibration_metrics = evaluate_suite(
        codec,
        simple_model,
        token_model,
        calibration_simple,
        calibration_token,
        calibration_suite,
    )
    learned_selection = select_learned_rankers(calibration_metrics)
    full_suite, stats_bundle = train_selector_suite(
        codec,
        simple_model,
        token_model,
        simple_train,
        token_train,
        args.seed,
        args.enable_pairwise_ranker,
    )
    metrics = evaluate_suite(codec, simple_model, token_model, simple_test, token_test, full_suite)
    best_learned_selection = attach_best_learned(metrics, calibration_metrics, learned_selection)
    results = {
        "dataset_shape": raw_shape,
        "train_sequences": args.train_sequences,
        "test_sequences": args.test_sequences,
        "selection_fit_sequences": len(fit_simple),
        "selection_calibration_sequences": len(calibration_simple),
        "pairwise_ranker_enabled": bool(args.enable_pairwise_ranker),
        "amf_scale": args.amf_scale,
        "simple_amf_cells": int(len(getattr(simple_model, "centers", []))),
        "token_amf_cells": int(len(getattr(token_model, "centers", []))),
        "candidate_names": full_suite["names"],
        "candidate_pair_count": len(pair_names(full_suite["names"])),
        "ranker_stats": stats_bundle["ranker_stats"],
        "frame_slot_ranker_stats": stats_bundle["frame_slot_ranker_stats"],
        "pair_ranker_stats": stats_bundle["pair_ranker_stats"],
        "calibration_metrics": calibration_metrics,
        "best_learned_selection": best_learned_selection,
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - start,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
