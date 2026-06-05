from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

from phase12a_physicalai_world_probe import (
    FPS,
    PhysicsTrack,
    fit_model_family,
    load_tracks,
    mae,
    mse,
    predict_family,
    select_candidate,
    split_train_validation,
)
from phase12b_identity_world_probe import identity_feature
from phase12b_orientation_world_probe import normalized_quat, orientation_delta, orientation_feature
from phase12c_energy_world_probe import energy_feature, object_bits
from phase12c_temporal_energy_world_probe import temporal_energy_feature


HORIZONS_12D = (1, 5, 15, 30, 60)
BASE_PREDICTORS = ("temporal_energy", "energy", "orientation", "identity")
ROUTER_PREDICTORS = BASE_PREDICTORS + ("static_ensemble",)
MODEL_CANDIDATES = ("ridge", "constant_velocity", "cv_amf", "ridge_amf_0.25", "ridge_amf_0.5", "ridge_amf_1.0")
ALPHA_GRID = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50)
ROUTER_BLEND_GRID = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)


@dataclass
class SampleSet:
    x_by_predictor: dict[str, dict[int, np.ndarray]]
    target: dict[int, np.ndarray]
    last: dict[int, np.ndarray]
    cv: dict[int, np.ndarray]
    keys: dict[int, np.ndarray]
    object_names: dict[int, list[str]]
    slot_indices: dict[int, list[int]]
    regimes: dict[int, list[str]]


def _window(track: PhysicsTrack, frame: int, window: int) -> tuple[int, np.ndarray, np.ndarray]:
    start = max(0, frame - window + 1)
    return start, track.com[start : frame + 1].astype(np.float32), track.velocity[start : frame + 1].astype(np.float32)


def _safe_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))


def _radial_parts(track: PhysicsTrack, frame: int) -> tuple[np.ndarray, float, float, float]:
    com = track.com[frame].astype(np.float32)
    velocity = track.velocity[frame].astype(np.float32)
    anchor = track.com[0].astype(np.float32)
    rel = com - anchor
    radius = _safe_norm(rel)
    radial_dir = rel / max(radius, 1e-6)
    radial_velocity = float(np.dot(velocity, radial_dir))
    tangential_velocity = velocity - radial_velocity * radial_dir
    return radial_dir.astype(np.float32), radius, radial_velocity, _safe_norm(tangential_velocity)


def _energy(track: PhysicsTrack, frame: int) -> float:
    velocity = track.velocity[frame].astype(np.float32)
    height = float(track.com[frame, 1] - track.com[0, 1])
    return _safe_norm(velocity) ** 2 + 9.81 * height


def _orientation_window_delta(track: PhysicsTrack, start: int, frame: int) -> float:
    if track.rot is None:
        return 0.0
    now = normalized_quat(track, frame)
    old = normalized_quat(track, start)
    return _safe_norm(now - old)


def episode_regime(track: PhysicsTrack, frame: int, window: int) -> str:
    start, com_window, vel_window = _window(track, frame, window)
    radial_dir, radius, radial_velocity, tangential_speed = _radial_parts(track, frame)
    radius_values = np.linalg.norm(com_window - track.com[0].astype(np.float32), axis=1)
    radius_std = float(np.std(radius_values))
    vertical_accel = 0.0
    if len(vel_window) >= 2:
        vertical_accel = float(np.mean(np.diff(vel_window[:, 1])))
    accel = np.diff(vel_window, axis=0) if len(vel_window) >= 2 else np.zeros((0, 3), dtype=np.float32)
    impact = float(np.max(np.linalg.norm(accel, axis=1))) if len(accel) else 0.0
    speed = _safe_norm(track.velocity[frame])
    energy_delta = _energy(track, frame) - _energy(track, start)
    radial_lock = radius_std / max(radius, 1e-6)
    pendulum_score = tangential_speed / max(abs(radial_velocity) + 1e-3, 1e-3)
    freefall_score = abs(vertical_accel + 9.81 / FPS) / (abs(vertical_accel) + 1e-3)

    if impact > 0.40 or abs(energy_delta) > 10.0:
        return "impact_or_regime_change"
    if speed < 0.05:
        return "rest_or_constraint_hold"
    if radial_lock < 0.05 and pendulum_score > 2.0:
        return "pendulum_radial_constraint"
    if freefall_score < 0.35 and abs(track.velocity[frame, 1]) > tangential_speed:
        return "freefall_like"
    if abs(radial_velocity) > tangential_speed:
        return "radial_transfer"
    return "mixed_workspace_motion"


def ltm_key(track: PhysicsTrack, frame: int, horizon: int, window: int) -> np.ndarray:
    start, com_window, vel_window = _window(track, frame, window)
    velocity = track.velocity[frame].astype(np.float32)
    prev_velocity = track.velocity[max(0, frame - 1)].astype(np.float32)
    prev2_velocity = track.velocity[max(0, frame - 2)].astype(np.float32)
    acceleration = velocity - prev_velocity
    jerk = velocity - 2.0 * prev_velocity + prev2_velocity

    radial_dir, radius, radial_velocity, tangential_speed = _radial_parts(track, frame)
    start_radial_dir, start_radius, start_radial_velocity, start_tangential_speed = _radial_parts(track, start)
    radius_values = np.linalg.norm(com_window - track.com[0].astype(np.float32), axis=1)
    radius_std = float(np.std(radius_values))
    energy_values = np.asarray([_energy(track, i) for i in range(start, frame + 1)], dtype=np.float32)
    energy_delta = float(energy_values[-1] - energy_values[0])
    energy_std = float(np.std(energy_values))

    if len(vel_window) >= 2:
        accel_window = np.diff(vel_window, axis=0)
        accel_norms = np.linalg.norm(accel_window, axis=1)
        max_impact = float(np.max(accel_norms))
        mean_impact = float(np.mean(accel_norms))
        vertical_flip = float(np.any(np.sign(vel_window[:-1, 1]) != np.sign(vel_window[1:, 1])))
    else:
        max_impact = 0.0
        mean_impact = 0.0
        vertical_flip = 0.0

    radial_flip = float(np.sign(radial_velocity) != np.sign(start_radial_velocity))
    orientation_now = orientation_delta(track, frame)
    orientation_window = _orientation_window_delta(track, start, frame)
    color = (
        track.segmentation_color.astype(np.float32) / 255.0
        if track.segmentation_color is not None
        else np.zeros(4, dtype=np.float32)
    )

    h_fast = np.concatenate(
        [
            velocity / 10.0,
            prev_velocity / 10.0,
            acceleration / 10.0,
            jerk / 10.0,
            np.asarray(
                [
                    _safe_norm(velocity) / 10.0,
                    _safe_norm(acceleration) / 10.0,
                    _safe_norm(jerk) / 10.0,
                    energy_delta / 100.0,
                    energy_std / 100.0,
                ],
                dtype=np.float32,
            ),
        ]
    )
    h_event = np.asarray(
        [
            max_impact / 10.0,
            mean_impact / 10.0,
            vertical_flip,
            radial_flip,
            abs(energy_delta) / 100.0,
            orientation_window,
        ],
        dtype=np.float32,
    )
    h_regime = np.concatenate(
        [
            radial_dir,
            np.asarray(
                [
                    radius / 10.0,
                    (radius - start_radius) / 10.0,
                    radius_std / 10.0,
                    radial_velocity / 10.0,
                    (radial_velocity - start_radial_velocity) / 10.0,
                    tangential_speed / 10.0,
                    (tangential_speed - start_tangential_speed) / 10.0,
                    horizon / 60.0,
                ],
                dtype=np.float32,
            ),
        ]
    )
    h_workspace = np.concatenate(
        [
            color,
            object_bits(track.object_name),
            np.asarray([track.slot_index / 64.0, track.com[0, 1] / 10.0], dtype=np.float32),
            orientation_now,
        ]
    )
    return np.concatenate([h_fast, h_event, h_regime, h_workspace]).astype(np.float32)


def make_samples(
    tracks: list[PhysicsTrack],
    sequences: set[str],
    horizons: tuple[int, ...],
    stride: int,
    memory_window: int,
) -> SampleSet:
    x_by_predictor: dict[str, dict[int, list[np.ndarray]]] = {
        name: {h: [] for h in horizons} for name in BASE_PREDICTORS
    }
    target: dict[int, list[np.ndarray]] = {h: [] for h in horizons}
    last: dict[int, list[np.ndarray]] = {h: [] for h in horizons}
    cv: dict[int, list[np.ndarray]] = {h: [] for h in horizons}
    keys: dict[int, list[np.ndarray]] = {h: [] for h in horizons}
    object_names: dict[int, list[str]] = {h: [] for h in horizons}
    slot_indices: dict[int, list[int]] = {h: [] for h in horizons}
    regimes: dict[int, list[str]] = {h: [] for h in horizons}

    for track in tracks:
        if track.sequence not in sequences:
            continue
        frames = min(track.com.shape[0], track.velocity.shape[0])
        if track.rot is not None:
            frames = min(frames, track.rot.shape[0])
        for horizon in horizons:
            for frame in range(0, frames - horizon, stride):
                com = track.com[frame].astype(np.float32)
                velocity = track.velocity[frame].astype(np.float32)
                x_by_predictor["temporal_energy"][horizon].append(temporal_energy_feature(track, frame, horizon))
                x_by_predictor["energy"][horizon].append(energy_feature(track, frame, horizon))
                x_by_predictor["orientation"][horizon].append(orientation_feature(track, frame, horizon))
                x_by_predictor["identity"][horizon].append(identity_feature(track, frame, horizon))
                target[horizon].append(track.com[frame + horizon].astype(np.float32))
                last[horizon].append(com)
                cv[horizon].append((com + velocity * (horizon / FPS)).astype(np.float32))
                keys[horizon].append(ltm_key(track, frame, horizon, memory_window))
                object_names[horizon].append(track.object_name)
                slot_indices[horizon].append(track.slot_index)
                regimes[horizon].append(episode_regime(track, frame, memory_window))

    out_x = {
        name: {
            h: np.stack(rows).astype(np.float32) if rows else np.zeros((0, 0), dtype=np.float32)
            for h, rows in by_h.items()
        }
        for name, by_h in x_by_predictor.items()
    }
    out_target = {
        h: np.stack(rows).astype(np.float32) if rows else np.zeros((0, 3), dtype=np.float32)
        for h, rows in target.items()
    }
    out_last = {
        h: np.stack(rows).astype(np.float32) if rows else np.zeros((0, 3), dtype=np.float32)
        for h, rows in last.items()
    }
    out_cv = {
        h: np.stack(rows).astype(np.float32) if rows else np.zeros((0, 3), dtype=np.float32)
        for h, rows in cv.items()
    }
    out_keys = {
        h: np.stack(rows).astype(np.float32) if rows else np.zeros((0, 0), dtype=np.float32)
        for h, rows in keys.items()
    }
    return SampleSet(out_x, out_target, out_last, out_cv, out_keys, object_names, slot_indices, regimes)


class EpisodicLtm:
    def __init__(self, top_k: int, radius: float) -> None:
        self.top_k = top_k
        self.radius = radius
        self.center: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.keys: np.ndarray | None = None
        self.residuals: np.ndarray | None = None
        self.best_ids: np.ndarray | None = None
        self.nn: NearestNeighbors | None = None

    def fit(self, keys: np.ndarray, residuals: np.ndarray, best_ids: np.ndarray) -> None:
        self.center = np.median(keys, axis=0).astype(np.float32)
        scale = np.percentile(np.abs(keys - self.center), 75, axis=0).astype(np.float32)
        self.scale = np.maximum(scale, 1e-4)
        self.keys = self._normalize(keys)
        self.residuals = residuals.astype(np.float32)
        self.best_ids = best_ids.astype(np.int64)
        self.nn = NearestNeighbors(n_neighbors=min(self.top_k + 1, len(keys)), metric="euclidean")
        self.nn.fit(self.keys)

    def _normalize(self, keys: np.ndarray) -> np.ndarray:
        if self.center is None or self.scale is None:
            raise RuntimeError("EpisodicLtm must be fitted first")
        return ((keys - self.center) / self.scale).astype(np.float32)

    def neighbors(self, keys: np.ndarray, leave_one_out: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if self.nn is None or self.keys is None:
            raise RuntimeError("EpisodicLtm must be fitted first")
        k = min(self.top_k + int(leave_one_out), len(self.keys))
        dists, idx = self.nn.kneighbors(self._normalize(keys), n_neighbors=k, return_distance=True)
        weights = np.exp(-dists / max(self.radius, 1e-6)).astype(np.float32)
        if leave_one_out:
            row_ids = np.arange(len(idx))[:, None]
            weights[idx == row_ids] = 0.0
        return idx, weights


def convex_weight_grid(dim: int, steps: int = 4) -> list[np.ndarray]:
    weights: list[np.ndarray] = []

    def rec(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            weights.append(np.asarray(prefix + [remaining], dtype=np.float32) / steps)
            return
        for value in range(remaining + 1):
            rec(prefix + [value], remaining - value, slots - 1)

    rec([], steps, dim)
    return weights


def route_predictions(
    predictor_preds: dict[str, np.ndarray],
    ltm: EpisodicLtm,
    query_keys: np.ndarray,
    leave_one_out: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ltm.residuals is None or ltm.best_ids is None:
        raise RuntimeError("EpisodicLtm must be fitted first")
    idx, weights = ltm.neighbors(query_keys, leave_one_out=leave_one_out)
    denom = np.maximum(weights.sum(axis=1, keepdims=True), 1e-9)
    residual = ((ltm.residuals[idx] * weights[:, :, None]).sum(axis=1) / denom).astype(np.float32)
    one_hot = np.eye(len(ROUTER_PREDICTORS), dtype=np.float32)[ltm.best_ids[idx]]
    mix = (one_hot * weights[:, :, None]).sum(axis=1)
    mix = mix / np.maximum(mix.sum(axis=1, keepdims=True), 1e-9)
    pred_stack = np.stack([predictor_preds[name] for name in ROUTER_PREDICTORS], axis=1)
    router = (pred_stack * mix[:, :, None]).sum(axis=1).astype(np.float32)
    confidence = np.max(mix, axis=1).astype(np.float32)
    return router, residual, confidence


def choose_static_ensemble(val_preds: dict[str, np.ndarray], target: np.ndarray) -> tuple[np.ndarray, float]:
    grid = convex_weight_grid(len(BASE_PREDICTORS), steps=4)
    stack = np.stack([val_preds[name] for name in BASE_PREDICTORS], axis=1)
    best_weights = grid[0]
    best_loss = float("inf")
    for weights in grid:
        pred = (stack * weights[None, :, None]).sum(axis=1)
        loss = mse(pred, target)
        if loss < best_loss:
            best_loss = loss
            best_weights = weights
    return best_weights.astype(np.float32), best_loss


def apply_static_ensemble(preds: dict[str, np.ndarray], weights: np.ndarray) -> np.ndarray:
    stack = np.stack([preds[name] for name in BASE_PREDICTORS], axis=1)
    return (stack * weights[None, :, None]).sum(axis=1).astype(np.float32)


def fit_predictor_models(
    fit_samples: SampleSet,
    val_samples: SampleSet,
    train_samples: SampleSet,
    test_samples: SampleSet,
    max_cells: int,
    ridge: float,
    radius: float,
    top_k: int,
    tie_tolerance: float,
) -> tuple[dict[str, Any], dict[str, dict[int, np.ndarray]], dict[str, dict[int, np.ndarray]]]:
    model_report: dict[str, Any] = {}
    val_preds: dict[str, dict[int, np.ndarray]] = {name: {} for name in BASE_PREDICTORS}
    test_preds: dict[str, dict[int, np.ndarray]] = {name: {} for name in BASE_PREDICTORS}

    for predictor in BASE_PREDICTORS:
        model_report[predictor] = {}
        for horizon in HORIZONS_12D:
            x_fit = fit_samples.x_by_predictor[predictor][horizon]
            target_fit = fit_samples.target[horizon]
            cv_fit = fit_samples.cv[horizon]
            selector_model = fit_model_family(
                x_fit, target_fit, cv_fit, max_cells=max_cells, ridge=ridge, radius=radius, top_k=top_k
            )
            selected, val_losses = select_candidate(
                selector_model,
                val_samples.x_by_predictor[predictor][horizon],
                val_samples.target[horizon],
                val_samples.cv[horizon],
                MODEL_CANDIDATES,
                tie_tolerance=tie_tolerance,
            )
            model = fit_model_family(
                train_samples.x_by_predictor[predictor][horizon],
                train_samples.target[horizon],
                train_samples.cv[horizon],
                max_cells=max_cells,
                ridge=ridge,
                radius=radius,
                top_k=top_k,
            )
            val_preds[predictor][horizon] = predict_family(
                model, val_samples.x_by_predictor[predictor][horizon], val_samples.cv[horizon], selected
            )
            test_preds[predictor][horizon] = predict_family(
                model, test_samples.x_by_predictor[predictor][horizon], test_samples.cv[horizon], selected
            )
            model_report[predictor][f"h{horizon}"] = {
                "selected_candidate": selected,
                "validation_losses": val_losses,
                "feature_dim": int(train_samples.x_by_predictor[predictor][horizon].shape[1]),
            }
    return model_report, val_preds, test_preds


def summarize_regimes(regimes: list[str], best_ids: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for regime, best_id in zip(regimes, best_ids):
        label = ROUTER_PREDICTORS[int(best_id)]
        if regime not in out:
            out[regime] = {"count": 0, "best_predictor_counts": {name: 0 for name in ROUTER_PREDICTORS}}
        out[regime]["count"] += 1
        out[regime]["best_predictor_counts"][label] += 1
    return out


def calibrate_and_evaluate_horizon(
    horizon: int,
    val_samples: SampleSet,
    test_samples: SampleSet,
    val_preds: dict[str, dict[int, np.ndarray]],
    test_preds: dict[str, dict[int, np.ndarray]],
    top_k: int,
    radius: float,
) -> dict[str, Any]:
    val_target = val_samples.target[horizon]
    test_target = test_samples.target[horizon]
    val_pred_h = {name: val_preds[name][horizon] for name in ROUTER_PREDICTORS}
    test_pred_h = {name: test_preds[name][horizon] for name in ROUTER_PREDICTORS}

    val_errors = np.stack(
        [np.sum(np.square(val_pred_h[name] - val_target), axis=1) for name in ROUTER_PREDICTORS], axis=1
    )
    best_ids = np.argmin(val_errors, axis=1).astype(np.int64)
    temporal_residual = (val_target - val_pred_h["temporal_energy"]).astype(np.float32)

    ltm = EpisodicLtm(top_k=top_k, radius=radius)
    ltm.fit(val_samples.keys[horizon], temporal_residual, best_ids)

    raw_router_val, residual_val, confidence_val = route_predictions(
        val_pred_h, ltm, val_samples.keys[horizon], leave_one_out=True
    )
    raw_router_test, residual_test, confidence_test = route_predictions(
        test_pred_h, ltm, test_samples.keys[horizon], leave_one_out=False
    )

    router_blend_alpha = min(
        ROUTER_BLEND_GRID,
        key=lambda alpha: mse(
            val_pred_h["temporal_energy"] + alpha * (raw_router_val - val_pred_h["temporal_energy"]),
            val_target,
        ),
    )
    router_val = val_pred_h["temporal_energy"] + router_blend_alpha * (
        raw_router_val - val_pred_h["temporal_energy"]
    )
    router_test = test_pred_h["temporal_energy"] + router_blend_alpha * (
        raw_router_test - test_pred_h["temporal_energy"]
    )

    residual_alpha = min(
        ALPHA_GRID,
        key=lambda alpha: mse(val_pred_h["temporal_energy"] + alpha * residual_val, val_target),
    )
    router_residual_alpha = min(
        ALPHA_GRID,
        key=lambda alpha: mse(router_val + alpha * residual_val, val_target),
    )

    temporal_test = test_pred_h["temporal_energy"]
    residual_test_pred = temporal_test + residual_alpha * residual_test
    router_residual_test_pred = router_test + router_residual_alpha * residual_test
    oracle_errors = np.stack(
        [np.sum(np.square(test_pred_h[name] - test_target), axis=1) for name in ROUTER_PREDICTORS], axis=1
    )
    oracle_choice = np.argmin(oracle_errors, axis=1)
    oracle_stack = np.stack([test_pred_h[name] for name in ROUTER_PREDICTORS], axis=1)
    oracle_pred = oracle_stack[np.arange(len(test_target)), oracle_choice]

    ablations = {
        "temporal_energy": temporal_test,
        "ltm_router_no_residual": router_test,
        "ltm_residual_no_router": residual_test_pred,
        "ltm_router_plus_residual": router_residual_test_pred,
        "oracle_selector_test_only_invalid": oracle_pred,
    }
    metrics = {
        name: {
            "mse": mse(pred, test_target),
            "mae": mae(pred, test_target),
            "gain_vs_temporal_energy": (mse(temporal_test, test_target) - mse(pred, test_target))
            / max(mse(temporal_test, test_target), 1e-9),
        }
        for name, pred in ablations.items()
    }
    valid_names = ("temporal_energy", "ltm_router_no_residual", "ltm_residual_no_router", "ltm_router_plus_residual")
    best_valid = min(valid_names, key=lambda name: metrics[name]["mse"])
    return {
        "samples": int(len(test_target)),
        "validation_samples": int(len(val_target)),
        "memory_window_frames": None,
        "top_k": top_k,
        "router_radius": radius,
        "residual_alpha": residual_alpha,
        "router_blend_alpha": router_blend_alpha,
        "router_residual_alpha": router_residual_alpha,
        "confidence_mean_validation": float(np.mean(confidence_val)),
        "confidence_mean_test": float(np.mean(confidence_test)),
        "best_valid_ablation": best_valid,
        "best_valid_mse": metrics[best_valid]["mse"],
        "best_valid_gain_vs_temporal_energy": metrics[best_valid]["gain_vs_temporal_energy"],
        "ablation_metrics": metrics,
        "memory_best_predictor_counts": {
            name: int(np.sum(best_ids == idx)) for idx, name in enumerate(ROUTER_PREDICTORS)
        },
        "regime_memory_summary": summarize_regimes(val_samples.regimes[horizon], best_ids),
    }


def run_ltm_router_probe(
    tar_path: Path,
    train_fraction: float,
    stride: int,
    max_cells: int,
    ridge: float,
    model_radius: float,
    model_top_k: int,
    router_radius: float,
    router_top_k: int,
    tie_tolerance: float,
    split_seed: int,
    memory_window: int,
) -> dict[str, Any]:
    started = time.time()
    tracks = load_tracks(tar_path)
    sequences = sorted({track.sequence for track in tracks})
    fit_sequences, val_sequences, train_sequences, test_sequences = split_train_validation(sequences, train_fraction, split_seed)

    fit_samples = make_samples(tracks, fit_sequences, HORIZONS_12D, stride, memory_window)
    val_samples = make_samples(tracks, val_sequences, HORIZONS_12D, stride, memory_window)
    train_samples = make_samples(tracks, train_sequences, HORIZONS_12D, stride, memory_window)
    test_samples = make_samples(tracks, test_sequences, HORIZONS_12D, stride, memory_window)

    model_report, val_preds, test_preds = fit_predictor_models(
        fit_samples,
        val_samples,
        train_samples,
        test_samples,
        max_cells=max_cells,
        ridge=ridge,
        radius=model_radius,
        top_k=model_top_k,
        tie_tolerance=tie_tolerance,
    )

    static_ensemble_weights: dict[str, list[float]] = {}
    val_preds["static_ensemble"] = {}
    test_preds["static_ensemble"] = {}
    for horizon in HORIZONS_12D:
        weights, val_loss = choose_static_ensemble(
            {name: val_preds[name][horizon] for name in BASE_PREDICTORS}, val_samples.target[horizon]
        )
        val_preds["static_ensemble"][horizon] = apply_static_ensemble(
            {name: val_preds[name][horizon] for name in BASE_PREDICTORS}, weights
        )
        test_preds["static_ensemble"][horizon] = apply_static_ensemble(
            {name: test_preds[name][horizon] for name in BASE_PREDICTORS}, weights
        )
        static_ensemble_weights[f"h{horizon}"] = [float(v) for v in weights]
        model_report.setdefault("static_ensemble", {})[f"h{horizon}"] = {
            "weights": {name: float(weight) for name, weight in zip(BASE_PREDICTORS, weights)},
            "validation_mse": val_loss,
        }

    metrics = {}
    for horizon in HORIZONS_12D:
        metrics[f"h{horizon}"] = calibrate_and_evaluate_horizon(
            horizon,
            val_samples,
            test_samples,
            val_preds,
            test_preds,
            top_k=router_top_k,
            radius=router_radius,
        )
        metrics[f"h{horizon}"]["memory_window_frames"] = memory_window

    return {
        "probe": "phase12d_ltm_router_probe",
        "architecture": "AMF-LTM episodic router/retriever over frozen temporal-energy baseline",
        "tar_path": str(tar_path),
        "track_count": len(tracks),
        "sequence_count": len(sequences),
        "fit_sequences": len(fit_sequences),
        "validation_sequences": len(val_sequences),
        "train_sequences": len(train_sequences),
        "test_sequences": len(test_sequences),
        "horizons": list(HORIZONS_12D),
        "stride": stride,
        "max_cells": max_cells,
        "model_radius": model_radius,
        "model_top_k": model_top_k,
        "router_radius": router_radius,
        "router_top_k": router_top_k,
        "tie_tolerance": tie_tolerance,
        "split_seed": split_seed,
        "memory_window_frames": memory_window,
        "no_leakage_rule": "predictors, static ensemble, router alphas, and episodic memories are calibrated only on fit/train/validation sequences; oracle selector is test-only and marked invalid",
        "ltm_levels": {
            "H_fast": "velocity, short trend, acceleration, jerk, energy volatility",
            "H_event": "impact proxy, bounce/sign flips, abrupt energy/orientation change",
            "H_regime": "radial constraint, tangential/radial velocity, pendulum/freefall/impact/rest labels",
            "H_workspace": "object identity, slot, segmentation color, anchor constraint, orientation",
        },
        "model_report": model_report,
        "static_ensemble_weights": static_ensemble_weights,
        "test_metrics": metrics,
        "elapsed_seconds": time.time() - started,
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Fase 12D - AMF-LTM router/retriever",
        "",
        f"Tar: `{result['tar_path']}`",
        f"Tracks: {result['track_count']}",
        f"Sequences: {result['sequence_count']} ({result['train_sequences']} train / {result['test_sequences']} test)",
        f"Fit/validation/test: {result['fit_sequences']} / {result['validation_sequences']} / {result['test_sequences']} (seed {result['split_seed']})",
        f"Stride: {result['stride']}; window: {result['memory_window_frames']} frames; router top-k: {result['router_top_k']}",
        "",
        "## Discipline",
        "",
        "- Temporal-energy is frozen as the strong baseline.",
        "- H_event and H_workspace are used for routing/retrieval/confidence, not as dense predictor features.",
        "- Memories, selector weights, and residual alphas are calibrated only on train/validation sequences.",
        "- `oracle_selector_test_only_invalid` is a diagnostic ceiling only and is not a valid model.",
        "",
        "## Metrics",
        "",
        "| horizon | temporal-energy MSE | router MSE | residual MSE | router+residual MSE | router blend | best valid | gain vs temporal | oracle invalid MSE |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for horizon in [f"h{h}" for h in result["horizons"]]:
        metrics = result["test_metrics"][horizon]
        ab = metrics["ablation_metrics"]
        lines.append(
            f"| {horizon} | {ab['temporal_energy']['mse']:.6f} | "
            f"{ab['ltm_router_no_residual']['mse']:.6f} | "
            f"{ab['ltm_residual_no_router']['mse']:.6f} | "
            f"{ab['ltm_router_plus_residual']['mse']:.6f} | "
            f"{metrics['router_blend_alpha']:.2f} | "
            f"{metrics['best_valid_ablation']} | "
            f"{metrics['best_valid_gain_vs_temporal_energy']:.6f} | "
            f"{ab['oracle_selector_test_only_invalid']['mse']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## LTM interpretation",
            "",
            "AMF-LTM 12D changes the role of long-term memory: it no longer feeds a wider dense vector into the predictor. "
            "It writes episodic validation memories with physical regime, energy trend, radial/tangential state, orientation change, impact/change proxies, object identity, slot/color workspace identity, residual surprise, and the predictor that won locally.",
            "",
            "At test time, each state retrieves nearby episodes and uses them either as a router over temporal-energy/energy/orientation/identity/static-ensemble or as a small residual on top of temporal-energy.",
            "",
            "## Memory summaries",
            "",
        ]
    )
    for horizon in [f"h{h}" for h in result["horizons"]]:
        counts = result["test_metrics"][horizon]["memory_best_predictor_counts"]
        compact = ", ".join(f"{name}={count}" for name, count in counts.items())
        lines.append(f"- {horizon}: {compact}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate Phase 12D AMF-LTM episodic router/retriever.")
    parser.add_argument("--tar-path", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--max-cells", type=int, default=8000)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--model-radius", type=float, default=0.75)
    parser.add_argument("--model-top-k", type=int, default=32)
    parser.add_argument("--router-radius", type=float, default=1.25)
    parser.add_argument("--router-top-k", type=int, default=32)
    parser.add_argument("--tie-tolerance", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--memory-window", type=int, default=20)
    parser.add_argument("--out-json", default="results/phase12d_ltm_router_probe.json")
    parser.add_argument("--out-report", default="results/FASE12D_LTM_ROUTER_PROBE.md")
    args = parser.parse_args()

    result = run_ltm_router_probe(
        Path(args.tar_path),
        args.train_fraction,
        args.stride,
        args.max_cells,
        args.ridge,
        args.model_radius,
        args.model_top_k,
        args.router_radius,
        args.router_top_k,
        args.tie_tolerance,
        args.split_seed,
        args.memory_window,
    )
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(render_report(result), encoding="utf-8")
    print(json.dumps({"out_json": str(out_json), "out_report": str(out_report)}, indent=2))


if __name__ == "__main__":
    main()
