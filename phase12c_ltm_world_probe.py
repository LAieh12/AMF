from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from phase12a_physicalai_world_probe import (
    FPS,
    HORIZONS,
    PhysicsTrack,
    evaluate,
    fit_model_family,
    load_tracks,
    render_report as render_base_report,
    select_candidate,
    split_train_validation,
)
from phase12c_energy_world_probe import energy_feature, object_bits
from phase12c_temporal_energy_world_probe import temporal_scalars


H_FAST_SCALE = 1.0
H_EVENT_SCALE = 0.20
H_REGIME_SCALE = 1.0
H_WORKSPACE_SCALE = 0.20


def _track_frames(track: PhysicsTrack) -> int:
    frames = min(track.com.shape[0], track.velocity.shape[0])
    if track.rot is not None:
        frames = min(frames, track.rot.shape[0])
    return frames


def h_fast(track: PhysicsTrack, frame: int) -> np.ndarray:
    return temporal_scalars(track, frame)


def h_event(sequence_tracks: list[PhysicsTrack], frame: int, index: int) -> np.ndarray:
    positions = np.stack([track.com[frame] for track in sequence_tracks]).astype(np.float32)
    velocities = np.stack([track.velocity[frame] for track in sequence_tracks]).astype(np.float32)
    track = sequence_tracks[index]
    prev = max(0, frame - 1)
    velocity = velocities[index]
    prev_velocity = track.velocity[prev].astype(np.float32)
    acceleration = velocity - prev_velocity

    if len(sequence_tracks) > 1:
        rel_pos = positions - positions[index][None, :]
        dists = np.linalg.norm(rel_pos, axis=1)
        dists[index] = np.inf
        nearest = int(np.argmin(dists))
        nearest_dist = float(dists[nearest])
        rel = rel_pos[nearest]
        rel_vel = velocities[nearest] - velocity
        direction = rel / max(nearest_dist, 1e-6)
        closing_speed = -float(np.dot(rel_vel, direction))
        density_025 = float(np.count_nonzero(dists < 0.25))
        density_05 = float(np.count_nonzero(dists < 0.5))
        density_1 = float(np.count_nonzero(dists < 1.0))
    else:
        rel = np.zeros(3, dtype=np.float32)
        rel_vel = np.zeros(3, dtype=np.float32)
        nearest_dist = 0.0
        closing_speed = 0.0
        density_025 = density_05 = density_1 = 0.0

    sign_flip = np.sign(velocity) != np.sign(prev_velocity)
    impact_score = float(np.linalg.norm(acceleration))
    bounce_score = float(np.count_nonzero(sign_flip)) / 3.0
    contact_score = max(0.0, closing_speed) / max(nearest_dist + 1e-3, 1e-3)

    return np.concatenate(
        [
            rel / 10.0,
            rel_vel / 10.0,
            acceleration / 10.0,
            np.asarray(
                [
                    nearest_dist / 10.0,
                    closing_speed / 10.0,
                    density_025 / 16.0,
                    density_05 / 16.0,
                    density_1 / 16.0,
                    impact_score / 10.0,
                    bounce_score,
                    contact_score / 10.0,
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)


def h_regime(track: PhysicsTrack, frame: int) -> np.ndarray:
    prev = max(0, frame - 1)
    com = track.com[frame].astype(np.float32)
    velocity = track.velocity[frame].astype(np.float32)
    prev_velocity = track.velocity[prev].astype(np.float32)
    acceleration = velocity - prev_velocity
    anchor = track.com[0].astype(np.float32)
    rel = com - anchor
    radius = float(np.linalg.norm(rel))
    radial_dir = rel / max(radius, 1e-6)
    radial_velocity = abs(float(np.dot(velocity, radial_dir)))
    speed = float(np.linalg.norm(velocity))
    tangential_speed = float(np.linalg.norm(velocity - radial_velocity * radial_dir))
    accel_norm = float(np.linalg.norm(acceleration))
    height = float(com[1] - anchor[1])

    pendulum = tangential_speed / max(speed + radial_velocity + 1e-6, 1e-6)
    freefall = max(0.0, -velocity[1]) / 10.0
    impact = accel_norm / 10.0
    rest = 1.0 / (1.0 + speed)
    radial_stretch = radius / 10.0
    rising = 1.0 if velocity[1] > 0 else 0.0
    falling = 1.0 if velocity[1] < 0 else 0.0
    high_energy = (speed * speed + 9.81 * height) / 100.0
    return np.asarray(
        [pendulum, freefall, impact, rest, radial_stretch, rising, falling, high_energy],
        dtype=np.float32,
    )


def h_workspace(sequence_tracks: list[PhysicsTrack], frame: int, index: int) -> np.ndarray:
    track = sequence_tracks[index]
    positions = np.stack([item.com[frame] for item in sequence_tracks]).astype(np.float32)
    center = positions.mean(axis=0)
    spread = positions.std(axis=0)
    color = (
        track.segmentation_color.astype(np.float32) / 255.0
        if track.segmentation_color is not None
        else np.zeros(4, dtype=np.float32)
    )
    return np.concatenate(
        [
            color,
            object_bits(track.object_name),
            np.asarray([track.slot_index / 32.0, len(sequence_tracks) / 64.0], dtype=np.float32),
            (track.com[frame].astype(np.float32) - center) / 10.0,
            center / 10.0,
            spread / 10.0,
        ]
    ).astype(np.float32)


def ltm_feature(sequence_tracks: list[PhysicsTrack], frame: int, horizon: int, index: int) -> np.ndarray:
    track = sequence_tracks[index]
    return np.concatenate(
        [
            energy_feature(track, frame, horizon),
            H_FAST_SCALE * h_fast(track, frame),
            H_EVENT_SCALE * h_event(sequence_tracks, frame, index),
            H_REGIME_SCALE * h_regime(track, frame),
            H_WORKSPACE_SCALE * h_workspace(sequence_tracks, frame, index),
        ]
    ).astype(np.float32)


def make_ltm_samples(
    tracks: list[PhysicsTrack],
    sequences: set[str],
    horizons: tuple[int, ...],
    stride: int,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    by_sequence_view: dict[tuple[str, str], list[PhysicsTrack]] = defaultdict(list)
    for track in tracks:
        if track.sequence in sequences:
            by_sequence_view[(track.sequence, track.object_name)].append(track)

    by_h: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {h: [] for h in horizons}
    for sequence_tracks in by_sequence_view.values():
        frames = min(_track_frames(track) for track in sequence_tracks)
        for horizon in horizons:
            for frame in range(0, frames - horizon, stride):
                for index, track in enumerate(sequence_tracks):
                    com = track.com[frame].astype(np.float32)
                    velocity = track.velocity[frame].astype(np.float32)
                    target = track.com[frame + horizon].astype(np.float32)
                    cv = (com + velocity * (horizon / FPS)).astype(np.float32)
                    by_h[horizon].append((ltm_feature(sequence_tracks, frame, horizon, index), target, com, cv))

    out = {}
    for horizon, rows in by_h.items():
        if not rows:
            out[horizon] = (
                np.zeros((0, 116), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
            )
            continue
        feats, targets, last, cv = zip(*rows)
        out[horizon] = (np.stack(feats), np.stack(targets), np.stack(last), np.stack(cv))
    return out


def run_ltm_probe(
    tar_path: Path,
    train_fraction: float,
    stride: int,
    max_cells: int,
    ridge: float,
    radius: float,
    top_k: int,
    tie_tolerance: float,
    split_seed: int,
) -> dict[str, Any]:
    started = time.time()
    tracks = load_tracks(tar_path)
    sequences = sorted({track.sequence for track in tracks})
    fit_sequences, val_sequences, train_sequences, test_sequences = split_train_validation(sequences, train_fraction, split_seed)
    fit_samples = make_ltm_samples(tracks, fit_sequences, HORIZONS, stride)
    val_samples = make_ltm_samples(tracks, val_sequences, HORIZONS, stride)
    train_samples = make_ltm_samples(tracks, train_sequences, HORIZONS, stride)
    test_samples = make_ltm_samples(tracks, test_sequences, HORIZONS, stride)

    models: dict[int, dict[str, Any]] = {}
    candidates = ("ridge", "constant_velocity", "cv_amf", "ridge_amf_0.25", "ridge_amf_0.5", "ridge_amf_1.0")
    for horizon, (x_fit, target_fit, _last_fit, cv_fit) in fit_samples.items():
        val_x, val_target, _val_last, val_cv = val_samples[horizon]
        selector_model = fit_model_family(
            x_fit, target_fit, cv_fit, max_cells=max_cells, ridge=ridge, radius=radius, top_k=top_k
        )
        selected_candidate, val_losses = select_candidate(
            selector_model, val_x, val_target, val_cv, candidates, tie_tolerance=tie_tolerance
        )
        x_train, target_train, _last_train, cv_train = train_samples[horizon]
        model = fit_model_family(
            x_train, target_train, cv_train, max_cells=max_cells, ridge=ridge, radius=radius, top_k=top_k
        )
        model["selected_candidate"] = selected_candidate
        model["validation_losses"] = val_losses
        models[horizon] = model

    metrics = evaluate(test_samples, models)
    return {
        "probe": "phase12c_ltm_world_probe",
        "tar_path": str(tar_path),
        "track_count": len(tracks),
        "sequence_count": len(sequences),
        "fit_sequences": len(fit_sequences),
        "validation_sequences": len(val_sequences),
        "train_sequences": len(train_sequences),
        "test_sequences": len(test_sequences),
        "horizons": list(HORIZONS),
        "stride": stride,
        "max_cells": max_cells,
        "radius": radius,
        "top_k": top_k,
        "tie_tolerance": tie_tolerance,
        "split_seed": split_seed,
        "feature_dim": int(next(iter(train_samples.values()))[0].shape[1]),
        "memory_levels": ["H_fast", "H_event", "H_regime", "H_workspace"],
        "memory_scales": {
            "H_fast": H_FAST_SCALE,
            "H_event": H_EVENT_SCALE,
            "H_regime": H_REGIME_SCALE,
            "H_workspace": H_WORKSPACE_SCALE,
        },
        "test_metrics": metrics,
        "elapsed_seconds": time.time() - started,
    }


def render_report(result: dict[str, Any]) -> str:
    base = render_base_report(result)
    return base.replace(
        "La arquitectura activa combina encoder fisico enriquecido, Ridge global y memorias AMF locales normalizadas para corregir residuales.",
        "La arquitectura activa combina AMF-LTM con H_fast, H_event, H_regime, H_workspace y memorias AMF locales normalizadas para corregir residuales.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate an AMF-LTM PhysicalAI world probe.")
    parser.add_argument("--tar-path", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-cells", type=int, default=20000)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--radius", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--tie-tolerance", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--out-json", default="results/phase12c_ltm_world_probe.json")
    parser.add_argument("--out-report", default="results/FASE12C_LTM_WORLD_PROBE.md")
    args = parser.parse_args()

    result = run_ltm_probe(
        Path(args.tar_path),
        args.train_fraction,
        args.stride,
        args.max_cells,
        args.ridge,
        args.radius,
        args.top_k,
        args.tie_tolerance,
        args.split_seed,
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
