from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from phase12a_physicalai_world_probe import (
    HORIZONS,
    PhysicsTrack,
    evaluate,
    fit_model_family,
    load_tracks,
    render_report as render_base_report,
    select_candidate,
    split_train_validation,
)
from phase12c_energy_world_probe import energy_feature


def temporal_scalars(track: PhysicsTrack, frame: int) -> np.ndarray:
    prev = max(0, frame - 1)
    prev2 = max(0, frame - 2)

    com = track.com[frame].astype(np.float32)
    prev_com = track.com[prev].astype(np.float32)
    velocity = track.velocity[frame].astype(np.float32)
    prev_velocity = track.velocity[prev].astype(np.float32)
    prev2_velocity = track.velocity[prev2].astype(np.float32)

    anchor = track.com[0].astype(np.float32)
    rel = com - anchor
    prev_rel = prev_com - anchor
    radius = float(np.linalg.norm(rel))
    prev_radius = float(np.linalg.norm(prev_rel))

    acceleration = velocity - prev_velocity
    jerk = velocity - 2.0 * prev_velocity + prev2_velocity
    radial_dir = rel / max(radius, 1e-6)
    radial_velocity = float(np.dot(velocity, radial_dir))
    prev_radial_dir = prev_rel / max(prev_radius, 1e-6)
    prev_radial_velocity = float(np.dot(prev_velocity, prev_radial_dir))

    speed = float(np.linalg.norm(velocity))
    prev_speed = float(np.linalg.norm(prev_velocity))
    height = float(com[1] - anchor[1])
    prev_height = float(prev_com[1] - anchor[1])
    energy = speed * speed + 9.81 * height
    prev_energy = prev_speed * prev_speed + 9.81 * prev_height

    return np.concatenate(
        [
            prev_velocity / 10.0,
            acceleration / 10.0,
            jerk / 10.0,
            np.asarray(
                [
                    (radius - prev_radius) / 10.0,
                    (radial_velocity - prev_radial_velocity) / 10.0,
                    (energy - prev_energy) / 100.0,
                    float(np.linalg.norm(acceleration)) / 10.0,
                    float(np.linalg.norm(jerk)) / 10.0,
                    1.0,
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)


def temporal_energy_feature(track: PhysicsTrack, frame: int, horizon: int) -> np.ndarray:
    return np.concatenate([energy_feature(track, frame, horizon), temporal_scalars(track, frame)]).astype(np.float32)


def make_temporal_energy_samples(
    tracks: list[PhysicsTrack],
    sequences: set[str],
    horizons: tuple[int, ...],
    stride: int,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    by_h: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {h: [] for h in horizons}
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
                target = track.com[frame + horizon].astype(np.float32)
                cv = (com + velocity * (horizon / 30.0)).astype(np.float32)
                by_h[horizon].append((temporal_energy_feature(track, frame, horizon), target, com, cv))

    out = {}
    for horizon, rows in by_h.items():
        if not rows:
            out[horizon] = (
                np.zeros((0, 68), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
            )
            continue
        feats, targets, last, cv = zip(*rows)
        out[horizon] = (np.stack(feats), np.stack(targets), np.stack(last), np.stack(cv))
    return out


def run_temporal_energy_probe(
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
    fit_samples = make_temporal_energy_samples(tracks, fit_sequences, HORIZONS, stride)
    val_samples = make_temporal_energy_samples(tracks, val_sequences, HORIZONS, stride)
    train_samples = make_temporal_energy_samples(tracks, train_sequences, HORIZONS, stride)
    test_samples = make_temporal_energy_samples(tracks, test_sequences, HORIZONS, stride)

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
        "probe": "phase12c_temporal_energy_world_probe",
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
        "test_metrics": metrics,
        "elapsed_seconds": time.time() - started,
    }


def render_report(result: dict[str, Any]) -> str:
    base = render_base_report(result)
    return base.replace(
        "La arquitectura activa combina encoder fisico enriquecido, Ridge global y memorias AMF locales normalizadas para corregir residuales.",
        "La arquitectura activa combina energia/constraint, historia temporal corta, aceleracion, Ridge global y memorias AMF locales normalizadas para corregir residuales.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate a temporal-energy AMF PhysicalAI world probe.")
    parser.add_argument("--tar-path", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-cells", type=int, default=20000)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--radius", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--tie-tolerance", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--out-json", default="results/phase12c_temporal_energy_world_probe.json")
    parser.add_argument("--out-report", default="results/FASE12C_TEMPORAL_ENERGY_WORLD_PROBE.md")
    args = parser.parse_args()

    result = run_temporal_energy_probe(
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
