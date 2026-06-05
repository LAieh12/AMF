from __future__ import annotations

import argparse
import json
import time
import zlib
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


def object_code(name: str, slot_index: int) -> np.ndarray:
    crc = zlib.crc32(name.encode("utf-8"))
    bits = [1.0 if (crc >> bit) & 1 else -1.0 for bit in range(8)]
    return np.asarray(bits + [slot_index / 16.0], dtype=np.float32)


def base_feature(track: PhysicsTrack, frame: int, horizon: int) -> np.ndarray:
    dt = horizon / FPS
    com = track.com[frame]
    velocity = track.velocity[frame]
    spin = track.spin[frame] if track.spin is not None else np.zeros(3, dtype=np.float32)
    cv = com + velocity * dt
    return np.concatenate(
        [
            com,
            velocity / 10.0,
            spin / 1000.0,
            cv,
            np.asarray(
                [
                    dt,
                    dt * dt,
                    horizon / 30.0,
                    np.linalg.norm(velocity) / 10.0,
                    com[1] / 10.0,
                    np.linalg.norm(com[[0, 2]]) / 10.0,
                ],
                dtype=np.float32,
            ),
            object_code(track.object_name, track.slot_index),
        ]
    ).astype(np.float32)


def contact_features(
    positions: np.ndarray,
    velocities: np.ndarray,
    index: int,
    horizon: int,
) -> np.ndarray:
    if len(positions) <= 1:
        return np.zeros(13, dtype=np.float32)
    rel_pos = positions - positions[index][None, :]
    dists = np.linalg.norm(rel_pos, axis=1)
    dists[index] = np.inf
    nearest = int(np.argmin(dists))
    nearest_dist = float(dists[nearest])
    rel = rel_pos[nearest]
    rel_vel = velocities[nearest] - velocities[index]
    direction = rel / max(nearest_dist, 1e-6)
    closing_speed = -float(np.dot(rel_vel, direction))
    density_025 = float(np.count_nonzero(dists < 0.25))
    density_05 = float(np.count_nonzero(dists < 0.5))
    density_1 = float(np.count_nonzero(dists < 1.0))
    return np.concatenate(
        [
            rel / 10.0,
            rel_vel / 10.0,
            np.asarray(
                [
                    nearest_dist / 10.0,
                    closing_speed / 10.0,
                    density_025 / 16.0,
                    density_05 / 16.0,
                    density_1 / 16.0,
                    min(nearest_dist / max(horizon / FPS, 1e-6), 100.0) / 100.0,
                    1.0,
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)


def make_contact_samples(
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
        if not sequence_tracks:
            continue
        frames = min(min(track.com.shape[0], track.velocity.shape[0]) for track in sequence_tracks)
        for horizon in horizons:
            for frame in range(0, frames - horizon, stride):
                positions = np.stack([track.com[frame] for track in sequence_tracks]).astype(np.float32)
                velocities = np.stack([track.velocity[frame] for track in sequence_tracks]).astype(np.float32)
                for index, track in enumerate(sequence_tracks):
                    com = track.com[frame].astype(np.float32)
                    velocity = track.velocity[frame].astype(np.float32)
                    target = track.com[frame + horizon].astype(np.float32)
                    cv = (com + velocity * (horizon / FPS)).astype(np.float32)
                    feat = np.concatenate([base_feature(track, frame, horizon), contact_features(positions, velocities, index, horizon)])
                    by_h[horizon].append((feat.astype(np.float32), target, com, cv))

    out = {}
    for horizon, rows in by_h.items():
        if not rows:
            out[horizon] = (
                np.zeros((0, 40), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
            )
            continue
        feats, targets, last, cv = zip(*rows)
        out[horizon] = (np.stack(feats), np.stack(targets), np.stack(last), np.stack(cv))
    return out


def run_contact_probe(
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
    fit_samples = make_contact_samples(tracks, fit_sequences, HORIZONS, stride)
    val_samples = make_contact_samples(tracks, val_sequences, HORIZONS, stride)
    train_samples = make_contact_samples(tracks, train_sequences, HORIZONS, stride)
    test_samples = make_contact_samples(tracks, test_sequences, HORIZONS, stride)

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
        "probe": "phase12b_contact_world_probe",
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
        "La arquitectura activa combina encoder fisico enriquecido, contexto de contacto multi-objeto, Ridge global y memorias AMF locales normalizadas para corregir residuales.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate a contact-aware AMF PhysicalAI world probe.")
    parser.add_argument("--tar-path", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-cells", type=int, default=20000)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--radius", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--tie-tolerance", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--out-json", default="results/phase12b_contact_world_probe.json")
    parser.add_argument("--out-report", default="results/FASE12B_CONTACT_WORLD_PROBE.md")
    args = parser.parse_args()

    result = run_contact_probe(
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
