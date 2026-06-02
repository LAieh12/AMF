from __future__ import annotations

import argparse
import io
import json
import math
import tarfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


HORIZONS = (1, 5, 15, 30)
FPS = 30.0


@dataclass(frozen=True)
class PhysicsTrack:
    sequence: str
    object_name: str
    slot_index: int
    com: np.ndarray
    velocity: np.ndarray
    spin: np.ndarray | None


def _read_npz(tar: tarfile.TarFile, member: tarfile.TarInfo) -> dict[str, np.ndarray]:
    fileobj = tar.extractfile(member)
    if fileobj is None:
        raise RuntimeError(f"Could not extract {member.name}")
    with np.load(io.BytesIO(fileobj.read()), allow_pickle=False) as npz:
        return {key: np.asarray(npz[key]) for key in npz.files}


def load_tracks(tar_path: Path) -> list[PhysicsTrack]:
    grouped: dict[tuple[str, str], dict[str, np.ndarray]] = defaultdict(dict)
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.endswith(".npz"):
                continue
            sequence, filename = member.name.split("/", 1)
            stem = Path(filename).stem
            if "_" not in stem:
                continue
            object_name, field = stem.rsplit("_", 1)
            if field not in {"com", "velocity", "spin"}:
                continue
            data = _read_npz(tar, member)
            if "data" in data:
                grouped[(sequence, object_name)][field] = np.asarray(data["data"], dtype=np.float32)

    tracks: list[PhysicsTrack] = []
    for (sequence, object_name), fields in grouped.items():
        if "com" not in fields or "velocity" not in fields:
            continue
        com = fields["com"]
        velocity = fields["velocity"]
        spin = fields.get("spin")
        slot_count = min(com.shape[0], velocity.shape[0])
        if spin is not None:
            slot_count = min(slot_count, spin.shape[0])
        for slot_index in range(slot_count):
            tracks.append(
                PhysicsTrack(
                    sequence=sequence,
                    object_name=object_name,
                    slot_index=slot_index,
                    com=com[slot_index],
                    velocity=velocity[slot_index],
                    spin=spin[slot_index] if spin is not None else None,
                )
            )
    return tracks


def _feature(track: PhysicsTrack, frame: int, horizon: int) -> np.ndarray:
    spin = track.spin[frame] if track.spin is not None else np.zeros(3, dtype=np.float32)
    return np.concatenate(
        [
            track.com[frame],
            track.velocity[frame] / 10.0,
            spin / 1000.0,
            np.array([horizon / 30.0, 1.0], dtype=np.float32),
        ]
    ).astype(np.float32)


def make_samples(
    tracks: list[PhysicsTrack], sequences: set[str], horizons: tuple[int, ...], stride: int
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    by_h: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {h: [] for h in horizons}
    for track in tracks:
        if track.sequence not in sequences:
            continue
        frames = min(track.com.shape[0], track.velocity.shape[0])
        for horizon in horizons:
            for frame in range(0, frames - horizon, stride):
                com = track.com[frame]
                velocity = track.velocity[frame]
                target = track.com[frame + horizon]
                cv = com + velocity * (horizon / FPS)
                by_h[horizon].append((_feature(track, frame, horizon), target.astype(np.float32), com.astype(np.float32), cv.astype(np.float32)))
    out = {}
    for horizon, rows in by_h.items():
        if not rows:
            out[horizon] = (
                np.zeros((0, 11), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
            )
            continue
        feats, targets, last, cv = zip(*rows)
        out[horizon] = (np.stack(feats), np.stack(targets), np.stack(last), np.stack(cv))
    return out


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    xtx = x64.T @ x64
    reg = ridge * np.eye(xtx.shape[0], dtype=np.float64)
    rhs = x64.T @ y64
    try:
        weights = np.linalg.solve(xtx + reg, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(xtx + reg) @ rhs
    return weights.astype(np.float32)


class ResidualMemory:
    def __init__(self, radius: float, top_k: int) -> None:
        self.radius = radius
        self.top_k = top_k
        self.x: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def fit(self, x: np.ndarray, residual: np.ndarray, max_cells: int) -> None:
        if len(x) <= max_cells:
            self.x = x.astype(np.float32)
            self.y = residual.astype(np.float32)
            return
        step = max(1, int(math.ceil(len(x) / max_cells)))
        self.x = x[::step].astype(np.float32)
        self.y = residual[::step].astype(np.float32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.x is None or self.y is None or len(self.x) == 0:
            return np.zeros((len(x), 3), dtype=np.float32)
        preds = []
        for row in x:
            dists = np.linalg.norm(self.x - row[None, :], axis=1)
            k = min(self.top_k, len(dists))
            idx = np.argpartition(dists, k - 1)[:k]
            weights = np.exp(-dists[idx] / max(self.radius, 1e-6))
            denom = float(weights.sum())
            preds.append((self.y[idx] * weights[:, None]).sum(axis=0) / max(denom, 1e-9))
        return np.asarray(preds, dtype=np.float32)


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.square(pred - target)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def evaluate(samples: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], models: dict[int, dict[str, Any]]) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for horizon, (x, target, last, cv) in samples.items():
        if len(x) == 0:
            continue
        model = models[horizon]
        ridge_pred = x @ model["ridge"]
        amf_pred = cv + model["memory"].predict(x)
        hybrid = 0.5 * ridge_pred + 0.5 * amf_pred
        candidate_losses = {
            "last": mse(last, target),
            "constant_velocity": mse(cv, target),
            "ridge": mse(ridge_pred, target),
            "amf_residual": mse(amf_pred, target),
            "ridge_amf_mean": mse(hybrid, target),
        }
        best_name = min(candidate_losses.items(), key=lambda item: item[1])[0]
        best_pred = {
            "last": last,
            "constant_velocity": cv,
            "ridge": ridge_pred,
            "amf_residual": amf_pred,
            "ridge_amf_mean": hybrid,
        }[best_name]
        results[f"h{horizon}"] = {
            "samples": int(len(x)),
            "never12a_physics_mse": candidate_losses[best_name],
            "never12a_physics_mae": mae(best_pred, target),
            "best_candidate": best_name,
            "last_mse": candidate_losses["last"],
            "constant_velocity_mse": candidate_losses["constant_velocity"],
            "ridge_mse": candidate_losses["ridge"],
            "amf_residual_mse": candidate_losses["amf_residual"],
            "ridge_amf_mean_mse": candidate_losses["ridge_amf_mean"],
            "mse_skill_vs_last": (candidate_losses["last"] - candidate_losses[best_name]) / max(candidate_losses["last"], 1e-9),
        }
    return results


def run_probe(tar_path: Path, train_fraction: float, stride: int, max_cells: int, ridge: float) -> dict[str, Any]:
    started = time.time()
    tracks = load_tracks(tar_path)
    sequences = sorted({track.sequence for track in tracks})
    split = max(1, min(len(sequences) - 1, int(len(sequences) * train_fraction)))
    train_sequences = set(sequences[:split])
    test_sequences = set(sequences[split:])
    train_samples = make_samples(tracks, train_sequences, HORIZONS, stride)
    test_samples = make_samples(tracks, test_sequences, HORIZONS, stride)

    models: dict[int, dict[str, Any]] = {}
    for horizon, (x, target, _last, cv) in train_samples.items():
        residual = target - cv
        memory = ResidualMemory(radius=0.25, top_k=16)
        memory.fit(x, residual, max_cells=max_cells)
        models[horizon] = {"ridge": fit_ridge(x, target, ridge=ridge), "memory": memory}

    metrics = evaluate(test_samples, models)
    return {
        "probe": "phase12a_physicalai_world_probe",
        "tar_path": str(tar_path),
        "track_count": len(tracks),
        "sequence_count": len(sequences),
        "train_sequences": len(train_sequences),
        "test_sequences": len(test_sequences),
        "horizons": list(HORIZONS),
        "stride": stride,
        "max_cells": max_cells,
        "test_metrics": metrics,
        "elapsed_seconds": time.time() - started,
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Fase 12A - PhysicalAI world probe",
        "",
        f"Tar: `{result['tar_path']}`",
        f"Tracks: {result['track_count']}",
        f"Sequences: {result['sequence_count']} ({result['train_sequences']} train / {result['test_sequences']} test)",
        "",
        "## Metrics",
        "",
        "| horizon | candidate | MSE | MAE | last MSE | skill vs last |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for horizon, metrics in result["test_metrics"].items():
        lines.append(
            f"| {horizon} | {metrics['best_candidate']} | {metrics['never12a_physics_mse']:.6f} | "
            f"{metrics['never12a_physics_mae']:.6f} | {metrics['last_mse']:.6f} | {metrics['mse_skill_vs_last']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Lectura",
            "",
            "Este probe usa ground truth fisico real (`com` y `velocity`) del dataset NVIDIA PhysicalAI.",
            "La primera meta de 12A no es renderizar RGB: es que Never prediga estados fisicos multi-slot con memoria AMF y baselines claros.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate a small Never-style physics world probe on PhysicalAI NPZ tracks.")
    parser.add_argument("--tar-path", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-cells", type=int, default=12000)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--out-json", default="results/phase12a_physicalai_world_probe.json")
    parser.add_argument("--out-report", default="results/FASE12A_PHYSICALAI_WORLD_PROBE.md")
    args = parser.parse_args()

    result = run_probe(Path(args.tar_path), args.train_fraction, args.stride, args.max_cells, args.ridge)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(render_report(result), encoding="utf-8")
    print(json.dumps({"out_json": str(out_json), "out_report": str(out_report)}, indent=2))


if __name__ == "__main__":
    main()
