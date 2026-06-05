from __future__ import annotations

import argparse
import io
import json
import tarfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors


HORIZONS = (1, 5, 15, 30)
FPS = 30.0
FEATURE_DIM = 18


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
            np.array(
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
                np.zeros((0, FEATURE_DIM), dtype=np.float32),
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
        self.center: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.nn: NearestNeighbors | None = None

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.center is None or self.scale is None:
            raise RuntimeError("ResidualMemory must be fitted before predict")
        return ((x - self.center) / self.scale).astype(np.float32)

    def fit(self, x: np.ndarray, residual: np.ndarray, max_cells: int) -> None:
        if len(x) == 0:
            self.x = np.zeros((0, FEATURE_DIM), dtype=np.float32)
            self.y = np.zeros((0, 3), dtype=np.float32)
            return

        self.center = np.median(x, axis=0).astype(np.float32)
        scale = np.percentile(np.abs(x - self.center), 75, axis=0).astype(np.float32)
        self.scale = np.maximum(scale, 1e-4)

        if len(x) > max_cells:
            uniform_count = max_cells // 2
            hard_count = max_cells - uniform_count
            uniform_idx = np.linspace(0, len(x) - 1, uniform_count, dtype=np.int64)
            hard_score = np.linalg.norm(residual, axis=1)
            hard_idx = np.argpartition(hard_score, -hard_count)[-hard_count:]
            idx = np.unique(np.concatenate([uniform_idx, hard_idx]))
            if len(idx) > max_cells:
                idx = idx[:max_cells]
            x_fit = x[idx]
            y_fit = residual[idx]
        else:
            x_fit = x
            y_fit = residual

        self.x = self._normalize(x_fit)
        self.y = y_fit.astype(np.float32)
        k = min(self.top_k, len(self.x))
        self.nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
        self.nn.fit(self.x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.x is None or self.y is None or len(self.x) == 0 or self.nn is None:
            return np.zeros((len(x), 3), dtype=np.float32)
        x_norm = self._normalize(x)
        dists, idx = self.nn.kneighbors(x_norm, return_distance=True)
        weights = np.exp(-dists / max(self.radius, 1e-6)).astype(np.float32)
        denom = np.maximum(weights.sum(axis=1, keepdims=True), 1e-9)
        return ((self.y[idx] * weights[:, :, None]).sum(axis=1) / denom).astype(np.float32)


def split_train_validation(sequences: list[str], train_fraction: float, split_seed: int) -> tuple[set[str], set[str], set[str]]:
    shuffled = list(sequences)
    rng = np.random.default_rng(split_seed)
    rng.shuffle(shuffled)
    split = max(1, min(len(shuffled) - 1, int(len(shuffled) * train_fraction)))
    train_all = shuffled[:split]
    test_sequences = set(shuffled[split:])
    val_count = max(1, min(len(train_all) - 1, int(len(train_all) * 0.2)))
    fit_sequences = set(train_all[:-val_count])
    val_sequences = set(train_all[-val_count:])
    return fit_sequences, val_sequences, set(train_all), test_sequences


def fit_model_family(
    x: np.ndarray,
    target: np.ndarray,
    cv: np.ndarray,
    max_cells: int,
    ridge: float,
    radius: float,
    top_k: int,
) -> dict[str, Any]:
    ridge_weights = fit_ridge(x, target, ridge=ridge)
    ridge_pred = x @ ridge_weights

    cv_memory = ResidualMemory(radius=radius, top_k=top_k)
    cv_memory.fit(x, target - cv, max_cells=max_cells)

    ridge_memory = ResidualMemory(radius=radius, top_k=top_k)
    ridge_memory.fit(x, target - ridge_pred, max_cells=max_cells)
    return {
        "ridge": ridge_weights,
        "cv_memory": cv_memory,
        "ridge_memory": ridge_memory,
    }


def predict_family(model: dict[str, Any], x: np.ndarray, cv: np.ndarray, candidate: str) -> np.ndarray:
    ridge_pred = x @ model["ridge"]
    if candidate == "ridge":
        return ridge_pred
    if candidate == "constant_velocity":
        return cv
    if candidate == "cv_amf":
        return cv + model["cv_memory"].predict(x)
    if candidate.startswith("ridge_amf_"):
        alpha = float(candidate.rsplit("_", 1)[-1])
        return ridge_pred + alpha * model["ridge_memory"].predict(x)
    raise KeyError(candidate)


def select_candidate(
    model: dict[str, Any],
    x: np.ndarray,
    target: np.ndarray,
    cv: np.ndarray,
    candidates: tuple[str, ...],
    tie_tolerance: float,
) -> tuple[str, dict[str, float]]:
    losses = {name: mse(predict_family(model, x, cv, name), target) for name in candidates}
    best_name, best = min(losses.items(), key=lambda item: item[1])
    threshold = best * (1.0 + tie_tolerance)
    if best_name == "constant_velocity":
        for preferred in ("ridge", "ridge_amf_0.25", "ridge_amf_0.5"):
            if preferred in losses and losses[preferred] <= threshold:
                return preferred, losses
    return best_name, losses


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.square(pred - target)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def evaluate(
    samples: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    models: dict[int, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for horizon, (x, target, last, cv) in samples.items():
        if len(x) == 0:
            continue
        model = models[horizon]
        selected = model["selected_candidate"]
        pred = predict_family(model, x, cv, selected)
        ridge_pred = predict_family(model, x, cv, "ridge")
        cv_amf_pred = predict_family(model, x, cv, "cv_amf")
        ridge_amf_pred = predict_family(model, x, cv, "ridge_amf_1.0")
        candidate_losses = {
            "last": mse(last, target),
            "constant_velocity": mse(cv, target),
            "ridge": mse(ridge_pred, target),
            "cv_amf": mse(cv_amf_pred, target),
            "ridge_amf_0.25": mse(predict_family(model, x, cv, "ridge_amf_0.25"), target),
            "ridge_amf_0.5": mse(predict_family(model, x, cv, "ridge_amf_0.5"), target),
            "ridge_amf_1.0": mse(ridge_amf_pred, target),
        }
        results[f"h{horizon}"] = {
            "samples": int(len(x)),
            "never12a_physics_mse": candidate_losses[selected],
            "never12a_physics_mae": mae(pred, target),
            "selected_candidate": selected,
            "validation_losses": model["validation_losses"],
            "last_mse": candidate_losses["last"],
            "constant_velocity_mse": candidate_losses["constant_velocity"],
            "ridge_mse": candidate_losses["ridge"],
            "cv_amf_mse": candidate_losses["cv_amf"],
            "ridge_amf_0.25_mse": candidate_losses["ridge_amf_0.25"],
            "ridge_amf_0.5_mse": candidate_losses["ridge_amf_0.5"],
            "ridge_amf_1.0_mse": candidate_losses["ridge_amf_1.0"],
            "mse_skill_vs_last": (candidate_losses["last"] - candidate_losses[selected]) / max(candidate_losses["last"], 1e-9),
            "mse_gain_vs_ridge": (candidate_losses["ridge"] - candidate_losses[selected]) / max(candidate_losses["ridge"], 1e-9),
        }
    return results


def run_probe(
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
    fit_samples = make_samples(tracks, fit_sequences, HORIZONS, stride)
    val_samples = make_samples(tracks, val_sequences, HORIZONS, stride)
    train_samples = make_samples(tracks, train_sequences, HORIZONS, stride)
    test_samples = make_samples(tracks, test_sequences, HORIZONS, stride)

    models: dict[int, dict[str, Any]] = {}
    candidates = ("ridge", "constant_velocity", "cv_amf", "ridge_amf_0.25", "ridge_amf_0.5", "ridge_amf_1.0")
    for horizon, (x_fit, target_fit, _last_fit, cv_fit) in fit_samples.items():
        val_x, val_target, _val_last, val_cv = val_samples[horizon]
        selector_model = fit_model_family(x_fit, target_fit, cv_fit, max_cells=max_cells, ridge=ridge, radius=radius, top_k=top_k)
        selected_candidate, val_losses = select_candidate(
            selector_model, val_x, val_target, val_cv, candidates, tie_tolerance=tie_tolerance
        )

        x_train, target_train, _last_train, cv_train = train_samples[horizon]
        model = fit_model_family(x_train, target_train, cv_train, max_cells=max_cells, ridge=ridge, radius=radius, top_k=top_k)
        model["selected_candidate"] = selected_candidate
        model["validation_losses"] = val_losses
        models[horizon] = model

    metrics = evaluate(test_samples, models)
    return {
        "probe": "phase12a_physicalai_world_probe",
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
        "test_metrics": metrics,
        "elapsed_seconds": time.time() - started,
    }


def render_report(result: dict[str, Any]) -> str:
    tar_path = str(result["tar_path"])
    phase = "12B" if any(scene in tar_path for scene in ("bowling", "dominoes", "rolling_ramp", "obstruction")) else "12A"
    lines = [
        f"# Fase {phase} - PhysicalAI world probe",
        "",
        f"Tar: `{tar_path}`",
        f"Tracks: {result['track_count']}",
        f"Sequences: {result['sequence_count']} ({result['train_sequences']} train / {result['test_sequences']} test)",
        f"Fit/validation/test: {result['fit_sequences']} / {result['validation_sequences']} / {result['test_sequences']} (seed {result['split_seed']})",
        "",
        "## Metrics",
        "",
        "| horizon | selected | MSE | MAE | last MSE | Ridge MSE | gain vs Ridge | skill vs last |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon, metrics in result["test_metrics"].items():
        lines.append(
            f"| {horizon} | {metrics['selected_candidate']} | {metrics['never12a_physics_mse']:.6f} | "
            f"{metrics['never12a_physics_mae']:.6f} | {metrics['last_mse']:.6f} | {metrics['ridge_mse']:.6f} | "
            f"{metrics['mse_gain_vs_ridge']:.6f} | {metrics['mse_skill_vs_last']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Selector",
            "",
            "El candidato activo se elige en validacion y luego se reentrena sobre todo el bloque train antes del test.",
            "Si velocidad constante gana validacion pero Ridge o Ridge+AMF quedan cerca, el selector prefiere el candidato global/interpolado mas estable.",
            "Esto evita elegir el mejor metodo mirando el test y hace mas justa la comparacion contra Ridge.",
            "",
            "## Lectura",
            "",
            "Este probe usa ground truth fisico real (`com` y `velocity`) del dataset NVIDIA PhysicalAI.",
            "La arquitectura activa combina encoder fisico enriquecido, Ridge global y memorias AMF locales normalizadas para corregir residuales.",
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
    parser.add_argument("--radius", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--tie-tolerance", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--out-json", default="results/phase12a_physicalai_world_probe.json")
    parser.add_argument("--out-report", default="results/FASE12A_PHYSICALAI_WORLD_PROBE.md")
    args = parser.parse_args()

    result = run_probe(
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
