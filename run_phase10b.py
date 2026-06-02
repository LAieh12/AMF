from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from phase10a_amf_world_model import AMFDynamicsWorldModel
from phase10a_toy_simulator import generate_trajectories, split_trajectories, transitions_from_trajectories
from phase10b_visual_codec import VisualWorldCodec, mask_iou, pixel_mse


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.square(a.astype(np.float32) - b.astype(np.float32))))


def evaluate_codec_resolution(
    resolution: int,
    model: AMFDynamicsWorldModel,
    transitions: list[Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    codec = VisualWorldCodec(resolution=resolution, dt=args.dt)
    rng = np.random.default_rng(args.seed + resolution)
    rows = []
    encode_seconds = 0.0
    decode_seconds = 0.0
    amf_seconds = 0.0
    sample_frames: dict[str, np.ndarray] = {}
    for i, transition in enumerate(transitions[: args.eval_transitions]):
        radius = 0.055 + 0.015 * ((transition.trajectory_id % 3) / 2.0)
        shape_code = 0.25 if transition.trajectory_id % 2 == 0 else 1.0
        visual = codec.render_from_state(transition.state, rng=rng, radius=radius, shape_code=shape_code)
        next_latent_true = codec.state_to_latent(transition.next_state, radius=radius, shape_code=shape_code)
        next_clean = codec.decode(next_latent_true)

        start = time.perf_counter()
        latent = codec.encode(visual.frame)
        encode_seconds += time.perf_counter() - start

        start = time.perf_counter()
        reconstructed = codec.decode(latent)
        decode_seconds += time.perf_counter() - start

        start = time.perf_counter()
        predicted_state = model.predict_next(latent[:4], transition.action)
        amf_seconds += time.perf_counter() - start

        start = time.perf_counter()
        predicted_latent, predicted_frame = codec.decode_predicted(latent, predicted_state)
        decode_seconds += time.perf_counter() - start

        true_latent = codec.state_to_latent(transition.state, radius=radius, shape_code=shape_code)
        rows.append(
            {
                "latent_encode_mse": mse(latent[:4], true_latent[:4]),
                "identity_mse": mse(latent[4:6], true_latent[4:6]),
                "reconstruction_mse": pixel_mse(reconstructed, visual.clean_frame),
                "reconstruction_iou": mask_iou(reconstructed, visual.clean_frame),
                "predicted_latent_mse": mse(predicted_latent[:4], next_latent_true[:4]),
                "predicted_frame_mse": pixel_mse(predicted_frame, next_clean),
                "predicted_frame_iou": mask_iou(predicted_frame, next_clean),
            }
        )
        if i == 0:
            sample_frames = {
                "input_noisy": visual.frame,
                "input_clean": visual.clean_frame,
                "reconstructed": reconstructed,
                "next_clean": next_clean,
                "predicted_next": predicted_frame,
            }
    frame_bytes = codec.frame_bytes()
    latent_bytes = codec.latent_bytes()
    raw_pixel_amf_mb_est = model.centers.shape[0] * (2 * frame_bytes + 2 * 4 + 4) / (1024.0 * 1024.0)
    codec_path = codec.export(args.export_dir)
    return {
        "resolution": resolution,
        "latent_dim": 8,
        "latent_bytes": latent_bytes,
        "frame_bytes": frame_bytes,
        "compression_ratio": frame_bytes / latent_bytes,
        "eval_transitions": len(rows),
        "latent_encode_mse": mean([row["latent_encode_mse"] for row in rows]),
        "identity_mse": mean([row["identity_mse"] for row in rows]),
        "reconstruction_mse": mean([row["reconstruction_mse"] for row in rows]),
        "reconstruction_iou": mean([row["reconstruction_iou"] for row in rows]),
        "predicted_latent_mse": mean([row["predicted_latent_mse"] for row in rows]),
        "predicted_frame_mse": mean([row["predicted_frame_mse"] for row in rows]),
        "predicted_frame_iou": mean([row["predicted_frame_iou"] for row in rows]),
        "encode_ms": 1000.0 * encode_seconds / max(1, len(rows)),
        "decode_ms": 1000.0 * decode_seconds / max(1, len(rows)),
        "amf_predict_ms": 1000.0 * amf_seconds / max(1, len(rows)),
        "amf_cells": int(model.centers.shape[0]),
        "amf_memory_mb": model.memory_mb(),
        "raw_pixel_amf_memory_mb_est": raw_pixel_amf_mb_est,
        "raw_pixel_memory_multiplier": raw_pixel_amf_mb_est / max(model.memory_mb(), 1e-9),
        "codec_metadata": codec_path,
        "sample_frames": sample_frames,
    }


def evaluate_invariants(resolution: int, args: argparse.Namespace) -> dict[str, float]:
    codec = VisualWorldCodec(resolution=resolution, dt=args.dt, noise_std=0.010)
    rng = np.random.default_rng(args.seed + 404)

    base_state = np.array([-0.55, -0.15, 0.35, 0.12], dtype=np.float32)
    shifted_state = np.array([0.35, 0.45, 0.35, 0.12], dtype=np.float32)
    base = codec.encode(codec.render_from_state(base_state, rng=rng, radius=0.065, shape_code=0.25).frame)
    shifted = codec.encode(codec.render_from_state(shifted_state, rng=rng, radius=0.065, shape_code=0.25).frame)
    expected_shift = shifted_state[:2] - base_state[:2]
    measured_shift = shifted[:2] - base[:2]

    object_a = codec.encode(codec.render_from_state(base_state, rng=rng, radius=0.055, shape_code=0.25).frame)
    object_b = codec.encode(codec.render_from_state(base_state, rng=rng, radius=0.095, shape_code=1.0).frame)
    separability = float(np.linalg.norm(object_a[4:6] - object_b[4:6]))

    latent_steps = []
    state = np.array([-0.65, 0.55, 0.12, 0.05], dtype=np.float32)
    action = np.array([0.00, 0.35], dtype=np.float32)
    from phase10a_toy_simulator import ToyGravityBounceSimulator

    simulator = ToyGravityBounceSimulator(dt=args.dt)
    for _ in range(18):
        latent_steps.append(codec.encode(codec.render_from_state(state, rng=rng, radius=0.065, shape_code=0.25).frame))
        state = simulator.step(state, action)
    diffs = [float(np.linalg.norm(b[:4] - a[:4])) for a, b in zip(latent_steps, latent_steps[1:])]
    return {
        "permanence_identity_distance": float(np.linalg.norm(base[4:6] - shifted[4:6])),
        "permanence_shift_error": float(np.linalg.norm(measured_shift - expected_shift)),
        "separability_identity_distance": separability,
        "continuity_mean_step": mean(diffs),
        "continuity_max_step": max(diffs) if diffs else 0.0,
    }


def run_phase10b(args: argparse.Namespace) -> dict[str, Any]:
    model = AMFDynamicsWorldModel.load(args.amf_npz)
    trajectories = generate_trajectories(args.trajectories, args.steps, seed=args.seed)
    _, test_trajectories = split_trajectories(trajectories, test_fraction=0.35)
    transitions = transitions_from_trajectories(test_trajectories)
    results = []
    sample_npz: dict[str, np.ndarray] = {}
    for resolution in args.resolutions:
        record = evaluate_codec_resolution(resolution, model, transitions, args)
        if resolution == max(args.resolutions):
            for key, value in record["sample_frames"].items():
                sample_npz[f"{resolution}_{key}"] = value
        record.pop("sample_frames", None)
        results.append(record)
    invariants = evaluate_invariants(max(args.resolutions), args)
    sample_path = Path(args.export_dir) / "phase10b_sample_frames.npz"
    Path(args.export_dir).mkdir(exist_ok=True)
    np.savez_compressed(sample_path, **sample_npz)
    return {
        "title": "Phase 10b - Visual latent codec for AMF world model",
        "amf_npz": args.amf_npz,
        "rules": {
            "encoder_decoder_no_backprop": True,
            "amf_sees_latent_not_pixels": True,
            "latent_dim_constant": 8,
            "physically_consistent_not_photorealistic": True,
        },
        "data": {
            "trajectories": args.trajectories,
            "steps": args.steps,
            "eval_transitions_per_resolution": args.eval_transitions,
            "resolutions": args.resolutions,
        },
        "invariants": invariants,
        "scaling_results": results,
        "sample_frames_npz": str(sample_path),
    }


def _fmt(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_reports(results: dict[str, Any], out_dir: str | Path = "results") -> None:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    (out / "phase10b_latest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    rows = [
        "| res | frame KB | latent bytes | compression | enc ms | dec ms | AMF ms | recon MSE | recon IoU | pred latent MSE | pred frame MSE | pred IoU | raw pixel AMF est MB | actual AMF MB |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results["scaling_results"]:
        rows.append(
            f"| {row['resolution']} | {_fmt(row['frame_bytes'] / 1024.0)} | {row['latent_bytes']} | "
            f"{_fmt(row['compression_ratio'])}x | {_fmt(row['encode_ms'])} | {_fmt(row['decode_ms'])} | "
            f"{_fmt(row['amf_predict_ms'])} | {_fmt(row['reconstruction_mse'])} | {_fmt(row['reconstruction_iou'])} | "
            f"{_fmt(row['predicted_latent_mse'])} | {_fmt(row['predicted_frame_mse'])} | {_fmt(row['predicted_frame_iou'])} | "
            f"{_fmt(row['raw_pixel_amf_memory_mb_est'])} | {_fmt(row['amf_memory_mb'])} |"
        )
    inv = results["invariants"]
    report = f"""# Fase 10b - Encoder/Decoder visual para AMF world model

Objetivo: comprimir frames visuales grandes a un vector latente compacto `S(t)`,
usar el AMF calentado para predecir `S(t+1)` y decodificar un frame fisicamente
coherente sin hacer que AMF cargue pixeles.

Flujo:

```text
frame visual -> encoder -> S(t) -> AMF world model -> S(t+1) -> decoder -> frame futuro
```

Reglas: encoder/decoder sin backprop = {results['rules']['encoder_decoder_no_backprop']},
AMF ve latent y no pixeles = {results['rules']['amf_sees_latent_not_pixels']},
latent dim constante = {results['rules']['latent_dim_constant']}.

## Escalabilidad por resolucion

{chr(10).join(rows)}

## Propiedades del latent

- permanencia identity distance: {inv['permanence_identity_distance']:.6f}
- permanencia shift error: {inv['permanence_shift_error']:.6f}
- separabilidad identity distance: {inv['separability_identity_distance']:.6f}
- continuidad mean step: {inv['continuity_mean_step']:.6f}
- continuidad max step: {inv['continuity_max_step']:.6f}

## Export

- AMF usado: `{results['amf_npz']}`
- sample frames: `{results['sample_frames_npz']}`
- codec metadata por resolucion en `data/phase10b_visual_codec_<res>.json`

## Lectura

La escala visual ya no entra al AMF como pixeles. Un frame `256x256x4` pesa
1048576 bytes en float32, pero el estado que ve AMF pesa 32 bytes. El estimado
de memoria si AMF guardara celdas crudas de pixeles llega a miles de MB, contra
menos de 1 MB de arrays del AMF latent actual. El encoder conserva posicion,
velocidad, identidad visual, distancia a paredes y continuidad; el decoder no
busca fotorrealismo, sino coherencia fisica para el mundo simulado.
"""
    (out / "FASE10B_RESULTADOS.md").write_text(report, encoding="utf-8")
    biggest = results["scaling_results"][-1]
    complete = f"""# FASE10B_COMPLETADA

Fase 10b implementa el encoder/decoder visual para el AMF world model.

Entregables:

- `phase10b_visual_codec.py`
- `run_phase10b.py`
- `results/phase10b_latest.json`
- `results/FASE10B_RESULTADOS.md`
- `data/phase10b_sample_frames.npz`
- `data/phase10b_visual_codec_*.json`

Resultado en resolucion maxima ({biggest['resolution']}):

- latent_dim: {biggest['latent_dim']}
- frame_bytes: {biggest['frame_bytes']}
- latent_bytes: {biggest['latent_bytes']}
- compression_ratio: {biggest['compression_ratio']:.1f}x
- reconstruction_iou: {biggest['reconstruction_iou']:.4f}
- predicted_frame_iou: {biggest['predicted_frame_iou']:.4f}
- predicted_latent_mse: {biggest['predicted_latent_mse']:.6f}
- AMF predict ms: {biggest['amf_predict_ms']:.6f}
- raw_pixel_amf_memory_mb_est: {biggest['raw_pixel_amf_memory_mb_est']:.2f}
- actual_amf_memory_mb: {biggest['amf_memory_mb']:.6f}

Invariantes:

- permanence_identity_distance: {inv['permanence_identity_distance']:.6f}
- permanence_shift_error: {inv['permanence_shift_error']:.6f}
- separability_identity_distance: {inv['separability_identity_distance']:.6f}
- continuity_mean_step: {inv['continuity_mean_step']:.6f}
- continuity_max_step: {inv['continuity_max_step']:.6f}

El AMF ya no escala con pixeles: escala con `S(t)` compacto.
"""
    Path("FASE10B_COMPLETADA.md").write_text(complete, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 10b visual latent codec benchmark.")
    parser.add_argument("--amf-npz", default="data/phase10a_warm_amf.npz")
    parser.add_argument("--trajectories", type=int, default=180)
    parser.add_argument("--steps", type=int, default=65)
    parser.add_argument("--eval-transitions", type=int, default=900)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--seed", type=int, default=2201)
    parser.add_argument("--dt", type=float, default=0.08)
    parser.add_argument("--export-dir", default="data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_phase10b(args)
    write_reports(results)
    print("report: results/FASE10B_RESULTADOS.md")
    for row in results["scaling_results"]:
        print(
            f"res={row['resolution']} latent={row['latent_dim']} compression={row['compression_ratio']:.1f}x "
            f"pred_iou={row['predicted_frame_iou']:.4f} amf_ms={row['amf_predict_ms']:.4f} "
            f"raw_est={row['raw_pixel_amf_memory_mb_est']:.1f}MB actual={row['amf_memory_mb']:.3f}MB"
        )


if __name__ == "__main__":
    main()
