from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from phase12a_physicalai_world_probe import (
    HORIZONS,
    fit_model_family,
    load_tracks,
    mae,
    mse,
    predict_family,
    render_report as render_base_report,
    select_candidate,
    split_train_validation,
)
from phase12a_physicalai_world_probe import make_samples as make_base_samples
from phase12b_identity_world_probe import make_identity_samples
from phase12b_orientation_world_probe import make_orientation_samples


SampleMaker = Callable[[list[Any], set[str], tuple[int, ...], int], dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]


ENCODERS: dict[str, SampleMaker] = {
    "base": make_base_samples,
    "identity": make_identity_samples,
    "orientation": make_orientation_samples,
}


def run_selector_probe(
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
    candidates = ("ridge", "constant_velocity", "cv_amf", "ridge_amf_0.25", "ridge_amf_0.5", "ridge_amf_1.0")

    encoder_samples = {}
    for name, maker in ENCODERS.items():
        encoder_samples[name] = {
            "fit": maker(tracks, fit_sequences, HORIZONS, stride),
            "val": maker(tracks, val_sequences, HORIZONS, stride),
            "train": maker(tracks, train_sequences, HORIZONS, stride),
            "test": maker(tracks, test_sequences, HORIZONS, stride),
        }

    results: dict[str, dict[str, Any]] = {}
    for horizon in HORIZONS:
        validation_records = {}
        selected = None
        selected_loss = float("inf")
        for encoder_name, splits in encoder_samples.items():
            x_fit, target_fit, _last_fit, cv_fit = splits["fit"][horizon]
            val_x, val_target, _val_last, val_cv = splits["val"][horizon]
            selector_model = fit_model_family(
                x_fit, target_fit, cv_fit, max_cells=max_cells, ridge=ridge, radius=radius, top_k=top_k
            )
            candidate, losses = select_candidate(
                selector_model, val_x, val_target, val_cv, candidates, tie_tolerance=tie_tolerance
            )
            validation_records[encoder_name] = {"selected_candidate": candidate, "validation_losses": losses}
            loss = losses[candidate]
            if loss < selected_loss:
                selected_loss = loss
                selected = (encoder_name, candidate)

        if selected is None:
            continue
        encoder_name, candidate = selected
        x_train, target_train, _last_train, cv_train = encoder_samples[encoder_name]["train"][horizon]
        model = fit_model_family(
            x_train, target_train, cv_train, max_cells=max_cells, ridge=ridge, radius=radius, top_k=top_k
        )
        x_test, target_test, last_test, cv_test = encoder_samples[encoder_name]["test"][horizon]
        pred = predict_family(model, x_test, cv_test, candidate)
        ridge_pred = predict_family(model, x_test, cv_test, "ridge")
        selected_mse = mse(pred, target_test)
        ridge_mse = mse(ridge_pred, target_test)
        last_mse = mse(last_test, target_test)
        results[f"h{horizon}"] = {
            "samples": int(len(x_test)),
            "selected_encoder": encoder_name,
            "selected_candidate": candidate,
            "never12b_selector_mse": selected_mse,
            "never12b_selector_mae": mae(pred, target_test),
            "ridge_mse": ridge_mse,
            "last_mse": last_mse,
            "mse_gain_vs_ridge": (ridge_mse - selected_mse) / max(ridge_mse, 1e-9),
            "mse_skill_vs_last": (last_mse - selected_mse) / max(last_mse, 1e-9),
            "validation_records": validation_records,
        }

    return {
        "probe": "phase12b_encoder_selector_probe",
        "tar_path": str(tar_path),
        "track_count": len(tracks),
        "sequence_count": len(sequences),
        "fit_sequences": len(fit_sequences),
        "validation_sequences": len(val_sequences),
        "train_sequences": len(train_sequences),
        "test_sequences": len(test_sequences),
        "horizons": list(HORIZONS),
        "encoders": list(ENCODERS),
        "stride": stride,
        "max_cells": max_cells,
        "radius": radius,
        "top_k": top_k,
        "tie_tolerance": tie_tolerance,
        "split_seed": split_seed,
        "test_metrics": results,
        "elapsed_seconds": time.time() - started,
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Fase 12B - Encoder selector world probe",
        "",
        f"Tar: `{result['tar_path']}`",
        f"Tracks: {result['track_count']}",
        f"Sequences: {result['sequence_count']} ({result['train_sequences']} train / {result['test_sequences']} test)",
        f"Fit/validation/test: {result['fit_sequences']} / {result['validation_sequences']} / {result['test_sequences']} (seed {result['split_seed']})",
        f"Encoders: {', '.join(result['encoders'])}",
        "",
        "## Metrics",
        "",
        "| horizon | encoder | candidate | MSE | MAE | Ridge MSE | gain vs Ridge | skill vs last |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for horizon, metrics in result["test_metrics"].items():
        lines.append(
            f"| {horizon} | {metrics['selected_encoder']} | {metrics['selected_candidate']} | "
            f"{metrics['never12b_selector_mse']:.6f} | {metrics['never12b_selector_mae']:.6f} | "
            f"{metrics['ridge_mse']:.6f} | {metrics['mse_gain_vs_ridge']:.6f} | {metrics['mse_skill_vs_last']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Lectura",
            "",
            "El encoder activo se elige en validacion entre base, identidad de slot y orientacion.",
            "Despues se reentrena el encoder elegido sobre todo train y se evalua en test separado.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Select among AMF PhysicalAI encoders using validation, then evaluate on test.")
    parser.add_argument("--tar-path", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-cells", type=int, default=20000)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--radius", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--tie-tolerance", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--out-json", default="results/phase12b_encoder_selector_probe.json")
    parser.add_argument("--out-report", default="results/FASE12B_ENCODER_SELECTOR_PROBE.md")
    args = parser.parse_args()

    result = run_selector_probe(
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
