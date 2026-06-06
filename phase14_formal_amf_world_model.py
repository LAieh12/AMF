from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from phase13_scene_loader import SceneShard, load_scene_data
from phase13b_regime_expert_selector import evaluate_scene_selector


ARCHITECTURE_FREEZE = {
    "name": "AMF World Model Phase 14 frozen architecture",
    "frozen_after": "phase13b_regime_expert_selector",
    "components": [
        "AMF residual/local transition cells",
        "temporal-energy",
        "identity/orientation",
        "energy/constraint",
        "AMF-LTM residual",
        "Regime Expert Selector 13B",
        "ridge safety fallback",
    ],
    "forbidden_without_bug": [
        "new dense H_event/H_workspace feature concatenation",
        "new physical features",
        "selector formula changes",
        "residual formula changes",
        "test-driven weight calibration",
    ],
}


def maybe_limit_tracks(scene_data, max_tracks: int | None):
    if not max_tracks or len(scene_data.tracks) <= max_tracks:
        return scene_data
    buckets = [
        scene_data.fit_sequences,
        scene_data.validation_sequences,
        scene_data.test_sequences,
    ]
    per_bucket = max(1, max_tracks // len(buckets))
    limited_tracks = []
    seen_sequences: set[str] = set()
    for bucket in buckets:
        count = 0
        for track in scene_data.tracks:
            if track.sequence not in bucket:
                continue
            limited_tracks.append(track)
            seen_sequences.add(track.sequence)
            count += 1
            if count >= per_bucket:
                break
    if len(limited_tracks) < max_tracks:
        used_ids = {id(track) for track in limited_tracks}
        for track in scene_data.tracks:
            if id(track) in used_ids:
                continue
            limited_tracks.append(track)
            seen_sequences.add(track.sequence)
            if len(limited_tracks) >= max_tracks:
                break
    return replace(
        scene_data,
        tracks=limited_tracks,
        sequences=sorted(seen_sequences),
        fit_sequences=scene_data.fit_sequences & seen_sequences,
        validation_sequences=scene_data.validation_sequences & seen_sequences,
        train_sequences=scene_data.train_sequences & seen_sequences,
        test_sequences=scene_data.test_sequences & seen_sequences,
    )


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_checkpoint(
    checkpoint_dir: Path,
    shard_id: str,
    config: dict[str, Any],
    completed: list[str],
    partial_metrics: list[dict[str, Any]],
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "architecture_freeze": ARCHITECTURE_FREEZE,
        "config": config,
        "completed_shards": completed,
        "partial_metrics": partial_metrics,
        "timestamp": time.time(),
        "checkpoint_contains": [
            "AMF cell counts by horizon",
            "LTM memory counts and memory MB by horizon",
            "selector weights/context bins by horizon",
            "config snapshot",
            "scene/shard progress",
            "partial metrics",
        ],
    }
    shard_path = checkpoint_dir / f"epoch_or_shard_{shard_id}.ckpt"
    latest_path = checkpoint_dir / "latest.ckpt"
    text = json.dumps(checkpoint, indent=2)
    shard_path.write_text(text, encoding="utf-8")
    latest_path.write_text(text, encoding="utf-8")
    return latest_path


def load_checkpoint(checkpoint_dir: Path) -> dict[str, Any] | None:
    latest = checkpoint_dir / "latest.ckpt"
    if not latest.exists():
        return None
    return json.loads(latest.read_text(encoding="utf-8"))


def scene_result_to_log_records(scene_result: dict[str, Any], split: str = "test") -> list[dict[str, Any]]:
    records = []
    for horizon, hrec in scene_result["horizon_results"].items():
        previous_name = hrec["previous_best_model"]
        records.append(
            {
                "time": time.time(),
                "scene": scene_result["scene"],
                "shard": scene_result["tar_path"],
                "split": split,
                "horizon": horizon,
                "model": "regime_expert_selector_13b",
                "expert_selected": max(hrec["selector"]["average_weights_test"], key=hrec["selector"]["average_weights_test"].get),
                "mse": hrec["final_metrics"]["mse"],
                "gain_vs_ridge": hrec["final_metrics"]["gain_vs_ridge"],
                "gain_vs_temporal_energy": hrec["final_metrics"]["gain_vs_temporal_energy"],
                "gain_vs_best_previous_amf": hrec["final_metrics"]["gain_vs_best_previous_amf"],
                "memory_mb": hrec["ltm"]["memory_mb"],
                "cells": hrec["samples"],
                "ltm_memories": hrec["ltm"]["memories_created"],
                "selector_confidence": hrec["ltm"]["confidence_mean_test"],
                "residual_on": hrec["ltm"]["optional_corrected_count"],
                "residual_off": hrec["ltm"]["optional_off_count"],
                "time_seconds": scene_result["elapsed_seconds"],
                "previous_best_model": previous_name,
                "previous_best_mse": hrec["expert_metrics"][previous_name]["mse"],
                "model_export_path": scene_result.get("model_export_path"),
            }
        )
    return records


def run_formal_world_model(config: dict[str, Any], resume: bool = False, stop_after_scenes: int | None = None) -> dict[str, Any]:
    output_paths = config["output_paths"]
    checkpoint_dir = Path(config["checkpoint"]["dir"])
    log_path = Path(output_paths["train_log_jsonl"])
    latest_path = Path(output_paths["latest_json"])
    live_path = Path(output_paths["metrics_live_json"])
    export_dir = Path(output_paths.get("model_export_dir", "models/phase14"))
    export_index_path = export_dir / "model_index.json"
    checkpoint = load_checkpoint(checkpoint_dir) if resume else None
    completed = list(checkpoint.get("completed_shards", [])) if checkpoint else []
    partial_metrics = list(checkpoint.get("partial_metrics", [])) if checkpoint else []
    scene_results = []
    started = time.time()

    for scene_cfg in config["scenes"]:
        scene_name = scene_cfg["scene"]
        shard_path = Path(scene_cfg["shard"])
        shard_id = f"{scene_name}_{shard_path.stem}"
        if shard_id in completed:
            continue
        scene_data = load_scene_data(
            SceneShard(scene=scene_name, tar_path=shard_path, tier=int(scene_cfg.get("tier", 1))),
            train_fraction=float(config["split"]["train_fraction"]),
            split_seed=int(config["random_seed"]),
        )
        scene_data = maybe_limit_tracks(scene_data, config.get("preflight", {}).get("max_tracks_per_scene"))
        scene_result = evaluate_scene_selector(
            scene_data,
            horizons=tuple(int(h) for h in config["horizons"]),
            stride=int(config["training"]["stride"]),
            memory_window=int(config["model"]["memory_window"]),
            max_cells=int(config["model"]["max_cells"]),
            ridge=float(config["model"]["ridge"]),
            model_radius=float(config["model"]["model_radius"]),
            model_top_k=int(config["model"]["model_top_k"]),
            ltm_radius=float(config["model"]["ltm_radius"]),
            ltm_top_k=int(config["model"]["ltm_top_k"]),
            tie_tolerance=float(config["model"]["tie_tolerance"]),
            selector_step=float(config["selector"]["step"]),
            min_group=int(config["selector"]["min_group"]),
            previous_matrix={},
            export_dir=export_dir,
        )
        scene_results.append(scene_result)
        completed.append(shard_id)
        for record in scene_result_to_log_records(scene_result):
            append_jsonl(log_path, record)
            partial_metrics.append(record)
        write_checkpoint(checkpoint_dir, shard_id, config, completed, partial_metrics)
        live = {
            "architecture_freeze": ARCHITECTURE_FREEZE,
            "completed_shards": completed,
            "latest_scene": scene_name,
            "latest_shard": str(shard_path),
            "records": partial_metrics[-len(scene_result["horizon_results"]) :],
            "model_export_path": scene_result.get("model_export_path"),
        }
        live_path.parent.mkdir(parents=True, exist_ok=True)
        live_path.write_text(json.dumps(live, indent=2), encoding="utf-8")
        if stop_after_scenes is not None and len(scene_results) >= stop_after_scenes:
            break

    result = {
        "phase": "14_prepared_or_formal",
        "architecture_freeze": ARCHITECTURE_FREEZE,
        "resume_used": bool(checkpoint),
        "completed_shards": completed,
        "scene_results": scene_results,
        "partial_metrics": partial_metrics,
        "model_export_dir": str(export_dir),
        "model_exports": [
            scene.get("model_export_path")
            for scene in scene_results
            if scene.get("model_export_path")
        ],
        "elapsed_seconds": time.time() - started,
        "no_test_calibration": True,
    }
    export_dir.mkdir(parents=True, exist_ok=True)
    export_index = {
        "format": "phase14_amf_world_model_export_index_v1",
        "architecture_freeze": ARCHITECTURE_FREEZE,
        "completed_shards": completed,
        "exports": result["model_exports"],
        "latest_json": str(latest_path),
        "checkpoint": str(checkpoint_dir / "latest.ckpt"),
    }
    export_index_path.write_text(json.dumps(export_index, indent=2), encoding="utf-8")
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
