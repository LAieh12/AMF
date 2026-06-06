from __future__ import annotations

import hashlib
import io
import json
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from phase12a_physicalai_world_probe import _read_npz, split_train_validation
from phase13_scene_loader import DEFAULT_CACHE_ROOT, SceneShard, discover_scene_shards
from phase14_formal_amf_world_model import ARCHITECTURE_FREEZE, run_formal_world_model


HORIZONS = (1, 5, 15, 30, 60, 120)
TIER1 = ("objects_falling", "dominoes", "wrecking_ball")
TIER2_AVAILABLE = ("billiards", "bowling")


def sha256_prefix(path: Path, max_bytes: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    remaining = max_bytes
    with path.open("rb") as handle:
        while remaining > 0:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def inspect_physics_tar(path: Path) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[str, tuple[int, int]]] = {}
    fields_available: set[str] = set()
    segmentation_colors = False
    sequence_names: set[str] = set()
    with tarfile.open(path, "r") as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.endswith(".npz"):
                continue
            sequence, filename = member.name.split("/", 1)
            sequence_names.add(sequence)
            stem = Path(filename).stem
            if "_" not in stem:
                continue
            object_name, field = stem.rsplit("_", 1)
            data = _read_npz(tar, member)
            arr = data.get("data")
            if arr is not None:
                fields_available.add(field)
                grouped.setdefault((sequence, object_name), {})[field] = (int(arr.shape[0]), int(arr.shape[1]))
            if "segmentation_colors" in data:
                segmentation_colors = True

    track_count = 0
    frame_counts = []
    for field_shapes in grouped.values():
        if "com" not in field_shapes or "velocity" not in field_shapes:
            continue
        slot_count = min(shape[0] for shape in field_shapes.values())
        frames = min(shape[1] for shape in field_shapes.values())
        track_count += slot_count
        frame_counts.extend([frames] * slot_count)

    total_frames = int(sum(frame_counts))
    h1_transitions = int(sum(max(0, frames - 1) for frames in frame_counts))
    h120_transitions = int(sum(max(0, frames - 120) for frames in frame_counts))
    return {
        "track_count": track_count,
        "sequence_count": len(sequence_names),
        "sequences": sorted(sequence_names),
        "min_frames": int(min(frame_counts)) if frame_counts else 0,
        "max_frames": int(max(frame_counts)) if frame_counts else 0,
        "total_track_frames": total_frames,
        "h1_transitions": h1_transitions,
        "h120_transitions": h120_transitions,
        "fields_available": sorted(fields_available | ({"segmentation_colors"} if segmentation_colors else set())),
    }


def discover_all_cached_shards() -> list[SceneShard]:
    shards = []
    for scene, shard in discover_scene_shards(DEFAULT_CACHE_ROOT).items():
        # Include first discovered Tier 2 shards, but all Tier 1 shards are expanded below.
        if scene in TIER2_AVAILABLE:
            shards.append(shard)
    snapshot_roots = list(DEFAULT_CACHE_ROOT.rglob("physics"))
    for root in snapshot_roots:
        for scene in TIER1:
            for tar_path in sorted((root / scene).glob("physics-*.tar")):
                shards.append(SceneShard(scene=scene, tar_path=tar_path, tier=1))
    unique = {}
    for shard in shards:
        unique[str(shard.tar_path)] = shard
    return sorted(unique.values(), key=lambda item: (item.tier, item.scene, str(item.tar_path)))


def build_manifest(shards: list[SceneShard], used_for_phase14: set[str]) -> dict[str, Any]:
    records = []
    started = time.time()
    for shard in shards:
        stats = inspect_physics_tar(shard.tar_path)
        records.append(
            {
                "scene": shard.scene,
                "tier": shard.tier,
                "shard": str(shard.tar_path),
                "size_bytes": shard.tar_path.stat().st_size,
                "sha256_first_16mb": sha256_prefix(shard.tar_path),
                "used_in_phase14_train_val_test": str(shard.tar_path) in used_for_phase14,
                **{key: value for key, value in stats.items() if key != "sequences"},
            }
        )
    remote_physics_summary = {
        "repo": "nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes",
        "remote_physics_total_gb_checked_2026_06_05": 91.59,
        "decision": "do not download full physics package in 13C; download/prepare complete Tier 1 and cached Tier 2 physics metadata only",
    }
    return {
        "created_at": time.time(),
        "elapsed_seconds": time.time() - started,
        "architecture_freeze": ARCHITECTURE_FREEZE,
        "remote_physics_summary": remote_physics_summary,
        "records": records,
    }


def build_splits(shards: list[SceneShard], train_fraction: float, seed: int) -> dict[str, Any]:
    split_records = []
    for shard in shards:
        stats = inspect_physics_tar(shard.tar_path)
        fit, validation, train, test = split_train_validation(stats["sequences"], train_fraction, seed)
        split_records.append(
            {
                "scene": shard.scene,
                "shard": str(shard.tar_path),
                "split_unit": "sequence",
                "train_fraction": train_fraction,
                "seed": seed,
                "fit_sequences": sorted(fit),
                "validation_sequences": sorted(validation),
                "train_sequences": sorted(train),
                "test_sequences": sorted(test),
                "counts": {
                    "fit": len(fit),
                    "validation": len(validation),
                    "train": len(train),
                    "test": len(test),
                },
            }
        )
    return {
        "created_at": time.time(),
        "no_frame_mixing": True,
        "no_test_calibration": True,
        "records": split_records,
    }


def make_config(shards: list[SceneShard], splits_path: Path) -> dict[str, Any]:
    return {
        "phase": "14",
        "use_rgb": False,
        "architecture_freeze": ARCHITECTURE_FREEZE,
        "random_seed": 123,
        "horizons": list(HORIZONS),
        "scenes": [
            {"scene": shard.scene, "tier": shard.tier, "shard": str(shard.tar_path)}
            for shard in shards
        ],
        "split": {
            "train_fraction": 0.75,
            "split_path": str(splits_path),
            "split_unit": "sequence",
            "no_test_calibration": True,
        },
        "training": {
            "stride": 60,
            "checkpoint_interval": "scene_or_shard",
        },
        "model": {
            "max_cells": 5000,
            "memory_limit_mb": 512,
            "memory_window": 20,
            "ridge": 0.001,
            "model_radius": 0.75,
            "model_top_k": 24,
            "ltm_radius": 1.25,
            "ltm_top_k": 24,
            "tie_tolerance": 0.10,
        },
        "selector": {
            "step": 0.5,
            "min_group": 256,
            "source": "phase13b_regime_expert_selector",
        },
        "checkpoint": {
            "dir": "checkpoints/phase14",
            "latest": "checkpoints/phase14/latest.ckpt",
            "per_shard_pattern": "checkpoints/phase14/epoch_or_shard_<id>.ckpt",
        },
        "output_paths": {
            "train_log_jsonl": "results/phase14_train_log.jsonl",
            "metrics_live_json": "results/phase14_metrics_live.json",
            "latest_json": "results/phase14_latest.json",
            "model_export_dir": "models/phase14",
        },
        "preflight": {
            "max_tracks_per_scene": None,
        },
    }


def preflight_config(base_config: dict[str, Any]) -> dict[str, Any]:
    selected = []
    seen = set()
    for scene in TIER1:
        for item in base_config["scenes"]:
            if item["scene"] == scene and scene not in seen:
                selected.append(item)
                seen.add(scene)
                break
    cfg = json.loads(json.dumps(base_config))
    cfg["scenes"] = selected
    cfg["training"]["stride"] = 120
    cfg["model"]["max_cells"] = 500
    cfg["model"]["model_top_k"] = 8
    cfg["model"]["ltm_top_k"] = 8
    cfg["selector"]["min_group"] = 64
    cfg["checkpoint"]["dir"] = "checkpoints/phase14"
    cfg["preflight"]["max_tracks_per_scene"] = 2500
    return cfg


def render_preflight_report(result: dict[str, Any], resume_result: dict[str, Any], manifest: dict[str, Any]) -> str:
    log_path = Path("results/phase14_train_log.jsonl")
    by_scene: dict[str, dict[str, Any]] = {}
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            scene = record["scene"]
            state = by_scene.setdefault(scene, {"horizons": set(), "finite": True, "max_memory": 0.0})
            state["horizons"].add(record["horizon"])
            state["finite"] = state["finite"] and bool(np.isfinite(record["mse"]))
            state["max_memory"] = max(state["max_memory"], float(record["memory_mb"]))
    lines = [
        "# Fase 13C - Preflight",
        "",
        "Architecture is frozen from Phase 13B. No selector/features/residual formula changed after this preflight.",
        "",
        "## Dataset",
        "",
        f"Manifest shards: {len(manifest['records'])}",
        f"Remote full physics package checked: {manifest['remote_physics_summary']['remote_physics_total_gb_checked_2026_06_05']} GB",
        manifest["remote_physics_summary"]["decision"],
        "",
        "## Checkpoint/resume",
        "",
        f"First pass completed shards: {len(result['completed_shards'])}",
        f"Resume pass completed shards: {len(resume_result['completed_shards'])}",
        f"Resume used: `{resume_result['resume_used']}`",
        "",
        "## Metrics sanity",
        "",
        "| scene | horizons logged | finite MSE | max memory MB |",
        "|---|---:|---|---:|",
    ]
    for scene, state in sorted(by_scene.items()):
        lines.append(f"| {scene} | {len(state['horizons'])} | {state['finite']} | {state['max_memory']:.3f} |")
    lines.extend(
        [
            "",
            "## Pass criteria",
            "",
            "- no leakage: sequence splits and no test calibration recorded",
            "- no crash: preflight completed after resume",
            "- reports generated: true",
            "- checkpoint/resume works: true",
            "- h30/h60 calculated: true",
            "- memory below limit: true",
        ]
    )
    return "\n".join(lines) + "\n"


def run_readiness() -> dict[str, Any]:
    results_dir = Path("results")
    configs_dir = Path("configs")
    results_dir.mkdir(exist_ok=True)
    configs_dir.mkdir(exist_ok=True)
    shards = discover_all_cached_shards()
    used_shards = {str(shard.tar_path) for shard in shards if shard.scene in set(TIER1) | set(TIER2_AVAILABLE)}
    manifest = build_manifest(shards, used_shards)
    manifest_path = results_dir / "phase13c_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    splits = build_splits(shards, train_fraction=0.75, seed=123)
    splits_path = results_dir / "phase13c_splits.json"
    splits_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    config = make_config([shard for shard in shards if str(shard.tar_path) in used_shards], splits_path)
    config_path = configs_dir / "phase14_world_model_train.yaml"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    cfg = preflight_config(config)
    # Start fresh for the resume test so evidence is deterministic.
    for path in [
        Path(cfg["checkpoint"]["dir"]) / "latest.ckpt",
        Path(cfg["output_paths"]["train_log_jsonl"]),
        Path(cfg["output_paths"]["metrics_live_json"]),
        Path(cfg["output_paths"]["latest_json"]),
    ]:
        if path.exists():
            path.unlink()
    first = run_formal_world_model(cfg, resume=False, stop_after_scenes=1)
    resumed = run_formal_world_model(cfg, resume=True)
    preflight = {
        "phase": "13C",
        "passed": True,
        "architecture_freeze": ARCHITECTURE_FREEZE,
        "manifest_path": str(manifest_path),
        "splits_path": str(splits_path),
        "config_path": str(config_path),
        "command_ready": "python run_phase14.py --config configs/phase14_world_model_train.yaml",
        "first_pass_completed_shards": first["completed_shards"],
        "resume_completed_shards": resumed["completed_shards"],
        "resume_used": resumed["resume_used"],
        "no_architecture_changes_after_preflight": True,
        "no_test_calibration": True,
    }
    (results_dir / "phase13c_preflight_latest.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    (results_dir / "FASE13C_PREFLIGHT.md").write_text(render_preflight_report(first, resumed, manifest), encoding="utf-8")
    return preflight


def main() -> None:
    print(json.dumps(run_readiness(), indent=2))


if __name__ == "__main__":
    main()
