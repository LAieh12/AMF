from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from phase13_amf_ltm_model import HORIZONS_13, evaluate_scene
from phase13_cross_scene_eval import aggregate_cross_scene, render_cross_scene_report
from phase13_ltm_diagnostics import collect_ltm_diagnostics, render_ltm_diagnostic_report
from phase13_scene_loader import SceneShard, load_scene_data


def run_phase13_training(
    scene_shards: list[SceneShard],
    train_fraction: float,
    split_seed: int,
    stride: int,
    memory_window: int,
    max_cells: int,
    ridge: float,
    model_radius: float,
    model_top_k: int,
    ltm_radius: float,
    ltm_top_k: int,
    tie_tolerance: float,
    horizons: tuple[int, ...] = HORIZONS_13,
) -> dict[str, Any]:
    started = time.time()
    scene_results = []
    for shard in scene_shards:
        scene_data = load_scene_data(shard, train_fraction=train_fraction, split_seed=split_seed)
        scene_results.append(
            evaluate_scene(
                scene_data,
                horizons=horizons,
                stride=stride,
                memory_window=memory_window,
                max_cells=max_cells,
                ridge=ridge,
                model_radius=model_radius,
                model_top_k=model_top_k,
                ltm_radius=ltm_radius,
                ltm_top_k=ltm_top_k,
                tie_tolerance=tie_tolerance,
            )
        )

    result = {
        "phase": "13",
        "objective": "Formal AMF World Model multi-scene training over real PhysicalAI physics shards",
        "architecture": "AMF-LTM residual/router over temporal-energy; LTM levels are retrieval/residual selectors, not dense appended features",
        "no_leakage_rule": "Sequence split per scene; candidates, static ensembles, LTM thresholds and residual alphas calibrated on fit/validation only; test used once for final metrics; oracle_no_valid is diagnostic only.",
        "scenes": [shard.scene for shard in scene_shards],
        "horizons": list(horizons),
        "train_fraction": train_fraction,
        "split_seed": split_seed,
        "stride": stride,
        "memory_window": memory_window,
        "max_cells": max_cells,
        "model_radius": model_radius,
        "model_top_k": model_top_k,
        "ltm_radius": ltm_radius,
        "ltm_top_k": ltm_top_k,
        "tie_tolerance": tie_tolerance,
        "scene_results": scene_results,
        "elapsed_seconds": time.time() - started,
    }
    result["cross_scene"] = aggregate_cross_scene(scene_results)
    result["ltm_diagnostics"] = collect_ltm_diagnostics(scene_results)
    result["success_audit"] = success_audit(result)
    return result


def success_audit(result: dict[str, Any]) -> dict[str, Any]:
    long_horizons = [h for h in ("h30", "h60", "h120") if h in result["cross_scene"]["horizon_summary"]]
    wins_vs_previous = 0
    total_vs_previous = 0
    wins_vs_temporal = 0
    total_vs_temporal = 0
    for scene in result["scene_results"]:
        for horizon in long_horizons:
            if horizon not in scene["horizon_results"]:
                continue
            record = scene["horizon_results"][horizon]
            full = record["metrics"]["amf_ltm_selected"]["mse"]
            temporal = record["metrics"]["temporal_energy"]["mse"]
            previous = record["metrics"][record["best_previous_amf"]]["mse"]
            total_vs_temporal += 1
            total_vs_previous += 1
            wins_vs_temporal += int(full <= temporal)
            wins_vs_previous += int(full <= previous)

    passes_majority_previous = wins_vs_previous >= max(1, (total_vs_previous + 1) // 2)
    improves_long_temporal = wins_vs_temporal >= max(1, (total_vs_temporal + 1) // 2)
    multi_scene = len(result["scene_results"]) >= 3
    memory_ok = True
    for scene in result["scene_results"]:
        for record in scene["horizon_results"].values():
            diag = record["ltm_diagnostics"]["amf_ltm_full"]
            if diag["memory_mb"] > 512:
                memory_ok = False
    passed = passes_majority_previous and improves_long_temporal and multi_scene and memory_ok
    return {
        "phase13_passed": passed,
        "long_horizons_checked": long_horizons,
        "wins_or_ties_vs_best_previous_amf_long": wins_vs_previous,
        "total_vs_best_previous_amf_long": total_vs_previous,
        "wins_or_ties_vs_temporal_energy_long": wins_vs_temporal,
        "total_vs_temporal_energy_long": total_vs_temporal,
        "multi_scene": multi_scene,
        "memory_under_512mb_per_horizon": memory_ok,
        "interpretation": (
        "AMF-LTM validation-selected residual passes the formal long-horizon gate."
            if passed
            else "AMF-LTM validation-selected residual did not clear every formal gate; inspect diagnostics for detector/retrieval/residual failure modes."
        ),
    }


def render_world_model_report(result: dict[str, Any]) -> str:
    lines = [
        "# Fase 13 - Formal AMF World Model Training",
        "",
        f"Scenes: {', '.join(result['scenes'])}",
        f"Horizons: {', '.join('h' + str(h) for h in result['horizons'])}",
        f"Stride: {result['stride']}; memory window: {result['memory_window']}; max cells: {result['max_cells']}",
        "",
        "## Architecture",
        "",
        result["architecture"],
        "",
        "Temporal-energy remains the base predictor. AMF-LTM stores episodic validation memories and applies calibrated residual corrections only when confidence passes validation thresholds.",
        "",
        "## Success audit",
        "",
        f"Passed: `{result['success_audit']['phase13_passed']}`",
        f"Long-horizon W/T vs temporal-energy: {result['success_audit']['wins_or_ties_vs_temporal_energy_long']}/{result['success_audit']['total_vs_temporal_energy_long']}",
        f"Long-horizon W/T vs best previous AMF: {result['success_audit']['wins_or_ties_vs_best_previous_amf_long']}/{result['success_audit']['total_vs_best_previous_amf_long']}",
        "",
        "## Scene metrics",
        "",
    ]
    for scene in result["scene_results"]:
        lines.append(f"### {scene['scene']}")
        lines.append("")
        lines.append(
            f"Tracks: {scene['track_count']}; sequences: {scene['sequence_count']} "
            f"({scene['train_sequences']} train / {scene['test_sequences']} test); elapsed: {scene['elapsed_seconds']:.2f}s"
        )
        lines.append("")
        lines.append("| horizon | temporal | best previous | AMF-LTM selected | selected branch | gain selected vs temporal | gain selected vs previous |")
        lines.append("|---|---:|---:|---:|---|---:|---:|")
        for horizon, record in scene["horizon_results"].items():
            metrics = record["metrics"]
            best_previous = record["best_previous_amf"]
            full = metrics["amf_ltm_selected"]
            lines.append(
                f"| {horizon} | {metrics['temporal_energy']['mse']:.6f} | "
                f"{metrics[best_previous]['mse']:.6f} ({best_previous}) | "
                f"{full['mse']:.6f} | {record['selected_ltm_branch']} | "
                f"{full['gain_vs_temporal_energy']:.6f} | {full['gain_vs_best_previous_amf']:.6f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Notes",
            "",
            "- `oracle_no_valid` is present in JSON only as a test-only diagnostic ceiling.",
            "- If AMF-LTM full loses, the report keeps the loss and the diagnostics identify whether confidence, regime retrieval, or residual aggression caused it.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_phase13_outputs(result: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase13_latest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "FASE13_WORLD_MODEL_TRAINING.md").write_text(render_world_model_report(result), encoding="utf-8")
    (out_dir / "FASE13_CROSS_SCENE_EVAL.md").write_text(render_cross_scene_report(result), encoding="utf-8")
    (out_dir / "FASE13_LTM_DIAGNOSTIC.md").write_text(render_ltm_diagnostic_report(result), encoding="utf-8")
