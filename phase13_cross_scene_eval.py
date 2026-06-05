from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from phase13_eval_horizons import win_tie_loss


def aggregate_cross_scene(scene_results: list[dict[str, Any]]) -> dict[str, Any]:
    by_horizon: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scene in scene_results:
        for horizon, record in scene["horizon_results"].items():
            by_horizon[horizon].append({"scene": scene["scene"], **record})

    horizon_summary: dict[str, Any] = {}
    h30_h60_gains: list[float] = []
    ltm_scene_gains: list[tuple[str, str, float]] = []
    noisy_scene_gains: list[tuple[str, str, float]] = []

    for horizon, records in sorted(by_horizon.items(), key=lambda item: int(item[0][1:])):
        temporal_scores = []
        previous_scores = []
        full_scores = []
        best_scores = []
        wtl_temporal = {"win": 0, "tie": 0, "loss": 0}
        wtl_previous = {"win": 0, "tie": 0, "loss": 0}
        per_scene = []
        for record in records:
            metrics = record["metrics"]
            temporal = metrics["temporal_energy"]["mse"]
            previous = metrics[record["best_previous_amf"]]["mse"]
            full = metrics["amf_ltm_selected"]["mse"]
            best = metrics[record["best_valid_model"]]["mse"]
            temporal_scores.append(temporal)
            previous_scores.append(previous)
            full_scores.append(full)
            best_scores.append(best)
            wtl_temporal[win_tie_loss(full, temporal)] += 1
            wtl_previous[win_tie_loss(full, previous)] += 1
            gain_temporal = metrics["amf_ltm_selected"]["gain_vs_temporal_energy"]
            gain_previous = metrics["amf_ltm_selected"]["gain_vs_best_previous_amf"]
            if horizon in {"h30", "h60"}:
                h30_h60_gains.append(gain_temporal)
            ltm_scene_gains.append((record["scene"], horizon, gain_temporal))
            noisy_scene_gains.append((record["scene"], horizon, gain_previous))
            per_scene.append(
                {
                    "scene": record["scene"],
                    "temporal_energy_mse": temporal,
                    "best_previous_amf": record["best_previous_amf"],
                    "best_previous_mse": previous,
                    "amf_ltm_selected_mse": full,
                    "selected_ltm_branch": record["selected_ltm_branch"],
                    "best_valid_model": record["best_valid_model"],
                    "best_valid_mse": best,
                    "full_gain_vs_temporal": gain_temporal,
                    "full_gain_vs_best_previous": gain_previous,
                }
            )

        horizon_summary[horizon] = {
            "scene_count": len(records),
            "mean_temporal_energy_mse": float(np.mean(temporal_scores)),
            "mean_best_previous_amf_mse": float(np.mean(previous_scores)),
            "mean_amf_ltm_selected_mse": float(np.mean(full_scores)),
            "mean_best_valid_mse": float(np.mean(best_scores)),
            "win_tie_loss_vs_temporal_energy": wtl_temporal,
            "win_tie_loss_vs_best_previous_amf": wtl_previous,
            "per_scene": per_scene,
        }

    best_ltm = max(ltm_scene_gains, key=lambda item: item[2]) if ltm_scene_gains else None
    worst_ltm = min(noisy_scene_gains, key=lambda item: item[2]) if noisy_scene_gains else None
    return {
        "horizon_summary": horizon_summary,
        "mean_h30_h60_gain_vs_temporal_energy": float(np.mean(h30_h60_gains)) if h30_h60_gains else 0.0,
        "scene_where_ltm_helps_most": {
            "scene": best_ltm[0],
            "horizon": best_ltm[1],
            "gain_vs_temporal": best_ltm[2],
        }
        if best_ltm
        else None,
        "scene_where_ltm_adds_most_noise": {
            "scene": worst_ltm[0],
            "horizon": worst_ltm[1],
            "gain_vs_best_previous": worst_ltm[2],
        }
        if worst_ltm
        else None,
    }


def render_cross_scene_report(result: dict[str, Any]) -> str:
    lines = [
        "# Fase 13 - Cross-scene evaluation",
        "",
        f"Scenes: {', '.join(scene['scene'] for scene in result['scene_results'])}",
        f"Mean h30/h60 gain vs temporal-energy: {result['cross_scene']['mean_h30_h60_gain_vs_temporal_energy']:.6f}",
        "",
        "| horizon | mean temporal | mean best previous | mean AMF-LTM selected | W/T/L vs temporal | W/T/L vs previous |",
        "|---|---:|---:|---:|---|---|",
    ]
    for horizon, record in result["cross_scene"]["horizon_summary"].items():
        wt = record["win_tie_loss_vs_temporal_energy"]
        wp = record["win_tie_loss_vs_best_previous_amf"]
        lines.append(
            f"| {horizon} | {record['mean_temporal_energy_mse']:.6f} | "
            f"{record['mean_best_previous_amf_mse']:.6f} | {record['mean_amf_ltm_selected_mse']:.6f} | "
            f"{wt['win']}/{wt['tie']}/{wt['loss']} | {wp['win']}/{wp['tie']}/{wp['loss']} |"
        )
    lines.extend(["", "## Per scene", ""])
    for horizon, record in result["cross_scene"]["horizon_summary"].items():
        lines.append(f"### {horizon}")
        lines.append("")
        lines.append("| scene | temporal | best previous | AMF-LTM selected | selected branch | gain vs temporal | gain vs previous | best valid |")
        lines.append("|---|---:|---:|---:|---|---:|---:|---|")
        for row in record["per_scene"]:
            lines.append(
                f"| {row['scene']} | {row['temporal_energy_mse']:.6f} | {row['best_previous_mse']:.6f} | "
                f"{row['amf_ltm_selected_mse']:.6f} | {row['selected_ltm_branch']} | "
                f"{row['full_gain_vs_temporal']:.6f} | {row['full_gain_vs_best_previous']:.6f} | "
                f"{row['best_valid_model']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"
