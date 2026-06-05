from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def collect_ltm_diagnostics(scene_results: list[dict[str, Any]]) -> dict[str, Any]:
    by_scene: dict[str, Any] = {}
    global_regimes: Counter[str] = Counter()
    global_safety: dict[str, Counter[str]] = defaultdict(Counter)

    for scene in scene_results:
        scene_name = scene["scene"]
        scene_record = {
            "regime_counts_test": Counter(),
            "event_counts_test": Counter(),
            "safety_by_horizon": {},
            "ltm_full_gain_by_horizon": {},
        }
        for horizon, hrec in scene["horizon_results"].items():
            scene_record["regime_counts_test"].update(hrec["regimes_detected_test"])
            scene_record["event_counts_test"].update(hrec["events_detected_test"])
            selected_branch = hrec["selected_ltm_branch"]
            full_diag = hrec["ltm_diagnostics"][selected_branch]
            scene_record["safety_by_horizon"][horizon] = {
                "corrected": full_diag["ltm_corrected_count"],
                "off": full_diag["ltm_off_count"],
                "improved": full_diag["ltm_improved_count"],
                "worsened": full_diag["ltm_worsened_count"],
                "confidence_mean": full_diag["confidence_mean"],
                "alpha": full_diag["alpha"],
                "threshold": full_diag["confidence_threshold"],
                "retrieved_per_prediction": full_diag["memories_retrieved_per_prediction"],
                "memories_created": full_diag["ltm_memories_created"],
                "memory_mb": full_diag["memory_mb"],
            }
            scene_record["ltm_full_gain_by_horizon"][horizon] = hrec["metrics"]["amf_ltm_selected"][
                "gain_vs_temporal_energy"
            ]
            scene_record.setdefault("selected_branch_by_horizon", {})[horizon] = selected_branch
            global_safety[horizon].update(
                {
                    "corrected": full_diag["ltm_corrected_count"],
                    "off": full_diag["ltm_off_count"],
                    "improved": full_diag["ltm_improved_count"],
                    "worsened": full_diag["ltm_worsened_count"],
                }
            )
        global_regimes.update(scene_record["regime_counts_test"])
        scene_record["regime_counts_test"] = dict(scene_record["regime_counts_test"])
        scene_record["event_counts_test"] = dict(scene_record["event_counts_test"])
        by_scene[scene_name] = scene_record

    return {
        "by_scene": by_scene,
        "global_regime_counts_test": dict(global_regimes),
        "global_safety_by_horizon": {horizon: dict(counter) for horizon, counter in global_safety.items()},
    }


def render_ltm_diagnostic_report(result: dict[str, Any]) -> str:
    diagnostics = result["ltm_diagnostics"]
    lines = [
        "# Fase 13 - LTM diagnostic",
        "",
        "## Global regime counts",
        "",
        "| regime | count |",
        "|---|---:|",
    ]
    for regime, count in sorted(diagnostics["global_regime_counts_test"].items()):
        lines.append(f"| {regime} | {count} |")
    lines.extend(["", "## Residual safety", "", "| horizon | corrected | off | improved | worsened |", "|---|---:|---:|---:|---:|"])
    for horizon, record in sorted(diagnostics["global_safety_by_horizon"].items(), key=lambda item: int(item[0][1:])):
        lines.append(
            f"| {horizon} | {record.get('corrected', 0)} | {record.get('off', 0)} | "
            f"{record.get('improved', 0)} | {record.get('worsened', 0)} |"
        )
    lines.extend(["", "## By scene", ""])
    for scene, record in diagnostics["by_scene"].items():
        lines.append(f"### {scene}")
        lines.append("")
        lines.append("| horizon | selected branch | gain vs temporal | corrected | off | improved | worsened | alpha | confidence |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for horizon, safety in sorted(record["safety_by_horizon"].items(), key=lambda item: int(item[0][1:])):
            lines.append(
                f"| {horizon} | {record['selected_branch_by_horizon'][horizon]} | "
                f"{record['ltm_full_gain_by_horizon'][horizon]:.6f} | "
                f"{safety['corrected']} | {safety['off']} | {safety['improved']} | {safety['worsened']} | "
                f"{safety['alpha']:.2f} | {safety['confidence_mean']:.6f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"
