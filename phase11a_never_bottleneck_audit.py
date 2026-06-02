from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HORIZON_RE = re.compile(r"^h(\d+)$")
LOWER_IS_BETTER = (
    "mse",
    "mae",
    "error",
    "loss",
)
HIGHER_IS_BETTER = (
    "skill",
    "iou",
    "score",
    "rate",
    "accuracy",
)


@dataclass(frozen=True)
class MetricHit:
    path: Path
    horizon: str
    metric: str
    value: float


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _walk_metrics(obj: Any, prefix: str = "") -> list[tuple[str, float]]:
    hits: list[tuple[str, float]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            hits.extend(_walk_metrics(value, name))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        hits.append((prefix, float(obj)))
    return hits


def _horizon_dicts(obj: Any, prefix: str = "") -> list[tuple[str, str, dict[str, Any]]]:
    found: list[tuple[str, str, dict[str, Any]]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(key, str) and HORIZON_RE.match(key) and isinstance(value, dict):
                found.append((key, prefix, value))
            found.extend(_horizon_dicts(value, name))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            name = f"{prefix}.{idx}" if prefix else str(idx)
            found.extend(_horizon_dicts(item, name))
    return found


def collect_hits(results_dir: Path) -> list[MetricHit]:
    hits: list[MetricHit] = []
    for path in sorted(results_dir.glob("*.json")):
        obj = _load_json(path)
        if obj is None:
            continue
        for horizon, scope, metrics in _horizon_dicts(obj):
            for metric, value in _walk_metrics(metrics):
                if not metric:
                    continue
                if scope:
                    metric = f"{scope}.{metric}"
                hits.append(MetricHit(path=path, horizon=horizon, metric=metric, value=value))
    return hits


def best_by_metric(hits: list[MetricHit]) -> dict[tuple[str, str], MetricHit]:
    best: dict[tuple[str, str], MetricHit] = {}
    for hit in hits:
        key = (hit.horizon, hit.metric)
        current = best.get(key)
        metric_name = hit.metric.lower()
        lower = any(token in metric_name for token in LOWER_IS_BETTER) and not any(
            token in metric_name for token in HIGHER_IS_BETTER
        )
        if current is None:
            best[key] = hit
        elif lower and hit.value < current.value:
            best[key] = hit
        elif not lower and hit.value > current.value:
            best[key] = hit
    return best


def summarize_sources(root: Path) -> dict[str, Any]:
    py_files = sorted(root.glob("phase11a*.py"))
    source_text = {}
    for path in py_files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        source_text[path.name] = {
            "mentions_never": "never" in text.lower(),
            "mentions_world": "world" in text.lower(),
            "mentions_amf": "amf" in text.lower(),
            "mentions_decoder": "decoder" in text.lower(),
            "mentions_encoder": "encoder" in text.lower(),
            "mentions_slot": "slot" in text.lower(),
            "mentions_tile": "tile" in text.lower(),
            "mentions_download": "urlretrieve" in text.lower() or "download" in text.lower(),
        }
    return source_text


def render_markdown(root: Path, results_dir: Path, hits: list[MetricHit], source_summary: dict[str, Any]) -> str:
    best = best_by_metric(hits)
    interesting = []
    for (horizon, metric), hit in sorted(best.items(), key=lambda item: (int(item[0][0][1:]), item[0][1])):
        metric_l = metric.lower()
        if any(token in metric_l for token in ("hybrid", "slot", "oracle", "mse", "score", "iou", "skill")):
            interesting.append(hit)

    lines: list[str] = []
    lines.append("# Fase 11A - Never/AMF bottleneck audit")
    lines.append("")
    lines.append("Generado por `phase11a_never_bottleneck_audit.py`.")
    lines.append("")
    lines.append("## Mejor evidencia encontrada")
    lines.append("")
    if not interesting:
        lines.append("No se encontraron metricas comparables en JSON.")
    else:
        lines.append("| horizonte | metrica | valor | archivo |")
        lines.append("|---|---:|---:|---|")
        for hit in interesting[:80]:
            lines.append(f"| {hit.horizon} | `{hit.metric}` | {hit.value:.6f} | `{hit.path.as_posix()}` |")

    lines.append("")
    lines.append("## Uso real de Never / AMF")
    lines.append("")
    lines.append("| archivo | never | world | amf | encoder | decoder | slot | tile | descarga real |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for filename, flags in source_summary.items():
        lines.append(
            "| "
            + filename
            + " | "
            + " | ".join("si" if flags[key] else "no" for key in (
                "mentions_never",
                "mentions_world",
                "mentions_amf",
                "mentions_encoder",
                "mentions_decoder",
                "mentions_slot",
                "mentions_tile",
                "mentions_download",
            ))
            + " |"
        )

    lines.append("")
    lines.append("## Diagnostico")
    lines.append("")
    lines.append("- Si los mejores valores aparecen como `oracle`, el encoder/decoder no es el unico cuello de botella: el sistema aun no aprende a escoger la rama correcta con suficiente fiabilidad.")
    lines.append("- Si `Never` o `world` casi no aparecen en los scripts medidos, las pruebas estan usando piezas AMF/slot parciales, no un world model Never completo.")
    lines.append("- Si los probes con decoder nuevo no superan al slot-hybrid, el problema principal es modelado temporal/seleccion de estado, no solo reconstruccion visual.")
    lines.append("- Si aumentar `max_cells` no aumenta celulas efectivas, el cuello de botella es densidad/cobertura de memoria y politica de creacion, no el limite nominal.")
    lines.append("")
    lines.append("## Recomendacion")
    lines.append("")
    lines.append("El siguiente paso no debe ser otro encoder aislado. Debe ser un codec Never completo: encoder de estado, memoria AMF de transiciones, selector global por horizonte y decoder copy-skip.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase 11A results and source usage for Never/AMF bottlenecks.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="results/FASE11A_NEVER_BOTTLENECK_AUDIT.md")
    args = parser.parse_args()

    root = Path(args.root)
    results_dir = root / args.results_dir
    hits = collect_hits(results_dir)
    source_summary = summarize_sources(root)
    report = render_markdown(root, results_dir, hits, source_summary)
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
