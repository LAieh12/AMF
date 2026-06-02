from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

from phase8_morphogenic_decoder import tokenize
from phase9_baselines import build_phase9_baselines
from phase9_corpus_builder import Phase9Example, build_phase9_corpus, corpus_domain_counts
from phase9_decoder_scaling import GenerationResult, Phase9MorphogenicAssistant
from phase9_eval_prompts import Phase9EvalPrompt, build_phase9_eval_prompts


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def rough_sizeof(obj: Any, seen: set[int] | None = None) -> int:
    seen = seen or set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if hasattr(obj, "nbytes"):
        return int(obj.nbytes)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(rough_sizeof(k, seen) + rough_sizeof(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(rough_sizeof(v, seen) for v in obj)
    elif hasattr(obj, "__dict__"):
        size += rough_sizeof(vars(obj), seen)
    return int(size)


def format_success(output: str, expected_format: str) -> float:
    stripped = output.strip()
    if expected_format == "json":
        try:
            json.loads(stripped)
            return 1.0
        except json.JSONDecodeError:
            return 0.0
    if expected_format == "table":
        return float("|" in output and "---" in output)
    if expected_format == "steps":
        return float("1." in output and "2." in output)
    if expected_format == "pseudocode":
        return float("funcion" in output and "retornar" in output)
    if expected_format == "experiment_plan":
        return float("Hipotesis:" in output and "Metrica:" in output and "Baseline:" in output)
    if expected_format == "diagnosis":
        return float("Causa:" in output and "Siguiente paso:" in output)
    return float(stripped.endswith((".", "!", "?")))


def score_generation(result: GenerationResult, item: Phase9EvalPrompt) -> dict[str, float]:
    output = result.output
    tokens = tokenize(output)
    token_set = set(tokens)
    keyword_set = set(item.keywords)
    expected_domains = set(item.domains)
    predicted_domains = set(result.domains)
    domain_accuracy = len(expected_domains & predicted_domains) / max(1, len(expected_domains))
    keyword_hit = float(bool(token_set & keyword_set) or any(keyword in output.lower() for keyword in keyword_set))
    relevance = len(token_set & keyword_set) / max(1, len(keyword_set))
    repeated_pairs = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
    repetition_penalty = repeated_pairs / max(1, len(tokens) - 1)
    fmt = format_success(output, item.output_format)
    complete = float((len(tokens) >= 6 and output.strip().endswith((".", "!", "?"))) or fmt == 1.0)
    length_ok = float(4 <= len(tokens) <= 120)
    if item.requires_composition:
        composition = float(len(expected_domains & predicted_domains) >= min(2, len(expected_domains)))
    else:
        composition = 1.0
    return {
        "domain_accuracy": domain_accuracy,
        "format_success": fmt,
        "complete": complete,
        "keyword_hit": keyword_hit,
        "relevance": min(1.0, relevance),
        "length_ok": length_ok,
        "repetition_penalty": repetition_penalty,
        "composition_success": composition,
        "tokens": float(len(tokens)),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    outputs = [row["output"] for row in rows]
    diversity = len(set(outputs)) / max(1, len(outputs))
    summary = {
        "domain_accuracy": mean([row["domain_accuracy"] for row in rows]),
        "format_success": mean([row["format_success"] for row in rows]),
        "complete_rate": mean([row["complete"] for row in rows]),
        "keyword_hit_rate": mean([row["keyword_hit"] for row in rows]),
        "relevance": mean([row["relevance"] for row in rows]),
        "length_ok_rate": mean([row["length_ok"] for row in rows]),
        "repetition_penalty": mean([row["repetition_penalty"] for row in rows]),
        "composition_success": mean([row["composition_success"] for row in rows]),
        "avg_tokens": mean([row["tokens"] for row in rows]),
        "diversity": diversity,
        "avg_latency_ms": mean([row["latency_ms"] for row in rows]),
    }
    summary["talk_score"] = (
        0.20 * summary["complete_rate"]
        + 0.18 * summary["keyword_hit_rate"]
        + 0.18 * summary["relevance"]
        + 0.14 * summary["format_success"]
        + 0.12 * summary["composition_success"]
        + 0.08 * summary["domain_accuracy"]
        + 0.06 * summary["diversity"]
        + 0.04 * summary["length_ok_rate"]
        - 0.10 * summary["repetition_penalty"]
    )
    summary["service_score"] = (
        0.18 * summary["domain_accuracy"]
        + 0.18 * summary["format_success"]
        + 0.16 * summary["composition_success"]
        + 0.14 * summary["keyword_hit_rate"]
        + 0.12 * summary["relevance"]
        + 0.10 * summary["complete_rate"]
        + 0.07 * summary["diversity"]
        + 0.05 * summary["length_ok_rate"]
        - 0.10 * summary["repetition_penalty"]
    )
    return summary


def evaluate_system(system: Any, suite: list[Phase9EvalPrompt]) -> dict[str, Any]:
    rows = []
    for item in suite:
        start = time.perf_counter()
        result = system.generate_result(item.prompt)
        latency_ms = (time.perf_counter() - start) * 1000.0
        metrics = score_generation(result, item)
        rows.append(
            {
                "prompt": item.prompt,
                "expected_domains": item.domains,
                "predicted_domains": result.domains,
                "expected_format": item.output_format,
                "predicted_format": result.output_format,
                "source": result.source,
                "output": result.output,
                "latency_ms": latency_ms,
                **metrics,
            }
        )
    return {"summary": summarize_rows(rows), "rows": rows}


def online_learning_probe(system: Phase9MorphogenicAssistant) -> dict[str, Any]:
    prompt = "explica hotcell memoria online para recuperar un caso nuevo"
    keywords = ("hotcell", "online", "caso", "nuevo")
    expected = Phase9EvalPrompt(
        prompt=prompt,
        domains=("architecture", "research"),
        intents=("online_memory",),
        output_format="normal",
        keywords=keywords,
        requires_composition=True,
    )
    before = system.generate_result(prompt)
    before_score = score_generation(before, expected)
    example = Phase9Example(
        prompt=prompt,
        domains=("architecture", "research"),
        intent="online_memory",
        response="Hotcell online guarda el caso nuevo y refuerza la ruta de respuesta.",
        output_format="normal",
        keywords=keywords,
    )
    start = time.perf_counter()
    system.learn(example)
    learn_seconds = time.perf_counter() - start
    after = system.generate_result(prompt)
    after_score = score_generation(after, expected)
    return {
        "prompt": prompt,
        "learn_seconds": learn_seconds,
        "before_output": before.output,
        "after_output": after.output,
        "before_relevance": before_score["relevance"],
        "after_relevance": after_score["relevance"],
        "before_keyword_hit": before_score["keyword_hit"],
        "after_keyword_hit": after_score["keyword_hit"],
        "improved": bool(
            after_score["relevance"] > before_score["relevance"]
            or after_score["keyword_hit"] > before_score["keyword_hit"]
        ),
    }


def fit_phase9_assistant(examples: list[Phase9Example]) -> tuple[Phase9MorphogenicAssistant, float, float]:
    tracemalloc.start()
    start = time.perf_counter()
    system = Phase9MorphogenicAssistant(dims=256, top_k=9, radius=0.42).fit(examples)
    fit_seconds = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_mb = max(rough_sizeof(system) / (1024.0 * 1024.0), peak / (1024.0 * 1024.0))
    return system, fit_seconds, memory_mb


def run_one_size(size: int, suite: list[Phase9EvalPrompt], include_baselines: bool) -> dict[str, Any]:
    examples = build_phase9_corpus(size)
    system, fit_seconds, memory_mb = fit_phase9_assistant(examples)
    evaluation = evaluate_system(system, suite)
    evaluation["summary"].update(
        {
            "examples": size,
            "fit_seconds": fit_seconds,
            "model_memory_mb": memory_mb,
            "memory_counts": system.field.memory_counts(),
            "domain_counts": corpus_domain_counts(examples),
        }
    )
    baselines = []
    if include_baselines:
        for baseline in build_phase9_baselines():
            start = time.perf_counter()
            baseline.fit(examples)
            fit_time = time.perf_counter() - start
            baseline_eval = evaluate_system(baseline, suite)
            baseline_eval["summary"].update(
                {
                    "name": baseline.name,
                    "examples": size,
                    "fit_seconds": fit_time,
                    "model_memory_mb": rough_sizeof(baseline) / (1024.0 * 1024.0),
                }
            )
            baselines.append(baseline_eval)
    online_probe = online_learning_probe(system) if size >= 1000 else None
    return {
        "size": size,
        "assistant": evaluation,
        "baselines": baselines,
        "online_probe": online_probe,
        "sample_outputs": evaluation["rows"][:8],
    }


def run_phase9(args: argparse.Namespace) -> dict[str, Any]:
    sizes = [int(part) for part in args.sizes.split(",") if part.strip()]
    suite = build_phase9_eval_prompts()
    results = []
    for size in sizes:
        include_baselines = size in {sizes[0], sizes[-1]} or size <= args.baseline_until
        results.append(run_one_size(size, suite, include_baselines=include_baselines))
    return {
        "title": "Phase 9 - AMF8 Domain Expansion",
        "architecture": "prompt -> domain router -> domain memory -> resonant morphogenic assistant -> structured output",
        "rules": {
            "no_llm": True,
            "no_dense_decoder": True,
            "no_backprop": True,
            "scales_by": ["domains", "memories", "output_cells", "composition", "online_learning"],
        },
        "sizes": sizes,
        "eval_prompts": len(suite),
        "results": results,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "not measured"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_reports(results: dict[str, Any], out_dir: str | Path = "results") -> None:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    (out / "phase9_latest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    scaling_rows = [
        "| examples | service | talk | domain | format | comp | relevance | rep | diversity | fit s | ms/prompt | memory MB |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in results["results"]:
        s = record["assistant"]["summary"]
        scaling_rows.append(
            f"| {record['size']} | {_fmt(s['service_score'])} | {_fmt(s['talk_score'])} | "
            f"{_fmt(s['domain_accuracy'])} | {_fmt(s['format_success'])} | {_fmt(s['composition_success'])} | "
            f"{_fmt(s['relevance'])} | {_fmt(s['repetition_penalty'])} | {_fmt(s['diversity'])} | "
            f"{_fmt(s['fit_seconds'])} | {_fmt(s['avg_latency_ms'])} | {_fmt(s['model_memory_mb'])} |"
        )

    baseline_rows = [
        "| examples | system | service | talk | domain | format | comp | ms/prompt | memory MB |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in results["results"]:
        for baseline in record["baselines"]:
            s = baseline["summary"]
            baseline_rows.append(
                f"| {record['size']} | {s['name']} | {_fmt(s['service_score'])} | {_fmt(s['talk_score'])} | "
                f"{_fmt(s['domain_accuracy'])} | {_fmt(s['format_success'])} | {_fmt(s['composition_success'])} | "
                f"{_fmt(s['avg_latency_ms'])} | {_fmt(s['model_memory_mb'])} |"
            )

    last = results["results"][-1]
    examples = [
        f"- Prompt: `{row['prompt']}`\n  Domains: {', '.join(row['predicted_domains'])}\n  Output: {row['output']}"
        for row in last["sample_outputs"]
    ]
    online_rows = []
    for record in results["results"]:
        probe = record.get("online_probe")
        if not probe:
            continue
        online_rows.append(
            f"| {record['size']} | {_fmt(probe['before_relevance'])} | {_fmt(probe['after_relevance'])} | "
            f"{_fmt(probe['learn_seconds'])} | {probe['improved']} |"
        )
    report = f"""# Fase 9 - AMF8 Domain Expansion

Objetivo: convertir la demo de habla controlada en un asistente morfogenetico
servible, escalando por dominios y memorias, no por capas densas.

Arquitectura:

```text
{results['architecture']}
```

Reglas: no LLM = {results['rules']['no_llm']}, no decoder denso =
{results['rules']['no_dense_decoder']}, no backprop = {results['rules']['no_backprop']}.

Dominios: conversation, architecture, research, code, structured, safety.

## Escala AMF9

{chr(10).join(scaling_rows)}

## Baselines

{chr(10).join(baseline_rows)}

## Aprendizaje online

| examples | relevance antes | relevance despues | learn seconds | improved |
|---:|---:|---:|---:|---|
{chr(10).join(online_rows) if online_rows else '| not measured | not measured | not measured | not measured | not measured |'}

## Salidas de ejemplo en el tamano mayor

{chr(10).join(examples)}

## Lectura

Fase 9 separa memorias por dominio: conversation, architecture, research, code,
structured y safety. El prompt activa un router de dominio, luego solo consulta
las memorias relevantes y finalmente compone una salida normal o estructurada.
Esto reduce interferencia entre saludos, arquitectura, research, codigo y
formatos utiles.

El score sigue siendo local y auditable, no una metrica universal de calidad de
lenguaje. La evidencia que importa aqui es la curva de escala: score, dominio,
formato, composicion, repeticion, latencia, memoria y aprendizaje online.
"""
    (out / "FASE9_RESULTADOS.md").write_text(report, encoding="utf-8")

    best = results["results"][-1]["assistant"]["summary"]
    complete = f"""# FASE9_COMPLETADA

Fase 9 implementa AMF8 Domain Expansion.

Entregables:

- `phase9_corpus_builder.py`
- `phase9_domain_memory.py`
- `phase9_decoder_scaling.py`
- `phase9_eval_prompts.py`
- `phase9_baselines.py`
- `run_phase9.py`
- `results/phase9_latest.json`
- `results/FASE9_RESULTADOS.md`

Resultado en el tamano mayor ({results['results'][-1]['size']} ejemplos):

- service_score: {best['service_score']:.4f}
- talk_score local: {best['talk_score']:.4f}
- domain_accuracy: {best['domain_accuracy']:.4f}
- format_success: {best['format_success']:.4f}
- composition_success: {best['composition_success']:.4f}
- repetition_penalty: {best['repetition_penalty']:.4f}
- diversity: {best['diversity']:.4f}
- avg_latency_ms: {best['avg_latency_ms']:.4f}
- model_memory_mb: {best['model_memory_mb']:.4f}

La arquitectura escala por dominios, memorias separadas, celulas de salida,
composicion, formatos estructurados y aprendizaje online. No usa LLM, decoder
denso ni backprop.
"""
    Path("FASE9_COMPLETADA.md").write_text(complete, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 9 domain expansion benchmark.")
    parser.add_argument("--sizes", default="90,300,1000,3000,10000", help="Comma-separated corpus sizes.")
    parser.add_argument("--baseline-until", type=int, default=1000, help="Run baselines for sizes up to this value plus first/last.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_phase9(args)
    write_reports(results)
    print("report: results/FASE9_RESULTADOS.md")
    for record in results["results"]:
        summary = record["assistant"]["summary"]
        print(
            f"size={record['size']} service={summary['service_score']:.4f} "
            f"talk={summary['talk_score']:.4f} memory={summary['model_memory_mb']:.2f}MB "
            f"latency={summary['avg_latency_ms']:.2f}ms"
        )


if __name__ == "__main__":
    main()
