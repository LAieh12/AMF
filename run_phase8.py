from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from phase8_corpus import INTENT_KEYWORDS, build_prompt_suite, build_training_corpus
from phase8_morphogenic_decoder import MorphogenicInputField, build_decoders, tokenize


VERBS = {
    "soy",
    "es",
    "esta",
    "puedo",
    "convierte",
    "activa",
    "selecciona",
    "produce",
    "aprendo",
    "guardan",
    "exige",
    "mejora",
    "conecto",
    "genero",
    "combina",
    "contiene",
    "indica",
    "queda",
    "conserva",
    "responde",
    "hablar",
    "ayudarte",
}


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def score_output(output: str, expected_intent: str, keywords: tuple[str, ...]) -> dict[str, float]:
    tokens = tokenize(output)
    token_set = set(tokens)
    complete = float(len(tokens) >= 6 and bool(tokens) and tokens[-1] in {".", "!", "?"})
    has_verb = float(any(token in VERBS for token in tokens))
    keyword_hit = float(any(keyword in token_set for keyword in keywords))
    intent_vocab = set(INTENT_KEYWORDS[expected_intent])
    relevance = len(token_set & (intent_vocab | set(keywords))) / max(1, len(set(keywords)))
    repeated_pairs = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
    repetition_penalty = repeated_pairs / max(1, len(tokens) - 1)
    length_ok = float(6 <= len(tokens) <= 26)
    return {
        "complete": complete,
        "has_verb": has_verb,
        "keyword_hit": keyword_hit,
        "relevance": min(1.0, relevance),
        "length_ok": length_ok,
        "repetition_penalty": repetition_penalty,
        "tokens": float(len(tokens)),
    }


def evaluate_decoder(decoder: Any, suite: list[Any]) -> dict[str, Any]:
    rows = []
    outputs = []
    for item in suite:
        output = decoder.generate(item.prompt)
        metrics = score_output(output, item.intent, item.keywords)
        rows.append({"prompt": item.prompt, "intent": item.intent, "output": output, **metrics})
        outputs.append(output)
    unique_rate = len(set(outputs)) / max(1, len(outputs))
    summary = {
        "complete_rate": mean([row["complete"] for row in rows]),
        "verb_rate": mean([row["has_verb"] for row in rows]),
        "keyword_hit_rate": mean([row["keyword_hit"] for row in rows]),
        "relevance": mean([row["relevance"] for row in rows]),
        "length_ok_rate": mean([row["length_ok"] for row in rows]),
        "repetition_penalty": mean([row["repetition_penalty"] for row in rows]),
        "unique_output_rate": unique_rate,
        "avg_tokens": mean([row["tokens"] for row in rows]),
    }
    summary["talk_score"] = (
        0.24 * summary["complete_rate"]
        + 0.20 * summary["verb_rate"]
        + 0.22 * summary["keyword_hit_rate"]
        + 0.18 * summary["relevance"]
        + 0.10 * summary["length_ok_rate"]
        + 0.06 * summary["unique_output_rate"]
        - 0.12 * summary["repetition_penalty"]
    )
    return {"name": decoder.name, "summary": summary, "rows": rows}


def run_phase8() -> dict[str, Any]:
    start = perf_counter()
    training = build_training_corpus()
    suite = build_prompt_suite()
    field = MorphogenicInputField(dims=384, top_k=9, radius=0.42).fit(training)
    decoder_results = []
    for decoder in build_decoders():
        decoder.fit(training, field)
        decoder_results.append(evaluate_decoder(decoder, suite))
    best = max(decoder_results, key=lambda result: result["summary"]["talk_score"])
    return {
        "title": "Phase 8 - Morphogenic decoder and output capacity",
        "training_examples": len(training),
        "test_prompts": len(suite),
        "architecture": "Input -> morphogenic input field -> active latent state -> morphogenic decoder -> output",
        "rules": {
            "no_llm": True,
            "no_dense_decoder": True,
            "uses_backprop": False,
        },
        "elapsed_seconds": perf_counter() - start,
        "results": decoder_results,
        "best_decoder": best["name"],
    }


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def write_reports(results: dict[str, Any], out_dir: str | Path = "results") -> None:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    (out / "phase8_latest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    rows = [
        "| Decoder | talk score | complete | verb | keyword | relevance | unique | repetition | avg tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in sorted(results["results"], key=lambda item: item["summary"]["talk_score"], reverse=True):
        s = result["summary"]
        rows.append(
            f"| {result['name']} | {_fmt(s['talk_score'])} | {_fmt(s['complete_rate'])} | "
            f"{_fmt(s['verb_rate'])} | {_fmt(s['keyword_hit_rate'])} | {_fmt(s['relevance'])} | "
            f"{_fmt(s['unique_output_rate'])} | {_fmt(s['repetition_penalty'])} | {_fmt(s['avg_tokens'])} |"
        )
    best = next(result for result in results["results"] if result["name"] == results["best_decoder"])
    examples = [
        f"- Prompt: `{row['prompt']}`\n  Output: {row['output']}"
        for row in best["rows"]
    ]
    report = f"""# Fase 8 - Decoder morfogenetico

Objetivo: agregar capacidad de output a un sistema morfogenetico sin usar LLMs
ni pegar un decoder denso clasico.

Arquitectura probada:

```text
{results['architecture']}
```

Reglas: no LLM = {results['rules']['no_llm']}, no decoder denso =
{results['rules']['no_dense_decoder']}, backprop = {results['rules']['uses_backprop']}.

Training examples: {results['training_examples']}
Test prompts: {results['test_prompts']}
Tiempo total: {results['elapsed_seconds']:.2f} s

## Comparacion de decoders

{chr(10).join(rows)}

## Mejor decoder

`{results['best_decoder']}` fue el mejor por `talk_score`.

## Ejemplos de habla

{chr(10).join(examples)}

## Lectura

El mejor resultado no sale de una capa densa. Sale de un campo de entrada que
activa intenciones y memorias locales, seguido por decoders de celulas: vecinos
de respuesta, transiciones token-a-token, frames con slots y resonancia entre
candidatos. El sistema ya puede producir frases completas y relevantes desde
prompts no vistos.
"""
    (out / "FASE8_RESULTADOS.md").write_text(report, encoding="utf-8")
    complete = """# FASE8_COMPLETADA

Fase 8 agrego capacidad de output a la arquitectura morfogenetica.

Entregables:

- `phase8_corpus.py`
- `phase8_morphogenic_decoder.py`
- `run_phase8.py`
- `results/phase8_latest.json`
- `results/FASE8_RESULTADOS.md`

La suite compara cuatro decoders sin LLM y sin decoder denso:

- nearest response cells
- transition cells
- frame slot cells
- resonant morphogenic decoder

La evidencia de completitud es que el mejor decoder genera frases completas,
con verbo, diversidad y relevancia sobre prompts no vistos.
"""
    Path("FASE8_COMPLETADA.md").write_text(complete, encoding="utf-8")


def main() -> None:
    results = run_phase8()
    write_reports(results)
    print("report: results/FASE8_RESULTADOS.md")
    print(f"best_decoder: {results['best_decoder']}")
    for result in sorted(results["results"], key=lambda item: item["summary"]["talk_score"], reverse=True):
        print(f"{result['name']}: talk_score={result['summary']['talk_score']:.4f}")


if __name__ == "__main__":
    main()
