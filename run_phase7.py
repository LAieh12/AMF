from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tracemalloc
import warnings
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import sklearn
from sklearn.neural_network import MLPClassifier

from phase6_datasets import Phase6Dataset, make_split, select_datasets
from phase6_metrics import (
    accuracy,
    balanced_accuracy,
    macro_f1,
    model_memory_mb,
    predict_measure,
    summarize,
    timed_fit,
)
from phase6_sklearn_baselines import ModelCandidate, official_model_grids
from phase7_architecture import AMF7SuperField


DEFAULT_DATASETS = ["iris", "wine", "wdbc", "ionosphere", "sonar", "spambase", "madelon"]
SUMMARY_KEYS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "fit_seconds",
    "predict_seconds",
    "model_mb",
    "peak_fit_ram_mb",
    "param_count",
    "amf_cells",
    "experts",
]


def _fmt(stat: dict[str, float]) -> str:
    return f"{stat['mean']:.3f} +- {stat['std']:.3f}"


def _mlp_param_count(n_features: int, n_classes: int, hidden: tuple[int, ...]) -> int:
    sizes = (n_features,) + hidden + (n_classes,)
    return int(sum((sizes[i] + 1) * sizes[i + 1] for i in range(len(sizes) - 1)))


def _dataset_meta(dataset: Phase6Dataset) -> dict[str, Any]:
    return {
        "name": dataset.name,
        "samples": int(len(dataset.x)),
        "features": int(dataset.n_features),
        "classes": int(dataset.n_classes),
        "source": dataset.source,
        "kind": dataset.kind,
    }


def load_phase7_datasets(names: list[str] | None) -> list[Phase6Dataset]:
    return select_datasets(names or DEFAULT_DATASETS)


def baseline_grids(
    seed: int,
    n_features: int,
    n_classes: int,
    n_train: int,
    include_million: bool,
) -> dict[str, list[ModelCandidate]]:
    grids = official_model_grids(
        seed=seed,
        n_features=n_features,
        n_train=n_train,
        include_mlp=True,
        include_rbf=True,
    )
    if include_million:
        hidden = (1024, 1024)
        grids["sk_mlp_million"] = [
            ModelCandidate(
                "sk_mlp_million",
                f"1024x1024_{_mlp_param_count(n_features, n_classes, hidden)}params",
                lambda: MLPClassifier(
                    hidden_layer_sizes=hidden,
                    activation="relu",
                    alpha=5e-5,
                    batch_size=128,
                    early_stopping=True,
                    validation_fraction=0.18,
                    n_iter_no_change=10,
                    max_iter=110 if n_train < 2600 and n_features <= 180 else 80,
                    random_state=seed,
                ),
            )
        ]
    return grids


def score_prediction(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
    }


def select_candidate(candidates: list[ModelCandidate], x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    validations = []
    for candidate in candidates:
        try:
            fitted = timed_fit(candidate.factory, x_train, y_train)
            pred, pred_seconds = predict_measure(fitted.model, x_val)
            scores = score_prediction(y_val, pred)
            record = {
                "family": candidate.family,
                "variant": candidate.variant,
                "validation_predict_seconds": pred_seconds,
                "validation_fit_seconds": fitted.fit_seconds,
                **scores,
            }
            validations.append(record)
            key = (scores["macro_f1"], scores["accuracy"], -fitted.fit_seconds)
            if best is None or key > best["key"]:
                best = {"candidate": candidate, "record": record, "key": key}
        except Exception as exc:
            validations.append({"family": candidate.family, "variant": candidate.variant, "error": repr(exc)})
    if best is None:
        raise RuntimeError(f"All variants failed for {candidates[0].family if candidates else 'empty family'}")
    return {"candidate": best["candidate"], "validation": best["record"], "all_validations": validations}


def evaluate_baseline(candidate: ModelCandidate, x_full: np.ndarray, y_full: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    fitted = timed_fit(candidate.factory, x_full, y_full)
    pred, predict_seconds = predict_measure(fitted.model, x_test)
    record = score_prediction(y_test, pred)
    record.update(
        {
            "fit_seconds": fitted.fit_seconds,
            "predict_seconds": predict_seconds,
            "peak_fit_ram_mb": fitted.peak_ram_mb,
            "model_mb": model_memory_mb(fitted.model),
            "param_count": None,
            "amf_cells": None,
            "experts": None,
        }
    )
    if candidate.family == "sk_mlp_million":
        parts = candidate.variant.split("_")
        for part in parts:
            if part.endswith("params"):
                record["param_count"] = int(part[:-6])
    return record


def evaluate_amf7(
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    include_million: bool,
) -> dict[str, Any]:
    model = AMF7SuperField(seed=seed, include_million_mlp=include_million)
    tracemalloc.start()
    start = perf_counter()
    model.fit(x_train, y_train, x_val, y_val)
    fit_seconds = perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    pred, predict_seconds = predict_measure(model, x_test)
    record = score_prediction(y_test, pred)
    summary = model.summary()
    record.update(
        {
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "peak_fit_ram_mb": peak / (1024.0 * 1024.0),
            "model_mb": model.memory_bytes() / (1024.0 * 1024.0),
            "param_count": None,
            "amf_cells": summary.get("amf_cells"),
            "experts": summary.get("experts"),
            "strategy": summary.get("strategy"),
            "expert_results": summary.get("expert_results"),
            "strategy_scores": summary.get("strategy_scores"),
        }
    )
    return record


def run_phase7(args: argparse.Namespace) -> dict[str, Any]:
    start = perf_counter()
    seeds = list(range(args.seeds))
    datasets = load_phase7_datasets(args.datasets)
    records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for dataset in datasets:
            for seed in seeds:
                split = make_split(dataset, seed)
                x_full, y_full = split.train_val()
                amf7_record = evaluate_amf7(
                    seed=seed,
                    x_train=split.x_train,
                    y_train=split.y_train,
                    x_val=split.x_val,
                    y_val=split.y_val,
                    x_test=split.x_test,
                    y_test=split.y_test,
                    include_million=not args.no_million,
                )
                amf7_record.update(
                    {
                        "dataset": dataset.name,
                        "seed": seed,
                        "family": "AMF7_superfield",
                        "selected_variant": "validated_hybrid",
                        "n_train": int(len(split.y_train)),
                        "n_val": int(len(split.y_val)),
                        "n_test": int(len(split.y_test)),
                        "n_features": int(dataset.n_features),
                        "n_classes": int(dataset.n_classes),
                    }
                )
                records.append(amf7_record)

                grids = baseline_grids(
                    seed=seed,
                    n_features=dataset.n_features,
                    n_classes=dataset.n_classes,
                    n_train=len(split.y_train),
                    include_million=not args.no_million,
                )
                for family, candidates in grids.items():
                    selected_info = select_candidate(candidates, split.x_train, split.y_train, split.x_val, split.y_val)
                    selected = selected_info["candidate"]
                    for val_record in selected_info["all_validations"]:
                        validation_records.append(
                            {
                                **val_record,
                                "dataset": dataset.name,
                                "seed": seed,
                                "selected": val_record.get("variant") == selected.variant,
                            }
                        )
                    record = evaluate_baseline(selected, x_full, y_full, split.x_test, split.y_test)
                    record.update(
                        {
                            "dataset": dataset.name,
                            "seed": seed,
                            "family": family,
                            "selected_variant": selected.variant,
                            "validation_macro_f1": selected_info["validation"]["macro_f1"],
                            "validation_accuracy": selected_info["validation"]["accuracy"],
                            "n_train": int(len(split.y_train)),
                            "n_val": int(len(split.y_val)),
                            "n_test": int(len(split.y_test)),
                            "n_features": int(dataset.n_features),
                            "n_classes": int(dataset.n_classes),
                        }
                    )
                    records.append(record)

    summary: dict[str, dict[str, Any]] = defaultdict(dict)
    for dataset in sorted({record["dataset"] for record in records}):
        for family in sorted({record["family"] for record in records if record["dataset"] == dataset}):
            subset = [record for record in records if record["dataset"] == dataset and record["family"] == family]
            summary[dataset][family] = summarize(subset, SUMMARY_KEYS)
    return {
        "title": "Phase 7 - AMF7 superfield against strong classical models",
        "seeds": seeds,
        "datasets": [_dataset_meta(dataset) for dataset in datasets],
        "config": {
            "include_million_parameter_models": not args.no_million,
            "datasets_requested": args.datasets or DEFAULT_DATASETS,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "elapsed_seconds": perf_counter() - start,
        "records": records,
        "validation_records": validation_records,
        "summary": json.loads(json.dumps(summary)),
    }


def make_paper_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for meta in results["datasets"]:
        dataset = meta["name"]
        summary = results["summary"][dataset]
        amf = summary["AMF7_superfield"]
        baseline_rows = [
            (family, stats["accuracy"]["mean"])
            for family, stats in summary.items()
            if family not in {"AMF7_superfield", "AMF5"} and not family.startswith("sk_dummy")
        ]
        best_family, best_acc = max(baseline_rows, key=lambda item: item[1])
        rows.append(
            {
                "dataset": dataset,
                "n": meta["samples"],
                "d": meta["features"],
                "classes": meta["classes"],
                "amf7_acc": amf["accuracy"]["mean"],
                "amf7_std": amf["accuracy"]["std"],
                "amf7_f1": amf["macro_f1"]["mean"],
                "best_baseline": best_family,
                "best_baseline_acc": best_acc,
                "gap": amf["accuracy"]["mean"] - best_acc,
                "amf7_fit": amf["fit_seconds"]["mean"],
                "amf7_predict": amf["predict_seconds"]["mean"],
                "amf7_mb": amf["model_mb"]["mean"],
                "amf7_cells": amf["amf_cells"]["mean"],
                "amf7_experts": amf["experts"]["mean"],
            }
        )
    return rows


def write_reports(results: dict[str, Any], out_dir: str | Path = "results") -> None:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    (out / "phase7_latest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    rows = make_paper_rows(results)
    table = [
        "| Dataset | n | d | C | AMF7 acc | Best classic | Gap | AMF7 F1 | fit s | pred s | MB | cells | experts |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    wins = 0
    gaps = []
    for row in rows:
        if row["gap"] > 0:
            wins += 1
        gaps.append(row["gap"])
        table.append(
            f"| {row['dataset']} | {row['n']} | {row['d']} | {row['classes']} | "
            f"{row['amf7_acc']:.3f} +- {row['amf7_std']:.3f} | "
            f"{row['best_baseline']} {row['best_baseline_acc']:.3f} | "
            f"{row['gap']:+.3f} | {row['amf7_f1']:.3f} | {row['amf7_fit']:.2f} | "
            f"{row['amf7_predict']:.4f} | {row['amf7_mb']:.2f} | "
            f"{row['amf7_cells']:.1f} | {row['amf7_experts']:.1f} |"
        )
    avg_gap = float(np.mean(gaps)) if gaps else 0.0
    report = f"""# Fase 7 - AMF7 SuperField

Objetivo: modificar la arquitectura hasta que supere a modelos clasicos fuertes.

Seeds: {results['seeds']}
Modelos de millones de parametros incluidos: {results['config']['include_million_parameter_models']}
scikit-learn: {results['environment']['sklearn']}
Tiempo total: {results['elapsed_seconds']:.1f} s

## Tabla principal

{chr(10).join(table)}

## Score global

- Wins AMF7 vs mejor clasico por dataset: {wins}/{len(rows)}
- Gap promedio AMF7 - mejor clasico: {avg_gap:+.4f}

## Lectura

AMF7 es un supercampo hibrido: conserva memorias morfogenicas locales AMF,
agrega expertos globales fuertes y aprende pesos/estrategia en validation. El
test permanece sellado hasta la evaluacion final. Si el gap sigue negativo,
todavia no se debe declarar la fase ganada.
"""
    (out / "FASE7_RESULTADOS.md").write_text(report, encoding="utf-8")
    (out / "FASE7_TABLA_PAPER.md").write_text("# Fase 7 - Tabla tipo paper\n\n" + "\n".join(table) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 7 AMF7 superfield benchmark.")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--no-million", action="store_true", help="Disable million-parameter MLP expert/baseline.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_phase7(args)
    write_reports(results)
    print("report: results/FASE7_RESULTADOS.md")
    print(f"elapsed_seconds: {results['elapsed_seconds']:.2f}")
    for row in make_paper_rows(results):
        print(
            f"{row['dataset']}: AMF7 {row['amf7_acc']:.4f} "
            f"best {row['best_baseline']} {row['best_baseline_acc']:.4f} "
            f"gap {row['gap']:+.4f}"
        )


if __name__ == "__main__":
    main()
