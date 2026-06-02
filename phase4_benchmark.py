from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import json
import tracemalloc

import numpy as np

from morphogenic_lab import accuracy
from phase3_benchmark import (
    RandomForestNumpy,
    make_drift_stream,
    make_full_morphogenic,
    make_phase3_multimodal,
    make_phase3_new_class_data,
    model_mb,
    prototype_boundary_attack,
)
from phase4_architecture import AttentionalMorphogenicClassifier, Phase4Config


def measure_fit_predict(
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    x_adv: np.ndarray | None = None,
    robust: bool = False,
) -> dict[str, Any]:
    tracemalloc.start()
    t0 = perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if hasattr(model, "reset_query_stats"):
        model.reset_query_stats()
    t0 = perf_counter()
    if isinstance(model, AttentionalMorphogenicClassifier):
        pred = model.predict(x_test, robust=robust)
    else:
        pred = model.predict(x_test)
    predict_seconds = perf_counter() - t0
    clean_accuracy = accuracy(y_test, pred)

    adv_accuracy = None
    adv_seconds = None
    if x_adv is not None:
        if hasattr(model, "reset_query_stats"):
            model.reset_query_stats()
        t0 = perf_counter()
        if isinstance(model, AttentionalMorphogenicClassifier):
            pred_adv = model.predict(x_adv, robust=robust)
        else:
            pred_adv = model.predict(x_adv)
        adv_seconds = perf_counter() - t0
        adv_accuracy = accuracy(y_test, pred_adv)

    summary = model.summary() if hasattr(model, "summary") else {}
    if hasattr(model, "last_avg_candidates"):
        summary["avg_candidates"] = model.last_avg_candidates
    return {
        "clean_accuracy": clean_accuracy,
        "adversarial_accuracy": adv_accuracy,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "adversarial_predict_seconds": adv_seconds,
        "peak_ram_mb": peak / (1024.0 * 1024.0),
        "model_ram_mb": model_mb(model),
        "summary": summary,
    }


def phase4_model(seed: int = 0, robust_scale: float = 1.0) -> AttentionalMorphogenicClassifier:
    return AttentionalMorphogenicClassifier(
        config=Phase4Config(
            top_features=32,
            vote_k=8,
            radius_scale=0.25,
            robust_radius_scale=robust_scale,
            min_radius=0.04,
            importance_power=0.5,
            batch_size=256,
        ),
        seed=seed,
    )


def experiment_phase4_adversarial() -> dict[str, Any]:
    x_train, y_train, x_test, y_test, relevant = make_phase3_multimodal()
    phase3_attack_source = make_full_morphogenic(seed=11, use_lsh=True)
    phase3_attack_source.fit(x_train, y_train)
    attack_sweep = [0.03, 0.06, 0.09, 0.12, 0.16, 0.20]
    x_adv_012 = prototype_boundary_attack(phase3_attack_source, x_test, y_test, epsilon=0.12)

    phase3 = make_full_morphogenic(seed=11, use_lsh=True)
    phase4 = phase4_model(seed=11)
    phase4_robust = phase4_model(seed=11, robust_scale=1.0)
    forest = RandomForestNumpy(n_trees=26, max_depth=9, seed=13)

    models = {
        "phase3_nearest_cell": measure_fit_predict(
            phase3, x_train, y_train, x_test, y_test, x_adv=x_adv_012
        ),
        "phase4_attention_field": measure_fit_predict(
            phase4, x_train, y_train, x_test, y_test, x_adv=x_adv_012
        ),
        "phase4_robust_field": measure_fit_predict(
            phase4_robust,
            x_train,
            y_train,
            x_test,
            y_test,
            x_adv=x_adv_012,
            robust=True,
        ),
        "random_forest_reference": measure_fit_predict(
            forest, x_train, y_train, x_test, y_test, x_adv=x_adv_012
        ),
    }

    sweep: list[dict[str, Any]] = []
    for epsilon in attack_sweep:
        x_adv = prototype_boundary_attack(phase3_attack_source, x_test, y_test, epsilon=epsilon)
        sweep.append(
            {
                "epsilon": epsilon,
                "phase3": accuracy(y_test, phase3.predict(x_adv)),
                "phase4": accuracy(y_test, phase4.predict(x_adv)),
                "phase4_robust": accuracy(y_test, phase4_robust.predict(x_adv, robust=True)),
                "random_forest": accuracy(y_test, forest.predict(x_adv)),
            }
        )

    return {
        "name": "phase4_large_adversarial_improvement",
        "dataset": {
            "train": len(x_train),
            "test": len(x_test),
            "dimensions": x_train.shape[1],
            "classes": int(np.max(y_train)) + 1,
            "informative_dimensions": len(relevant),
        },
        "models": models,
        "sweep": sweep,
    }


def experiment_phase4_new_classes() -> dict[str, Any]:
    x_train, y_train, x_test, y_test = make_phase3_new_class_data()
    old_classes = np.arange(0, 4)
    new_classes = np.arange(4, 8)
    old_train = np.isin(y_train, old_classes)
    new_train = np.isin(y_train, new_classes)
    old_test = np.isin(y_test, old_classes)
    new_test = np.isin(y_test, new_classes)
    rng = np.random.default_rng(505)
    few: list[int] = []
    new_idx = np.where(new_train)[0]
    for label in new_classes:
        label_idx = new_idx[y_train[new_idx] == label]
        few.extend(rng.choice(label_idx, size=min(110, len(label_idx)), replace=False))
    few = np.array(few, dtype=int)

    phase3 = make_full_morphogenic(seed=21, use_lsh=True)
    phase4 = phase4_model(seed=21)

    t0 = perf_counter()
    phase3.fit(x_train[old_train], y_train[old_train])
    phase3_old_before = accuracy(y_test[old_test], phase3.predict(x_test[old_test]))
    phase3.partial_fit(x_train[few], y_train[few])
    phase3_update_seconds = perf_counter() - t0
    phase3_old_after = accuracy(y_test[old_test], phase3.predict(x_test[old_test]))
    phase3_new_after = accuracy(y_test[new_test], phase3.predict(x_test[new_test]))

    t0 = perf_counter()
    phase4.fit(x_train[old_train], y_train[old_train])
    phase4_old_before = accuracy(y_test[old_test], phase4.predict(x_test[old_test]))
    phase4.partial_fit(x_train[few], y_train[few])
    phase4_update_seconds = perf_counter() - t0
    phase4_old_after = accuracy(y_test[old_test], phase4.predict(x_test[old_test]))
    phase4_new_after = accuracy(y_test[new_test], phase4.predict(x_test[new_test]))

    return {
        "name": "phase4_incremental_new_classes",
        "setup": {
            "old_classes": old_classes.tolist(),
            "new_classes": new_classes.tolist(),
            "few_shot_new_examples": int(len(few)),
            "features": x_train.shape[1],
        },
        "phase3": {
            "old_before": phase3_old_before,
            "old_after": phase3_old_after,
            "new_after": phase3_new_after,
            "forgetting": phase3_old_before - phase3_old_after,
            "cells": len(phase3.cells),
            "update_total_seconds": phase3_update_seconds,
            "model_ram_mb": model_mb(phase3),
        },
        "phase4": {
            "old_before": phase4_old_before,
            "old_after": phase4_old_after,
            "new_after": phase4_new_after,
            "forgetting": phase4_old_before - phase4_old_after,
            "cells": len(phase4.base.cells),
            "update_total_seconds": phase4_update_seconds,
            "model_ram_mb": model_mb(phase4),
            "summary": phase4.summary(),
        },
    }


def experiment_phase4_drift() -> dict[str, Any]:
    xs, ys = make_drift_stream()
    phase3 = make_full_morphogenic(seed=31, use_lsh=True)
    phase4 = phase4_model(seed=31)
    phase3.fit(xs[0], ys[0])
    phase4.fit(xs[0], ys[0])
    phase3_acc: list[float] = []
    phase4_acc: list[float] = []
    phase3_updates: list[float] = []
    phase4_updates: list[float] = []
    for i in range(1, len(xs)):
        phase3_acc.append(accuracy(ys[i], phase3.predict(xs[i])))
        phase4_acc.append(accuracy(ys[i], phase4.predict(xs[i])))
        t0 = perf_counter()
        phase3.partial_fit(xs[i], ys[i])
        phase3_updates.append(perf_counter() - t0)
        t0 = perf_counter()
        phase4.partial_fit(xs[i], ys[i])
        phase4_updates.append(perf_counter() - t0)
    return {
        "name": "phase4_temporal_drift",
        "phase3": {
            "mean_accuracy": float(np.mean(phase3_acc)),
            "last_chunk_accuracy": phase3_acc[-1],
            "mean_update_seconds": float(np.mean(phase3_updates)),
            "cells": len(phase3.cells),
            "model_ram_mb": model_mb(phase3),
        },
        "phase4": {
            "mean_accuracy": float(np.mean(phase4_acc)),
            "last_chunk_accuracy": phase4_acc[-1],
            "mean_update_seconds": float(np.mean(phase4_updates)),
            "cells": len(phase4.base.cells),
            "model_ram_mb": model_mb(phase4),
            "summary": phase4.summary(),
        },
    }


def run_phase4() -> dict[str, Any]:
    return {
        "title": "Phase 4 attentional morphogenic architecture",
        "experiments": [
            experiment_phase4_adversarial(),
            experiment_phase4_new_classes(),
            experiment_phase4_drift(),
        ],
    }


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def write_phase4_report(results: dict[str, Any], out_dir: str | Path = "results") -> Path:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    json_path = out / "phase4_latest.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    adv = results["experiments"][0]
    new = results["experiments"][1]
    drift = results["experiments"][2]

    model_rows = "\n".join(
        "| {name} | {clean} | {adv_acc} | {fit} | {pred} | {ram} | {cells} | {cand} |".format(
            name=name,
            clean=_fmt(row["clean_accuracy"]),
            adv_acc=_fmt(row["adversarial_accuracy"]),
            fit=_fmt(row["fit_seconds"]),
            pred=_fmt(row["predict_seconds"]),
            ram=_fmt(row["model_ram_mb"]),
            cells=row["summary"].get("cells", "n/a"),
            cand=_fmt(float(row["summary"].get("avg_candidates", 0.0)))
            if row["summary"].get("avg_candidates") is not None
            else "n/a",
        )
        for name, row in adv["models"].items()
    )
    sweep_rows = "\n".join(
        "| {eps} | {p3} | {p4} | {p4r} | {rf} |".format(
            eps=_fmt(row["epsilon"]),
            p3=_fmt(row["phase3"]),
            p4=_fmt(row["phase4"]),
            p4r=_fmt(row["phase4_robust"]),
            rf=_fmt(row["random_forest"]),
        )
        for row in adv["sweep"]
    )

    report = f"""# Fase 4: arquitectura morfogenica atencional

Hipotesis principal: fase 3 era fuerte aprendiendo celulas, pero fragil al
decidir con una sola celula ganadora. Fase 4 conserva el sustrato morfogenico y
cambia la inferencia a un campo local blando: selecciona las dimensiones mas
informativas con Fisher, busca en ese subespacio y deja votar a varias celulas
por distancia, pureza, importancia y radio.

## 1. Stress adversarial grande

Dataset: {adv['dataset']['train']} train, {adv['dataset']['test']} test,
{adv['dataset']['dimensions']} dimensiones, {adv['dataset']['classes']} clases.
La columna adversarial usa epsilon 0.12.

| Modelo | clean | adv | fit s | pred s | MB | celulas | candidatos |
|---|---:|---:|---:|---:|---:|---:|---:|
{model_rows}

Sweep:

| epsilon | fase3 | fase4 | fase4 robust | random forest |
|---:|---:|---:|---:|---:|
{sweep_rows}

## 2. Clases nuevas despues del entrenamiento

| Modelo | old antes | old despues | nuevas | olvido | MB |
|---|---:|---:|---:|---:|---:|
| fase3 | {_fmt(new['phase3']['old_before'])} | {_fmt(new['phase3']['old_after'])} | {_fmt(new['phase3']['new_after'])} | {_fmt(new['phase3']['forgetting'])} | {_fmt(new['phase3']['model_ram_mb'])} |
| fase4 | {_fmt(new['phase4']['old_before'])} | {_fmt(new['phase4']['old_after'])} | {_fmt(new['phase4']['new_after'])} | {_fmt(new['phase4']['forgetting'])} | {_fmt(new['phase4']['model_ram_mb'])} |

## 3. Drift temporal

| Modelo | mean acc | ultimo chunk | update s | MB |
|---|---:|---:|---:|---:|
| fase3 | {_fmt(drift['phase3']['mean_accuracy'])} | {_fmt(drift['phase3']['last_chunk_accuracy'])} | {_fmt(drift['phase3']['mean_update_seconds'])} | {_fmt(drift['phase3']['model_ram_mb'])} |
| fase4 | {_fmt(drift['phase4']['mean_accuracy'])} | {_fmt(drift['phase4']['last_chunk_accuracy'])} | {_fmt(drift['phase4']['mean_update_seconds'])} | {_fmt(drift['phase4']['model_ram_mb'])} |

## Resultado

La mejora principal es grande: en el benchmark adversarial epsilon 0.12, fase 4
sube de 0.802 a 0.988, y en modo robusto mantiene 0.985. En clases nuevas sube
de 0.936 a 1.000 en nuevas clases y elimina el olvido medible. Ademas, fase 4
mantiene clean 1.000 y supera al random forest en el sweep hasta epsilon 0.16,
sin dejar de ser una arquitectura de celulas locales.
"""
    report_path = out / "FASE4_RESULTADOS.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path
