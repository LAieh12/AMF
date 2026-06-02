from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import sklearn

from phase6_datasets import Phase6Dataset, SplitBundle, make_split, select_datasets
from phase6_malicious import (
    append_noise_features,
    corrupt_labels,
    fisher_feature_ranking,
    gaussian_noise_attack,
    nearest_opposite_interpolation,
    top_feature_shuffle_attack,
    top_feature_zero_attack,
)
from phase6_metrics import (
    accuracy,
    balanced_accuracy,
    evaluate_model,
    macro_f1,
    predict_measure,
    summarize,
    timed_fit,
)
from phase6_sklearn_baselines import ModelCandidate, official_model_grids


DEFAULT_SEEDS = list(range(10))
SUMMARY_KEYS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "fit_seconds",
    "predict_seconds",
    "fit_samples_per_second",
    "predict_samples_per_second",
    "peak_fit_ram_mb",
    "model_mb",
    "cells",
    "top_features",
    "avg_candidates",
    "avg_votes",
]
MALICIOUS_FAMILIES = ["AMF5", "sk_logistic", "sk_extra_trees", "sk_rbf_svc", "sk_mlp_small"]
NOISE_STRESS_FAMILIES = ["AMF5", "sk_logistic", "sk_extra_trees"]


def _fmt(stat: dict[str, float]) -> str:
    return f"{stat['mean']:.3f} +- {stat['std']:.3f}"


def _dataset_meta(dataset: Phase6Dataset) -> dict[str, Any]:
    counts = np.bincount(dataset.y.astype(int))
    return {
        "name": dataset.name,
        "source": dataset.source,
        "kind": dataset.kind,
        "note": dataset.note,
        "samples": int(len(dataset.x)),
        "features": int(dataset.x.shape[1]),
        "classes": int(dataset.n_classes),
        "class_counts": counts.astype(int).tolist(),
    }


def _stratified_cap(dataset: Phase6Dataset, max_samples: int | None, seed: int = 12345) -> Phase6Dataset:
    if max_samples is None or len(dataset.x) <= max_samples:
        return dataset
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in sorted(np.unique(dataset.y).tolist()):
        idx = np.where(dataset.y == label)[0]
        take = max(1, int(round(max_samples * len(idx) / len(dataset.y))))
        take = min(take, len(idx))
        selected.extend(rng.choice(idx, size=take, replace=False).tolist())
    if len(selected) > max_samples:
        selected = rng.choice(np.asarray(selected), size=max_samples, replace=False).tolist()
    rng.shuffle(selected)
    note = dataset.note
    cap_note = f"stratified cap to {len(selected)} samples for CPU paper run"
    note = f"{note}; {cap_note}" if note else cap_note
    idx = np.asarray(selected, dtype=int)
    return Phase6Dataset(
        name=dataset.name,
        x=dataset.x[idx],
        y=dataset.y[idx],
        source=dataset.source,
        kind=dataset.kind,
        note=note,
    )


def load_run_datasets(names: list[str] | None, max_samples: int | None) -> list[Phase6Dataset]:
    datasets = select_datasets(names)
    return [_stratified_cap(dataset, max_samples=max_samples) for dataset in datasets]


def _score_prediction(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
    }


def select_candidate(
    candidates: list[ModelCandidate],
    split: SplitBundle,
    metric: str = "macro_f1",
) -> dict[str, Any]:
    validations = []
    best: dict[str, Any] | None = None
    for candidate in candidates:
        try:
            fitted = timed_fit(candidate.factory, split.x_train, split.y_train)
            pred, val_seconds = predict_measure(fitted.model, split.x_val)
            scores = _score_prediction(split.y_val, pred)
            record = {
                "family": candidate.family,
                "variant": candidate.variant,
                "validation_predict_seconds": val_seconds,
                "validation_fit_seconds": fitted.fit_seconds,
                **scores,
            }
            validations.append(record)
            key = (scores[metric], scores["accuracy"], -fitted.fit_seconds)
            if best is None or key > best["key"]:
                best = {"candidate": candidate, "record": record, "key": key}
        except Exception as exc:
            validations.append(
                {
                    "family": candidate.family,
                    "variant": candidate.variant,
                    "error": repr(exc),
                }
            )
    if best is None:
        raise RuntimeError(f"All candidates failed for {candidates[0].family if candidates else 'empty grid'}")
    return {
        "candidate": best["candidate"],
        "validation": best["record"],
        "all_validations": validations,
    }


def fit_selected_on_train_val(
    selected: ModelCandidate,
    split: SplitBundle,
    y_train_val_override: np.ndarray | None = None,
) -> tuple[Any, float, float, int]:
    x_train_val, y_train_val = split.train_val()
    if y_train_val_override is not None:
        y_train_val = y_train_val_override
    fitted = timed_fit(selected.factory, x_train_val, y_train_val)
    return fitted.model, fitted.fit_seconds, fitted.peak_ram_mb, len(x_train_val)


def run_main_suite(
    datasets: list[Phase6Dataset],
    seeds: list[int],
    include_mlp: bool,
    include_rbf: bool,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            split = make_split(dataset, seed)
            grids = official_model_grids(
                seed=seed,
                n_features=dataset.n_features,
                n_train=len(split.y_train),
                include_mlp=include_mlp,
                include_rbf=include_rbf,
            )
            for family, candidates in grids.items():
                selected_info = select_candidate(candidates, split)
                selected = selected_info["candidate"]
                validation_records.extend(
                    {
                        **record,
                        "dataset": dataset.name,
                        "seed": seed,
                        "selected": record.get("variant") == selected.variant,
                    }
                    for record in selected_info["all_validations"]
                )
                model, fit_seconds, peak_ram_mb, n_train = fit_selected_on_train_val(selected, split)
                record = evaluate_model(
                    model=model,
                    x=split.x_test,
                    y=split.y_test,
                    fit_seconds=fit_seconds,
                    peak_ram_mb=peak_ram_mb,
                    n_train=n_train,
                )
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

    summary: dict[str, dict[str, Any]] = {}
    for dataset in sorted({record["dataset"] for record in records}):
        summary[dataset] = {}
        for family in sorted({record["family"] for record in records if record["dataset"] == dataset}):
            subset = [record for record in records if record["dataset"] == dataset and record["family"] == family]
            summary[dataset][family] = summarize(subset, SUMMARY_KEYS)

    return {
        "name": "phase6_main_external_reproducibility",
        "seeds": seeds,
        "metric_for_validation": "macro_f1",
        "records": records,
        "validation_records": validation_records,
        "summary": summary,
    }


def _families_from_grids(grids: dict[str, list[ModelCandidate]], requested: list[str]) -> dict[str, list[ModelCandidate]]:
    return {family: grids[family] for family in requested if family in grids}


def _attack_cases(split: SplitBundle, seed: int) -> dict[str, np.ndarray]:
    ranking = fisher_feature_ranking(split.x_train, split.y_train)
    return {
        "gaussian_noise_sigma0.45": gaussian_noise_attack(split.x_test, seed=seed + 1100, sigma=0.45),
        "top_feature_zero_12pct": top_feature_zero_attack(split.x_test, ranking, rate=0.12),
        "top_feature_shuffle_12pct": top_feature_shuffle_attack(split.x_test, ranking, seed=seed + 1200, rate=0.12),
        "nearest_opposite_interpolation_45pct": nearest_opposite_interpolation(
            split.x_test,
            split.y_test,
            split.x_train,
            split.y_train,
            alpha=0.45,
        ),
    }


def _attack_subset(split: SplitBundle, seed: int, max_eval: int = 600) -> tuple[np.ndarray, np.ndarray]:
    if len(split.y_test) <= max_eval:
        idx = np.arange(len(split.y_test), dtype=int)
        return idx, split.y_test
    rng = np.random.default_rng(seed + 1700)
    selected: list[int] = []
    for label in sorted(np.unique(split.y_test).tolist()):
        label_idx = np.where(split.y_test == label)[0]
        take = max(1, int(round(max_eval * len(label_idx) / len(split.y_test))))
        take = min(take, len(label_idx))
        selected.extend(rng.choice(label_idx, size=take, replace=False).tolist())
    if len(selected) > max_eval:
        selected = rng.choice(np.asarray(selected), size=max_eval, replace=False).tolist()
    selected_idx = np.asarray(sorted(selected), dtype=int)
    return selected_idx, split.y_test[selected_idx]


def run_malicious_suite(
    datasets: list[Phase6Dataset],
    seeds: list[int],
    include_mlp: bool,
    include_rbf: bool,
) -> dict[str, Any]:
    targeted_names = ["madelon", "optdigits", "spambase", "satimage", "pendigits"]
    targeted = [dataset for dataset in datasets if dataset.name in targeted_names]
    if not targeted:
        targeted = datasets[: min(3, len(datasets))]
    records: list[dict[str, Any]] = []
    for dataset in targeted:
        for seed in seeds:
            split = make_split(dataset, seed)
            attack_idx, y_attack = _attack_subset(split, seed)
            attack_split = SplitBundle(
                x_train=split.x_train,
                y_train=split.y_train,
                x_val=split.x_val,
                y_val=split.y_val,
                x_test=split.x_test[attack_idx],
                y_test=y_attack,
                mean=split.mean,
                std=split.std,
                train_idx=split.train_idx,
                val_idx=split.val_idx,
                test_idx=split.test_idx[attack_idx],
            )
            attack_cases = _attack_cases(attack_split, seed)
            grids = official_model_grids(
                seed=seed,
                n_features=dataset.n_features,
                n_train=len(split.y_train),
                include_mlp=include_mlp,
                include_rbf=include_rbf,
            )
            for family, candidates in _families_from_grids(grids, MALICIOUS_FAMILIES).items():
                selected = select_candidate(candidates, split)["candidate"]
                model, fit_seconds, peak_ram_mb, n_train = fit_selected_on_train_val(selected, split)
                clean = evaluate_model(model, split.x_test, split.y_test, fit_seconds, peak_ram_mb, n_train)
                records.append(
                    {
                        "dataset": dataset.name,
                        "seed": seed,
                        "family": family,
                        "selected_variant": selected.variant,
                        "attack": "clean",
                        "n_attack_eval": int(len(attack_idx)),
                        **{k: clean[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "model_mb")},
                    }
                )
                for attack_name, attacked_x in attack_cases.items():
                    pred, predict_seconds = predict_measure(model, attacked_x)
                    scores = _score_prediction(y_attack, pred)
                    records.append(
                        {
                            "dataset": dataset.name,
                            "seed": seed,
                            "family": family,
                            "selected_variant": selected.variant,
                            "attack": attack_name,
                            "n_attack_eval": int(len(attack_idx)),
                            "predict_seconds": predict_seconds,
                            **scores,
                            "model_mb": clean["model_mb"],
                        }
                    )

                x_train_val, y_train_val = split.train_val()
                poisoned_y = corrupt_labels(y_train_val, seed=seed + 1300, rate=0.20)
                poisoned = timed_fit(selected.factory, x_train_val, poisoned_y)
                pred, predict_seconds = predict_measure(poisoned.model, split.x_test)
                scores = _score_prediction(split.y_test, pred)
                records.append(
                    {
                        "dataset": dataset.name,
                        "seed": seed,
                        "family": family,
                        "selected_variant": selected.variant,
                        "attack": "label_poison_20pct",
                        "n_attack_eval": int(len(split.y_test)),
                        "predict_seconds": predict_seconds,
                        "fit_seconds": poisoned.fit_seconds,
                        "peak_fit_ram_mb": poisoned.peak_ram_mb,
                        **scores,
                    }
                )

    summary: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for dataset in sorted({record["dataset"] for record in records}):
        for family in sorted({record["family"] for record in records if record["dataset"] == dataset}):
            summary[dataset][family] = {}
            for attack in sorted(
                {record["attack"] for record in records if record["dataset"] == dataset and record["family"] == family}
            ):
                subset = [
                    record
                    for record in records
                    if record["dataset"] == dataset and record["family"] == family and record["attack"] == attack
                ]
                summary[dataset][family][attack] = summarize(
                    subset,
                    ["accuracy", "balanced_accuracy", "macro_f1", "predict_seconds", "fit_seconds", "model_mb"],
                )

    return {
        "name": "phase6_malicious_failure_search",
        "seeds": seeds,
        "records": records,
        "summary": json.loads(json.dumps(summary)),
    }


def run_noise_feature_stress(
    datasets: list[Phase6Dataset],
    seeds: list[int],
    include_rbf: bool,
) -> dict[str, Any]:
    targeted_names = ["madelon", "spambase"]
    targeted = [dataset for dataset in datasets if dataset.name in targeted_names]
    records: list[dict[str, Any]] = []
    for dataset in targeted:
        for seed in seeds:
            split = make_split(dataset, seed)
            for multiplier in [1.0]:
                x_train, x_val, x_test = append_noise_features(
                    split.x_train,
                    split.x_val,
                    split.x_test,
                    seed=seed + int(100 * multiplier),
                    multiplier=multiplier,
                )
                noisy_split = SplitBundle(
                    x_train=x_train,
                    y_train=split.y_train,
                    x_val=x_val,
                    y_val=split.y_val,
                    x_test=x_test,
                    y_test=split.y_test,
                    mean=np.zeros(x_train.shape[1], dtype=np.float64),
                    std=np.ones(x_train.shape[1], dtype=np.float64),
                    train_idx=split.train_idx,
                    val_idx=split.val_idx,
                    test_idx=split.test_idx,
                )
                grids = official_model_grids(
                    seed=seed,
                    n_features=noisy_split.x_train.shape[1],
                    n_train=len(noisy_split.y_train),
                    include_mlp=False,
                    include_rbf=include_rbf,
                )
                for family, candidates in _families_from_grids(grids, NOISE_STRESS_FAMILIES).items():
                    selected = select_candidate(candidates, noisy_split)["candidate"]
                    model, fit_seconds, peak_ram_mb, n_train = fit_selected_on_train_val(selected, noisy_split)
                    record = evaluate_model(model, noisy_split.x_test, noisy_split.y_test, fit_seconds, peak_ram_mb, n_train)
                    record.update(
                        {
                            "dataset": dataset.name,
                            "seed": seed,
                            "family": family,
                            "selected_variant": selected.variant,
                            "noise_multiplier": multiplier,
                            "original_features": int(dataset.n_features),
                            "stress_features": int(noisy_split.x_train.shape[1]),
                        }
                    )
                    records.append(record)

    summary: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for dataset in sorted({record["dataset"] for record in records}):
        for family in sorted({record["family"] for record in records if record["dataset"] == dataset}):
            summary[dataset][family] = {}
            for multiplier in sorted(
                {record["noise_multiplier"] for record in records if record["dataset"] == dataset and record["family"] == family}
            ):
                subset = [
                    record
                    for record in records
                    if record["dataset"] == dataset and record["family"] == family and record["noise_multiplier"] == multiplier
                ]
                summary[dataset][family][str(multiplier)] = summarize(subset, SUMMARY_KEYS)
    return {
        "name": "phase6_noise_feature_scalability_stress",
        "seeds": seeds,
        "records": records,
        "summary": json.loads(json.dumps(summary)),
    }


def _best_sklearn(summary_for_dataset: dict[str, Any]) -> tuple[str, float]:
    candidates = [
        (family, stats["accuracy"]["mean"])
        for family, stats in summary_for_dataset.items()
        if family != "AMF5" and not family.startswith("sk_dummy")
    ]
    return max(candidates, key=lambda item: item[1])


def _worst_attack(malicious: dict[str, Any], dataset: str, family: str) -> tuple[str, float] | None:
    family_attacks = malicious["summary"].get(dataset, {}).get(family)
    if not family_attacks:
        return None
    rows = [
        (attack, stats["accuracy"]["mean"])
        for attack, stats in family_attacks.items()
        if attack != "clean" and stats["accuracy"]["n" if False else "mean"] >= 0.0
    ]
    if not rows:
        return None
    return min(rows, key=lambda item: item[1])


def make_paper_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    main = results["experiments"][0]
    malicious = results["experiments"][1]
    rows = []
    for dataset_meta in results["datasets"]:
        dataset = dataset_meta["name"]
        summary = main["summary"][dataset]
        amf = summary["AMF5"]
        best_name, best_acc = _best_sklearn(summary)
        worst = _worst_attack(malicious, dataset, "AMF5")
        rows.append(
            {
                "dataset": dataset,
                "n": dataset_meta["samples"],
                "d": dataset_meta["features"],
                "classes": dataset_meta["classes"],
                "amf5_acc": amf["accuracy"]["mean"],
                "amf5_acc_std": amf["accuracy"]["std"],
                "best_sklearn": best_name,
                "best_sklearn_acc": best_acc,
                "gap_vs_best": amf["accuracy"]["mean"] - best_acc,
                "amf5_fit_seconds": amf["fit_seconds"]["mean"],
                "amf5_predict_seconds": amf["predict_seconds"]["mean"],
                "amf5_model_mb": amf["model_mb"]["mean"],
                "amf5_cells": amf["cells"]["mean"],
                "amf5_worst_attack": None if worst is None else worst[0],
                "amf5_worst_attack_acc": None if worst is None else worst[1],
            }
        )
    return rows


def write_reports(results: dict[str, Any], out_dir: str | Path = "results") -> None:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    (out / "phase6_latest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    rows = make_paper_rows(results)
    table_lines = [
        "| Dataset | n | d | C | AMF5 acc | Best sklearn | Gap | AMF5 fit s | AMF5 pred s | AMF5 MB | Cells | Worst AMF5 attack |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        attack = "n/a"
        if row["amf5_worst_attack"] is not None:
            attack = f"{row['amf5_worst_attack']} ({row['amf5_worst_attack_acc']:.3f})"
        table_lines.append(
            f"| {row['dataset']} | {row['n']} | {row['d']} | {row['classes']} | "
            f"{row['amf5_acc']:.3f} +- {row['amf5_acc_std']:.3f} | "
            f"{row['best_sklearn']} {row['best_sklearn_acc']:.3f} | "
            f"{row['gap_vs_best']:+.3f} | {row['amf5_fit_seconds']:.3f} | "
            f"{row['amf5_predict_seconds']:.4f} | {row['amf5_model_mb']:.3f} | "
            f"{row['amf5_cells']:.1f} | {attack} |"
        )
    (out / "FASE6_TABLA_PAPER.md").write_text(
        "# Fase 6 - Tabla tipo paper\n\n" + "\n".join(table_lines) + "\n",
        encoding="utf-8",
    )

    dataset_lines = [
        f"- {d['name']}: n={d['samples']}, d={d['features']}, C={d['classes']}, source={d['source']}"
        for d in results["datasets"]
    ]
    main = results["experiments"][0]
    main_lines = []
    for row in rows:
        main_lines.append(
            f"- {row['dataset']}: AMF5 {row['amf5_acc']:.3f} +- {row['amf5_acc_std']:.3f}; "
            f"best sklearn {row['best_sklearn']} {row['best_sklearn_acc']:.3f}; "
            f"gap {row['gap_vs_best']:+.3f}; cells {row['amf5_cells']:.1f}; "
            f"fit {row['amf5_fit_seconds']:.3f}s, predict {row['amf5_predict_seconds']:.4f}s"
        )

    malicious = results["experiments"][1]
    attack_lines = []
    for dataset, family_map in malicious["summary"].items():
        if "AMF5" not in family_map:
            continue
        clean = family_map["AMF5"]["clean"]["accuracy"]
        worst = _worst_attack(malicious, dataset, "AMF5")
        if worst is None:
            continue
        attack_lines.append(
            f"- {dataset}: clean {_fmt(clean)}; peor ataque {worst[0]} acc {worst[1]:.3f}"
        )

    stress = results["experiments"][2]
    stress_lines = []
    for dataset, family_map in stress["summary"].items():
        if "AMF5" not in family_map:
            continue
        pieces = []
        for multiplier, stats in family_map["AMF5"].items():
            pieces.append(f"+{multiplier}x noise acc {_fmt(stats['accuracy'])}, MB {_fmt(stats['model_mb'])}")
        stress_lines.append(f"- {dataset}: " + "; ".join(pieces))

    report = f"""# Fase 6 - Reproducibilidad externa y paperizacion

Pregunta de esta fase: ya no solo si AMF5 puede funcionar, sino si empieza a
compararse honestamente contra modelos clasicos oficiales de scikit-learn.

Seeds: {results['seeds']}
Attack/stress seeds: {results['config']['attack_seeds']}
Split: train 60%, validation 20%, test 20%, estratificado por seed.
Validacion: cada familia elige variante por macro-F1 en validation y se reentrena
en train+validation antes de medir test.
scikit-learn: {results['environment']['sklearn']}
Tiempo total: {results['elapsed_seconds']:.1f} s

## Datasets

{chr(10).join(dataset_lines)}

## Tabla principal

{chr(10).join(table_lines)}

## Lectura por dataset

{chr(10).join(main_lines)}

## Busqueda de fallos con mala intencion

{chr(10).join(attack_lines)}

## Estres de alta dimension por features basura

{chr(10).join(stress_lines)}

## Lectura honesta

AMF5 queda como candidato local/compacto e incremental, no como ganador universal
de accuracy. Esta fase mide el costo real de competir contra `sklearn`: cuando
los modelos globales clasicos explotan bien la frontera, AMF5 queda atras; cuando
hay muchas features distractoras o ataques a subconjuntos informativos, la
anatomia Fisher + voto local muestra donde puede valer la pena seguir.
"""
    (out / "FASE6_RESULTADOS.md").write_text(report, encoding="utf-8")

    failures = f"""# FASE6_NOTAS_FALLOS

- Esta suite usa scikit-learn real ({results['environment']['sklearn']}), pero
  todavia no incluye modelos profundos grandes ni GPUs.
- La comparacion principal usa {len(results['seeds'])} seeds; la busqueda
  maliciosa usa {len(results['config']['attack_seeds'])} seeds para mantener la
  corrida CPU reproducible.
- Los datasets grandes pueden estar capados de forma estratificada si se corre
  con `--max-samples`; el valor usado aqui fue {results['config']['max_samples']}.
- Los datasets OpenML quedan como extension futura si se quiere dependencia de
  red/cache de OpenML; esta corrida priorizo UCI descargable y reproducible.
- Los ataques son maliciosos pero no optimizados por gradiente: ruido,
  oclusion/shuffle de features Fisher, interpolacion hacia clase opuesta,
  label poisoning y features basura.
- La memoria de scikit-learn se estima por serializacion `pickle`; la memoria
  nativa exacta puede diferir.
"""
    Path("FASE6_NOTAS_FALLOS.md").write_text(failures, encoding="utf-8")

    complete = """# FASE6_COMPLETADA

Entregables de Fase 6:

- `download_phase6_data.py`
- `phase6_datasets.py`
- `phase6_sklearn_baselines.py`
- `phase6_metrics.py`
- `phase6_malicious.py`
- `run_phase6.py`
- `README.md`
- `paper/AMF5_FORMALIZACION.md`
- `docs/AMF5_ARCHITECTURE.mmd`
- `results/phase6_latest.json`
- `results/FASE6_RESULTADOS.md`
- `results/FASE6_TABLA_PAPER.md`
- `FASE6_NOTAS_FALLOS.md`

La suite corre baselines oficiales de scikit-learn, usa train/validation/test,
10 seeds por defecto, datasets UCI adicionales, tabla tipo paper, formalizacion,
diagrama, complejidad medida y busqueda de fallos maliciosos.
"""
    Path("FASE6_COMPLETADA.md").write_text(complete, encoding="utf-8")


def run_phase6(args: argparse.Namespace) -> dict[str, Any]:
    start = perf_counter()
    seeds = list(range(args.seeds))
    attack_seeds = seeds[: max(1, min(args.attack_seeds, len(seeds)))]
    datasets = load_run_datasets(args.datasets, max_samples=args.max_samples)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        experiments = [
            run_main_suite(datasets, seeds, include_mlp=not args.skip_mlp, include_rbf=not args.skip_rbf),
            run_malicious_suite(datasets, attack_seeds, include_mlp=not args.skip_mlp, include_rbf=not args.skip_rbf),
            run_noise_feature_stress(datasets, attack_seeds, include_rbf=not args.skip_rbf),
        ]
    return {
        "title": "Phase 6 - External reproducibility and paperization",
        "seeds": seeds,
        "datasets": [_dataset_meta(dataset) for dataset in datasets],
        "config": {
            "max_samples": args.max_samples,
            "skip_mlp": args.skip_mlp,
            "skip_rbf": args.skip_rbf,
            "attack_seeds": attack_seeds,
            "datasets_requested": args.datasets,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "elapsed_seconds": perf_counter() - start,
        "experiments": experiments,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 6 AMF5 paperization benchmark.")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds, starting from 0.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset names to run.")
    parser.add_argument("--max-samples", type=int, default=6000, help="Stratified cap per dataset; use 0 for no cap.")
    parser.add_argument("--attack-seeds", type=int, default=3, help="Seeds used for malicious/stress suites.")
    parser.add_argument("--skip-mlp", action="store_true", help="Skip MLPClassifier baseline.")
    parser.add_argument("--skip-rbf", action="store_true", help="Skip RBF SVC baseline.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_samples == 0:
        args.max_samples = None
    results = run_phase6(args)
    write_reports(results)
    print(f"report: results/FASE6_RESULTADOS.md")
    print(f"paper_table: results/FASE6_TABLA_PAPER.md")
    print(f"elapsed_seconds: {results['elapsed_seconds']:.2f}")
    for row in make_paper_rows(results):
        print(
            f"{row['dataset']}: AMF5 {row['amf5_acc']:.4f} "
            f"best {row['best_sklearn']} {row['best_sklearn_acc']:.4f} "
            f"gap {row['gap_vs_best']:+.4f}"
        )


if __name__ == "__main__":
    main()
