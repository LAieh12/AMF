from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import math
import tracemalloc
import numpy as np


EPS = 1e-9


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None) -> list[list[int]]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = int(max(np.max(y_true), np.max(y_pred))) + 1 if n_classes is None else n_classes
    cm = np.zeros((n, n), dtype=int)
    for a, b in zip(y_true, y_pred):
        cm[int(a), int(b)] += 1
    return cm.tolist()


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = np.asarray(confusion_matrix(y_true, y_pred), dtype=np.float64)
    recalls = []
    for i in range(len(cm)):
        denom = cm[i].sum()
        if denom > 0:
            recalls.append(cm[i, i] / denom)
    return float(np.mean(recalls)) if recalls else 0.0


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = np.asarray(confusion_matrix(y_true, y_pred), dtype=np.float64)
    scores = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)
        scores.append(2.0 * precision * recall / (precision + recall + EPS))
    return float(np.mean(scores)) if scores else 0.0


def mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)) if len(arr) else 0.0,
        "std": float(np.std(arr)) if len(arr) else 0.0,
    }


def summarize_records(records: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {"n": len(records)}
    for key in keys:
        vals = [float(r[key]) for r in records if key in r and r[key] is not None and math.isfinite(float(r[key]))]
        summary[key] = mean_std(vals)
    return summary


def model_memory_mb(model: Any) -> float:
    if hasattr(model, "memory_bytes"):
        return float(model.memory_bytes()) / (1024.0 * 1024.0)
    return 0.0


def model_summary(model: Any) -> dict[str, Any]:
    if hasattr(model, "summary"):
        return dict(model.summary())
    if hasattr(model, "last_avg_candidates"):
        return {"avg_candidates": float(model.last_avg_candidates)}
    return {}


def morphogenic_efficiency_score(
    acc: float,
    robust_acc: float | None,
    model_mb: float,
    predict_seconds: float,
    incremental_score: float | None = None,
) -> float:
    robust = acc if robust_acc is None else robust_acc
    incremental = acc if incremental_score is None else incremental_score
    denom = max(model_mb, 1e-4) * max(predict_seconds, 1e-4)
    return float((acc * robust * incremental) / denom)


@dataclass
class FitPredictResult:
    model: Any
    record: dict[str, Any]


def fit_predict_measure(
    name: str,
    factory: Callable[[], Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    predict_kwargs: dict[str, Any] | None = None,
) -> FitPredictResult:
    predict_kwargs = predict_kwargs or {}
    model = factory()
    tracemalloc.start()
    fit_start = perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = perf_counter() - fit_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    pred_start = perf_counter()
    pred = model.predict(x_test, **predict_kwargs) if predict_kwargs else model.predict(x_test)
    predict_seconds = perf_counter() - pred_start
    acc = accuracy(y_test, pred)
    summary = model_summary(model)
    record = {
        "name": name,
        "accuracy": acc,
        "balanced_accuracy": balanced_accuracy(y_test, pred),
        "macro_f1": macro_f1(y_test, pred),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "peak_ram_mb": peak / (1024.0 * 1024.0),
        "model_mb": model_memory_mb(model),
        "cells": summary.get("cells"),
        "avg_candidates": summary.get("avg_candidates"),
        "avg_votes": summary.get("avg_votes"),
        "mes": morphogenic_efficiency_score(
            acc=acc,
            robust_acc=None,
            model_mb=model_memory_mb(model),
            predict_seconds=predict_seconds,
        ),
        "confusion_matrix": confusion_matrix(y_test, pred),
    }
    return FitPredictResult(model=model, record=record)
