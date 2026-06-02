from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import pickle
import tracemalloc

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


EPS = 1e-9


@dataclass(frozen=True)
class TimedFit:
    model: Any
    fit_seconds: float
    peak_ram_mb: float


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}


def summarize(records: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(records)}
    for key in keys:
        vals = []
        for record in records:
            value = record.get(key)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                vals.append(numeric)
        out[key] = mean_std(vals)
    return out


def timed_fit(factory: Callable[[], Any], x: np.ndarray, y: np.ndarray) -> TimedFit:
    model = factory()
    tracemalloc.start()
    start = perf_counter()
    model.fit(x, y)
    fit_seconds = perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return TimedFit(model=model, fit_seconds=fit_seconds, peak_ram_mb=peak / (1024.0 * 1024.0))


def predict_measure(model: Any, x: np.ndarray) -> tuple[np.ndarray, float]:
    start = perf_counter()
    pred = model.predict(x)
    return np.asarray(pred, dtype=int), perf_counter() - start


def model_memory_mb(model: Any) -> float:
    if hasattr(model, "memory_bytes"):
        return float(model.memory_bytes()) / (1024.0 * 1024.0)
    try:
        return float(len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def model_complexity_summary(model: Any) -> dict[str, Any]:
    if hasattr(model, "summary"):
        return dict(model.summary())
    out: dict[str, Any] = {}
    for attr in ("n_features_in_", "n_iter_", "n_estimators", "classes_"):
        if hasattr(model, attr):
            value = getattr(model, attr)
            if isinstance(value, np.ndarray):
                value = value.tolist()
            out[attr] = value
    if hasattr(model, "estimators_"):
        out["estimators_fit"] = len(getattr(model, "estimators_"))
    return out


def evaluate_model(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    fit_seconds: float,
    peak_ram_mb: float,
    n_train: int,
) -> dict[str, Any]:
    pred, predict_seconds = predict_measure(model, x)
    summary = model_complexity_summary(model)
    model_mb = model_memory_mb(model)
    return {
        "accuracy": accuracy(y, pred),
        "balanced_accuracy": balanced_accuracy(y, pred),
        "macro_f1": macro_f1(y, pred),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "fit_samples_per_second": float(n_train / max(fit_seconds, EPS)),
        "predict_samples_per_second": float(len(x) / max(predict_seconds, EPS)),
        "peak_fit_ram_mb": float(peak_ram_mb),
        "model_mb": float(model_mb),
        "cells": summary.get("cells"),
        "top_features": summary.get("top_features"),
        "avg_candidates": summary.get("avg_candidates"),
        "avg_votes": summary.get("avg_votes"),
        "summary": summary,
    }
