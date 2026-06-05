from __future__ import annotations

import time
from typing import Any

import numpy as np

from phase12a_physicalai_world_probe import mae, mse


def timed_call(fn, *args, **kwargs):
    started = time.time()
    value = fn(*args, **kwargs)
    return value, time.time() - started


def gain(reference: float, value: float) -> float:
    return (reference - value) / max(reference, 1e-9)


def prediction_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    last: np.ndarray,
    ridge_mse: float,
    temporal_mse: float,
    best_previous_mse: float,
) -> dict[str, float]:
    pred_mse = mse(pred, target)
    return {
        "mse": pred_mse,
        "mae": mae(pred, target),
        "skill_vs_last": gain(mse(last, target), pred_mse),
        "gain_vs_ridge": gain(ridge_mse, pred_mse),
        "gain_vs_temporal_energy": gain(temporal_mse, pred_mse),
        "gain_vs_best_previous_amf": gain(best_previous_mse, pred_mse),
    }


def horizon_bucket(horizon: int) -> str:
    if horizon in {1, 5}:
        return "short"
    if horizon == 15:
        return "medium"
    return "long"


def best_valid_model(metrics: dict[str, dict[str, Any]], exclude_prefixes: tuple[str, ...] = ("oracle",)) -> str:
    valid = {
        name: record
        for name, record in metrics.items()
        if not any(name.startswith(prefix) for prefix in exclude_prefixes)
    }
    return min(valid, key=lambda name: valid[name]["mse"])


def win_tie_loss(a: float, b: float, tolerance: float = 0.005) -> str:
    if a <= b * (1.0 - tolerance):
        return "win"
    if a >= b * (1.0 + tolerance):
        return "loss"
    return "tie"
