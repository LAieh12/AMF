from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from phase5_architecture import AMF5, AMF5Config
from phase5_attacks import top_fisher_perturbation
from phase5_metrics import accuracy, balanced_accuracy, macro_f1, model_memory_mb


def _eval_model(name: str, model: AMF5, x_test: np.ndarray, y_test: np.ndarray, x_adv: np.ndarray) -> dict[str, Any]:
    t0 = perf_counter()
    pred = model.predict(x_test)
    pred_s = perf_counter() - t0
    adv_pred = model.predict(x_adv)
    summary = model.summary()
    return {
        "name": name,
        "clean_accuracy": accuracy(y_test, pred),
        "balanced_accuracy": balanced_accuracy(y_test, pred),
        "macro_f1": macro_f1(y_test, pred),
        "top_feature_attack_accuracy": accuracy(y_test, adv_pred),
        "cells": summary["cells"],
        "pred_seconds": pred_s,
        "model_mb": model_memory_mb(model),
        "avg_candidates": summary["avg_candidates"],
        "avg_votes": summary["avg_votes"],
    }


def run_field_anatomy(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> list[dict[str, Any]]:
    base_config = AMF5Config()
    full = AMF5(config=base_config, seed=seed)
    full.fit(x_train, y_train)
    x_adv = top_fisher_perturbation(x_test, full.selected_features, seed=seed + 99)

    variants: list[tuple[str, AMF5]] = [
        ("AMF5_full", full),
    ]
    for k in [1, 3, 5, 8, 16, 32]:
        variants.append((f"vote_k={k}", full.with_config(AMF5Config(vote_k=k))))
    for top in [8, 16, 32, 64, 128]:
        variants.append((f"top_features={top}", full.with_config(AMF5Config(top_features=top))))
    variants.extend(
        [
            ("no_distance_weight", full.with_config(AMF5Config(use_distance=False))),
            ("no_radius", full.with_config(AMF5Config(use_radius=False))),
            ("no_importance", full.with_config(AMF5Config(use_importance=False))),
            ("no_purity", full.with_config(AMF5Config(use_purity=False))),
            ("uniform_vote", full.with_config(AMF5Config(uniform_vote=True))),
            ("class_normalized", full.with_config(AMF5Config(class_normalize=True))),
        ]
    )

    no_fisher = AMF5(config=AMF5Config(use_fisher=False), seed=seed)
    no_fisher.fit(x_train, y_train)
    variants.append(("no_Fisher", no_fisher))

    return [_eval_model(name, model, x_test, y_test, x_adv) for name, model in variants]
