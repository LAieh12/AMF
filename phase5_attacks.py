from __future__ import annotations

from typing import Any

import numpy as np


def gaussian_noise(x: np.ndarray, seed: int, sigma: float = 0.35) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(x, dtype=np.float64) + rng.normal(0.0, sigma, size=x.shape)


def feature_dropout(x: np.ndarray, seed: int, rate: float = 0.18) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = rng.random(x.shape) >= rate
    return np.asarray(x, dtype=np.float64) * mask


def random_direction_attack(x: np.ndarray, seed: int, epsilon: float = 0.55) -> np.ndarray:
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=x.shape)
    norm = np.sqrt(np.mean(np.square(direction), axis=1, keepdims=True)) + 1e-9
    return np.asarray(x, dtype=np.float64) + epsilon * direction / norm


def feature_swap(x: np.ndarray, features: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.asarray(x, dtype=np.float64).copy()
    if len(features) == 0:
        return out
    features = np.asarray(features, dtype=int)
    for feature in features:
        out[:, feature] = out[rng.permutation(len(out)), feature]
    return out


def top_fisher_perturbation(
    x: np.ndarray,
    features: np.ndarray,
    seed: int,
    epsilon: float = 0.75,
    top_n: int = 16,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.asarray(x, dtype=np.float64).copy()
    selected = np.asarray(features, dtype=int)[: min(top_n, len(features))]
    if len(selected) == 0:
        return out
    signs = rng.choice([-1.0, 1.0], size=(len(out), len(selected)))
    out[:, selected] += epsilon * signs
    return out


def generic_boundary_attack(
    model: Any,
    x: np.ndarray,
    seed: int,
    max_trials: int = 18,
    max_epsilon: float = 1.4,
) -> np.ndarray:
    """Black-box search: random directions only, no prototype/cell internals."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    out = x.copy()
    base_pred = model.predict(x)
    epsilons = np.linspace(0.08, max_epsilon, max_trials)
    for i, row in enumerate(x):
        best = row
        for eps in epsilons:
            direction = rng.normal(size=row.shape)
            direction /= np.sqrt(np.mean(np.square(direction))) + 1e-9
            candidate = row + eps * direction
            if model.predict(candidate[None, :])[0] != base_pred[i]:
                best = candidate
                break
        out[i] = best
    return out


def corrupt_labels(y: np.ndarray, seed: int, rate: float = 0.12) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.asarray(y, dtype=int).copy()
    labels = np.unique(out)
    mask = rng.random(len(out)) < rate
    for idx in np.where(mask)[0]:
        choices = labels[labels != out[idx]]
        out[idx] = int(rng.choice(choices))
    return out
