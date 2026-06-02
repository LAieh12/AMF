from __future__ import annotations

import numpy as np


EPS = 1e-9


def fisher_feature_ranking(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    labels = np.unique(y)
    overall = np.mean(x, axis=0)
    between = np.zeros(x.shape[1], dtype=np.float64)
    within = np.zeros(x.shape[1], dtype=np.float64)
    for label in labels:
        rows = x[y == label]
        if len(rows) == 0:
            continue
        mean = np.mean(rows, axis=0)
        between += len(rows) * np.square(mean - overall)
        within += np.sum(np.square(rows - mean), axis=0)
    score = between / (within + EPS)
    return np.argsort(score)[::-1]


def top_feature_zero_attack(
    x: np.ndarray,
    ranking: np.ndarray,
    rate: float = 0.12,
) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).copy()
    n_features = max(1, int(round(out.shape[1] * rate)))
    out[:, ranking[:n_features]] = 0.0
    return out


def top_feature_shuffle_attack(
    x: np.ndarray,
    ranking: np.ndarray,
    seed: int,
    rate: float = 0.12,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.asarray(x, dtype=np.float64).copy()
    n_features = max(1, int(round(out.shape[1] * rate)))
    for feature in ranking[:n_features]:
        out[:, feature] = out[rng.permutation(len(out)), feature]
    return out


def nearest_opposite_interpolation(
    x_test: np.ndarray,
    y_test: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 0.45,
    batch_size: int = 96,
) -> np.ndarray:
    out = np.asarray(x_test, dtype=np.float64).copy()
    for start in range(0, len(out), batch_size):
        end = min(start + batch_size, len(out))
        xb = out[start:end]
        yb = y_test[start:end]
        moved = xb.copy()
        for local_i, label in enumerate(yb):
            candidates = x_train[y_train != label]
            if len(candidates) == 0:
                continue
            d2 = np.mean(np.square(candidates - xb[local_i]), axis=1)
            nearest = candidates[int(np.argmin(d2))]
            moved[local_i] = (1.0 - alpha) * xb[local_i] + alpha * nearest
        out[start:end] = moved
    return out


def gaussian_noise_attack(x: np.ndarray, seed: int, sigma: float = 0.45) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(x, dtype=np.float64) + rng.normal(0.0, sigma, size=x.shape)


def corrupt_labels(y: np.ndarray, seed: int, rate: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.asarray(y, dtype=int).copy()
    labels = np.unique(out)
    for idx in np.where(rng.random(len(out)) < rate)[0]:
        choices = labels[labels != out[idx]]
        if len(choices):
            out[idx] = int(rng.choice(choices))
    return out


def append_noise_features(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    multiplier: float,
    max_extra: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    extra = min(max_extra, max(1, int(round(x_train.shape[1] * multiplier))))

    def add(x: np.ndarray) -> np.ndarray:
        return np.hstack([x, rng.normal(0.0, 1.0, size=(len(x), extra))])

    return add(x_train), add(x_val), add(x_test)
