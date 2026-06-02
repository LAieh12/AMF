from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


DATA_DIR = Path("data") / "uci"


@dataclass
class DatasetBundle:
    name: str
    x: np.ndarray
    y: np.ndarray
    kind: str
    source: str
    note: str = ""


def _label_encode(values: list[str] | np.ndarray) -> np.ndarray:
    vals = np.asarray(values)
    labels = {v: i for i, v in enumerate(sorted(set(vals.tolist())))}
    return np.array([labels[v] for v in vals], dtype=int)


def _read_csv_numeric(path: Path, label_col: int, skip_empty: bool = True) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[str]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if skip_empty and not line.strip():
            continue
        rows.append([part.strip() for part in line.split(",")])
    labels = [row[label_col] for row in rows]
    features = []
    for row in rows:
        vals = row[:label_col] + row[label_col + 1 :]
        features.append([float(v) for v in vals])
    return np.asarray(features, dtype=np.float64), _label_encode(labels)


def load_iris() -> DatasetBundle:
    x, y = _read_csv_numeric(DATA_DIR / "iris.data", label_col=4)
    return DatasetBundle(
        name="iris",
        x=x,
        y=y,
        kind="real_tabular_classic",
        source="UCI Iris",
        note="small sanity-check dataset",
    )


def load_wine() -> DatasetBundle:
    data = np.loadtxt(DATA_DIR / "wine.data", delimiter=",")
    y = data[:, 0].astype(int) - 1
    x = data[:, 1:].astype(np.float64)
    return DatasetBundle(
        name="wine",
        x=x,
        y=y,
        kind="real_tabular_classic",
        source="UCI Wine",
    )


def load_wdbc() -> DatasetBundle:
    rows = []
    for line in (DATA_DIR / "wdbc.data").read_text(encoding="utf-8").splitlines():
        parts = line.split(",")
        rows.append(parts)
    y = _label_encode([row[1] for row in rows])
    x = np.asarray([[float(v) for v in row[2:]] for row in rows], dtype=np.float64)
    return DatasetBundle(
        name="wdbc",
        x=x,
        y=y,
        kind="real_tabular_classic",
        source="UCI Breast Cancer Wisconsin Diagnostic",
    )


def load_optdigits() -> DatasetBundle:
    train = np.loadtxt(DATA_DIR / "optdigits.tra", delimiter=",")
    test = np.loadtxt(DATA_DIR / "optdigits.tes", delimiter=",")
    data = np.vstack([train, test])
    x = data[:, :-1].astype(np.float64)
    y = data[:, -1].astype(int)
    return DatasetBundle(
        name="optdigits",
        x=x,
        y=y,
        kind="real_vision_pixels",
        source="UCI Optical Recognition of Handwritten Digits",
        note="64 raw pixel features; simple vision without deep extractor",
    )


def load_madelon(max_rows: int | None = 1800) -> DatasetBundle:
    x = np.loadtxt(DATA_DIR / "madelon_train.data")
    y = np.loadtxt(DATA_DIR / "madelon_train.labels").astype(int)
    y = (y > 0).astype(int)
    if max_rows is not None and len(x) > max_rows:
        x = x[:max_rows]
        y = y[:max_rows]
    return DatasetBundle(
        name="madelon",
        x=x.astype(np.float64),
        y=y,
        kind="real_high_dimensional_noise",
        source="UCI Madelon",
        note="500-feature artificial-real benchmark with many distractor features",
    )


def load_real_datasets() -> list[DatasetBundle]:
    loaders = [load_iris, load_wine, load_wdbc, load_optdigits, load_madelon]
    return [loader() for loader in loaders]


def stratified_split(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    test_fraction: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for label in sorted(np.unique(y).tolist()):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_fraction)))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-8] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std


def pca_fit_transform(
    x_train: np.ndarray,
    x_test: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    centered = x_train - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[: min(n_components, vt.shape[0])]
    return centered @ components.T, (x_test - mean) @ components.T
