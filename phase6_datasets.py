from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from phase5_datasets import (
    load_iris,
    load_madelon,
    load_optdigits,
    load_wdbc,
    load_wine,
)


UCI6_DIR = Path("data") / "uci_phase6"


@dataclass(frozen=True)
class Phase6Dataset:
    name: str
    x: np.ndarray
    y: np.ndarray
    source: str
    kind: str
    note: str = ""

    @property
    def n_classes(self) -> int:
        return int(len(np.unique(self.y)))

    @property
    def n_features(self) -> int:
        return int(self.x.shape[1])


@dataclass(frozen=True)
class SplitBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    def train_val(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.vstack([self.x_train, self.x_val]),
            np.concatenate([self.y_train, self.y_val]),
        )


def _label_encode(values: Iterable[object]) -> np.ndarray:
    vals = np.asarray(list(values))
    mapping = {value: i for i, value in enumerate(sorted(set(vals.tolist())))}
    return np.asarray([mapping[value] for value in vals], dtype=int)


def _numeric_or_encoded_columns(rows: list[list[str]], label_col: int) -> tuple[np.ndarray, np.ndarray]:
    labels = _label_encode(row[label_col].strip() for row in rows)
    feature_rows = [row[:label_col] + row[label_col + 1 :] for row in rows]
    cols = list(zip(*feature_rows))
    numeric_cols = []
    for col in cols:
        try:
            numeric_cols.append([float(value) for value in col])
        except ValueError:
            encoded = _label_encode(value.strip() for value in col)
            numeric_cols.append(encoded.astype(float).tolist())
    x = np.asarray(list(zip(*numeric_cols)), dtype=np.float64)
    return x, labels


def _read_csv(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip():
            rows.append([part.strip() for part in line.split(",")])
    return rows


def _from_phase5(bundle: object) -> Phase6Dataset:
    return Phase6Dataset(
        name=getattr(bundle, "name"),
        x=np.asarray(getattr(bundle, "x"), dtype=np.float64),
        y=np.asarray(getattr(bundle, "y"), dtype=int),
        source=getattr(bundle, "source"),
        kind=getattr(bundle, "kind"),
        note=getattr(bundle, "note", ""),
    )


def load_spambase() -> Phase6Dataset:
    data = np.loadtxt(UCI6_DIR / "spambase.data", delimiter=",")
    return Phase6Dataset(
        name="spambase",
        x=data[:, :-1].astype(np.float64),
        y=data[:, -1].astype(int),
        source="UCI Spambase",
        kind="real_tabular_email",
        note="57 engineered email features; binary spam classification",
    )


def load_ionosphere() -> Phase6Dataset:
    rows = _read_csv(UCI6_DIR / "ionosphere.data")
    x, y = _numeric_or_encoded_columns(rows, label_col=len(rows[0]) - 1)
    return Phase6Dataset(
        name="ionosphere",
        x=x,
        y=y,
        source="UCI Ionosphere",
        kind="real_tabular_signal",
        note="radar returns with 34 features and noisy class boundary",
    )


def load_sonar() -> Phase6Dataset:
    rows = _read_csv(UCI6_DIR / "sonar.all-data")
    x, y = _numeric_or_encoded_columns(rows, label_col=len(rows[0]) - 1)
    return Phase6Dataset(
        name="sonar",
        x=x,
        y=y,
        source="UCI Sonar",
        kind="real_tabular_signal",
        note="small high-variance benchmark with 60 continuous attributes",
    )


def load_pendigits() -> Phase6Dataset:
    train = np.loadtxt(UCI6_DIR / "pendigits.tra", delimiter=",")
    test = np.loadtxt(UCI6_DIR / "pendigits.tes", delimiter=",")
    data = np.vstack([train, test])
    return Phase6Dataset(
        name="pendigits",
        x=data[:, :-1].astype(np.float64),
        y=data[:, -1].astype(int),
        source="UCI Pen-Based Recognition of Handwritten Digits",
        kind="real_vision_strokes",
        note="16 stylus trajectory features; multiclass digit classification",
    )


def load_satimage() -> Phase6Dataset:
    train = np.loadtxt(UCI6_DIR / "sat.trn")
    test = np.loadtxt(UCI6_DIR / "sat.tst")
    data = np.vstack([train, test])
    labels = _label_encode(data[:, -1].astype(int).tolist())
    return Phase6Dataset(
        name="satimage",
        x=data[:, :-1].astype(np.float64),
        y=labels,
        source="UCI Statlog Landsat Satellite",
        kind="real_remote_sensing",
        note="36 multispectral image features; 6 observed classes",
    )


def available_phase6_datasets() -> list[Phase6Dataset]:
    datasets = [
        _from_phase5(load_iris()),
        _from_phase5(load_wine()),
        _from_phase5(load_wdbc()),
        _from_phase5(load_optdigits()),
        _from_phase5(load_madelon(max_rows=1800)),
    ]
    optional_loaders = [
        load_spambase,
        load_ionosphere,
        load_sonar,
        load_pendigits,
        load_satimage,
    ]
    for loader in optional_loaders:
        try:
            datasets.append(loader())
        except OSError:
            continue
    return datasets


def select_datasets(names: list[str] | None = None) -> list[Phase6Dataset]:
    datasets = available_phase6_datasets()
    if not names:
        return datasets
    requested = {name.lower() for name in names}
    selected = [dataset for dataset in datasets if dataset.name.lower() in requested]
    missing = sorted(requested - {dataset.name.lower() for dataset in selected})
    if missing:
        raise ValueError(f"Datasets not available: {', '.join(missing)}")
    return selected


def stratified_train_val_test_split(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    train_fraction: float = 0.60,
    val_fraction: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train_fraction <= 0.0 or val_fraction <= 0.0 or train_fraction + val_fraction >= 1.0:
        raise ValueError("Fractions must leave a positive test split.")
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for label in sorted(np.unique(y).tolist()):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = max(1, int(round(n * train_fraction)))
        n_val = max(1, int(round(n * val_fraction)))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train : n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val :].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return (
        np.asarray(train_idx, dtype=int),
        np.asarray(val_idx, dtype=int),
        np.asarray(test_idx, dtype=int),
    )


def make_split(dataset: Phase6Dataset, seed: int) -> SplitBundle:
    train_idx, val_idx, test_idx = stratified_train_val_test_split(dataset.x, dataset.y, seed)
    x_train_raw = dataset.x[train_idx]
    mean = np.nanmean(x_train_raw, axis=0)
    std = np.nanstd(x_train_raw, axis=0)
    std[std < 1e-8] = 1.0

    def transform(raw: np.ndarray) -> np.ndarray:
        x = np.asarray(raw, dtype=np.float64)
        x = np.where(np.isfinite(x), x, mean)
        return (x - mean) / std

    return SplitBundle(
        x_train=transform(dataset.x[train_idx]),
        y_train=dataset.y[train_idx].astype(int),
        x_val=transform(dataset.x[val_idx]),
        y_val=dataset.y[val_idx].astype(int),
        x_test=transform(dataset.x[test_idx]),
        y_test=dataset.y[test_idx].astype(int),
        mean=mean,
        std=std,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
