from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC

from phase5_architecture import AMF5, AMF5Config


@dataclass(frozen=True)
class ModelCandidate:
    family: str
    variant: str
    factory: Callable[[], object]


def _amf_factory(seed: int, config: AMF5Config, max_cells: int) -> Callable[[], AMF5]:
    return lambda: AMF5(
        config=config,
        seed=seed,
        base_kwargs={
            "max_cells": max_cells,
            "min_candidates": 48,
            "index_tables": 14,
            "index_planes": 9,
        },
    )


def official_model_grids(
    seed: int,
    n_features: int,
    n_train: int,
    include_mlp: bool = True,
    include_rbf: bool = True,
) -> dict[str, list[ModelCandidate]]:
    """Small validation grids using official scikit-learn estimators."""
    large = n_train > 2500 or n_features > 120
    top_small = min(max(8, n_features // 4), 24)
    top_default = min(max(16, n_features // 2), 48)
    top_wide = min(max(32, n_features), 96)

    amf_grid = [
        ModelCandidate(
            "AMF5",
            f"default_tf{top_default}_k8",
            _amf_factory(
                seed,
                AMF5Config(top_features=top_default, vote_k=8, radius_scale=0.25, min_radius=0.04),
                max_cells=560,
            ),
        ),
    ]
    if not large:
        amf_grid = [
            ModelCandidate(
                "AMF5",
                f"compact_tf{top_small}_k5",
                _amf_factory(
                    seed,
                    AMF5Config(top_features=top_small, vote_k=5, radius_scale=0.28, min_radius=0.05),
                    max_cells=360,
                ),
            ),
            *amf_grid,
            ModelCandidate(
                "AMF5",
                f"wide_tf{top_wide}_k16",
                _amf_factory(
                    seed,
                    AMF5Config(top_features=top_wide, vote_k=16, radius_scale=0.22, min_radius=0.04),
                    max_cells=760,
                ),
            ),
        ]

    forest_estimators = 90 if large else 120
    extra_estimators = 110 if large else 160
    hist_iters = 55 if large else 80

    grids: dict[str, list[ModelCandidate]] = {
        "AMF5": amf_grid,
        "sk_dummy": [
            ModelCandidate(
                "sk_dummy",
                "most_frequent",
                lambda: DummyClassifier(strategy="most_frequent", random_state=seed),
            ),
            ModelCandidate(
                "sk_dummy",
                "stratified",
                lambda: DummyClassifier(strategy="stratified", random_state=seed),
            ),
        ],
        "sk_gaussian_nb": [
            ModelCandidate("sk_gaussian_nb", "default", lambda: GaussianNB()),
        ],
        "sk_knn": [
            ModelCandidate("sk_knn", "k3_distance", lambda: KNeighborsClassifier(n_neighbors=3, weights="distance")),
            ModelCandidate("sk_knn", "k7_distance", lambda: KNeighborsClassifier(n_neighbors=7, weights="distance")),
            ModelCandidate("sk_knn", "k15_distance", lambda: KNeighborsClassifier(n_neighbors=15, weights="distance")),
        ],
        "sk_logistic": [
            ModelCandidate(
                "sk_logistic",
                "C0.3",
                lambda: LogisticRegression(C=0.3, max_iter=2000, solver="lbfgs", random_state=seed),
            ),
            ModelCandidate(
                "sk_logistic",
                "C1.0",
                lambda: LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=seed),
            ),
            ModelCandidate(
                "sk_logistic",
                "C3.0",
                lambda: LogisticRegression(C=3.0, max_iter=2000, solver="lbfgs", random_state=seed),
            ),
        ],
        "sk_linear_svc": [
            ModelCandidate("sk_linear_svc", "C0.3", lambda: LinearSVC(C=0.3, dual="auto", max_iter=6000, random_state=seed)),
            ModelCandidate("sk_linear_svc", "C1.0", lambda: LinearSVC(C=1.0, dual="auto", max_iter=6000, random_state=seed)),
        ],
        "sk_random_forest": [
            ModelCandidate(
                "sk_random_forest",
                    "trees120_leaf1",
                    lambda: RandomForestClassifier(
                    n_estimators=forest_estimators,
                    max_features="sqrt",
                    min_samples_leaf=1,
                    n_jobs=1,
                    random_state=seed,
                ),
            ),
            ModelCandidate(
                "sk_random_forest",
                    "trees120_leaf2",
                    lambda: RandomForestClassifier(
                    n_estimators=forest_estimators,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    n_jobs=1,
                    random_state=seed,
                ),
            ),
        ],
        "sk_extra_trees": [
            ModelCandidate(
                "sk_extra_trees",
                    "trees160_leaf1",
                    lambda: ExtraTreesClassifier(
                    n_estimators=extra_estimators,
                    max_features="sqrt",
                    min_samples_leaf=1,
                    n_jobs=1,
                    random_state=seed,
                ),
            ),
            ModelCandidate(
                "sk_extra_trees",
                    "trees160_leaf2",
                    lambda: ExtraTreesClassifier(
                    n_estimators=extra_estimators,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    n_jobs=1,
                    random_state=seed,
                ),
            ),
        ],
        "sk_hist_gradient_boosting": [
            ModelCandidate(
                "sk_hist_gradient_boosting",
                    "iter80_l2_0",
                    lambda: HistGradientBoostingClassifier(
                    max_iter=hist_iters,
                    learning_rate=0.08,
                    max_leaf_nodes=31,
                    l2_regularization=0.0,
                    random_state=seed,
                ),
            ),
            ModelCandidate(
                "sk_hist_gradient_boosting",
                    "iter80_l2_0.05",
                    lambda: HistGradientBoostingClassifier(
                    max_iter=hist_iters,
                    learning_rate=0.08,
                    max_leaf_nodes=31,
                    l2_regularization=0.05,
                    random_state=seed,
                ),
            ),
        ],
    }

    if large:
        grids["sk_random_forest"] = grids["sk_random_forest"][:1]
        grids["sk_extra_trees"] = grids["sk_extra_trees"][:1]
        grids["sk_hist_gradient_boosting"] = grids["sk_hist_gradient_boosting"][:1]

    if include_rbf and n_train <= 3500:
        grids["sk_rbf_svc"] = [
            ModelCandidate("sk_rbf_svc", "C1_scale", lambda: SVC(C=1.0, gamma="scale", cache_size=512, random_state=seed)),
            ModelCandidate("sk_rbf_svc", "C3_scale", lambda: SVC(C=3.0, gamma="scale", cache_size=512, random_state=seed)),
        ]

    if include_mlp:
        grids["sk_mlp_small"] = [
            ModelCandidate(
                "sk_mlp_small",
                "h64",
                lambda: MLPClassifier(
                    hidden_layer_sizes=(64,),
                    alpha=1e-4,
                    batch_size=128,
                    early_stopping=True,
                    max_iter=140,
                    random_state=seed,
                ),
            ),
            ModelCandidate(
                "sk_mlp_small",
                "h128",
                lambda: MLPClassifier(
                    hidden_layer_sizes=(128,),
                    alpha=1e-4,
                    batch_size=128,
                    early_stopping=True,
                    max_iter=140,
                    random_state=seed,
                ),
            ),
        ]
        if large:
            grids["sk_mlp_small"] = grids["sk_mlp_small"][:1]

    return grids
