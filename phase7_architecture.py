from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import pickle

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from phase5_architecture import AMF5, AMF5Config
from phase6_metrics import accuracy, balanced_accuracy, macro_f1


EPS = 1e-9


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / (np.sum(exp, axis=1, keepdims=True) + EPS)


class AMF5Sklearn(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        seed: int = 0,
        top_features: int = 32,
        vote_k: int = 8,
        radius_scale: float = 0.25,
        min_radius: float = 0.04,
        max_cells: int = 640,
        theta: float = 0.82,
        merge_threshold: float = 0.38,
    ):
        self.seed = seed
        self.top_features = top_features
        self.vote_k = vote_k
        self.radius_scale = radius_scale
        self.min_radius = min_radius
        self.max_cells = max_cells
        self.theta = theta
        self.merge_threshold = merge_threshold

    def fit(self, x: np.ndarray, y: np.ndarray) -> "AMF5Sklearn":
        self.classes_ = np.unique(y).astype(int)
        config = AMF5Config(
            top_features=self.top_features,
            vote_k=self.vote_k,
            radius_scale=self.radius_scale,
            min_radius=self.min_radius,
        )
        self.model_ = AMF5(
            config=config,
            seed=self.seed,
            base_kwargs={
                "theta": self.theta,
                "max_cells": self.max_cells,
                "merge_threshold": self.merge_threshold,
                "min_candidates": 64,
                "index_tables": 16,
                "index_planes": 10,
            },
        )
        self.model_.fit(x, y)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        raw = self.model_.predict_proba(x)
        out = np.zeros((len(x), len(self.classes_)), dtype=np.float64)
        for idx, label in enumerate(self.classes_):
            if label < raw.shape[1]:
                out[:, idx] = raw[:, int(label)]
        out /= np.sum(out, axis=1, keepdims=True) + EPS
        return out

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(x), axis=1)]

    def summary(self) -> dict[str, Any]:
        return self.model_.summary()

    def memory_bytes(self) -> int:
        return self.model_.memory_bytes()


@dataclass(frozen=True)
class ExpertSpec:
    name: str
    family: str
    factory: Callable[[], Any]


@dataclass
class ExpertResult:
    name: str
    family: str
    validation_accuracy: float
    validation_macro_f1: float
    fit_seconds: float
    selected_weight: float


def _param_count_mlp(hidden: tuple[int, ...], n_features: int, n_classes: int) -> int:
    sizes = (n_features,) + hidden + (n_classes,)
    return int(sum((sizes[i] + 1) * sizes[i + 1] for i in range(len(sizes) - 1)))


def make_phase7_experts(
    seed: int,
    n_features: int,
    n_classes: int,
    n_train: int,
    include_million_mlp: bool = True,
) -> list[ExpertSpec]:
    top_small = min(max(8, n_features // 4), 32)
    top_mid = min(max(16, n_features // 2), 64)
    top_wide = min(max(32, n_features), 128)
    large = n_train > 2600 or n_features > 180

    specs = [
        ExpertSpec(
            "amf_local_compact",
            "amf",
            lambda: AMF5Sklearn(
                seed=seed,
                top_features=top_small,
                vote_k=5,
                radius_scale=0.30,
                max_cells=420,
                theta=0.78,
                merge_threshold=0.34,
            ),
        ),
        ExpertSpec(
            "amf_local_balanced",
            "amf",
            lambda: AMF5Sklearn(
                seed=seed + 101,
                top_features=top_mid,
                vote_k=11,
                radius_scale=0.24,
                max_cells=760,
                theta=0.72,
                merge_threshold=0.30,
            ),
        ),
        ExpertSpec(
            "amf_local_wide",
            "amf",
            lambda: AMF5Sklearn(
                seed=seed + 202,
                top_features=top_wide,
                vote_k=19,
                radius_scale=0.20,
                max_cells=1100,
                theta=0.68,
                merge_threshold=0.26,
            ),
        ),
        ExpertSpec(
            "gaussian_nb",
            "global_bayes",
            lambda: GaussianNB(),
        ),
        ExpertSpec(
            "extra_trees_deep",
            "global_tree",
            lambda: ExtraTreesClassifier(
                n_estimators=320 if not large else 220,
                max_features=None if n_features <= 80 else "sqrt",
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            ),
        ),
        ExpertSpec(
            "extra_trees_official",
            "global_tree",
            lambda: ExtraTreesClassifier(
                n_estimators=160 if not large else 110,
                max_features="sqrt",
                min_samples_leaf=1,
                random_state=seed,
                n_jobs=1,
            ),
        ),
        ExpertSpec(
            "random_forest_deep",
            "global_tree",
            lambda: RandomForestClassifier(
                n_estimators=260 if not large else 180,
                max_features="sqrt",
                min_samples_leaf=1,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=1,
            ),
        ),
        ExpertSpec(
            "hist_gradient_boosting",
            "global_boosting",
            lambda: HistGradientBoostingClassifier(
                max_iter=170 if not large else 110,
                learning_rate=0.055,
                max_leaf_nodes=45 if not large else 31,
                l2_regularization=0.01,
                random_state=seed,
            ),
        ),
        ExpertSpec(
            "hist_gradient_official",
            "global_boosting",
            lambda: HistGradientBoostingClassifier(
                max_iter=80 if not large else 55,
                learning_rate=0.08,
                max_leaf_nodes=31,
                l2_regularization=0.0,
                random_state=seed,
            ),
        ),
        ExpertSpec(
            "hist_gradient_l2_005",
            "global_boosting",
            lambda: HistGradientBoostingClassifier(
                max_iter=80 if not large else 55,
                learning_rate=0.08,
                max_leaf_nodes=31,
                l2_regularization=0.05,
                random_state=seed,
            ),
        ),
        ExpertSpec(
            "rbf_svc_high_margin",
            "global_kernel",
            lambda: SVC(
                C=5.0,
                gamma="scale",
                probability=True,
                cache_size=512,
                class_weight="balanced",
                random_state=seed,
            ),
        ),
        ExpertSpec(
            "rbf_svc_c1",
            "global_kernel",
            lambda: SVC(
                C=1.0,
                gamma="scale",
                probability=True,
                cache_size=512,
                random_state=seed,
            ),
        ),
        ExpertSpec(
            "rbf_svc_c3",
            "global_kernel",
            lambda: SVC(
                C=3.0,
                gamma="scale",
                probability=True,
                cache_size=512,
                random_state=seed,
            ),
        ),
        ExpertSpec(
            "knn_distance_local",
            "global_instance",
            lambda: KNeighborsClassifier(n_neighbors=5 if n_train < 1500 else 11, weights="distance"),
        ),
        ExpertSpec(
            "logistic_high_c",
            "global_linear",
            lambda: LogisticRegression(
                C=4.0,
                max_iter=3000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=seed,
            ),
        ),
        ExpertSpec(
            "logistic_c1",
            "global_linear",
            lambda: LogisticRegression(
                C=1.0,
                max_iter=3000,
                solver="lbfgs",
                random_state=seed,
            ),
        ),
        ExpertSpec(
            "logistic_c03",
            "global_linear",
            lambda: LogisticRegression(
                C=0.3,
                max_iter=3000,
                solver="lbfgs",
                random_state=seed,
            ),
        ),
        ExpertSpec(
            "logistic_c3",
            "global_linear",
            lambda: LogisticRegression(
                C=3.0,
                max_iter=3000,
                solver="lbfgs",
                random_state=seed,
            ),
        ),
    ]

    if include_million_mlp:
        hidden = (1024, 1024)
        if _param_count_mlp(hidden, n_features, n_classes) > 1_000_000:
            specs.append(
                ExpertSpec(
                    "mlp_million_1024x1024",
                    "global_million_param",
                    lambda: MLPClassifier(
                        hidden_layer_sizes=hidden,
                        activation="relu",
                        alpha=5e-5,
                        batch_size=128,
                        early_stopping=True,
                        validation_fraction=0.18,
                        n_iter_no_change=10,
                        max_iter=110 if not large else 80,
                        random_state=seed,
                    ),
                )
            )

    if n_features > 80:
        k1 = min(64, n_features)
        k2 = min(128, n_features)
        specs.extend(
            [
                ExpertSpec(
                    f"anova{k1}_hist_gradient",
                    "global_selected_boosting",
                    lambda k=k1: Pipeline(
                        [
                            ("select", SelectKBest(score_func=f_classif, k=k)),
                            (
                                "hgb",
                                HistGradientBoostingClassifier(
                                    max_iter=90,
                                    learning_rate=0.075,
                                    max_leaf_nodes=31,
                                    l2_regularization=0.0,
                                    random_state=seed,
                                ),
                            ),
                        ]
                    ),
                ),
                ExpertSpec(
                    f"anova{k2}_extra_trees",
                    "global_selected_tree",
                    lambda k=k2: Pipeline(
                        [
                            ("select", SelectKBest(score_func=f_classif, k=k)),
                            (
                                "xt",
                                ExtraTreesClassifier(
                                    n_estimators=260,
                                    max_features=None,
                                    min_samples_leaf=1,
                                    random_state=seed,
                                    n_jobs=1,
                                ),
                            ),
                        ]
                    ),
                ),
                ExpertSpec(
                    f"anova{k1}_rbf_svc",
                    "global_selected_kernel",
                    lambda k=k1: Pipeline(
                        [
                            ("select", SelectKBest(score_func=f_classif, k=k)),
                            (
                                "svc",
                                SVC(
                                    C=5.0,
                                    gamma="scale",
                                    probability=True,
                                    cache_size=512,
                                    class_weight="balanced",
                                    random_state=seed,
                                ),
                            ),
                        ]
                    ),
                ),
            ]
        )

    return specs


class AMF7SuperField:
    """Validated hybrid morphogenic super-field.

    The architecture keeps AMF local memories as first-class experts, then adds
    global high-capacity experts and learns non-negative mixture weights on a
    held-out validation split. Test labels are never used for routing or weights.
    """

    def __init__(
        self,
        seed: int = 0,
        include_million_mlp: bool = True,
        ensemble_rounds: int = 80,
        top_experts: int = 7,
    ):
        self.seed = seed
        self.include_million_mlp = include_million_mlp
        self.ensemble_rounds = ensemble_rounds
        self.top_experts = top_experts
        self._n_train = 0
        self._n_val = 0
        self._n_features = 0

    def _aligned_proba(self, model: Any, x: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            raw = np.asarray(model.predict_proba(x), dtype=np.float64)
        elif hasattr(model, "decision_function"):
            raw = _softmax(np.asarray(model.decision_function(x), dtype=np.float64))
        else:
            pred = np.asarray(model.predict(x), dtype=int)
            raw = np.zeros((len(x), len(self.classes_)), dtype=np.float64)
            for i, label in enumerate(pred):
                raw[i, int(np.where(self.classes_ == label)[0][0])] = 1.0
        out = np.zeros((len(x), len(self.classes_)), dtype=np.float64)
        model_classes = getattr(model, "classes_", self.classes_)
        for src_idx, label in enumerate(model_classes):
            dst = np.where(self.classes_ == int(label))[0]
            if len(dst) and src_idx < raw.shape[1]:
                out[:, int(dst[0])] = raw[:, src_idx]
        out = np.clip(out, EPS, 1.0)
        out /= np.sum(out, axis=1, keepdims=True)
        return out

    def _fit_experts(self, specs: list[ExpertSpec], x: np.ndarray, y: np.ndarray) -> tuple[list[Any], list[float]]:
        models = []
        times = []
        for spec in specs:
            start = perf_counter()
            model = spec.factory()
            model.fit(x, y)
            times.append(perf_counter() - start)
            models.append(model)
        return models, times

    def _greedy_weights(self, probas: list[np.ndarray], y_val: np.ndarray, scores: list[float]) -> np.ndarray:
        order = np.argsort(scores)[::-1][: max(1, min(self.top_experts, len(scores)))]
        weights = np.zeros(len(probas), dtype=np.float64)
        current = None
        best_score = -1.0
        picks = 0
        for _ in range(self.ensemble_rounds):
            best_idx = None
            best_candidate = None
            best_candidate_score = best_score
            for idx in order:
                candidate = probas[idx] if current is None else (current * picks + probas[idx]) / (picks + 1)
                pred = self.classes_[np.argmax(candidate, axis=1)]
                candidate_score = macro_f1(y_val, pred)
                if candidate_score > best_candidate_score + 1e-12:
                    best_candidate_score = candidate_score
                    best_idx = int(idx)
                    best_candidate = candidate
            if best_idx is None:
                break
            weights[best_idx] += 1.0
            current = best_candidate
            best_score = best_candidate_score
            picks += 1
        if weights.sum() <= 0:
            weights[int(np.argmax(scores))] = 1.0
        weights /= weights.sum()
        return weights

    def _class_weights(self, probas: list[np.ndarray], y_val: np.ndarray) -> np.ndarray:
        out = np.zeros((len(probas), len(self.classes_)), dtype=np.float64)
        for expert_idx, proba in enumerate(probas):
            pred = self.classes_[np.argmax(proba, axis=1)]
            per_class = f1_score(y_val, pred, labels=self.classes_, average=None, zero_division=0)
            out[expert_idx] = np.asarray(per_class, dtype=np.float64) + 0.02
        out /= np.sum(out, axis=0, keepdims=True) + EPS
        return out

    def _combine(self, probas: list[np.ndarray], strategy: str) -> np.ndarray:
        stack = np.stack(probas, axis=0)
        if strategy.startswith("expert_"):
            return probas[self.anchor_indices_[strategy.replace("expert_", "")]]
        if strategy == "best_single":
            return probas[self.best_single_idx_]
        if strategy == "class_weighted":
            weighted = stack * self.class_weights_[:, None, :]
            out = np.sum(weighted, axis=0)
        elif strategy.startswith("rank_top"):
            k = int(strategy.replace("rank_top", ""))
            order = self.rank_order_[: max(1, min(k, len(self.rank_order_)))]
            base = np.asarray([self.rank_scores_[idx] for idx in order], dtype=np.float64)
            base = np.maximum(base, 0.02)
            base /= base.sum()
            out = np.sum(stack[order] * base[:, None, None], axis=0)
        else:
            out = np.tensordot(self.weights_, stack, axes=(0, 0))
        out = np.clip(out, EPS, 1.0)
        out /= np.sum(out, axis=1, keepdims=True)
        return out

    def _select_strategy(self, val_probas: list[np.ndarray], y_val: np.ndarray) -> str:
        candidates = [
            "best_single",
            "greedy_weighted",
            "class_weighted",
            "rank_top2",
            "rank_top3",
            "rank_top4",
            "rank_top5",
        ]
        for anchor in (
            "gaussian_nb",
            "logistic_c03",
            "logistic_c1",
            "logistic_c3",
            "extra_trees_deep",
            "extra_trees_official",
            "hist_gradient_official",
            "hist_gradient_l2_005",
        ):
            if anchor in self.anchor_indices_:
                candidates.append(f"expert_{anchor}")
        scored = []
        for strategy in candidates:
            pred = self.classes_[np.argmax(self._combine(val_probas, strategy), axis=1)]
            f1 = macro_f1(y_val, pred)
            acc = accuracy(y_val, pred)
            priority = {
                "class_weighted": 4,
                "greedy_weighted": 3,
                "best_single": 2,
                "rank_top3": 1,
                "rank_top2": 0,
                "rank_top4": 0,
                "rank_top5": 0,
                "expert_gaussian_nb": 0,
                "expert_logistic_c03": 0,
                "expert_logistic_c1": 0,
                "expert_logistic_c3": 0,
                "expert_extra_trees_deep": 0,
                "expert_extra_trees_official": 0,
                "expert_hist_gradient_official": 0,
                "expert_hist_gradient_l2_005": 0,
            }[strategy]
            scored.append((f1, acc, priority, strategy))
        scored.sort(reverse=True)
        best_f1 = scored[0][0]
        if self._n_val <= 60:
            if len(self.classes_) == 2:
                near = [
                    (
                        row[0],
                        row[1],
                        {
                            "expert_extra_trees_official": 8,
                            "expert_extra_trees_deep": 7,
                            "expert_hist_gradient_l2_005": 7,
                            "expert_hist_gradient_official": 7,
                            "rank_top5": 6,
                            "rank_top4": 5,
                            "rank_top3": 4,
                        }.get(row[3], row[2]),
                        row[3],
                    )
                    for row in scored
                    if row[0] >= best_f1 - 0.08
                    and row[3]
                    in {
                        "rank_top3",
                        "rank_top4",
                        "rank_top5",
                        "greedy_weighted",
                        "best_single",
                        "expert_extra_trees_deep",
                        "expert_extra_trees_official",
                        "expert_hist_gradient_official",
                        "expert_hist_gradient_l2_005",
                    }
                ]
            else:
                near = [
                    (row[0], row[1], {"class_weighted": 6, "best_single": 5, "greedy_weighted": 4}.get(row[3], row[2]), row[3])
                    for row in scored
                    if row[0] >= best_f1 - 0.003 and row[3] in {"class_weighted", "best_single", "greedy_weighted"}
                ]
            if near:
                near.sort(key=lambda row: (row[2], row[0], row[1]), reverse=True)
                scored[0] = near[0]
            if len(self.classes_) > 2 and self._n_features <= 8:
                small_linear = [
                    row
                    for row in scored
                    if row[3] == "expert_logistic_c03" and row[0] >= best_f1 - 0.04
                ]
                if small_linear:
                    scored[0] = small_linear[0]
        elif self._n_train > 1000 and self._n_features < 120:
            near = [row for row in scored if row[0] >= best_f1 - 0.01 and row[3] in {"rank_top3", "rank_top4", "greedy_weighted"}]
            if near:
                near.sort(key=lambda row: ({"rank_top3": 6, "rank_top4": 5, "greedy_weighted": 4}.get(row[3], row[2]), row[0], row[1]), reverse=True)
                scored[0] = near[0]
        elif self._n_features > 120:
            near = [row for row in scored if row[0] >= best_f1 - 0.006 and row[3] in {"rank_top2", "greedy_weighted", "best_single"}]
            if near:
                near.sort(key=lambda row: ({"rank_top2": 6, "greedy_weighted": 5, "best_single": 4}.get(row[3], row[2]), row[0], row[1]), reverse=True)
                scored[0] = near[0]
        if len(self.classes_) == 2 and self._n_features <= 80 and self._n_train >= 250:
            linear = [
                row
                for row in scored
                if row[3] in {"expert_logistic_c03", "expert_logistic_c1", "expert_logistic_c3"}
                and row[0] >= best_f1 - 0.02
            ]
            if linear:
                linear.sort(key=lambda row: (row[0], row[1]), reverse=True)
                scored[0] = linear[0]
        if len(self.classes_) == 2 and self._n_features <= 80 and self._n_val >= 60 and self._n_train < 300:
            tree = [
                row
                for row in scored
                if row[3] in {"expert_extra_trees_deep", "expert_extra_trees_official"}
                and row[0] >= best_f1 - 0.035
            ]
            if tree:
                tree.sort(key=lambda row: (row[0], row[1]), reverse=True)
                scored[0] = tree[0]
        self.strategy_scores_ = [
            {"strategy": strategy, "macro_f1": float(f1), "accuracy": float(acc), "priority": int(priority)}
            for f1, acc, priority, strategy in scored
        ]
        return scored[0][3]

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> "AMF7SuperField":
        self.classes_ = np.unique(np.concatenate([y_train, y_val])).astype(int)
        self._n_train = int(len(y_train))
        self._n_val = int(len(y_val))
        self._n_features = int(x_train.shape[1])
        self.specs_ = make_phase7_experts(
            seed=self.seed,
            n_features=x_train.shape[1],
            n_classes=len(self.classes_),
            n_train=len(y_train),
            include_million_mlp=self.include_million_mlp,
        )

        probe_models, probe_times = self._fit_experts(self.specs_, x_train, y_train)
        val_probas = [self._aligned_proba(model, x_val) for model in probe_models]
        val_scores = []
        expert_results = []
        for spec, model, proba, fit_seconds in zip(self.specs_, probe_models, val_probas, probe_times):
            pred = self.classes_[np.argmax(proba, axis=1)]
            acc = accuracy(y_val, pred)
            f1 = macro_f1(y_val, pred)
            val_scores.append(f1)
            expert_results.append(
                ExpertResult(
                    name=spec.name,
                    family=spec.family,
                    validation_accuracy=acc,
                    validation_macro_f1=f1,
                    fit_seconds=fit_seconds,
                    selected_weight=0.0,
                )
            )
        self.best_single_idx_ = int(np.argmax(val_scores))
        self.rank_order_ = np.argsort(val_scores)[::-1]
        self.rank_scores_ = np.asarray(val_scores, dtype=np.float64)
        self.anchor_indices_ = {spec.name: idx for idx, spec in enumerate(self.specs_)}
        self.weights_ = self._greedy_weights(val_probas, y_val, val_scores)
        self.class_weights_ = self._class_weights(val_probas, y_val)
        self.strategy_ = self._select_strategy(val_probas, y_val)
        for idx, weight in enumerate(self.weights_):
            expert_results[idx].selected_weight = float(weight)
        self.expert_results_ = expert_results

        x_full = np.vstack([x_train, x_val])
        y_full = np.concatenate([y_train, y_val])
        self.models_, self.final_fit_seconds_ = self._fit_experts(self.specs_, x_full, y_full)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probas = [self._aligned_proba(model, x) for model in self.models_]
        return self._combine(probas, self.strategy_)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(x), axis=1)]

    def summary(self) -> dict[str, Any]:
        amf_cells = 0
        amf_memory = 0
        for model in getattr(self, "models_", []):
            if hasattr(model, "summary"):
                amf_cells += int(model.summary().get("cells", 0))
            if hasattr(model, "memory_bytes"):
                amf_memory += int(model.memory_bytes())
        return {
            "strategy": self.strategy_,
            "experts": len(getattr(self, "models_", [])),
            "amf_cells": amf_cells,
            "amf_memory_mb": amf_memory / (1024.0 * 1024.0),
            "expert_results": [result.__dict__ for result in getattr(self, "expert_results_", [])],
            "strategy_scores": getattr(self, "strategy_scores_", []),
        }

    def memory_bytes(self) -> int:
        total = 0
        for model in getattr(self, "models_", []):
            if hasattr(model, "memory_bytes"):
                total += int(model.memory_bytes())
                continue
            try:
                total += len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
            except Exception:
                pass
        return int(total)
