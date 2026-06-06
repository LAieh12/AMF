from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

from phase12a_physicalai_world_probe import fit_model_family, mse, predict_family, select_candidate
from phase12d_ltm_router_probe import (
    ALPHA_GRID,
    BASE_PREDICTORS,
    MODEL_CANDIDATES,
    ROUTER_PREDICTORS,
    apply_static_ensemble,
    choose_static_ensemble,
    make_samples,
)
from phase13_eval_horizons import best_valid_model, prediction_metrics
from phase13_scene_loader import SceneData


HORIZONS_13 = (1, 5, 15, 30, 60, 120)
LTM_LEVEL_SLICES = {
    "H_fast": slice(0, 17),
    "H_event": slice(17, 23),
    "H_regime": slice(23, 34),
    "H_workspace": slice(34, 53),
}
LTM_VARIANTS = {
    "amf_ltm_no_h_event": ("H_fast", "H_regime", "H_workspace"),
    "amf_ltm_no_h_regime": ("H_fast", "H_event", "H_workspace"),
    "amf_ltm_no_h_workspace": ("H_fast", "H_event", "H_regime"),
    "amf_ltm_full": ("H_fast", "H_event", "H_regime", "H_workspace"),
}


@dataclass
class PredictionBundle:
    val_preds: dict[str, dict[int, np.ndarray]]
    test_preds: dict[str, dict[int, np.ndarray]]
    model_report: dict[str, Any]
    trained_models: dict[str, dict[int, dict[str, Any]]]


class LtmResidualMemory:
    def __init__(self, top_k: int, radius: float) -> None:
        self.top_k = top_k
        self.radius = radius
        self.center: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.keys: np.ndarray | None = None
        self.residuals: np.ndarray | None = None
        self.nn: NearestNeighbors | None = None

    def fit(self, keys: np.ndarray, residuals: np.ndarray) -> None:
        self.center = np.median(keys, axis=0).astype(np.float32)
        scale = np.percentile(np.abs(keys - self.center), 75, axis=0).astype(np.float32)
        self.scale = np.maximum(scale, 1e-4)
        self.keys = self._normalize(keys)
        self.residuals = residuals.astype(np.float32)
        self.nn = NearestNeighbors(n_neighbors=min(self.top_k + 1, len(keys)), metric="euclidean")
        self.nn.fit(self.keys)

    def _normalize(self, keys: np.ndarray) -> np.ndarray:
        if self.center is None or self.scale is None:
            raise RuntimeError("LtmResidualMemory must be fitted first")
        return ((keys - self.center) / self.scale).astype(np.float32)

    def predict(self, keys: np.ndarray, leave_one_out: bool) -> tuple[np.ndarray, np.ndarray]:
        if self.nn is None or self.keys is None or self.residuals is None:
            raise RuntimeError("LtmResidualMemory must be fitted first")
        k = min(self.top_k + int(leave_one_out), len(self.keys))
        dists, idx = self.nn.kneighbors(self._normalize(keys), n_neighbors=k, return_distance=True)
        weights = np.exp(-dists / max(self.radius, 1e-6)).astype(np.float32)
        if leave_one_out:
            row_ids = np.arange(len(idx))[:, None]
            weights[idx == row_ids] = 0.0
        denom = np.maximum(weights.sum(axis=1, keepdims=True), 1e-9)
        residual = ((self.residuals[idx] * weights[:, :, None]).sum(axis=1) / denom).astype(np.float32)
        confidence = (weights.sum(axis=1) / max(k, 1)).astype(np.float32)
        return residual, confidence


def key_subset(keys: np.ndarray, levels: tuple[str, ...]) -> np.ndarray:
    return np.concatenate([keys[:, LTM_LEVEL_SLICES[level]] for level in levels], axis=1).astype(np.float32)


def fit_predictors(
    fit_samples,
    val_samples,
    train_samples,
    test_samples,
    horizons: tuple[int, ...],
    max_cells: int,
    ridge: float,
    radius: float,
    top_k: int,
    tie_tolerance: float,
) -> PredictionBundle:
    val_preds: dict[str, dict[int, np.ndarray]] = {name: {} for name in BASE_PREDICTORS}
    test_preds: dict[str, dict[int, np.ndarray]] = {name: {} for name in BASE_PREDICTORS}
    model_report: dict[str, Any] = {}
    trained_models: dict[str, dict[int, dict[str, Any]]] = {name: {} for name in BASE_PREDICTORS}

    for predictor in BASE_PREDICTORS:
        model_report[predictor] = {}
        for horizon in horizons:
            x_fit = fit_samples.x_by_predictor[predictor][horizon]
            target_fit = fit_samples.target[horizon]
            cv_fit = fit_samples.cv[horizon]
            selector_model = fit_model_family(
                x_fit, target_fit, cv_fit, max_cells=max_cells, ridge=ridge, radius=radius, top_k=top_k
            )
            selected, val_losses = select_candidate(
                selector_model,
                val_samples.x_by_predictor[predictor][horizon],
                val_samples.target[horizon],
                val_samples.cv[horizon],
                MODEL_CANDIDATES,
                tie_tolerance=tie_tolerance,
            )
            val_preds[predictor][horizon] = predict_family(
                selector_model,
                val_samples.x_by_predictor[predictor][horizon],
                val_samples.cv[horizon],
                selected,
            )

            final_model = fit_model_family(
                train_samples.x_by_predictor[predictor][horizon],
                train_samples.target[horizon],
                train_samples.cv[horizon],
                max_cells=max_cells,
                ridge=ridge,
                radius=radius,
                top_k=top_k,
            )
            trained_models[predictor][horizon] = final_model
            test_preds[predictor][horizon] = predict_family(
                final_model,
                test_samples.x_by_predictor[predictor][horizon],
                test_samples.cv[horizon],
                selected,
            )

            if predictor == "temporal_energy":
                val_preds["ridge"] = val_preds.get("ridge", {})
                test_preds["ridge"] = test_preds.get("ridge", {})
                val_preds["amf_residual"] = val_preds.get("amf_residual", {})
                test_preds["amf_residual"] = test_preds.get("amf_residual", {})
                val_preds["ridge"][horizon] = predict_family(
                    selector_model,
                    val_samples.x_by_predictor[predictor][horizon],
                    val_samples.cv[horizon],
                    "ridge",
                )
                test_preds["ridge"][horizon] = predict_family(
                    final_model,
                    test_samples.x_by_predictor[predictor][horizon],
                    test_samples.cv[horizon],
                    "ridge",
                )
                val_preds["amf_residual"][horizon] = predict_family(
                    selector_model,
                    val_samples.x_by_predictor[predictor][horizon],
                    val_samples.cv[horizon],
                    "ridge_amf_1.0",
                )
                test_preds["amf_residual"][horizon] = predict_family(
                    final_model,
                    test_samples.x_by_predictor[predictor][horizon],
                    test_samples.cv[horizon],
                    "ridge_amf_1.0",
                )

            model_report[predictor][f"h{horizon}"] = {
                "selected_candidate": selected,
                "validation_losses": val_losses,
                "feature_dim": int(train_samples.x_by_predictor[predictor][horizon].shape[1]),
                "approx_amf_cells_used": int(min(max_cells, len(train_samples.target[horizon]))),
            }

    val_preds["static_ensemble"] = {}
    test_preds["static_ensemble"] = {}
    model_report["amf_ensemble_12c"] = {}
    for horizon in horizons:
        weights, val_loss = choose_static_ensemble(
            {name: val_preds[name][horizon] for name in BASE_PREDICTORS}, val_samples.target[horizon]
        )
        val_preds["static_ensemble"][horizon] = apply_static_ensemble(
            {name: val_preds[name][horizon] for name in BASE_PREDICTORS}, weights
        )
        test_preds["static_ensemble"][horizon] = apply_static_ensemble(
            {name: test_preds[name][horizon] for name in BASE_PREDICTORS}, weights
        )
        model_report["amf_ensemble_12c"][f"h{horizon}"] = {
            "weights": {name: float(weight) for name, weight in zip(BASE_PREDICTORS, weights)},
            "validation_mse": val_loss,
        }

    return PredictionBundle(
        val_preds=val_preds,
        test_preds=test_preds,
        model_report=model_report,
        trained_models=trained_models,
    )


def calibrate_ltm_variant(
    variant_name: str,
    levels: tuple[str, ...],
    horizon: int,
    val_samples,
    test_samples,
    val_temporal: np.ndarray,
    test_temporal: np.ndarray,
    val_target: np.ndarray,
    test_target: np.ndarray,
    top_k: int,
    radius: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    memory = LtmResidualMemory(top_k=top_k, radius=radius)
    val_keys = key_subset(val_samples.keys[horizon], levels)
    test_keys = key_subset(test_samples.keys[horizon], levels)
    memory.fit(val_keys, val_target - val_temporal)

    val_residual, val_confidence = memory.predict(val_keys, leave_one_out=True)
    test_residual, test_confidence = memory.predict(test_keys, leave_one_out=False)
    thresholds = sorted(
        {
            0.0,
            float(np.percentile(val_confidence, 10)),
            float(np.percentile(val_confidence, 25)),
            float(np.percentile(val_confidence, 50)),
            float(np.percentile(val_confidence, 75)),
            float(np.percentile(val_confidence, 90)),
        }
    )

    best_alpha = 0.0
    best_threshold = 0.0
    best_loss = mse(val_temporal, val_target)
    for alpha in ALPHA_GRID:
        for threshold in thresholds:
            mask = (val_confidence >= threshold).astype(np.float32)[:, None]
            pred = val_temporal + alpha * mask * val_residual
            loss = mse(pred, val_target)
            if loss < best_loss:
                best_loss = loss
                best_alpha = float(alpha)
                best_threshold = float(threshold)

    test_mask = test_confidence >= best_threshold
    pred = test_temporal + best_alpha * test_mask[:, None].astype(np.float32) * test_residual
    base_error = np.sum(np.square(test_temporal - test_target), axis=1)
    pred_error = np.sum(np.square(pred - test_target), axis=1)
    corrected = int(np.sum(test_mask & (best_alpha > 0.0)))
    diagnostics = {
        "variant": variant_name,
        "levels": list(levels),
        "alpha": best_alpha,
        "confidence_threshold": best_threshold,
        "validation_mse": best_loss,
        "ltm_memories_created": int(len(val_target)),
        "memories_retrieved_per_prediction": int(min(top_k, len(val_target))),
        "memory_mb": float((val_keys.nbytes + (val_target - val_temporal).nbytes) / (1024 * 1024)),
        "confidence_mean": float(np.mean(test_confidence)),
        "confidence_p10": float(np.percentile(test_confidence, 10)),
        "confidence_p90": float(np.percentile(test_confidence, 90)),
        "ltm_corrected_count": corrected,
        "ltm_off_count": int(len(test_target) - corrected),
        "ltm_improved_count": int(np.sum((pred_error < base_error) & test_mask)),
        "ltm_worsened_count": int(np.sum((pred_error > base_error) & test_mask)),
    }
    return pred.astype(np.float32), diagnostics


def regime_counter(regimes: list[str]) -> dict[str, int]:
    return dict(Counter(regimes))


def evaluate_scene(
    scene_data: SceneData,
    horizons: tuple[int, ...],
    stride: int,
    memory_window: int,
    max_cells: int,
    ridge: float,
    model_radius: float,
    model_top_k: int,
    ltm_radius: float,
    ltm_top_k: int,
    tie_tolerance: float,
) -> dict[str, Any]:
    scene_started = time.time()
    sample_started = time.time()
    fit_samples = make_samples(scene_data.tracks, scene_data.fit_sequences, horizons, stride, memory_window)
    val_samples = make_samples(scene_data.tracks, scene_data.validation_sequences, horizons, stride, memory_window)
    train_samples = make_samples(scene_data.tracks, scene_data.train_sequences, horizons, stride, memory_window)
    test_samples = make_samples(scene_data.tracks, scene_data.test_sequences, horizons, stride, memory_window)
    sample_seconds = time.time() - sample_started

    train_started = time.time()
    bundle = fit_predictors(
        fit_samples,
        val_samples,
        train_samples,
        test_samples,
        horizons,
        max_cells=max_cells,
        ridge=ridge,
        radius=model_radius,
        top_k=model_top_k,
        tie_tolerance=tie_tolerance,
    )
    train_seconds = time.time() - train_started

    horizon_results: dict[str, Any] = {}
    for horizon in horizons:
        eval_started = time.time()
        target = test_samples.target[horizon]
        last = test_samples.last[horizon]
        ridge_pred = bundle.test_preds["ridge"][horizon]
        temporal_pred = bundle.test_preds["temporal_energy"][horizon]
        base_predictions = {
            "last_state": last,
            "ridge": ridge_pred,
            "energy": bundle.test_preds["energy"][horizon],
            "temporal_energy": temporal_pred,
            "amf_residual": bundle.test_preds["amf_residual"][horizon],
            "amf_ensemble_12c": bundle.test_preds["static_ensemble"][horizon],
        }
        ridge_mse = mse(ridge_pred, target)
        temporal_mse = mse(temporal_pred, target)
        previous_amf_names = ("energy", "temporal_energy", "amf_residual", "amf_ensemble_12c")
        best_previous_name = min(previous_amf_names, key=lambda name: mse(base_predictions[name], target))
        best_previous_mse = mse(base_predictions[best_previous_name], target)

        metrics = {
            name: prediction_metrics(pred, target, last, ridge_mse, temporal_mse, best_previous_mse)
            for name, pred in base_predictions.items()
        }
        metrics["last_state"]["selected_branch"] = "last_state"
        metrics["ridge"]["selected_branch"] = "temporal_energy_ridge"
        metrics["amf_ensemble_12c"]["selected_branch"] = "validation_static_ensemble"

        ltm_diagnostics: dict[str, Any] = {}
        ltm_predictions: dict[str, np.ndarray] = {}
        for variant_name, levels in LTM_VARIANTS.items():
            pred, diagnostics = calibrate_ltm_variant(
                variant_name,
                levels,
                horizon,
                val_samples,
                test_samples,
                bundle.val_preds["temporal_energy"][horizon],
                temporal_pred,
                val_samples.target[horizon],
                target,
                top_k=ltm_top_k,
                radius=ltm_radius,
            )
            ltm_predictions[variant_name] = pred
            ltm_diagnostics[variant_name] = diagnostics
            metrics[variant_name] = prediction_metrics(pred, target, last, ridge_mse, temporal_mse, best_previous_mse)
            metrics[variant_name]["selected_branch"] = variant_name

        selected_ltm = min(LTM_VARIANTS, key=lambda name: ltm_diagnostics[name]["validation_mse"])
        ltm_predictions["amf_ltm_selected"] = ltm_predictions[selected_ltm]
        metrics["amf_ltm_selected"] = prediction_metrics(
            ltm_predictions[selected_ltm], target, last, ridge_mse, temporal_mse, best_previous_mse
        )
        metrics["amf_ltm_selected"]["selected_branch"] = selected_ltm

        oracle_names = tuple(base_predictions) + tuple(ltm_predictions)
        oracle_stack = np.stack([base_predictions.get(name, ltm_predictions.get(name)) for name in oracle_names], axis=1)
        oracle_errors = np.sum(np.square(oracle_stack - target[:, None, :]), axis=2)
        oracle_choice = np.argmin(oracle_errors, axis=1)
        oracle_pred = oracle_stack[np.arange(len(target)), oracle_choice]
        metrics["oracle_no_valid"] = prediction_metrics(oracle_pred, target, last, ridge_mse, temporal_mse, best_previous_mse)
        metrics["oracle_no_valid"]["selected_branch"] = "test_target_selector_invalid"

        best_model = best_valid_model(metrics)
        horizon_results[f"h{horizon}"] = {
            "samples": int(len(target)),
            "best_previous_amf": best_previous_name,
            "selected_ltm_branch": selected_ltm,
            "best_valid_model": best_model,
            "metrics": metrics,
            "model_report": {
                name: report.get(f"h{horizon}", {})
                for name, report in bundle.model_report.items()
            },
            "amf_cells_used": int(min(max_cells, len(train_samples.target[horizon]))),
            "ltm_diagnostics": ltm_diagnostics,
            "events_detected_test": {
                name: count
                for name, count in regime_counter(test_samples.regimes[horizon]).items()
                if "impact" in name or "freefall" in name or "change" in name
            },
            "regimes_detected_validation": regime_counter(val_samples.regimes[horizon]),
            "regimes_detected_test": regime_counter(test_samples.regimes[horizon]),
            "inference_seconds": time.time() - eval_started,
        }

    return {
        "scene": scene_data.shard.scene,
        "tier": scene_data.shard.tier,
        "tar_path": str(scene_data.shard.tar_path),
        "track_count": len(scene_data.tracks),
        "sequence_count": len(scene_data.sequences),
        "fit_sequences": len(scene_data.fit_sequences),
        "validation_sequences": len(scene_data.validation_sequences),
        "train_sequences": len(scene_data.train_sequences),
        "test_sequences": len(scene_data.test_sequences),
        "stride": stride,
        "memory_window": memory_window,
        "horizons": list(horizons),
        "sample_seconds": sample_seconds,
        "training_seconds": train_seconds,
        "elapsed_seconds": time.time() - scene_started,
        "horizon_results": horizon_results,
    }
