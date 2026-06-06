from __future__ import annotations

import itertools
import json
import pickle
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from phase12a_physicalai_world_probe import mae, mse
from phase12d_ltm_router_probe import BASE_PREDICTORS, make_samples
from phase13_amf_ltm_model import HORIZONS_13, LtmResidualMemory, fit_predictors
from phase13_eval_horizons import gain, win_tie_loss
from phase13_scene_loader import SceneData, SceneShard, load_scene_data


EXPERTS = (
    "temporal_energy",
    "energy_constraint",
    "identity_orientation",
    "ensemble_12c",
    "amf_ltm_residual",
    "ridge_safety",
    "amf_residual_base",
)


@dataclass
class LtmContext:
    val_pred: np.ndarray
    test_pred: np.ndarray
    val_delta: np.ndarray
    test_delta: np.ndarray
    val_confidence: np.ndarray
    test_confidence: np.ndarray
    val_residual_norm: np.ndarray
    test_residual_norm: np.ndarray
    alpha: float
    threshold: float
    memories_created: int
    retrieved_per_prediction: int
    memory_mb: float
    memory: Any = field(repr=False)


def weight_grid(expert_count: int, step: float) -> list[np.ndarray]:
    units = int(round(1.0 / step))
    out: list[np.ndarray] = []
    for parts in itertools.product(range(units + 1), repeat=expert_count):
        if sum(parts) == units:
            out.append(np.asarray(parts, dtype=np.float32) / units)
    return out


def weighted_prediction(pred_stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return (pred_stack * weights[None, :, None]).sum(axis=1).astype(np.float32)


def choose_weights(pred_stack: np.ndarray, target: np.ndarray, weights_to_try: list[np.ndarray]) -> tuple[np.ndarray, float]:
    best_weights = weights_to_try[0]
    best_loss = float("inf")
    for weights in weights_to_try:
        loss = mse(weighted_prediction(pred_stack, weights), target)
        if loss < best_loss:
            best_loss = loss
            best_weights = weights
    return best_weights.astype(np.float32), best_loss


def identity_orientation_mix(val_identity: np.ndarray, val_orientation: np.ndarray, target: np.ndarray) -> np.ndarray:
    options = [0.0, 0.25, 0.5, 0.75, 1.0]
    best_weight = min(
        options,
        key=lambda w: mse(w * val_identity + (1.0 - w) * val_orientation, target),
    )
    return np.asarray([best_weight, 1.0 - best_weight], dtype=np.float32)


def make_ltm_context(
    val_keys: np.ndarray,
    test_keys: np.ndarray,
    val_temporal: np.ndarray,
    test_temporal: np.ndarray,
    val_target: np.ndarray,
    top_k: int,
    radius: float,
) -> LtmContext:
    memory = LtmResidualMemory(top_k=top_k, radius=radius)
    residual = val_target - val_temporal
    memory.fit(val_keys, residual)
    val_delta, val_confidence = memory.predict(val_keys, leave_one_out=True)
    test_delta, test_confidence = memory.predict(test_keys, leave_one_out=False)

    thresholds = sorted(
        {
            0.0,
            float(np.percentile(val_confidence, 25)),
            float(np.percentile(val_confidence, 50)),
            float(np.percentile(val_confidence, 75)),
            float(np.percentile(val_confidence, 90)),
        }
    )
    alpha_grid = (0.0, 0.10, 0.25, 0.50)
    best_alpha = 0.0
    best_threshold = 0.0
    best_loss = mse(val_temporal, val_target)
    for alpha in alpha_grid:
        for threshold in thresholds:
            mask = (val_confidence >= threshold).astype(np.float32)[:, None]
            loss = mse(val_temporal + alpha * mask * val_delta, val_target)
            if loss < best_loss:
                best_loss = loss
                best_alpha = float(alpha)
                best_threshold = float(threshold)

    val_mask = (val_confidence >= best_threshold).astype(np.float32)[:, None]
    test_mask = (test_confidence >= best_threshold).astype(np.float32)[:, None]
    val_pred = val_temporal + best_alpha * val_mask * val_delta
    test_pred = test_temporal + best_alpha * test_mask * test_delta
    return LtmContext(
        val_pred=val_pred.astype(np.float32),
        test_pred=test_pred.astype(np.float32),
        val_delta=val_delta.astype(np.float32),
        test_delta=test_delta.astype(np.float32),
        val_confidence=val_confidence.astype(np.float32),
        test_confidence=test_confidence.astype(np.float32),
        val_residual_norm=np.linalg.norm(val_delta, axis=1).astype(np.float32),
        test_residual_norm=np.linalg.norm(test_delta, axis=1).astype(np.float32),
        alpha=best_alpha,
        threshold=best_threshold,
        memories_created=int(len(val_target)),
        retrieved_per_prediction=int(min(top_k, len(val_target))),
        memory_mb=float((val_keys.nbytes + residual.nbytes) / (1024 * 1024)),
        memory=memory,
    )


def bin_by_threshold(values: np.ndarray, thresholds: dict[str, float], name: str) -> list[str]:
    lo = thresholds[f"{name}_p33"]
    hi = thresholds[f"{name}_p66"]
    labels = []
    for value in values:
        if value <= lo:
            labels.append("lo")
        elif value <= hi:
            labels.append("mid")
        else:
            labels.append("hi")
    return labels


def context_thresholds(keys: np.ndarray, confidence: np.ndarray, residual_norm: np.ndarray) -> dict[str, float]:
    values = {
        "energy": np.abs(keys[:, 15]) + np.abs(keys[:, 21]),
        "orientation": np.abs(keys[:, 22]),
        "impact": np.abs(keys[:, 17]) + np.abs(keys[:, 18]),
        "confidence": confidence,
        "recent_error": residual_norm,
    }
    out: dict[str, float] = {}
    for name, arr in values.items():
        out[f"{name}_p33"] = float(np.percentile(arr, 33))
        out[f"{name}_p66"] = float(np.percentile(arr, 66))
    return out


def selector_groups(
    regimes: list[str],
    keys: np.ndarray,
    confidence: np.ndarray,
    residual_norm: np.ndarray,
    thresholds: dict[str, float],
) -> tuple[list[str], list[str]]:
    energy = bin_by_threshold(np.abs(keys[:, 15]) + np.abs(keys[:, 21]), thresholds, "energy")
    orientation = bin_by_threshold(np.abs(keys[:, 22]), thresholds, "orientation")
    impact = bin_by_threshold(np.abs(keys[:, 17]) + np.abs(keys[:, 18]), thresholds, "impact")
    conf = bin_by_threshold(confidence, thresholds, "confidence")
    err = bin_by_threshold(residual_norm, thresholds, "recent_error")
    specific = [
        f"{regime}|e={e}|o={o}|i={i}|c={c}|err={er}"
        for regime, e, o, i, c, er in zip(regimes, energy, orientation, impact, conf, err)
    ]
    regime_only = [f"{regime}|fallback" for regime in regimes]
    return specific, regime_only


def fit_group_weights(
    pred_stack: np.ndarray,
    target: np.ndarray,
    specific_groups: list[str],
    regime_groups: list[str],
    weights_to_try: list[np.ndarray],
    min_group: int,
) -> dict[str, Any]:
    weights: dict[str, list[float]] = {}
    losses: dict[str, float] = {}
    counts: dict[str, int] = {}

    global_weights, global_loss = choose_weights(pred_stack, target, weights_to_try)
    weights["global"] = [float(v) for v in global_weights]
    losses["global"] = global_loss
    counts["global"] = int(len(target))

    for groups in (regime_groups, specific_groups):
        for group in sorted(set(groups)):
            idx = np.asarray([i for i, value in enumerate(groups) if value == group], dtype=np.int64)
            if len(idx) < min_group:
                continue
            group_weights, group_loss = choose_weights(pred_stack[idx], target[idx], weights_to_try)
            weights[group] = [float(v) for v in group_weights]
            losses[group] = group_loss
            counts[group] = int(len(idx))

    return {"weights": weights, "validation_losses": losses, "counts": counts}


def apply_selector(
    pred_stack: np.ndarray,
    specific_groups: list[str],
    regime_groups: list[str],
    group_weights: dict[str, list[float]],
) -> tuple[np.ndarray, list[str], np.ndarray]:
    preds = np.zeros((len(pred_stack), 3), dtype=np.float32)
    sources: list[str] = []
    used_weights = np.zeros((len(pred_stack), len(EXPERTS)), dtype=np.float32)
    for i, (specific, regime) in enumerate(zip(specific_groups, regime_groups)):
        source = specific if specific in group_weights else regime if regime in group_weights else "global"
        weights = np.asarray(group_weights[source], dtype=np.float32)
        preds[i] = weighted_prediction(pred_stack[i : i + 1], weights)[0]
        sources.append(source)
        used_weights[i] = weights
    return preds, sources, used_weights


def choose_optional_ltm_beta(
    selected_pred: np.ndarray,
    ltm_delta: np.ndarray,
    confidence: np.ndarray,
    threshold: float,
    target: np.ndarray,
) -> tuple[float, float]:
    beta_grid = (0.0, 0.10, 0.25, 0.50)
    threshold_grid = sorted({threshold, float(np.percentile(confidence, 50)), float(np.percentile(confidence, 75)), 0.0})
    best_beta = 0.0
    best_threshold = threshold
    best_loss = mse(selected_pred, target)
    for beta in beta_grid:
        for candidate_threshold in threshold_grid:
            mask = (confidence >= candidate_threshold).astype(np.float32)[:, None]
            loss = mse(selected_pred + beta * mask * ltm_delta, target)
            if loss < best_loss:
                best_loss = loss
                best_beta = float(beta)
                best_threshold = float(candidate_threshold)
    return best_beta, best_threshold


def final_with_optional_residual(
    selected_pred: np.ndarray,
    ltm_delta: np.ndarray,
    confidence: np.ndarray,
    beta: float,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = confidence >= threshold
    pred = selected_pred + beta * mask[:, None].astype(np.float32) * ltm_delta
    return pred.astype(np.float32), mask


def expert_stack(
    bundle,
    horizon: int,
    target_val: np.ndarray,
    ltm: LtmContext,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    io_weights = identity_orientation_mix(
        bundle.val_preds["identity"][horizon],
        bundle.val_preds["orientation"][horizon],
        target_val,
    )
    val_identity_orientation = (
        io_weights[0] * bundle.val_preds["identity"][horizon]
        + io_weights[1] * bundle.val_preds["orientation"][horizon]
    )
    test_identity_orientation = (
        io_weights[0] * bundle.test_preds["identity"][horizon]
        + io_weights[1] * bundle.test_preds["orientation"][horizon]
    )
    val = {
        "temporal_energy": bundle.val_preds["temporal_energy"][horizon],
        "energy_constraint": bundle.val_preds["energy"][horizon],
        "identity_orientation": val_identity_orientation.astype(np.float32),
        "ensemble_12c": bundle.val_preds["static_ensemble"][horizon],
        "amf_ltm_residual": ltm.val_pred,
        "ridge_safety": bundle.val_preds["ridge"][horizon],
        "amf_residual_base": bundle.val_preds["amf_residual"][horizon],
    }
    test = {
        "temporal_energy": bundle.test_preds["temporal_energy"][horizon],
        "energy_constraint": bundle.test_preds["energy"][horizon],
        "identity_orientation": test_identity_orientation.astype(np.float32),
        "ensemble_12c": bundle.test_preds["static_ensemble"][horizon],
        "amf_ltm_residual": ltm.test_pred,
        "ridge_safety": bundle.test_preds["ridge"][horizon],
        "amf_residual_base": bundle.test_preds["amf_residual"][horizon],
    }
    return val, test, {"identity_orientation_weights": [float(v) for v in io_weights]}


def metrics_for(pred: np.ndarray, target: np.ndarray, last: np.ndarray, ridge: np.ndarray, temporal: np.ndarray, previous: np.ndarray) -> dict[str, float]:
    pred_mse = mse(pred, target)
    return {
        "mse": pred_mse,
        "mae": mae(pred, target),
        "skill_vs_last": gain(mse(last, target), pred_mse),
        "gain_vs_ridge": gain(mse(ridge, target), pred_mse),
        "gain_vs_temporal_energy": gain(mse(temporal, target), pred_mse),
        "gain_vs_best_previous_amf": gain(mse(previous, target), pred_mse),
    }


def normalize_expert_name(name: str | None) -> str | None:
    aliases = {
        "amf_ensemble_12c": "ensemble_12c",
        "energy": "energy_constraint",
        "amf_residual": "amf_residual_base",
    }
    if name is None:
        return None
    return aliases.get(name, name)


def choose_previous_expert(
    scene_name: str,
    horizon_name: str,
    previous_matrix: dict[str, dict[str, str]],
    val_experts: dict[str, np.ndarray],
    target_val: np.ndarray,
) -> str:
    previous_name = normalize_expert_name(previous_matrix.get(scene_name, {}).get(horizon_name))
    if previous_name in val_experts:
        return previous_name
    candidates = ("energy_constraint", "temporal_energy", "amf_residual_base", "ensemble_12c")
    return min(candidates, key=lambda name: mse(val_experts[name], target_val))


def load_previous_matrix(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    result = json.loads(path.read_text(encoding="utf-8"))
    matrix: dict[str, dict[str, str]] = {}
    for scene in result.get("scene_results", []):
        matrix[scene["scene"]] = {
            horizon: record["best_previous_amf"]
            for horizon, record in scene.get("horizon_results", {}).items()
        }
    return matrix


def evaluate_scene_selector(
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
    selector_step: float,
    min_group: int,
    previous_matrix: dict[str, dict[str, str]],
    export_dir: Path | None = None,
) -> dict[str, Any]:
    started = time.time()
    fit_samples = make_samples(scene_data.tracks, scene_data.fit_sequences, horizons, stride, memory_window)
    val_samples = make_samples(scene_data.tracks, scene_data.validation_sequences, horizons, stride, memory_window)
    train_samples = make_samples(scene_data.tracks, scene_data.train_sequences, horizons, stride, memory_window)
    test_samples = make_samples(scene_data.tracks, scene_data.test_sequences, horizons, stride, memory_window)
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
    weights_to_try = weight_grid(len(EXPERTS), selector_step)
    horizon_results: dict[str, Any] = {}
    export_horizons: dict[str, Any] = {}

    for horizon in horizons:
        hname = f"h{horizon}"
        target_val = val_samples.target[horizon]
        target_test = test_samples.target[horizon]
        ltm = make_ltm_context(
            val_samples.keys[horizon],
            test_samples.keys[horizon],
            bundle.val_preds["temporal_energy"][horizon],
            bundle.test_preds["temporal_energy"][horizon],
            target_val,
            top_k=ltm_top_k,
            radius=ltm_radius,
        )
        val_experts, test_experts, expert_report = expert_stack(bundle, horizon, target_val, ltm)
        val_stack = np.stack([val_experts[name] for name in EXPERTS], axis=1)
        test_stack = np.stack([test_experts[name] for name in EXPERTS], axis=1)

        thresholds = context_thresholds(val_samples.keys[horizon], ltm.val_confidence, ltm.val_residual_norm)
        val_specific, val_regime = selector_groups(
            val_samples.regimes[horizon], val_samples.keys[horizon], ltm.val_confidence, ltm.val_residual_norm, thresholds
        )
        test_specific, test_regime = selector_groups(
            test_samples.regimes[horizon],
            test_samples.keys[horizon],
            ltm.test_confidence,
            ltm.test_residual_norm,
            thresholds,
        )
        selector = fit_group_weights(val_stack, target_val, val_specific, val_regime, weights_to_try, min_group)
        val_selected, val_sources, val_used_weights = apply_selector(val_stack, val_specific, val_regime, selector["weights"])
        test_selected, test_sources, test_used_weights = apply_selector(test_stack, test_specific, test_regime, selector["weights"])
        beta, beta_threshold = choose_optional_ltm_beta(
            val_selected, ltm.val_delta, ltm.val_confidence, ltm.threshold, target_val
        )
        val_final_pred, _ = final_with_optional_residual(
            val_selected, ltm.val_delta, ltm.val_confidence, beta, beta_threshold
        )
        selector_final_pred, selector_ltm_mask = final_with_optional_residual(
            test_selected, ltm.test_delta, ltm.test_confidence, beta, beta_threshold
        )

        previous_name = choose_previous_expert(
            scene_data.shard.scene,
            hname,
            previous_matrix,
            val_experts,
            target_val,
        )
        previous_val_pred = val_experts[previous_name]
        previous_pred = test_experts[previous_name]
        selector_val_mse = mse(val_final_pred, target_val)
        previous_val_mse = mse(previous_val_pred, target_val)
        validation_guard_used = selector_val_mse > previous_val_mse * (1.0 + 1e-9)
        final_pred = previous_pred if validation_guard_used else selector_final_pred
        ltm_mask = np.zeros_like(selector_ltm_mask) if validation_guard_used else selector_ltm_mask
        final_model_selected = previous_name if validation_guard_used else "regime_expert_selector_13b"
        final_metrics = metrics_for(
            final_pred,
            target_test,
            test_samples.last[horizon],
            test_experts["ridge_safety"],
            test_experts["temporal_energy"],
            previous_pred,
        )
        selected_metrics = metrics_for(
            test_selected,
            target_test,
            test_samples.last[horizon],
            test_experts["ridge_safety"],
            test_experts["temporal_energy"],
            previous_pred,
        )
        expert_metrics = {
            name: metrics_for(
                pred,
                target_test,
                test_samples.last[horizon],
                test_experts["ridge_safety"],
                test_experts["temporal_energy"],
                previous_pred,
            )
            for name, pred in test_experts.items()
        }
        base_error = np.sum(np.square(test_selected - target_test), axis=1)
        final_error = np.sum(np.square(final_pred - target_test), axis=1)
        dominant_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for regime, weights in zip(test_samples.regimes[horizon], test_used_weights):
            dominant_counts[regime][EXPERTS[int(np.argmax(weights))]] += 1

        average_weights = {name: float(np.mean(test_used_weights[:, i])) for i, name in enumerate(EXPERTS)}
        horizon_results[hname] = {
            "samples": int(len(target_test)),
            "previous_best_model": previous_name,
            "final_model_selected": final_model_selected,
            "final_metrics": final_metrics,
            "selector_only_metrics": selected_metrics,
            "expert_metrics": expert_metrics,
            "selector": {
                "step": selector_step,
                "min_group": min_group,
                "specific_groups_learned": int(sum("|fallback" not in key and key != "global" for key in selector["weights"])),
                "regime_groups_learned": int(sum("|fallback" in key for key in selector["weights"])),
                "sources_used_test": dict(Counter(test_sources)),
                "average_weights_test": average_weights,
                "validation_group_counts": selector["counts"],
                "expert_report": expert_report,
                "context_thresholds": thresholds,
                "validation_guard": {
                    "used": bool(validation_guard_used),
                    "selector_final_mse": float(selector_val_mse),
                    "previous_best_mse": float(previous_val_mse),
                    "rule": "Use selector only when validation MSE is no worse than the previous best expert.",
                },
            },
            "ltm": {
                "alpha": ltm.alpha,
                "threshold": ltm.threshold,
                "optional_beta": beta,
                "optional_beta_threshold": beta_threshold,
                "memories_created": ltm.memories_created,
                "retrieved_per_prediction": ltm.retrieved_per_prediction,
                "memory_mb": ltm.memory_mb,
                "confidence_mean_test": float(np.mean(ltm.test_confidence)),
                "optional_corrected_count": int(np.sum(ltm_mask & (beta > 0))),
                "optional_off_count": int(len(ltm_mask) - np.sum(ltm_mask & (beta > 0))),
                "optional_improved_count": int(np.sum((final_error < base_error) & ltm_mask)),
                "optional_worsened_count": int(np.sum((final_error > base_error) & ltm_mask)),
            },
            "regime_counts_test": dict(Counter(test_samples.regimes[horizon])),
            "dominant_expert_by_regime": {
                regime: dict(counter) for regime, counter in dominant_counts.items()
            },
        }
        export_horizons[hname] = {
            "previous_best_model": previous_name,
            "final_model_selected": final_model_selected,
            "validation_guard_used": bool(validation_guard_used),
            "selector_weights": selector["weights"],
            "selector_validation_losses": selector["validation_losses"],
            "selector_counts": selector["counts"],
            "context_thresholds": thresholds,
            "identity_orientation_weights": expert_report["identity_orientation_weights"],
            "ltm": {
                "alpha": ltm.alpha,
                "threshold": ltm.threshold,
                "optional_beta": beta,
                "optional_beta_threshold": beta_threshold,
                "memory": ltm.memory,
            },
            "expert_order": list(EXPERTS),
        }
    export_path = None
    if export_dir is not None:
        export_dir.mkdir(parents=True, exist_ok=True)
        shard_id = f"{scene_data.shard.scene}_{scene_data.shard.tar_path.stem}"
        export_path = export_dir / f"{shard_id}.pkl"
        export_payload = {
            "format": "phase14_amf_world_model_shard_export_v1",
            "scene": scene_data.shard.scene,
            "shard": str(scene_data.shard.tar_path),
            "experts": list(EXPERTS),
            "horizons": list(horizons),
            "trained_expert_models": bundle.trained_models,
            "horizon_state": export_horizons,
            "notes": [
                "Contains fitted Ridge/AMF residual memories for base experts.",
                "Contains fitted LTM residual memory per horizon.",
                "Selector weights were learned on validation only.",
                "H_event/H_workspace remain routing/context state, not dense predictor features.",
            ],
        }
        with export_path.open("wb") as handle:
            pickle.dump(export_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "scene": scene_data.shard.scene,
        "tar_path": str(scene_data.shard.tar_path),
        "track_count": len(scene_data.tracks),
        "sequence_count": len(scene_data.sequences),
        "fit_sequences": len(scene_data.fit_sequences),
        "validation_sequences": len(scene_data.validation_sequences),
        "train_sequences": len(scene_data.train_sequences),
        "test_sequences": len(scene_data.test_sequences),
        "horizon_results": horizon_results,
        "model_export_path": str(export_path) if export_path else None,
        "elapsed_seconds": time.time() - started,
    }


def aggregate_selector_results(scene_results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    long_records = []
    for horizon in [f"h{h}" for h in HORIZONS_13]:
        records = []
        wtl = {"win": 0, "tie": 0, "loss": 0}
        for scene in scene_results:
            if horizon not in scene["horizon_results"]:
                continue
            record = scene["horizon_results"][horizon]
            status = win_tie_loss(record["final_metrics"]["mse"], record["expert_metrics"][record["previous_best_model"]]["mse"])
            wtl[status] += 1
            records.append(
                {
                    "scene": scene["scene"],
                    "final_mse": record["final_metrics"]["mse"],
                    "previous_best_model": record["previous_best_model"],
                    "previous_best_mse": record["expert_metrics"][record["previous_best_model"]]["mse"],
                    "gain_vs_best_previous": record["final_metrics"]["gain_vs_best_previous_amf"],
                    "gain_vs_temporal": record["final_metrics"]["gain_vs_temporal_energy"],
                }
            )
            if horizon in {"h30", "h60", "h120"}:
                long_records.append(records[-1])
        if records:
            summary[horizon] = {
                "win_tie_loss_vs_best_previous": wtl,
                "mean_gain_vs_best_previous": float(np.mean([r["gain_vs_best_previous"] for r in records])),
                "mean_gain_vs_temporal": float(np.mean([r["gain_vs_temporal"] for r in records])),
                "per_scene": records,
            }
    long_wins = sum(1 for r in long_records if r["final_mse"] <= r["previous_best_mse"] * 1.005)
    return {
        "horizon_summary": summary,
        "long_wins_or_ties_vs_best_previous": long_wins,
        "long_total": len(long_records),
        "phase13b_passed": long_wins >= max(1, (len(long_records) + 1) // 2),
    }


def run_phase13b(
    scene_shards: list[SceneShard],
    previous_matrix_path: Path,
    train_fraction: float,
    split_seed: int,
    stride: int,
    memory_window: int,
    max_cells: int,
    ridge: float,
    model_radius: float,
    model_top_k: int,
    ltm_radius: float,
    ltm_top_k: int,
    tie_tolerance: float,
    selector_step: float,
    min_group: int,
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    started = time.time()
    previous_matrix = load_previous_matrix(previous_matrix_path)
    scene_results = []
    for shard in scene_shards:
        scene_data = load_scene_data(shard, train_fraction=train_fraction, split_seed=split_seed)
        scene_results.append(
            evaluate_scene_selector(
                scene_data,
                horizons,
                stride,
                memory_window,
                max_cells,
                ridge,
                model_radius,
                model_top_k,
                ltm_radius,
                ltm_top_k,
                tie_tolerance,
                selector_step,
                min_group,
                previous_matrix,
            )
        )
    result = {
        "phase": "13B",
        "architecture": "Regime Expert Selector: validation-learned expert mixtures by regime/context/horizon plus optional gated LTM residual",
        "no_leakage_rule": "Previous matrix is report-only; selector weights, context bins, LTM alpha, and optional residual beta are learned on fit/validation only. Test is used only for final metrics.",
        "experts": list(EXPERTS),
        "horizons": list(horizons),
        "scenes": [shard.scene for shard in scene_shards],
        "stride": stride,
        "memory_window": memory_window,
        "max_cells": max_cells,
        "selector_step": selector_step,
        "min_group": min_group,
        "previous_best_matrix": previous_matrix,
        "scene_results": scene_results,
        "cross_scene": aggregate_selector_results(scene_results),
        "elapsed_seconds": time.time() - started,
    }
    return result


def render_selector_report(result: dict[str, Any]) -> str:
    lines = [
        "# Fase 13B - Regime Expert Selector",
        "",
        f"Scenes: {', '.join(result['scenes'])}",
        f"Experts: {', '.join(result['experts'])}",
        f"Stride: {result['stride']}; memory window: {result['memory_window']}; selector step: {result['selector_step']}",
        "",
        "## Success",
        "",
        f"Passed: `{result['cross_scene']['phase13b_passed']}`",
        f"Long h30/h60/h120 W/T vs best previous AMF: {result['cross_scene']['long_wins_or_ties_vs_best_previous']}/{result['cross_scene']['long_total']}",
        "",
        "## Cross-scene",
        "",
        "| horizon | W/T/L vs best previous | mean gain vs previous | mean gain vs temporal |",
        "|---|---|---:|---:|",
    ]
    for horizon, record in result["cross_scene"]["horizon_summary"].items():
        wtl = record["win_tie_loss_vs_best_previous"]
        lines.append(
            f"| {horizon} | {wtl['win']}/{wtl['tie']}/{wtl['loss']} | "
            f"{record['mean_gain_vs_best_previous']:.6f} | {record['mean_gain_vs_temporal']:.6f} |"
        )
    lines.extend(["", "## Scene x horizon", ""])
    for scene in result["scene_results"]:
        lines.append(f"### {scene['scene']}")
        lines.append("")
        lines.append("| horizon | selector MSE | previous best | previous MSE | gain vs previous | gain vs temporal | top expert weight | LTM beta |")
        lines.append("|---|---:|---|---:|---:|---:|---|---:|")
        for horizon, record in scene["horizon_results"].items():
            weights = record["selector"]["average_weights_test"]
            top_expert = max(weights, key=weights.get)
            prev_name = record["previous_best_model"]
            lines.append(
                f"| {horizon} | {record['final_metrics']['mse']:.6f} | {prev_name} | "
                f"{record['expert_metrics'][prev_name]['mse']:.6f} | "
                f"{record['final_metrics']['gain_vs_best_previous_amf']:.6f} | "
                f"{record['final_metrics']['gain_vs_temporal_energy']:.6f} | "
                f"{top_expert}:{weights[top_expert]:.2f} | {record['ltm']['optional_beta']:.2f} |"
            )
        lines.append("")
    lines.extend(["## Dominant experts by regime", ""])
    for scene in result["scene_results"]:
        lines.append(f"### {scene['scene']}")
        lines.append("")
        for horizon, record in scene["horizon_results"].items():
            compact = []
            for regime, counts in record["dominant_expert_by_regime"].items():
                top = max(counts, key=counts.get)
                compact.append(f"{regime}:{top}")
            lines.append(f"- {horizon}: {', '.join(compact[:8])}")
        lines.append("")
    lines.extend(["## Average expert weights", ""])
    for scene in result["scene_results"]:
        lines.append(f"### {scene['scene']}")
        lines.append("")
        lines.append("| horizon | weights | confidence | specific/regime/global sources |")
        lines.append("|---|---|---:|---|")
        for horizon, record in scene["horizon_results"].items():
            weights = ", ".join(
                f"{name}:{value:.2f}"
                for name, value in record["selector"]["average_weights_test"].items()
                if value > 0.01
            )
            sources = record["selector"]["sources_used_test"]
            source_summary = ", ".join(
                f"{name.replace('|', '/')}:{count}" for name, count in list(sources.items())[:5]
            )
            lines.append(
                f"| {horizon} | {weights} | {record['ltm']['confidence_mean_test']:.6f} | {source_summary} |"
            )
        lines.append("")
    lines.extend(["## LTM residual effect", ""])
    lines.append("| scene | horizon | beta | corrected | off | improved | worsened | retrieved | memory MB |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for scene in result["scene_results"]:
        for horizon, record in scene["horizon_results"].items():
            ltm = record["ltm"]
            lines.append(
                f"| {scene['scene']} | {horizon} | {ltm['optional_beta']:.2f} | "
                f"{ltm['optional_corrected_count']} | {ltm['optional_off_count']} | "
                f"{ltm['optional_improved_count']} | {ltm['optional_worsened_count']} | "
                f"{ltm['retrieved_per_prediction']} | {ltm['memory_mb']:.3f} |"
            )
    return "\n".join(lines) + "\n"


def render_matrix_report(result: dict[str, Any]) -> str:
    lines = [
        "# Fase 13B - Previous best matrix",
        "",
        "This matrix is generated from previous Phase 13 results and is not used to learn selector weights.",
        "",
        "| scene | horizon | previous best model |",
        "|---|---|---|",
    ]
    for scene, horizons in sorted(result["previous_best_matrix"].items()):
        for horizon, model in sorted(horizons.items(), key=lambda item: int(item[0][1:])):
            lines.append(f"| {scene} | {horizon} | {model} |")
    return "\n".join(lines) + "\n"


def write_phase13b_outputs(result: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase13b_latest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "FASE13B_REGIME_EXPERT_SELECTOR.md").write_text(render_selector_report(result), encoding="utf-8")
    (out_dir / "FASE13B_PREVIOUS_BEST_MATRIX.md").write_text(render_matrix_report(result), encoding="utf-8")
