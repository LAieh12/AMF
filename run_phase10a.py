from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from phase10a_amf_world_model import AMFDynamicsWorldModel
from phase10a_toy_simulator import (
    ToyGravityBounceSimulator,
    Trajectory,
    Transition,
    generate_trajectories,
    split_trajectories,
    transitions_from_trajectories,
)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.square(a - b)))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a - b))))


def transition_arrays(transitions: list[Transition]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.vstack([transition.state for transition in transitions]).astype(np.float32)
    actions = np.vstack([transition.action for transition in transitions]).astype(np.float32)
    next_states = np.vstack([transition.next_state for transition in transitions]).astype(np.float32)
    return states, actions, next_states


class StaticBaseline:
    name = "static_state"

    def fit(self, transitions: list[Transition]) -> "StaticBaseline":
        return self

    def predict_next(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return state.astype(np.float32)

    def predict_batch(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return states.astype(np.float32)


class ConstantVelocityBaseline:
    name = "constant_velocity"

    def __init__(self, dt: float = 0.08):
        self.dt = dt

    def fit(self, transitions: list[Transition]) -> "ConstantVelocityBaseline":
        return self

    def predict_next(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        x, y, vx, vy = [float(v) for v in state]
        predicted = np.array([x + self.dt * vx, y + self.dt * vy, vx, vy], dtype=np.float32)
        predicted[:2] = np.clip(predicted[:2], -1.05, 1.05)
        return predicted

    def predict_batch(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return np.vstack([self.predict_next(state, action) for state, action in zip(states, actions)]).astype(np.float32)


class RidgeLinearDynamicsBaseline:
    name = "ridge_linear_dynamics"

    def __init__(self, ridge: float = 1e-3):
        self.ridge = ridge
        self.weights = np.zeros((16, 4), dtype=np.float32)

    def _features(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        rows = []
        for state, action in zip(states, actions):
            rows.append(np.concatenate([AMFDynamicsWorldModel.encode(state, action), np.ones(1, dtype=np.float32)]))
        return np.vstack(rows).astype(np.float32)

    def fit(self, transitions: list[Transition]) -> "RidgeLinearDynamicsBaseline":
        states, actions, next_states = transition_arrays(transitions)
        x = self._features(states, actions).astype(np.float64)
        y = (next_states - states).astype(np.float64)
        reg = self.ridge * np.eye(x.shape[1], dtype=np.float64)
        self.weights = np.linalg.solve(x.T @ x + reg, x.T @ y).astype(np.float32)
        return self

    def predict_next(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        feature = np.concatenate([AMFDynamicsWorldModel.encode(state, action), np.ones(1, dtype=np.float32)])
        delta = feature @ self.weights
        predicted = state.astype(np.float32) + delta.astype(np.float32)
        predicted[:2] = np.clip(predicted[:2], -1.05, 1.05)
        predicted[2:] = np.clip(predicted[2:], -3.0, 3.0)
        return predicted.astype(np.float32)

    def predict_batch(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        x = self._features(states, actions)
        predicted = states + x @ self.weights
        predicted[:, :2] = np.clip(predicted[:, :2], -1.05, 1.05)
        predicted[:, 2:] = np.clip(predicted[:, 2:], -3.0, 3.0)
        return predicted.astype(np.float32)


def bounce_mask(states: np.ndarray, next_states: np.ndarray) -> np.ndarray:
    near_wall = np.max(np.abs(next_states[:, :2]), axis=1) > 0.93
    velocity_flip = np.any(np.sign(states[:, 2:]) != np.sign(next_states[:, 2:]), axis=1)
    return near_wall | velocity_flip


def evaluate_one_step(model: Any, transitions: list[Transition], max_items: int) -> dict[str, float]:
    subset = transitions[: min(max_items, len(transitions))]
    states, actions, next_states = transition_arrays(subset)
    start = time.perf_counter()
    predicted = model.predict_batch(states, actions)
    predict_seconds = time.perf_counter() - start
    mask = bounce_mask(states, next_states)
    return {
        "one_step_mse": mse(predicted, next_states),
        "position_rmse": rmse(predicted[:, :2], next_states[:, :2]),
        "velocity_rmse": rmse(predicted[:, 2:], next_states[:, 2:]),
        "bounce_mse": mse(predicted[mask], next_states[mask]) if np.any(mask) else 0.0,
        "eval_transitions": len(subset),
        "predict_seconds": predict_seconds,
        "ms_per_transition": 1000.0 * predict_seconds / max(1, len(subset)),
    }


def rollout_model(model: Any, trajectory: Trajectory, horizon: int) -> np.ndarray:
    state = trajectory.states[0].astype(np.float32)
    predicted = [state]
    steps = min(horizon, len(trajectory.actions))
    for i in range(steps):
        state = model.predict_next(state, trajectory.actions[i])
        predicted.append(state)
    return np.vstack(predicted).astype(np.float32)


def evaluate_rollouts(model: Any, trajectories: list[Trajectory], horizon: int, limit: int) -> dict[str, float]:
    errors = []
    position_errors = []
    final_errors = []
    for trajectory in trajectories[: min(limit, len(trajectories))]:
        predicted = rollout_model(model, trajectory, horizon)
        actual = trajectory.states[: len(predicted)]
        errors.append(mse(predicted, actual))
        position_errors.append(mse(predicted[:, :2], actual[:, :2]))
        final_errors.append(mse(predicted[-1], actual[-1]))
    return {
        "rollout_mse": float(np.mean(errors)) if errors else 0.0,
        "rollout_position_mse": float(np.mean(position_errors)) if position_errors else 0.0,
        "rollout_final_mse": float(np.mean(final_errors)) if final_errors else 0.0,
        "rollout_trajectories": min(limit, len(trajectories)),
        "rollout_horizon": horizon,
    }


def fit_and_evaluate_system(
    model: Any,
    train_transitions: list[Transition],
    test_transitions: list[Transition],
    test_trajectories: list[Trajectory],
    args: argparse.Namespace,
) -> dict[str, Any]:
    start = time.perf_counter()
    model.fit(train_transitions)
    fit_seconds = time.perf_counter() - start
    one_step = evaluate_one_step(model, test_transitions, args.eval_transitions)
    rollout = evaluate_rollouts(model, test_trajectories, args.rollout_horizon, args.rollout_trajectories)
    result = {
        "name": model.name if hasattr(model, "name") else "amf_dynamics_world_model",
        "fit_seconds": fit_seconds,
        **one_step,
        **rollout,
    }
    if isinstance(model, AMFDynamicsWorldModel):
        result.update(
            {
                "cells": int(len(model.centers)),
                "cell_size": float(model.fit_cell_size),
                "activation_radius": float(model.activation_radius),
                "top_k": int(model.top_k),
                "memory_mb_arrays": model.memory_mb(),
                "metaplasticity_stats": model.metaplasticity_stats,
            }
        )
    return result


def clone_amf(model: AMFDynamicsWorldModel) -> AMFDynamicsWorldModel:
    cloned = AMFDynamicsWorldModel(
        cell_size=model.fit_cell_size,
        activation_radius=model.activation_radius,
        top_k=model.top_k,
        max_cells=model.max_cells,
        min_cell_usage=model.min_cell_usage,
        explain_error_threshold=model.explain_error_threshold,
        novelty_confirmations=model.novelty_confirmations,
        fast_dynamics_lr=model.fast_dynamics_lr,
        identity_lr=model.identity_lr,
    )
    cloned.centers = model.centers.copy()
    cloned.deltas = model.deltas.copy()
    cloned.usage = model.usage.copy()
    cloned.fit_cell_size = model.fit_cell_size
    cloned.metaplasticity_stats = dict(model.metaplasticity_stats)
    return cloned


def metaplasticity_probe(model: AMFDynamicsWorldModel, transition: Transition) -> dict[str, Any]:
    probe = clone_amf(model)
    start_cells = int(len(probe.centers))
    explained = probe.learn_transition(transition)
    after_explained_cells = int(len(probe.centers))
    noisy_next = transition.state + np.array([0.75, -0.65, 1.25, -1.10], dtype=np.float32)
    noisy = Transition(
        state=transition.state,
        action=transition.action,
        next_state=noisy_next,
        trajectory_id=transition.trajectory_id,
        step=transition.step,
    )
    first_noise = probe.learn_transition(noisy)
    after_noise_cells = int(len(probe.centers))
    last_status = first_noise
    for _ in range(probe.novelty_confirmations):
        last_status = probe.learn_transition(noisy)
    after_confirmed_cells = int(len(probe.centers))
    return {
        "start_cells": start_cells,
        "max_cells": int(probe.max_cells),
        "explained_status": explained,
        "after_explained_cells": after_explained_cells,
        "first_noise_status": first_noise,
        "after_first_noise_cells": after_noise_cells,
        "confirmed_novelty_status": last_status,
        "after_confirmed_cells": after_confirmed_cells,
        "identity_frozen": probe.identity_frozen,
        "identity_lr": probe.identity_lr,
        "passed": bool(
            explained == "explained_by_existing_cell"
            and after_explained_cells == start_cells
            and first_noise == "buffered_possible_noise"
            and after_noise_cells == after_explained_cells
            and last_status == "created_confirmed_novelty"
            and after_confirmed_cells >= after_noise_cells
            and after_confirmed_cells <= probe.max_cells
            and probe.identity_frozen
            and probe.identity_lr == 0.0
        ),
    }


def run_phase10a(args: argparse.Namespace) -> dict[str, Any]:
    simulator = ToyGravityBounceSimulator()
    trajectories = generate_trajectories(args.trajectories, args.steps, seed=args.seed, simulator=simulator)
    train_trajectories, test_trajectories = split_trajectories(trajectories, test_fraction=args.test_fraction)
    train_transitions = transitions_from_trajectories(train_trajectories)
    test_transitions = transitions_from_trajectories(test_trajectories)

    amf = AMFDynamicsWorldModel(
        cell_size=args.cell_size,
        activation_radius=args.activation_radius,
        top_k=args.top_k,
        max_cells=args.max_cells,
    )
    systems = [
        fit_and_evaluate_system(amf, train_transitions, test_transitions, test_trajectories, args),
    ]
    baselines = [StaticBaseline(), ConstantVelocityBaseline(dt=simulator.dt), RidgeLinearDynamicsBaseline()]
    for baseline in baselines:
        systems.append(fit_and_evaluate_system(baseline, train_transitions, test_transitions, test_trajectories, args))

    export = amf.export(args.export_dir)
    loaded = AMFDynamicsWorldModel.load(export.npz_path)
    sample_state = test_transitions[0].state
    sample_action = test_transitions[0].action
    reload_max_abs_diff = float(
        np.max(np.abs(amf.predict_next(sample_state, sample_action) - loaded.predict_next(sample_state, sample_action)))
    )
    amf_result = systems[0]
    best_baseline = min(systems[1:], key=lambda row: row["one_step_mse"])
    probe_transition = train_transitions[0]
    for candidate in train_transitions[: min(1000, len(train_transitions))]:
        actual_delta = candidate.next_state - candidate.state
        if mse(amf.predict_delta(candidate.state, candidate.action), actual_delta) <= amf.explain_error_threshold:
            probe_transition = candidate
            break
    meta_probe = metaplasticity_probe(amf, probe_transition)
    return {
        "title": "Phase 10a - AMF synthetic world model pretraining",
        "simulator": {
            "name": "ToyGravityBounceSimulator",
            "state": ["x", "y", "vx", "vy"],
            "action": ["ax", "ay"],
            "dt": simulator.dt,
            "gravity": simulator.gravity,
            "drag": simulator.drag,
            "restitution": simulator.restitution,
        },
        "data": {
            "trajectories": args.trajectories,
            "steps_per_trajectory": args.steps,
            "train_trajectories": len(train_trajectories),
            "test_trajectories": len(test_trajectories),
            "train_transitions": len(train_transitions),
            "test_transitions": len(test_transitions),
            "seed": args.seed,
        },
        "rules": {
            "no_llm": True,
            "no_dense_decoder": True,
            "no_backprop": True,
            "pure_python_numpy": True,
        },
        "systems": systems,
        "export": {
            "npz_path": export.npz_path,
            "metadata_path": export.metadata_path,
            "cells": export.cells,
            "memory_mb_arrays": export.memory_mb,
            "reload_max_abs_diff": reload_max_abs_diff,
        },
        "metaplasticity_probe": meta_probe,
        "comparison": {
            "amf_one_step_mse": amf_result["one_step_mse"],
            "best_baseline": best_baseline["name"],
            "best_baseline_one_step_mse": best_baseline["one_step_mse"],
            "amf_beats_best_baseline_one_step": bool(amf_result["one_step_mse"] < best_baseline["one_step_mse"]),
            "amf_rollout_mse": amf_result["rollout_mse"],
        },
    }


def _fmt(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_reports(results: dict[str, Any], out_dir: str | Path = "results") -> None:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    (out / "phase10a_latest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    rows = [
        "| system | one-step MSE | pos RMSE | vel RMSE | bounce MSE | rollout MSE | rollout final MSE | ms/trans | fit s | memory/cells |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system in results["systems"]:
        memory = system.get("memory_mb_arrays", "")
        cells = system.get("cells", "")
        memory_text = f"{_fmt(memory)} MB / {cells}" if memory != "" else "-"
        rows.append(
            f"| {system['name']} | {_fmt(system['one_step_mse'])} | {_fmt(system['position_rmse'])} | "
            f"{_fmt(system['velocity_rmse'])} | {_fmt(system['bounce_mse'])} | {_fmt(system['rollout_mse'])} | "
            f"{_fmt(system['rollout_final_mse'])} | {_fmt(system['ms_per_transition'])} | "
            f"{_fmt(system['fit_seconds'])} | {memory_text} |"
        )
    amf = results["systems"][0]
    report = f"""# Fase 10a - AMF world model sintetico

Objetivo: pretraining sintetico en Python puro para calentar un world model AMF.

Flujo:

```text
simulador juguete -> (S_t, accion, S_t+1) -> celdas AMF de dinamica -> export caliente
```

Simulador: estado `[x, y, vx, vy]`, accion `[ax, ay]`, gravedad, drag, viento
suave y rebote contra paredes.

Datos:

- trayectorias: {results['data']['trajectories']}
- pasos por trayectoria: {results['data']['steps_per_trajectory']}
- train transitions: {results['data']['train_transitions']}
- test transitions: {results['data']['test_transitions']}

Reglas: no LLM = {results['rules']['no_llm']}, no decoder denso =
{results['rules']['no_dense_decoder']}, no backprop = {results['rules']['no_backprop']}.

## Resultados

{chr(10).join(rows)}

## Export AMF calentado

- NPZ: `{results['export']['npz_path']}`
- metadata: `{results['export']['metadata_path']}`
- cells: {results['export']['cells']}
- arrays memory MB: {results['export']['memory_mb_arrays']:.6f}
- reload max abs diff: {results['export']['reload_max_abs_diff']:.10f}

## Metaplasticidad

- guarda delta, no estado completo: {amf['metaplasticity_stats'].get('stores_delta')}
- raw cells antes de regular: {amf['metaplasticity_stats'].get('raw_cells')}
- celdas podadas por bajo uso: {amf['metaplasticity_stats'].get('pruned_low_usage')}
- celdas fusionadas por similitud: {amf['metaplasticity_stats'].get('fused_similar')}
- celdas finales: {amf['metaplasticity_stats'].get('final_cells')}
- identidad congelada: {results['metaplasticity_probe']['identity_frozen']}
- identidad learning rate: {results['metaplasticity_probe']['identity_lr']}
- probe celda existente: `{results['metaplasticity_probe']['explained_status']}`
- probe ruido inicial: `{results['metaplasticity_probe']['first_noise_status']}`
- probe novedad confirmada: `{results['metaplasticity_probe']['confirmed_novelty_status']}`
- probe passed: {results['metaplasticity_probe']['passed']}

## Lectura

El AMF aprende deltas locales `S_t+1 - S_t` como celdas de dinamica sobre el
espacio `(estado, accion)`. En prediccion activa las celdas cercanas y mezcla
sus deltas por resonancia local. La metaplasticidad evita aprendizaje infinito
bruto: no crea celda si una existente explica bien, fusiona celdas parecidas,
poda celdas poco usadas, congela identidad y exige confirmacion antes de tratar
ruido como novedad. Esto lo deja listo para Fase 10b: cargar el NPZ caliente y
usarlo como world model inicial.
"""
    (out / "FASE10A_RESULTADOS.md").write_text(report, encoding="utf-8")
    complete = f"""# FASE10A_COMPLETADA

Fase 10a implementa pretraining sintetico de un world model AMF.

Entregables:

- `phase10a_toy_simulator.py`
- `phase10a_amf_world_model.py`
- `run_phase10a.py`
- `results/phase10a_latest.json`
- `results/FASE10A_RESULTADOS.md`
- `{results['export']['npz_path']}`
- `{results['export']['metadata_path']}`

Resultado AMF:

- train transitions: {results['data']['train_transitions']}
- test transitions: {results['data']['test_transitions']}
- cells: {amf['cells']}
- one_step_mse: {amf['one_step_mse']:.8f}
- rollout_mse: {amf['rollout_mse']:.8f}
- bounce_mse: {amf['bounce_mse']:.8f}
- memory_mb_arrays: {amf['memory_mb_arrays']:.6f}
- export reload max abs diff: {results['export']['reload_max_abs_diff']:.10f}
- metaplasticity_probe_passed: {results['metaplasticity_probe']['passed']}
- raw_cells: {amf['metaplasticity_stats'].get('raw_cells')}
- pruned_low_usage: {amf['metaplasticity_stats'].get('pruned_low_usage')}
- fused_similar: {amf['metaplasticity_stats'].get('fused_similar')}
- final_cells: {amf['metaplasticity_stats'].get('final_cells')}

El modelo fue calentado con miles de transiciones sinteticas y exportado para
Fase 10b. Guarda deltas, regula crecimiento por metaplasticidad y no usa LLM,
decoder denso ni backprop.
"""
    Path("FASE10A_COMPLETADA.md").write_text(complete, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 10a synthetic AMF world model pretraining.")
    parser.add_argument("--trajectories", type=int, default=900)
    parser.add_argument("--steps", type=int, default=70)
    parser.add_argument("--seed", type=int, default=1007)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--eval-transitions", type=int, default=3500)
    parser.add_argument("--rollout-trajectories", type=int, default=32)
    parser.add_argument("--rollout-horizon", type=int, default=55)
    parser.add_argument("--cell-size", type=float, default=0.135)
    parser.add_argument("--activation-radius", type=float, default=0.055)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--max-cells", type=int, default=9000)
    parser.add_argument("--export-dir", default="data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_phase10a(args)
    write_reports(results)
    print("report: results/FASE10A_RESULTADOS.md")
    for system in results["systems"]:
        print(
            f"{system['name']}: one_step_mse={system['one_step_mse']:.8f} "
            f"rollout_mse={system['rollout_mse']:.8f} ms={system['ms_per_transition']:.4f}"
        )
    print(f"export: {results['export']['npz_path']}")


if __name__ == "__main__":
    main()
