from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase10a_amf_world_model import AMFDynamicsWorldModel
from phase10a_toy_simulator import ToyGravityBounceSimulator, Transition
from phase10b_visual_codec import VisualWorldCodec, pixel_mse


@dataclass(frozen=True)
class ActionVector:
    force_x: float
    force_y: float
    camera_yaw_deg: float
    style_hold: float
    source: str

    def physics(self) -> np.ndarray:
        return np.array([self.force_x, self.force_y], dtype=np.float32)

    def to_json(self) -> dict[str, float | str]:
        return {
            "force_x": float(self.force_x),
            "force_y": float(self.force_y),
            "camera_yaw_deg": float(self.camera_yaw_deg),
            "style_hold": float(self.style_hold),
            "source": self.source,
        }


class PromptActionProvider:
    def __init__(self, offline: bool, model: str, timeout: float = 20.0):
        self.offline = offline
        self.model = model
        self.timeout = timeout
        self.api_key = os.environ.get("OPENAI_API_KEY", "")

    def action_for(self, prompt: str, step: int, latent: np.ndarray) -> ActionVector:
        if not self.offline and self.api_key:
            try:
                return self._api_action(prompt, step, latent)
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
                return self._local_action(prompt, step, latent, source=f"local_fallback_after_api_error:{type(exc).__name__}")
        return self._local_action(prompt, step, latent, source="local_deterministic")

    def _api_action(self, prompt: str, step: int, latent: np.ndarray) -> ActionVector:
        instruction = {
            "task": "Convert the user animation prompt into a bounded action vector.",
            "prompt": prompt,
            "frame_step": step,
            "latent_state": [float(x) for x in latent[:4]],
            "return_json_schema": {
                "force_x": "float in [-1, 1]",
                "force_y": "float in [-1, 1]",
                "camera_yaw_deg": "float, usually -90..90",
                "style_hold": "float in [0, 1], 1 means identity frozen",
            },
        }
        body = json.dumps(
            {
                "model": self.model,
                "input": (
                    "Return only JSON for a N.E.V.E.R. AMF action vector. "
                    f"{json.dumps(instruction, ensure_ascii=True)}"
                ),
                "temperature": 0,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        text = payload.get("output_text")
        if not text:
            parts = []
            for item in payload.get("output", []):
                for content in item.get("content", []):
                    if "text" in content:
                        parts.append(content["text"])
            text = "\n".join(parts)
        data = json.loads(text.strip())
        return ActionVector(
            force_x=float(np.clip(data["force_x"], -1.0, 1.0)),
            force_y=float(np.clip(data["force_y"], -1.0, 1.0)),
            camera_yaw_deg=float(data.get("camera_yaw_deg", 0.0)),
            style_hold=float(np.clip(data.get("style_hold", 1.0), 0.0, 1.0)),
            source=f"api:{self.model}",
        )

    def _local_action(self, prompt: str, step: int, latent: np.ndarray, source: str) -> ActionVector:
        lower = prompt.lower()
        force_x = 0.10 * np.sin(0.19 * step)
        force_y = 0.02 * np.cos(0.13 * step)
        camera = 0.0
        if "salt" in lower or "jump" in lower:
            force_y += 0.85 if step < 10 else 0.25 * np.sin(0.17 * step)
        if "izquierda" in lower or "left" in lower:
            force_x -= 0.45
        if "derecha" in lower or "right" in lower:
            force_x += 0.45
        if "rotando" in lower or "camera" in lower or "camara" in lower:
            camera = 45.0 * min(1.0, step / 24.0)
            force_x += 0.18 * np.sin(0.09 * step)
        if "quieto" in lower or "still" in lower:
            force_x *= 0.15
            force_y *= 0.15
        return ActionVector(
            force_x=float(np.clip(force_x, -1.0, 1.0)),
            force_y=float(np.clip(force_y, -1.0, 1.0)),
            camera_yaw_deg=float(camera),
            style_hold=1.0,
            source=source,
        )


class FrozenIdentityGeometry:
    def __init__(self, vertex_count: int = 2048, radius: float = 0.065, shape_code: float = 1.0):
        self.vertex_count = vertex_count
        self.radius = radius
        self.shape_code = shape_code
        self.identity_frozen = True

    def export(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "source": "procedural_blender_geometry_stub",
            "vertex_count": self.vertex_count,
            "radius": self.radius,
            "shape_code": self.shape_code,
            "identity_frozen": self.identity_frozen,
            "note": "In production this is populated from Blender vertices/UVs/textures.",
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class NeverSceneRuntime:
    """Procedural stand-in for the real NEVER frame stream.

    The important contract is the runtime one: the AMF receives a real rendered
    frame, encodes it to S(t), predicts S(t+1), then learns from the encoded
    next real frame without rebuilding the model from scratch.
    """

    def __init__(self, prompt: str, simulator: ToyGravityBounceSimulator):
        self.prompt = prompt.lower()
        self.simulator = simulator

    def step(self, state: np.ndarray, action: ActionVector, step: int) -> np.ndarray:
        action_physics = action.physics()
        next_state = self.simulator.step(state.astype(np.float32), action_physics).astype(np.float32)
        camera_drive = float(np.clip(action.camera_yaw_deg / 90.0, -1.0, 1.0))
        style_drive = float(np.clip(action.style_hold, 0.0, 1.0))

        scene_delta = np.zeros(4, dtype=np.float32)
        if "rotando" in self.prompt or "camara" in self.prompt or "camera" in self.prompt:
            scene_delta[0] += 0.020 * np.sin(0.31 * step) + 0.010 * camera_drive
            scene_delta[2] += 0.060 * camera_drive + 0.025 * np.cos(0.17 * step)
        if "salt" in self.prompt or "jump" in self.prompt:
            landing_phase = np.sin(0.23 * step)
            scene_delta[1] += 0.018 * max(0.0, landing_phase)
            scene_delta[3] += 0.085 * np.cos(0.23 * step) * style_drive

        next_state += scene_delta
        next_state[2:] *= np.array([0.992, 0.985], dtype=np.float32)
        next_state[:2] = np.clip(next_state[:2], -1.05, 1.05)
        next_state[2:] = np.clip(next_state[2:], -3.0, 3.0)
        return next_state.astype(np.float32)


def latent_mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(a, dtype=np.float32)[:4] - np.asarray(b, dtype=np.float32)[:4])))


def online_probe_score(mse_value: float, explain_threshold: float) -> float:
    if mse_value <= explain_threshold:
        return 1.0
    if mse_value <= explain_threshold * 2.0:
        return 0.75
    if mse_value <= explain_threshold * 4.0:
        return 0.50
    return 0.25


def run_online_learning_probe(
    amf: AMFDynamicsWorldModel,
    scene_runtime: NeverSceneRuntime,
    repeats: int,
) -> dict[str, Any]:
    probe_state = np.array([-0.72, 0.44, -0.31, 0.18], dtype=np.float32)
    probe_action = ActionVector(
        force_x=0.73,
        force_y=-0.64,
        camera_yaw_deg=37.0,
        style_hold=1.0,
        source="online_probe",
    )
    real_next = scene_runtime.step(probe_state, probe_action, step=777)
    real_next = real_next + np.array([0.16, -0.13, -0.46, 0.52], dtype=np.float32)
    real_next[:2] = np.clip(real_next[:2], -1.05, 1.05)
    real_next[2:] = np.clip(real_next[2:], -3.0, 3.0)

    before_prediction = amf.predict_next(probe_state, probe_action.physics())
    before_mse = latent_mse(before_prediction, real_next)
    statuses = []
    curve = []
    for i in range(max(repeats, amf.novelty_confirmations)):
        status = amf.learn_transition(
            Transition(
                state=probe_state,
                action=probe_action.physics(),
                next_state=real_next,
                trajectory_id=1001,
                step=i,
            )
        )
        current_prediction = amf.predict_next(probe_state, probe_action.physics())
        current_mse = latent_mse(current_prediction, real_next)
        statuses.append(status)
        curve.append(
            {
                "repeat": i + 1,
                "status": status,
                "mse": current_mse,
                "score": online_probe_score(current_mse, amf.explain_error_threshold),
                "cells": int(len(amf.centers)),
            }
        )

    after_prediction = amf.predict_next(probe_state, probe_action.physics())
    after_mse = latent_mse(after_prediction, real_next)
    before_score = online_probe_score(before_mse, amf.explain_error_threshold)
    after_score = online_probe_score(after_mse, amf.explain_error_threshold)
    return {
        "prompt": "online AMF NEVER scene-specific transition",
        "state": [float(x) for x in probe_state],
        "action": probe_action.to_json(),
        "real_next": [float(x) for x in real_next],
        "before_prediction": [float(x) for x in before_prediction],
        "after_prediction": [float(x) for x in after_prediction],
        "before_mse": before_mse,
        "after_mse": after_mse,
        "before_score": before_score,
        "after_score": after_score,
        "improved": bool(after_mse < before_mse and after_score > before_score),
        "target_reached": bool(before_score <= 0.25 and after_score >= 1.0),
        "statuses": statuses,
        "learning_curve": curve,
        "cells_after_probe": int(len(amf.centers)),
    }


def run_online_session_probe(
    amf: AMFDynamicsWorldModel,
    scene_runtime: NeverSceneRuntime,
    sessions: int,
) -> dict[str, Any]:
    probe_actions = [
        ActionVector(0.62, 0.74, 18.0, 1.0, "online_session_probe"),
        ActionVector(-0.48, 0.31, 32.0, 1.0, "online_session_probe"),
        ActionVector(0.38, -0.52, 45.0, 1.0, "online_session_probe"),
        ActionVector(-0.71, 0.68, 27.0, 1.0, "online_session_probe"),
    ]
    probe_states = [
        np.array([-0.56, -0.22, 0.28, -0.14], dtype=np.float32),
        np.array([0.34, -0.51, -0.16, 0.42], dtype=np.float32),
        np.array([-0.18, 0.33, 0.55, -0.37], dtype=np.float32),
        np.array([0.68, 0.18, -0.44, 0.11], dtype=np.float32),
    ]
    transitions = []
    for i, (state, action) in enumerate(zip(probe_states, probe_actions)):
        real_next = scene_runtime.step(state, action, step=1200 + i)
        scene_signature = np.array(
            [
                0.045 * np.sin(i + 0.4),
                -0.040 * np.cos(i + 0.2),
                0.120 * np.cos(i * 0.7),
                -0.110 * np.sin(i * 0.6 + 0.3),
            ],
            dtype=np.float32,
        )
        real_next = real_next + scene_signature
        real_next[:2] = np.clip(real_next[:2], -1.05, 1.05)
        real_next[2:] = np.clip(real_next[2:], -3.0, 3.0)
        transitions.append((state, action, real_next))

    curve = []
    for session in range(max(1, sessions)):
        before_errors = [
            latent_mse(amf.predict_next(state, action.physics()), real_next)
            for state, action, real_next in transitions
        ]
        statuses = []
        for idx, (state, action, real_next) in enumerate(transitions):
            statuses.append(
                amf.learn_transition(
                    Transition(
                        state=state,
                        action=action.physics(),
                        next_state=real_next,
                        trajectory_id=2000 + session,
                        step=idx,
                    )
                )
            )
        after_errors = [
            latent_mse(amf.predict_next(state, action.physics()), real_next)
            for state, action, real_next in transitions
        ]
        before_mse = float(np.mean(before_errors))
        after_mse = float(np.mean(after_errors))
        curve.append(
            {
                "session": session + 1,
                "before_mse": before_mse,
                "after_mse": after_mse,
                "score_before": online_probe_score(before_mse, amf.explain_error_threshold),
                "score_after": online_probe_score(after_mse, amf.explain_error_threshold),
                "statuses": statuses,
                "cells": int(len(amf.centers)),
            }
        )

    first_mse = float(curve[0]["before_mse"])
    last_mse = float(curve[-1]["after_mse"])
    return {
        "sessions": len(curve),
        "first_mse": first_mse,
        "last_mse": last_mse,
        "mse_drop": first_mse - last_mse,
        "improved": bool(last_mse < first_mse),
        "score_first": float(curve[0]["score_before"]),
        "score_last": float(curve[-1]["score_after"]),
        "curve": curve,
    }


def window_mean(values: list[float], first: bool, fraction: float = 0.25) -> float:
    if not values:
        return 0.0
    width = max(1, int(np.ceil(len(values) * fraction)))
    chunk = values[:width] if first else values[-width:]
    return float(np.mean(chunk))


def run_loop(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    amf = AMFDynamicsWorldModel.load(args.amf_npz)
    codec = VisualWorldCodec(resolution=args.resolution)
    simulator = ToyGravityBounceSimulator(dt=0.08)
    scene_runtime = NeverSceneRuntime(args.prompt, simulator)
    provider = PromptActionProvider(offline=args.offline, model=args.action_model)
    identity = FrozenIdentityGeometry(vertex_count=args.vertex_count, radius=args.radius, shape_code=args.shape_code)
    geometry_path = out_dir / "phase10c_identity_geometry.json"
    identity.export(geometry_path)

    rng = np.random.default_rng(args.seed)
    state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    frames = []
    predicted_frames = []
    actual_frames = []
    logs = []
    status_counts: dict[str, int] = {}
    error_values = []
    post_online_error_values = []

    start_cells = int(len(amf.centers))
    start = time.perf_counter()
    online_training = not args.disable_online_training
    for step in range(args.steps):
        current_visual = codec.render_from_state(
            state,
            rng=rng,
            radius=identity.radius,
            shape_code=identity.shape_code,
        )
        latent = codec.encode(current_visual.frame)
        action = provider.action_for(args.prompt, step, latent)

        predicted_state = amf.predict_next(latent[:4], action.physics())
        predicted_latent, predicted_frame = codec.decode_predicted(latent, predicted_state)

        actual_state = scene_runtime.step(latent[:4], action, step)
        if args.inject_novelty and step in {max(2, args.steps // 2), max(3, args.steps // 2 + 1), max(4, args.steps // 2 + 2)}:
            actual_state = actual_state + np.array([0.18, -0.12, 0.35, -0.28], dtype=np.float32)
            actual_state[:2] = np.clip(actual_state[:2], -1.05, 1.05)
        actual_visual = codec.render_from_state(
            actual_state,
            rng=rng,
            radius=identity.radius,
            shape_code=identity.shape_code,
        )
        actual_latent = codec.encode(actual_visual.frame)
        prediction_error = latent_mse(predicted_latent, actual_latent)
        error_values.append(prediction_error)

        if online_training:
            learn_status = amf.learn_transition(
                Transition(
                    state=latent[:4].astype(np.float32),
                    action=action.physics(),
                    next_state=actual_latent[:4].astype(np.float32),
                    trajectory_id=0,
                    step=step,
                )
            )
        else:
            learn_status = "online_training_disabled"
        status_counts[learn_status] = status_counts.get(learn_status, 0) + 1
        post_prediction_state = amf.predict_next(latent[:4], action.physics())
        post_prediction_error = latent_mse(post_prediction_state, actual_latent)
        post_online_error_values.append(post_prediction_error)

        frames.append(current_visual.frame.astype(np.float32))
        predicted_frames.append(predicted_frame.astype(np.float32))
        actual_frames.append(actual_visual.frame.astype(np.float32))
        logs.append(
            {
                "step": step,
                "action": action.to_json(),
                "latent": [float(x) for x in latent],
                "predicted_latent": [float(x) for x in predicted_latent],
                "actual_latent": [float(x) for x in actual_latent],
                "prediction_error": prediction_error,
                "post_online_prediction_error": post_prediction_error,
                "online_error_delta": prediction_error - post_prediction_error,
                "learn_status": learn_status,
                "frame_mse": pixel_mse(predicted_frame, actual_visual.clean_frame),
                "cells": int(len(amf.centers)),
            }
        )
        state = actual_latent[:4].astype(np.float32) if args.closed_loop_actual else predicted_state.astype(np.float32)

    elapsed = time.perf_counter() - start
    online_probe = run_online_learning_probe(amf, scene_runtime, repeats=args.online_probe_repeats)
    online_session_probe = run_online_session_probe(amf, scene_runtime, sessions=args.online_session_repeats)
    novelty_probe = None
    if args.inject_novelty:
        probe_state = np.array([0.42, -0.33, 0.12, -0.08], dtype=np.float32)
        probe_action = np.array([0.95, -0.85], dtype=np.float32)
        probe_next = np.array([0.88, -0.80, 0.92, -0.74], dtype=np.float32)
        probe_statuses = []
        for i in range(amf.novelty_confirmations):
            status = amf.learn_transition(
                Transition(
                    state=probe_state,
                    action=probe_action,
                    next_state=probe_next,
                    trajectory_id=999,
                    step=i,
                )
            )
            probe_statuses.append(status)
            status_counts[status] = status_counts.get(status, 0) + 1
        novelty_probe = {
            "state": [float(x) for x in probe_state],
            "action": [float(x) for x in probe_action],
            "next_state": [float(x) for x in probe_next],
            "statuses": probe_statuses,
            "created_confirmed_novelty": "created_confirmed_novelty" in probe_statuses,
            "cells_after_probe": int(len(amf.centers)),
        }

    frames_path = out_dir / "phase10c_never_loop_frames.npz"
    np.savez_compressed(
        frames_path,
        current=np.asarray(frames, dtype=np.float32),
        predicted=np.asarray(predicted_frames, dtype=np.float32),
        actual=np.asarray(actual_frames, dtype=np.float32),
    )
    summary = {
        "title": "Phase 10c - N.E.V.E.R. AMF orchestrator loop",
        "prompt": args.prompt,
        "steps": args.steps,
        "offline": args.offline,
        "action_model": args.action_model,
        "resolution": args.resolution,
        "amf_npz": args.amf_npz,
        "identity_geometry": str(geometry_path),
        "frames_npz": str(frames_path),
        "identity_frozen": identity.identity_frozen,
        "latent_dim": 8,
        "latent_bytes": codec.latent_bytes(),
        "frame_bytes": codec.frame_bytes(),
        "compression_ratio": codec.frame_bytes() / codec.latent_bytes(),
        "cells_start": start_cells,
        "cells_end": int(len(amf.centers)),
        "amf_memory_mb": amf.memory_mb(),
        "online_training_enabled": online_training,
        "feedback_events": int(args.steps if online_training else 0),
        "status_counts": status_counts,
        "online_probe": online_probe,
        "online_session_probe": online_session_probe,
        "novelty_probe": novelty_probe,
        "mean_prediction_error": float(np.mean(error_values)),
        "mean_prediction_error_before_online": float(np.mean(error_values)),
        "mean_prediction_error_after_online": float(np.mean(post_online_error_values)),
        "mean_online_error_delta": float(np.mean(np.asarray(error_values) - np.asarray(post_online_error_values))),
        "first_window_prediction_error": window_mean(error_values, first=True),
        "last_window_prediction_error": window_mean(error_values, first=False),
        "feedback_improves_mean_mse": bool(np.mean(post_online_error_values) < np.mean(error_values)),
        "online_probe_target_reached": bool(online_probe["target_reached"]),
        "online_session_mse_decreases": bool(online_session_probe["improved"]),
        "max_prediction_error": float(np.max(error_values)),
        "mean_frame_mse": float(np.mean([row["frame_mse"] for row in logs])),
        "elapsed_seconds": elapsed,
        "ms_per_step": 1000.0 * elapsed / max(1, args.steps),
        "logs": logs,
    }
    summary_path = out_dir / "phase10c_never_loop.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    results_path = REPO_ROOT / "results" / "phase10c_latest.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="N.E.V.E.R. Phase 10c AMF orchestrator.")
    parser.add_argument("--prompt", default="Personaje saltando, camara rotando 45 grados")
    parser.add_argument("--steps", type=int, default=36)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--amf-npz", default=str(REPO_ROOT / "data" / "phase10a_warm_amf.npz"))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "outputs"))
    parser.add_argument("--action-model", default=os.environ.get("NEVER_ACTION_MODEL", "gpt-5.5"))
    parser.add_argument("--offline", action="store_true", help="Use deterministic local action provider instead of API.")
    parser.add_argument("--closed-loop-actual", action="store_true", default=True)
    parser.add_argument("--disable-online-training", action="store_true")
    parser.add_argument("--online-probe-repeats", type=int, default=4)
    parser.add_argument("--online-session-repeats", type=int, default=6)
    parser.add_argument("--inject-novelty", action="store_true", default=True)
    parser.add_argument("--vertex-count", type=int, default=2048)
    parser.add_argument("--radius", type=float, default=0.065)
    parser.add_argument("--shape-code", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3301)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_loop(args)
    print("report:", str(Path(summary["frames_npz"]).with_name("phase10c_never_loop.json")))
    print("frames:", summary["frames_npz"])
    print("status_counts:", summary["status_counts"])
    print(
        "latent_bytes={latent_bytes} frame_bytes={frame_bytes} compression={compression_ratio:.1f}x "
        "cells={cells_start}->{cells_end} mean_error={mean_prediction_error:.6f} "
        "post_online_error={mean_prediction_error_after_online:.6f} "
        "online_probe={online_probe_before:.2f}->{online_probe_after:.2f} "
        "session_mse={session_mse_first:.6f}->{session_mse_last:.6f} "
        "ms_step={ms_per_step:.3f}".format(
            online_probe_before=summary["online_probe"]["before_score"],
            online_probe_after=summary["online_probe"]["after_score"],
            session_mse_first=summary["online_session_probe"]["first_mse"],
            session_mse_last=summary["online_session_probe"]["last_mse"],
            **summary
        )
    )


if __name__ == "__main__":
    main()
