from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    trajectory_id: int
    step: int


@dataclass(frozen=True)
class Trajectory:
    states: np.ndarray
    actions: np.ndarray
    trajectory_id: int


class ToyGravityBounceSimulator:
    """Small deterministic 2D world with gravity, drag, wind and wall bounces."""

    def __init__(
        self,
        dt: float = 0.08,
        gravity: float = -0.85,
        action_scale: float = 1.25,
        drag: float = 0.045,
        restitution: float = 0.82,
        bounds: tuple[float, float] = (-1.0, 1.0),
    ):
        self.dt = dt
        self.gravity = gravity
        self.action_scale = action_scale
        self.drag = drag
        self.restitution = restitution
        self.low, self.high = bounds

    def reset(self, rng: np.random.Generator) -> np.ndarray:
        x = rng.uniform(-0.75, 0.75)
        y = rng.uniform(-0.55, 0.85)
        vx = rng.uniform(-0.65, 0.65)
        vy = rng.uniform(-0.45, 0.75)
        return np.array([x, y, vx, vy], dtype=np.float32)

    def sample_action(self, rng: np.random.Generator, step: int, previous: np.ndarray | None = None) -> np.ndarray:
        base = np.array(
            [
                0.65 * np.sin(0.17 * step) + 0.25 * np.cos(0.07 * step),
                0.55 * np.cos(0.13 * step) - 0.15 * np.sin(0.11 * step),
            ],
            dtype=np.float32,
        )
        noise = rng.normal(0.0, 0.18, size=2).astype(np.float32)
        action = base + noise
        if previous is not None:
            action = 0.72 * previous + 0.28 * action
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        x, y, vx, vy = [float(v) for v in state]
        ax, ay = [float(v) for v in np.clip(action, -1.0, 1.0)]
        wind = 0.10 * np.sin(3.0 * y) + 0.04 * np.cos(5.0 * x)
        speed = np.sqrt(vx * vx + vy * vy)
        vx = vx + self.dt * (self.action_scale * ax + wind - self.drag * speed * vx)
        vy = vy + self.dt * (self.action_scale * ay + self.gravity - self.drag * speed * vy)
        x = x + self.dt * vx
        y = y + self.dt * vy

        if x < self.low:
            x = self.low + (self.low - x)
            vx = abs(vx) * self.restitution
        elif x > self.high:
            x = self.high - (x - self.high)
            vx = -abs(vx) * self.restitution
        if y < self.low:
            y = self.low + (self.low - y)
            vy = abs(vy) * self.restitution
            vx *= 0.94
        elif y > self.high:
            y = self.high - (y - self.high)
            vy = -abs(vy) * self.restitution
            vx *= 0.97
        return np.array([x, y, vx, vy], dtype=np.float32)


def generate_trajectories(
    n_trajectories: int,
    steps: int,
    seed: int = 1007,
    simulator: ToyGravityBounceSimulator | None = None,
) -> list[Trajectory]:
    simulator = simulator or ToyGravityBounceSimulator()
    rng = np.random.default_rng(seed)
    trajectories: list[Trajectory] = []
    for trajectory_id in range(n_trajectories):
        state = simulator.reset(rng)
        previous_action = None
        states = [state]
        actions = []
        for step in range(steps):
            action = simulator.sample_action(rng, step + trajectory_id, previous_action)
            next_state = simulator.step(state, action)
            actions.append(action)
            states.append(next_state)
            state = next_state
            previous_action = action
        trajectories.append(
            Trajectory(
                states=np.vstack(states).astype(np.float32),
                actions=np.vstack(actions).astype(np.float32),
                trajectory_id=trajectory_id,
            )
        )
    return trajectories


def transitions_from_trajectories(trajectories: list[Trajectory]) -> list[Transition]:
    transitions: list[Transition] = []
    for trajectory in trajectories:
        for step, action in enumerate(trajectory.actions):
            transitions.append(
                Transition(
                    state=trajectory.states[step],
                    action=action,
                    next_state=trajectory.states[step + 1],
                    trajectory_id=trajectory.trajectory_id,
                    step=step,
                )
            )
    return transitions


def split_trajectories(
    trajectories: list[Trajectory],
    test_fraction: float = 0.15,
) -> tuple[list[Trajectory], list[Trajectory]]:
    cutoff = max(1, int(round(len(trajectories) * (1.0 - test_fraction))))
    return trajectories[:cutoff], trajectories[cutoff:]
