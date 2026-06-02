from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


EPS = 1e-9
LATENT_NAMES = ("x", "y", "vx", "vy", "radius", "shape_code", "wall_x", "wall_y")


@dataclass(frozen=True)
class VisualFrame:
    frame: np.ndarray
    clean_frame: np.ndarray
    state: np.ndarray
    previous_state: np.ndarray
    radius: float
    shape_code: float


class VisualWorldCodec:
    """Hand-built visual encoder/decoder for the Phase 10 toy world.

    The codec keeps AMF dynamics in a tiny dense latent:
    [x, y, vx, vy, radius, shape_code, wall_x, wall_y].
    """

    def __init__(
        self,
        resolution: int = 64,
        dt: float = 0.08,
        radius: float = 0.065,
        shape_code: float = 0.25,
        noise_std: float = 0.015,
        lighting_jitter: float = 0.18,
    ):
        self.resolution = int(resolution)
        self.dt = float(dt)
        self.radius = float(radius)
        self.shape_code = float(shape_code)
        self.noise_std = float(noise_std)
        self.lighting_jitter = float(lighting_jitter)
        self.velocity_color_scale = 0.16
        axis = np.linspace(-1.0, 1.0, self.resolution, dtype=np.float32)
        self.xx, self.yy = np.meshgrid(axis, axis)

    def _blob(self, x: float, y: float, radius: float, shape_code: float) -> np.ndarray:
        if shape_code < 0.5:
            dist2 = (self.xx - x) ** 2 + (self.yy - y) ** 2
            return np.exp(-dist2 / (2.0 * radius * radius)).astype(np.float32)
        dx = np.abs(self.xx - x) / max(radius * 1.15, EPS)
        dy = np.abs(self.yy - y) / max(radius * 0.85, EPS)
        return np.exp(-(dx + dy) ** 2).astype(np.float32)

    @staticmethod
    def wall_features(x: float, y: float) -> tuple[float, float]:
        wall_x = 1.0 - min(abs(float(x)), 1.0)
        wall_y = 1.0 - min(abs(float(y)), 1.0)
        return wall_x, wall_y

    def state_to_latent(self, state: np.ndarray, radius: float | None = None, shape_code: float | None = None) -> np.ndarray:
        x, y, vx, vy = [float(v) for v in state]
        r = self.radius if radius is None else float(radius)
        s = self.shape_code if shape_code is None else float(shape_code)
        wall_x, wall_y = self.wall_features(x, y)
        return np.array([x, y, vx, vy, r, s, wall_x, wall_y], dtype=np.float32)

    def render_clean(self, latent: np.ndarray) -> np.ndarray:
        x, y, vx, vy, radius, shape_code, _, _ = [float(v) for v in latent]
        current = self._blob(x, y, radius, shape_code)
        vx_channel = current * np.clip(0.5 + self.velocity_color_scale * vx, 0.02, 0.98)
        vy_channel = current * np.clip(0.5 + self.velocity_color_scale * vy, 0.02, 0.98)
        identity = current * np.clip(0.25 + 0.50 * shape_code, 0.05, 0.95)
        frame = np.stack(
            [
                np.clip(current, 0.0, 1.0),
                np.clip(vx_channel, 0.0, 1.0),
                np.clip(vy_channel, 0.0, 1.0),
                np.clip(identity, 0.0, 1.0),
            ],
            axis=-1,
        )
        return frame.astype(np.float32)

    def render_from_state(
        self,
        state: np.ndarray,
        previous_state: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        radius: float | None = None,
        shape_code: float | None = None,
    ) -> VisualFrame:
        r = self.radius if radius is None else float(radius)
        s = self.shape_code if shape_code is None else float(shape_code)
        if previous_state is None:
            previous_state = state.copy()
            previous_state[:2] = state[:2] - state[2:] * self.dt
        latent = self.state_to_latent(state, radius=r, shape_code=s)
        clean = self.render_clean(latent)
        if rng is None:
            return VisualFrame(clean.copy(), clean, state.astype(np.float32), previous_state.astype(np.float32), r, s)
        lighting = 1.0 + rng.uniform(-self.lighting_jitter, self.lighting_jitter)
        bias = rng.uniform(-0.025, 0.035)
        noise = rng.normal(0.0, self.noise_std, size=clean.shape).astype(np.float32)
        noisy = np.clip(clean * lighting + bias + noise, 0.0, 1.0)
        return VisualFrame(noisy.astype(np.float32), clean, state.astype(np.float32), previous_state.astype(np.float32), r, s)

    def _weighted_centroid(self, channel: np.ndarray, floor_quantile: float = 0.80) -> tuple[float, float, float]:
        baseline = float(np.quantile(channel, 0.12))
        clean = np.clip(channel - baseline, 0.0, None)
        max_value = float(np.max(clean))
        if max_value <= EPS:
            weights = clean
        else:
            norm = clean / (max_value + EPS)
            weights = np.square(np.clip(norm - 0.20, 0.0, None))
        total = float(np.sum(weights))
        if total <= EPS:
            idx = int(np.argmax(channel))
            row, col = np.unravel_index(idx, channel.shape)
            x = float(self.xx[row, col])
            y = float(self.yy[row, col])
            return x, y, 0.0
        x = float(np.sum(weights * self.xx) / (total + EPS))
        y = float(np.sum(weights * self.yy) / (total + EPS))
        var = float(np.sum(weights * ((self.xx - x) ** 2 + (self.yy - y) ** 2)) / (total + EPS))
        return x, y, np.sqrt(max(var, 0.0))

    def _channel_signal(self, image: np.ndarray, weights: np.ndarray, channel_index: int) -> float:
        channel = image[..., channel_index]
        baseline = float(np.quantile(channel, 0.15))
        clean_channel = np.clip(channel - baseline, 0.0, None)
        clean_current = np.clip(image[..., 0] - float(np.quantile(image[..., 0], 0.15)), 0.0, None)
        numerator = float(np.sum(clean_channel * weights))
        denominator = float(np.sum(clean_current * weights)) + EPS
        return numerator / denominator

    def encode(self, frame: np.ndarray) -> np.ndarray:
        image = np.asarray(frame, dtype=np.float32)
        if image.ndim != 3 or image.shape[-1] != 4:
            raise ValueError("Expected frame with shape [H, W, 4].")
        current = image[..., 0]
        x, y, spread = self._weighted_centroid(current, floor_quantile=0.82)
        clean_current = np.clip(current - float(np.quantile(current, 0.15)), 0.0, None)
        max_current = float(np.max(clean_current))
        norm = clean_current / (max_current + EPS)
        weights = np.clip(norm - 0.18, 0.0, None)
        pixel_area = (2.0 / max(1, self.resolution - 1)) ** 2
        active_area = float(np.sum(norm > 0.45) * pixel_area)
        radius = float(np.clip(np.sqrt(active_area / np.pi) / 1.26, 0.035, 0.12))
        vx_signal = self._channel_signal(image, weights, 1)
        vy_signal = self._channel_signal(image, weights, 2)
        identity_signal = self._channel_signal(image, weights, 3)
        vx = float(np.clip((vx_signal - 0.5) / self.velocity_color_scale, -3.0, 3.0))
        vy = float(np.clip((vy_signal - 0.5) / self.velocity_color_scale, -3.0, 3.0))
        shape_code = float(np.clip((identity_signal - 0.25) / 0.50, 0.0, 1.0))
        shape_code = 1.0 if shape_code >= 0.55 else 0.25
        wall_x, wall_y = self.wall_features(x, y)
        return np.array([x, y, vx, vy, radius, shape_code, wall_x, wall_y], dtype=np.float32)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        return self.render_clean(np.asarray(latent, dtype=np.float32))

    def decode_predicted(self, current_latent: np.ndarray, predicted_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        latent = np.asarray(current_latent, dtype=np.float32).copy()
        latent[:4] = np.asarray(predicted_state, dtype=np.float32)[:4]
        latent[6], latent[7] = self.wall_features(float(latent[0]), float(latent[1]))
        return latent, self.decode(latent)

    @staticmethod
    def latent_bytes() -> int:
        return len(LATENT_NAMES) * 4

    def frame_bytes(self) -> int:
        return self.resolution * self.resolution * 4 * 4

    def export(self, out_dir: str | Path = "data", name: str = "phase10b_visual_codec") -> str:
        out = Path(out_dir)
        out.mkdir(exist_ok=True)
        path = out / f"{name}_{self.resolution}.json"
        metadata: dict[str, Any] = {
            "codec": "VisualWorldCodec",
            "resolution": self.resolution,
            "latent_names": LATENT_NAMES,
            "latent_dim": len(LATENT_NAMES),
            "latent_bytes_float32": self.latent_bytes(),
            "frame_bytes_float32": self.frame_bytes(),
            "compression_ratio_frame_to_latent": self.frame_bytes() / self.latent_bytes(),
            "dt": self.dt,
            "default_radius": self.radius,
            "default_shape_code": self.shape_code,
            "preserves": ["permanence", "separability", "continuity"],
            "discarded": ["noise", "lighting_jitter", "exact_pixel_texture"],
        }
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return str(path)


def mask_iou(a: np.ndarray, b: np.ndarray, threshold: float = 0.05) -> float:
    ma = a[..., 0] > threshold
    mb = b[..., 0] > threshold
    union = np.logical_or(ma, mb)
    if not np.any(union):
        return 1.0
    return float(np.sum(np.logical_and(ma, mb)) / np.sum(union))


def pixel_mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.square(a.astype(np.float32) - b.astype(np.float32))))
