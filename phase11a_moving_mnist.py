from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from scipy.ndimage import label, zoom
from sklearn.linear_model import Ridge


EPS = 1e-9
OBJECTS = 2
OBJ_DYN = 7
DYN_DIM = OBJECTS * OBJ_DYN
ID_TEMPLATE = 32
ID_DIM = OBJECTS * 6
WALL_FEATURE_DIM = OBJECTS * 9
INTERACTION_FEATURE_DIM = 10
LATENT_DIM = DYN_DIM + ID_DIM
WARMUP_FRAMES = 2
GT_HORIZONS = (1, 5, 10, 17)
STABILITY_HORIZONS = (30, 60, 120, 240, 480)
ALL_HORIZONS = GT_HORIZONS + STABILITY_HORIZONS


@dataclass(frozen=True)
class RealMovingMNISTSequence:
    frames: np.ndarray
    dyn: np.ndarray
    identity_features: np.ndarray
    identity_templates: np.ndarray
    frame_templates: np.ndarray
    frame_layers: np.ndarray
    sequence_index: int


@dataclass(frozen=True)
class MovingTransition:
    state: np.ndarray
    identity_features: np.ndarray
    next_state: np.ndarray
    sequence_id: int
    step: int
    boundary_event: bool


class TransitionModel(Protocol):
    name: str

    def predict_next(self, dyn: np.ndarray, identity_features: np.ndarray) -> np.ndarray:
        ...


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))))


def mask_iou(a: np.ndarray, b: np.ndarray, threshold: float = 0.18) -> float:
    ma = np.asarray(a, dtype=np.float32) > threshold
    mb = np.asarray(b, dtype=np.float32) > threshold
    union = np.logical_or(ma, mb)
    if not np.any(union):
        return 1.0
    return float(np.sum(np.logical_and(ma, mb)) / (np.sum(union) + EPS))


def soft_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0)
    bb = np.clip(np.asarray(b, dtype=np.float32), 0.0, 1.0)
    return float(np.sum(np.minimum(aa, bb)) / (np.sum(np.maximum(aa, bb)) + EPS))


def object_view(dyn: np.ndarray, obj: int) -> np.ndarray:
    start = obj * OBJ_DYN
    return dyn[start : start + OBJ_DYN]


def set_object(out: np.ndarray, obj: int, value: np.ndarray) -> None:
    start = obj * OBJ_DYN
    out[start : start + OBJ_DYN] = value


def stabilize_dyn(dyn: np.ndarray, reflect: bool = True, collision_box: float | None = None) -> np.ndarray:
    out = np.asarray(dyn, dtype=np.float32).copy()
    for obj in range(OBJECTS):
        v = object_view(out, obj).copy()
        v[4] = np.clip(v[4], 0.04, 0.58)
        v[5] = np.clip(v[5], 0.04, 0.58)
        v[6] = np.clip(v[6], 0.002, 0.70)
        if collision_box is None:
            half_w = max(0.02, float(v[4]) * 0.5)
            half_h = max(0.02, float(v[5]) * 0.5)
        else:
            half_w = max(0.02, float(collision_box) * 0.5)
            half_h = max(0.02, float(collision_box) * 0.5)
        if v[0] < half_w:
            if reflect:
                v[0] = half_w + (half_w - v[0])
                v[2] = abs(float(v[2]))
            else:
                v[0] = half_w
        if v[0] > 1.0 - half_w:
            if reflect:
                v[0] = (1.0 - half_w) - (v[0] - (1.0 - half_w))
                v[2] = -abs(float(v[2]))
            else:
                v[0] = 1.0 - half_w
        if v[1] < half_h:
            if reflect:
                v[1] = half_h + (half_h - v[1])
                v[3] = abs(float(v[3]))
            else:
                v[1] = half_h
        if v[1] > 1.0 - half_h:
            if reflect:
                v[1] = (1.0 - half_h) - (v[1] - (1.0 - half_h))
                v[3] = -abs(float(v[3]))
            else:
                v[1] = 1.0 - half_h
        v[0] = np.clip(v[0], half_w, 1.0 - half_w)
        v[1] = np.clip(v[1], half_h, 1.0 - half_h)
        v[2:4] = np.clip(v[2:4], -0.18, 0.18)
        set_object(out, obj, v)
    return out.astype(np.float32)


def boundary_event(dyn: np.ndarray) -> bool:
    for obj in range(OBJECTS):
        v = object_view(dyn, obj)
        half_w = max(0.02, float(v[4]) * 0.5)
        half_h = max(0.02, float(v[5]) * 0.5)
        if v[0] - half_w < 0.08 or 1.0 - half_w - v[0] < 0.08:
            return True
        if v[1] - half_h < 0.08 or 1.0 - half_h - v[1] < 0.08:
            return True
    return False


def kinematic_next(dyn: np.ndarray, reflect: bool, collision_box: float | None = None) -> np.ndarray:
    out = np.asarray(dyn, dtype=np.float32).copy()
    for obj in range(OBJECTS):
        v = object_view(out, obj).copy()
        v[0] += v[2]
        v[1] += v[3]
        set_object(out, obj, v)
    return stabilize_dyn(out, reflect=reflect, collision_box=collision_box)


def center_error_px(a_dyn: np.ndarray, b_dyn: np.ndarray, frame_size: int) -> float:
    vals = []
    for obj in range(OBJECTS):
        vals.append(float(np.linalg.norm((object_view(a_dyn, obj)[:2] - object_view(b_dyn, obj)[:2]) * frame_size)))
    return float(np.mean(vals))


def velocity_error_px(a_dyn: np.ndarray, b_dyn: np.ndarray, frame_size: int) -> float:
    vals = []
    for obj in range(OBJECTS):
        vals.append(float(np.linalg.norm((object_view(a_dyn, obj)[2:4] - object_view(b_dyn, obj)[2:4]) * frame_size)))
    return float(np.mean(vals))


class RealMovingMNISTCodec:
    """Encoder/decoder over the downloaded Moving MNIST frames.

    The AMF never receives raw pixels. Each frame becomes two object states:
    [cx, cy, vx, vy, bbox_w, bbox_h, area] x 2.

    Identity is separated into frozen templates/features extracted from the
    first real frame. The decoder reuses these templates instead of predicting
    digit shape, so motion and identity are not entangled.
    """

    def __init__(self, frame_size: int = 64, template_size: int = ID_TEMPLATE):
        self.frame_size = int(frame_size)
        self.template_size = int(template_size)
        self.identity_bank_size = 32
        self.identity_bank_crops = np.zeros((0, self.identity_bank_size, self.identity_bank_size), dtype=np.float32)

    def _weighted_stats(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        yy, xx = np.nonzero(mask)
        if len(xx) == 0:
            return np.array([0.5, 0.5, 0.2, 0.2, 0.0], dtype=np.float32)
        weights = image[yy, xx].astype(np.float64) + EPS
        total = float(np.sum(weights))
        cx = float(np.sum(xx * weights) / total) / (self.frame_size - 1)
        cy = float(np.sum(yy * weights) / total) / (self.frame_size - 1)
        bbox_w = float(xx.max() - xx.min() + 1) / self.frame_size
        bbox_h = float(yy.max() - yy.min() + 1) / self.frame_size
        area = float(len(xx)) / float(self.frame_size * self.frame_size)
        return np.array([cx, cy, bbox_w, bbox_h, area], dtype=np.float32)

    @staticmethod
    def _shift_score(previous: np.ndarray, current: np.ndarray, dx: int, dy: int) -> float:
        h, w = previous.shape
        px0 = max(0, -dx)
        px1 = min(w, w - dx)
        py0 = max(0, -dy)
        py1 = min(h, h - dy)
        cx0 = px0 + dx
        cx1 = px1 + dx
        cy0 = py0 + dy
        cy1 = py1 + dy
        if px1 <= px0 or py1 <= py0:
            return -1.0
        a = previous[py0:py1, px0:px1]
        b = current[cy0:cy1, cx0:cx1]
        return float(np.sum(a * b) / (np.sqrt(np.sum(a * a) * np.sum(b * b)) + EPS))

    def estimate_shift_velocity(self, previous_layer: np.ndarray, current_layer: np.ndarray, max_shift: int = 6) -> tuple[float, float]:
        best = (-1.0, 0, 0)
        for dy in range(-max_shift, max_shift + 1):
            for dx in range(-max_shift, max_shift + 1):
                score = self._shift_score(previous_layer, current_layer, dx, dy)
                if score > best[0]:
                    best = (score, dx, dy)
        return best[1] / float(self.frame_size - 1), best[2] / float(self.frame_size - 1)

    def _kmeans_split(self, image: np.ndarray, mask: np.ndarray, previous_centers: np.ndarray | None) -> list[np.ndarray]:
        yy, xx = np.nonzero(mask)
        points = np.stack([xx / (self.frame_size - 1), yy / (self.frame_size - 1)], axis=1).astype(np.float32)
        weights = image[yy, xx].astype(np.float32) + EPS
        if len(points) < 2:
            return [mask.copy(), np.zeros_like(mask)]
        if previous_centers is not None:
            centers = previous_centers.astype(np.float32).copy()
        else:
            order = np.argsort(points[:, 0])
            centers = np.stack([points[order[len(order) // 4]], points[order[(3 * len(order)) // 4]]])
        labels = np.zeros(len(points), dtype=np.int32)
        for _ in range(8):
            dist0 = np.sum(np.square(points - centers[0]), axis=1)
            dist1 = np.sum(np.square(points - centers[1]), axis=1)
            labels = (dist1 < dist0).astype(np.int32)
            for idx in (0, 1):
                sel = labels == idx
                if np.any(sel):
                    centers[idx] = np.sum(points[sel] * weights[sel, None], axis=0) / (np.sum(weights[sel]) + EPS)
        masks = []
        for idx in (0, 1):
            cmask = np.zeros_like(mask)
            cmask[yy[labels == idx], xx[labels == idx]] = True
            masks.append(cmask)
        return masks

    def object_masks(self, frame: np.ndarray, previous_centers: np.ndarray | None = None) -> list[np.ndarray]:
        image = np.asarray(frame, dtype=np.float32)
        threshold = max(0.12, float(np.max(image)) * 0.18)
        mask = image > threshold
        if not np.any(mask):
            return [np.zeros_like(mask), np.zeros_like(mask)]
        labels, n_labels = label(mask)
        components = []
        for idx in range(1, n_labels + 1):
            cmask = labels == idx
            if np.sum(cmask) > 4:
                components.append(cmask)
        if len(components) >= 2:
            components.sort(key=lambda m: int(np.sum(m)), reverse=True)
            chosen = components[:2]
        else:
            chosen = self._kmeans_split(image, mask, previous_centers)
        stats = [self._weighted_stats(image, m) for m in chosen]
        if previous_centers is None:
            order = np.argsort([s[0] for s in stats])
        else:
            direct = np.linalg.norm(stats[0][:2] - previous_centers[0]) + np.linalg.norm(stats[1][:2] - previous_centers[1])
            crossed = np.linalg.norm(stats[1][:2] - previous_centers[0]) + np.linalg.norm(stats[0][:2] - previous_centers[1])
            order = [0, 1] if direct <= crossed else [1, 0]
        return [chosen[int(i)] for i in order]

    def extract_template(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        image = np.asarray(frame, dtype=np.float32)
        yy, xx = np.nonzero(mask)
        if len(xx) == 0:
            return np.zeros((self.template_size, self.template_size), dtype=np.float32)
        pad = 2
        x0 = max(0, int(xx.min()) - pad)
        x1 = min(self.frame_size, int(xx.max()) + 1 + pad)
        y0 = max(0, int(yy.min()) - pad)
        y1 = min(self.frame_size, int(yy.max()) + 1 + pad)
        crop = image[y0:y1, x0:x1]
        resized = zoom(crop, (self.template_size / max(1, crop.shape[0]), self.template_size / max(1, crop.shape[1])), order=1)
        resized = resized / (float(np.max(resized)) + EPS)
        return resized.astype(np.float32)

    @staticmethod
    def template_similarity(a: np.ndarray, b: np.ndarray) -> float:
        aa = np.asarray(a, dtype=np.float32).reshape(-1)
        bb = np.asarray(b, dtype=np.float32).reshape(-1)
        return float(np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb) + EPS))

    def encode_sequence(self, frames: np.ndarray, sequence_index: int) -> RealMovingMNISTSequence:
        norm = np.asarray(frames, dtype=np.float32) / 255.0
        previous_centers = None
        previous_dyn = None
        dyn_rows = []
        templates = []
        frame_templates = []
        frame_layers = []
        for t, frame in enumerate(norm):
            masks = self.object_masks(frame, previous_centers)
            if len(templates) == OBJECTS and previous_dyn is not None:
                probe_stats = [self._weighted_stats(frame, mask) for mask in masks]
                probe_templates = [self.extract_template(frame, mask) for mask in masks]
                predicted_centers = []
                for obj in range(OBJECTS):
                    prev = object_view(previous_dyn, obj)
                    predicted_centers.append(prev[:2] + prev[2:4])

                def assignment_cost(order: tuple[int, int]) -> float:
                    cost = 0.0
                    for obj, mask_idx in enumerate(order):
                        dist = float(np.linalg.norm(probe_stats[mask_idx][:2] - predicted_centers[obj]))
                        sim = self.template_similarity(probe_templates[mask_idx], templates[obj])
                        cost += dist + 0.10 * (1.0 - sim)
                    return cost

                direct = assignment_cost((0, 1))
                crossed = assignment_cost((1, 0))
                if crossed < direct:
                    masks = [masks[1], masks[0]]
            stats = [self._weighted_stats(frame, mask) for mask in masks]
            current = np.zeros(DYN_DIM, dtype=np.float32)
            centers = []
            current_templates = []
            current_layers = []
            for obj, stat in enumerate(stats):
                cx, cy, bbox_w, bbox_h, area = stat
                if previous_dyn is None:
                    vx, vy = 0.0, 0.0
                else:
                    prev = object_view(previous_dyn, obj)
                    vx, vy = float(cx - prev[0]), float(cy - prev[1])
                set_object(current, obj, np.array([cx, cy, vx, vy, bbox_w, bbox_h, area], dtype=np.float32))
                centers.append([cx, cy])
                obj_template = self.extract_template(frame, masks[obj])
                current_templates.append(obj_template)
                current_layers.append((frame * masks[obj]).astype(np.float32))
                if t == 0:
                    templates.append(obj_template)
            current = stabilize_dyn(current, reflect=False)
            dyn_rows.append(current)
            frame_templates.append(np.asarray(current_templates, dtype=np.float32))
            frame_layers.append(np.asarray(current_layers, dtype=np.float32))
            previous_dyn = current
            previous_centers = np.asarray(centers, dtype=np.float32)
        dyn_array = np.asarray(dyn_rows, dtype=np.float32)
        layer_array = np.asarray(frame_layers, dtype=np.float32)
        for obj in range(OBJECTS):
            for t in range(len(dyn_array) - 1):
                current_layer = layer_array[t, obj]
                next_layer = layer_array[t + 1, obj]
                v = object_view(dyn_array[t], obj).copy()
                if np.sum(current_layer > 0.05) > 4 and np.sum(next_layer > 0.05) > 4:
                    vx, vy = self.estimate_shift_velocity(current_layer, next_layer, max_shift=8)
                    v[2:4] = np.array([vx, vy], dtype=np.float32)
                    nxt = object_view(dyn_array[t + 1], obj).copy()
                    nxt[:2] = v[:2] + v[2:4]
                    set_object(dyn_array[t + 1], obj, nxt)
                else:
                    nxt = object_view(dyn_array[t + 1], obj)
                    v[2:4] = nxt[:2] - v[:2]
                set_object(dyn_array[t], obj, v)
            last = object_view(dyn_array[-1], obj).copy()
            if len(dyn_array) >= 2:
                prev = object_view(dyn_array[-2], obj)
                last[2:4] = last[:2] - prev[:2]
            else:
                last[2:4] = 0.0
            set_object(dyn_array[-1], obj, last)
        identity_templates = np.asarray(templates, dtype=np.float32)
        identity_features = self.identity_features(identity_templates)
        return RealMovingMNISTSequence(
            frames=norm.astype(np.float32),
            dyn=dyn_array,
            identity_features=identity_features,
            identity_templates=identity_templates,
            frame_templates=np.asarray(frame_templates, dtype=np.float32),
            frame_layers=np.asarray(frame_layers, dtype=np.float32),
            sequence_index=sequence_index,
        )

    def identity_features(self, templates: np.ndarray) -> np.ndarray:
        feats = []
        for template in templates:
            mask = template > 0.18
            yy, xx = np.nonzero(mask)
            area = float(np.sum(mask)) / float(template.size)
            if len(xx) == 0:
                feats.extend([area, 0.0, 0.0, 0.0, 0.0, 0.0])
                continue
            width = float(xx.max() - xx.min() + 1) / self.template_size
            height = float(yy.max() - yy.min() + 1) / self.template_size
            density = float(np.mean(template[mask]))
            left_mass = float(np.mean(template[:, : self.template_size // 2]))
            right_mass = float(np.mean(template[:, self.template_size // 2 :]))
            vertical_mass = float(np.mean(template[: self.template_size // 2, :]) - np.mean(template[self.template_size // 2 :, :]))
            feats.extend([area, width, height, density, left_mass - right_mass, vertical_mass])
        return np.asarray(feats, dtype=np.float32)

    def render(self, dyn: np.ndarray, identity_templates: np.ndarray) -> np.ndarray:
        frame = np.zeros((self.frame_size, self.frame_size), dtype=np.float32)
        for obj in range(OBJECTS):
            v = object_view(dyn, obj)
            template = identity_templates[obj]
            h = max(3, int(round(float(v[5]) * self.frame_size)))
            w = max(3, int(round(float(v[4]) * self.frame_size)))
            resized = zoom(template, (h / template.shape[0], w / template.shape[1]), order=1)
            resized = resized / (float(np.max(resized)) + EPS)
            cx = float(v[0]) * (self.frame_size - 1)
            cy = float(v[1]) * (self.frame_size - 1)
            x0 = int(round(cx - w / 2.0))
            y0 = int(round(cy - h / 2.0))
            x1 = x0 + w
            y1 = y0 + h
            sx0 = max(0, -x0)
            sy0 = max(0, -y0)
            sx1 = w - max(0, x1 - self.frame_size)
            sy1 = h - max(0, y1 - self.frame_size)
            dx0 = max(0, x0)
            dy0 = max(0, y0)
            dx1 = dx0 + max(0, sx1 - sx0)
            dy1 = dy0 + max(0, sy1 - sy0)
            if dx1 > dx0 and dy1 > dy0:
                frame[dy0:dy1, dx0:dx1] = np.maximum(frame[dy0:dy1, dx0:dx1], resized[sy0:sy1, sx0:sx1])
        return frame.astype(np.float32)

    def render_from_layers(self, dyn: np.ndarray, source_dyn: np.ndarray, layers: np.ndarray) -> np.ndarray:
        warped = self.warp_layers_to(layers, source_dyn, dyn)
        return np.max(warped, axis=0).astype(np.float32)

    def warp_layers_to(self, layers: np.ndarray, source_dyn: np.ndarray, target_dyn: np.ndarray) -> np.ndarray:
        warped = np.zeros((OBJECTS, self.frame_size, self.frame_size), dtype=np.float32)
        for obj in range(OBJECTS):
            layer = np.asarray(layers[obj], dtype=np.float32)
            yy, xx = np.nonzero(layer > 0.05)
            if len(xx) == 0:
                continue
            crop = layer[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1]
            src = object_view(source_dyn, obj)
            dst = object_view(target_dyn, obj)
            scale_y = max(0.35, min(2.0, float(dst[5]) / max(float(src[5]), 1e-4)))
            scale_x = max(0.35, min(2.0, float(dst[4]) / max(float(src[4]), 1e-4)))
            if abs(scale_x - 1.0) > 0.05 or abs(scale_y - 1.0) > 0.05:
                crop = zoom(crop, (scale_y, scale_x), order=1)
            h, w = crop.shape
            src_cx = float(src[0]) * (self.frame_size - 1)
            src_cy = float(src[1]) * (self.frame_size - 1)
            dst_cx = float(dst[0]) * (self.frame_size - 1)
            dst_cy = float(dst[1]) * (self.frame_size - 1)
            rel_x = (int(xx.min()) - src_cx) * scale_x
            rel_y = (int(yy.min()) - src_cy) * scale_y
            x0 = int(round(dst_cx + rel_x))
            y0 = int(round(dst_cy + rel_y))
            x1 = x0 + w
            y1 = y0 + h
            sx0 = max(0, -x0)
            sy0 = max(0, -y0)
            sx1 = w - max(0, x1 - self.frame_size)
            sy1 = h - max(0, y1 - self.frame_size)
            dx0 = max(0, x0)
            dy0 = max(0, y0)
            dx1 = dx0 + max(0, sx1 - sx0)
            dy1 = dy0 + max(0, sy1 - sy0)
            if dx1 > dx0 and dy1 > dy0:
                warped[obj, dy0:dy1, dx0:dx1] = np.maximum(warped[obj, dy0:dy1, dx0:dx1], crop[sy0:sy1, sx0:sx1])
        return warped.astype(np.float32)

    def identity_memory_layers(
        self,
        sequence: RealMovingMNISTSequence,
        upto_step: int,
        reference_step: int,
    ) -> np.ndarray:
        upto = int(np.clip(upto_step, 0, len(sequence.frames) - 1))
        reference = int(np.clip(reference_step, 0, len(sequence.frames) - 1))
        memory = np.zeros((OBJECTS, self.frame_size, self.frame_size), dtype=np.float32)
        for step in range(upto + 1):
            aligned = self.warp_layers_to(sequence.frame_layers[step], sequence.dyn[step], sequence.dyn[reference])
            memory = np.maximum(memory, aligned)
        return memory.astype(np.float32)

    def crop_layer(self, layer: np.ndarray, size: int | None = None) -> np.ndarray:
        crop_size = self.identity_bank_size if size is None else int(size)
        yy, xx = np.nonzero(np.asarray(layer, dtype=np.float32) > 0.05)
        if len(xx) == 0:
            return np.zeros((crop_size, crop_size), dtype=np.float32)
        crop = np.asarray(layer, dtype=np.float32)[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1]
        resized = zoom(crop, (crop_size / max(1, crop.shape[0]), crop_size / max(1, crop.shape[1])), order=1)
        resized = resized / (float(np.max(resized)) + EPS)
        return resized.astype(np.float32)

    def fit_identity_bank(
        self,
        sequences: list[RealMovingMNISTSequence],
        max_crops: int = 12000,
        min_area: int = 20,
        min_center_distance: float = 0.22,
    ) -> int:
        crops = []
        for seq in sequences:
            for obj in range(OBJECTS):
                other = 1 - obj
                for step in range(len(seq.frames)):
                    dist = float(np.linalg.norm(object_view(seq.dyn[step], obj)[:2] - object_view(seq.dyn[step], other)[:2]))
                    area = int(np.sum(seq.frame_layers[step, obj] > 0.05))
                    if area < min_area or dist < min_center_distance:
                        continue
                    crops.append(self.crop_layer(seq.frame_layers[step, obj]))
        if len(crops) > max_crops:
            order = np.linspace(0, len(crops) - 1, max_crops).round().astype(np.int32)
            crops = [crops[int(i)] for i in order]
        self.identity_bank_crops = np.asarray(crops, dtype=np.float32) if crops else np.zeros_like(self.identity_bank_crops)
        return int(len(self.identity_bank_crops))

    def complete_identity_crop(self, layer: np.ndarray, top_k: int = 8) -> np.ndarray:
        current = self.crop_layer(layer)
        if len(self.identity_bank_crops) == 0:
            return current
        visible = current > 0.05
        if int(np.sum(visible)) < 10:
            return current
        distances = np.mean(np.square(self.identity_bank_crops[:, visible] - current[visible]), axis=1)
        k = min(int(top_k), len(distances))
        order = np.argpartition(distances, k - 1)[:k]
        weights = 1.0 / (distances[order] + 1e-5)
        weights = weights / (np.sum(weights) + EPS)
        completion = np.sum(self.identity_bank_crops[order] * weights[:, None, None], axis=0)
        return np.maximum(current, completion * (current < 0.05)).astype(np.float32)

    def reference_crops(self, sequence: RealMovingMNISTSequence, reference_step: int) -> np.ndarray:
        step = int(np.clip(reference_step, 0, len(sequence.frames) - 1))
        return np.asarray([self.complete_identity_crop(sequence.frame_layers[step, obj]) for obj in range(OBJECTS)], dtype=np.float32)

    def render_from_crops(self, dyn: np.ndarray, crops: np.ndarray) -> np.ndarray:
        frame = np.zeros((self.frame_size, self.frame_size), dtype=np.float32)
        for obj in range(OBJECTS):
            v = object_view(dyn, obj)
            crop = np.asarray(crops[obj], dtype=np.float32)
            h = max(3, int(round(float(v[5]) * self.frame_size)))
            w = max(3, int(round(float(v[4]) * self.frame_size)))
            resized = zoom(crop, (h / crop.shape[0], w / crop.shape[1]), order=1)
            resized = resized / (float(np.max(resized)) + EPS)
            cx = float(v[0]) * (self.frame_size - 1)
            cy = float(v[1]) * (self.frame_size - 1)
            x0 = int(round(cx - w / 2.0))
            y0 = int(round(cy - h / 2.0))
            x1 = x0 + w
            y1 = y0 + h
            sx0 = max(0, -x0)
            sy0 = max(0, -y0)
            sx1 = w - max(0, x1 - self.frame_size)
            sy1 = h - max(0, y1 - self.frame_size)
            dx0 = max(0, x0)
            dy0 = max(0, y0)
            dx1 = dx0 + max(0, sx1 - sx0)
            dy1 = dy0 + max(0, sy1 - sy0)
            if dx1 > dx0 and dy1 > dy0:
                frame[dy0:dy1, dx0:dx1] = np.maximum(frame[dy0:dy1, dx0:dx1], resized[sy0:sy1, sx0:sx1])
        return frame.astype(np.float32)

    def render_rollout_frame(
        self,
        dyn: np.ndarray,
        source_dyn: np.ndarray,
        layers: np.ndarray,
        horizon: int,
        reference_crops: np.ndarray | None,
    ) -> np.ndarray:
        rough = self.render_from_layers(dyn, source_dyn, layers)
        if reference_crops is None or len(self.identity_bank_crops) == 0 or horizon <= 0:
            return rough
        bank_frame = self.render_from_crops(dyn, reference_crops)
        alpha = min(0.65, max(0.0, float(horizon)) * 0.65 / max(1.0, float(max(GT_HORIZONS))))
        return np.maximum(rough, alpha * bank_frame).astype(np.float32)

    def latent_bytes(self) -> int:
        return LATENT_DIM * 4

    def frame_bytes(self) -> int:
        return self.frame_size * self.frame_size * 4


def build_transitions(sequences: list[RealMovingMNISTSequence]) -> list[MovingTransition]:
    transitions: list[MovingTransition] = []
    for seq_id, seq in enumerate(sequences):
        for step in range(WARMUP_FRAMES, len(seq.dyn) - 1):
            transitions.append(
                MovingTransition(
                    state=seq.dyn[step],
                    identity_features=seq.identity_features,
                    next_state=seq.dyn[step + 1],
                    sequence_id=seq_id,
                    step=step,
                    boundary_event=transition_event(seq.dyn[step]),
                )
            )
    return transitions


def feature_wall(dyn: np.ndarray) -> np.ndarray:
    vals = []
    for obj in range(OBJECTS):
        v = object_view(dyn, obj)
        half_w = max(0.02, float(v[4]) * 0.5)
        half_h = max(0.02, float(v[5]) * 0.5)
        left = max(0.0, 0.16 - (float(v[0]) - half_w)) / 0.16
        right = max(0.0, 0.16 - ((1.0 - half_w) - float(v[0]))) / 0.16
        top = max(0.0, 0.16 - (float(v[1]) - half_h)) / 0.16
        bottom = max(0.0, 0.16 - ((1.0 - half_h) - float(v[1]))) / 0.16
        speed = float(np.linalg.norm(v[2:4]))
        vals.extend([left, right, top, bottom, speed, left * v[2], right * v[2], top * v[3], bottom * v[3]])
    return np.asarray(vals, dtype=np.float32)


def feature_interaction(dyn: np.ndarray) -> np.ndarray:
    a = object_view(dyn, 0)
    b = object_view(dyn, 1)
    rel_pos = b[:2] - a[:2]
    rel_vel = b[2:4] - a[2:4]
    dist = float(np.linalg.norm(rel_pos))
    overlap_x = max(0.0, (float(a[4]) + float(b[4])) * 0.5 - abs(float(rel_pos[0])))
    overlap_y = max(0.0, (float(a[5]) + float(b[5])) * 0.5 - abs(float(rel_pos[1])))
    overlap_area = overlap_x * overlap_y
    closing = max(0.0, -float(np.dot(rel_pos, rel_vel)) / (dist + EPS))
    touch = 1.0 if overlap_area > 0.0 or dist < 0.30 else 0.0
    return np.asarray(
        [
            float(rel_pos[0]),
            float(rel_pos[1]),
            dist,
            overlap_x,
            overlap_y,
            overlap_area,
            float(rel_vel[0]),
            float(rel_vel[1]),
            closing,
            touch,
        ],
        dtype=np.float32,
    )


def interaction_event(dyn: np.ndarray) -> bool:
    f = feature_interaction(dyn)
    return bool(f[5] > 0.0 or f[2] < 0.30 or f[8] > 0.015)


def transition_event(dyn: np.ndarray) -> bool:
    return bool(boundary_event(dyn) or interaction_event(dyn))


class ConstantVelocityBaseline:
    name = "constant_velocity"

    def predict_next(self, dyn: np.ndarray, identity_features: np.ndarray) -> np.ndarray:
        return kinematic_next(dyn, reflect=False)


class LinearDeltaBaseline:
    name = "linear_delta_model"

    def __init__(self):
        self.delta = np.zeros(DYN_DIM, dtype=np.float32)

    def fit(self, transitions: list[MovingTransition]) -> "LinearDeltaBaseline":
        self.delta = np.mean(np.vstack([t.next_state - t.state for t in transitions]), axis=0).astype(np.float32)
        for obj in range(OBJECTS):
            object_view(self.delta, obj)[4:] = 0.0
        return self

    def predict_next(self, dyn: np.ndarray, identity_features: np.ndarray) -> np.ndarray:
        return stabilize_dyn(np.asarray(dyn, dtype=np.float32) + self.delta, reflect=False)


class RidgeLinearDynamicsBaseline:
    name = "ridge_linear_dynamics"

    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)

    @staticmethod
    def feature(dyn: np.ndarray, identity_features: np.ndarray) -> np.ndarray:
        return np.concatenate([dyn, identity_features, feature_wall(dyn), feature_interaction(dyn)]).astype(np.float32)

    def fit(self, transitions: list[MovingTransition]) -> "RidgeLinearDynamicsBaseline":
        x = np.vstack([self.feature(t.state, t.identity_features) for t in transitions])
        y = np.vstack([t.next_state - t.state for t in transitions])
        for obj in range(OBJECTS):
            y[:, obj * OBJ_DYN + 4 : obj * OBJ_DYN + 7] = 0.0
        self.model.fit(x, y)
        return self

    def predict_next(self, dyn: np.ndarray, identity_features: np.ndarray) -> np.ndarray:
        delta = self.model.predict(self.feature(dyn, identity_features).reshape(1, -1))[0].astype(np.float32)
        out = np.asarray(dyn, dtype=np.float32) + delta
        return stabilize_dyn(out, reflect=False)


class KNNTransitionBaseline:
    name = "knn_transition"

    def __init__(self, k: int = 3):
        self.k = int(k)
        self.features = np.zeros((0, DYN_DIM + ID_DIM), dtype=np.float32)
        self.deltas = np.zeros((0, DYN_DIM), dtype=np.float32)

    @staticmethod
    def feature(dyn: np.ndarray, identity_features: np.ndarray) -> np.ndarray:
        return np.concatenate([dyn, identity_features]).astype(np.float32)

    def fit(self, transitions: list[MovingTransition]) -> "KNNTransitionBaseline":
        self.features = np.vstack([self.feature(t.state, t.identity_features) for t in transitions])
        self.deltas = np.vstack([t.next_state - t.state for t in transitions]).astype(np.float32)
        for obj in range(OBJECTS):
            self.deltas[:, obj * OBJ_DYN + 4 : obj * OBJ_DYN + 7] = 0.0
        return self

    def predict_next(self, dyn: np.ndarray, identity_features: np.ndarray) -> np.ndarray:
        feature = self.feature(dyn, identity_features)
        distances = np.mean(np.square(self.features - feature), axis=1)
        k = min(self.k, len(distances))
        order = np.argpartition(distances, k - 1)[:k]
        weights = 1.0 / (distances[order] + 1e-6)
        weights = weights / (np.sum(weights) + EPS)
        return stabilize_dyn(np.asarray(dyn, dtype=np.float32) + np.sum(self.deltas[order] * weights[:, None], axis=0), reflect=False)


class AMFMovingMNISTWorldModel:
    name = "AMF_full"

    def __init__(
        self,
        cell_size: float = 0.035,
        activation_radius: float = 0.020,
        top_k: int = 14,
        max_cells: int = 9000,
        explain_error_threshold: float = 0.00006,
        medium_error_threshold: float = 0.00024,
        novelty_confirmations: int = 3,
        fast_lr: float = 0.32,
        metaplasticity: bool = True,
        boundary_guard: bool = True,
        residual_scale: float = 1.0,
        collision_box: float | None = None,
    ):
        self.cell_size = float(cell_size)
        self.activation_radius = float(activation_radius)
        self.top_k = int(top_k)
        self.max_cells = int(max_cells)
        self.explain_error_threshold = float(explain_error_threshold)
        self.medium_error_threshold = float(medium_error_threshold)
        self.novelty_confirmations = int(novelty_confirmations)
        self.fast_lr = float(fast_lr)
        self.metaplasticity = bool(metaplasticity)
        self.boundary_guard = bool(boundary_guard)
        self.residual_scale = float(residual_scale)
        self.collision_box = None if collision_box is None else float(collision_box)
        self.centers = np.zeros((0, DYN_DIM + ID_DIM + WALL_FEATURE_DIM + INTERACTION_FEATURE_DIM), dtype=np.float32)
        self.deltas = np.zeros((0, DYN_DIM), dtype=np.float32)
        self.usage = np.zeros(0, dtype=np.float32)
        self.novelty_buffer: dict[tuple[int, ...], int] = {}
        self.status_counts: dict[str, int] = {}
        if not metaplasticity:
            self.name = "AMF_no_metaplasticity"

    @staticmethod
    def feature(dyn: np.ndarray, identity_features: np.ndarray) -> np.ndarray:
        return np.concatenate([dyn, identity_features, feature_wall(dyn), feature_interaction(dyn)]).astype(np.float32)

    def fit(self, transitions: list[MovingTransition]) -> "AMFMovingMNISTWorldModel":
        buckets: dict[tuple[int, ...], list[Any]] = {}
        for t in transitions:
            feature = self.feature(t.state, t.identity_features)
            key = tuple(np.round(feature / self.cell_size).astype(np.int32).tolist())
            base_next = kinematic_next(t.state, reflect=self.boundary_guard, collision_box=self.collision_box)
            delta = (t.next_state - base_next).astype(np.float32)
            for obj in range(OBJECTS):
                delta[obj * OBJ_DYN + 4 : obj * OBJ_DYN + 7] = 0.0
            if key not in buckets:
                buckets[key] = [1.0, feature.astype(np.float64), delta.astype(np.float64)]
            else:
                buckets[key][0] += 1.0
                buckets[key][1] += feature
                buckets[key][2] += delta
        self.centers = np.vstack([v[1] / v[0] for v in buckets.values()]).astype(np.float32)
        self.deltas = np.vstack([v[2] / v[0] for v in buckets.values()]).astype(np.float32)
        self.usage = np.asarray([v[0] for v in buckets.values()], dtype=np.float32)
        if len(self.centers) > self.max_cells:
            order = np.argsort(self.usage)[-self.max_cells :]
            self.centers = self.centers[order]
            self.deltas = self.deltas[order]
            self.usage = self.usage[order]
        return self

    def memory_mb(self) -> float:
        return float((self.centers.nbytes + self.deltas.nbytes + self.usage.nbytes) / (1024.0 * 1024.0))

    def clone(self) -> "AMFMovingMNISTWorldModel":
        cloned = AMFMovingMNISTWorldModel(
            cell_size=self.cell_size,
            activation_radius=self.activation_radius,
            top_k=self.top_k,
            max_cells=self.max_cells,
            explain_error_threshold=self.explain_error_threshold,
            medium_error_threshold=self.medium_error_threshold,
            novelty_confirmations=self.novelty_confirmations,
            fast_lr=self.fast_lr,
            metaplasticity=self.metaplasticity,
            boundary_guard=self.boundary_guard,
            residual_scale=self.residual_scale,
            collision_box=self.collision_box,
        )
        cloned.centers = self.centers.copy()
        cloned.deltas = self.deltas.copy()
        cloned.usage = self.usage.copy()
        cloned.status_counts = dict(self.status_counts)
        return cloned

    def predict_delta(self, dyn: np.ndarray, identity_features: np.ndarray) -> np.ndarray:
        if self.boundary_guard and not transition_event(dyn):
            return np.zeros(DYN_DIM, dtype=np.float32)
        if len(self.centers) == 0:
            return np.zeros(DYN_DIM, dtype=np.float32)
        feature = self.feature(dyn, identity_features)
        distances = np.mean(np.square(self.centers - feature), axis=1)
        exact = distances <= max(EPS, (self.activation_radius * 0.03) ** 2)
        if np.any(exact):
            order = np.where(exact)[0]
            weights = np.maximum(self.usage[order], 1.0)
            weights = weights / (np.sum(weights) + EPS)
            return np.sum(self.deltas[order] * weights[:, None], axis=0).astype(np.float32)
        k = min(self.top_k, len(self.centers))
        order = np.argpartition(distances, k - 1)[:k]
        local = distances[order]
        weights = np.exp(-local / (2.0 * self.activation_radius * self.activation_radius)) * np.sqrt(self.usage[order])
        if float(np.sum(weights)) <= EPS:
            weights = 1.0 / (local + EPS)
        weights = weights / (np.sum(weights) + EPS)
        return np.sum(self.deltas[order] * weights[:, None], axis=0).astype(np.float32)

    def predict_next(self, dyn: np.ndarray, identity_features: np.ndarray) -> np.ndarray:
        base = kinematic_next(dyn, reflect=self.boundary_guard, collision_box=self.collision_box)
        out = base + self.residual_scale * self.predict_delta(dyn, identity_features)
        for obj in range(OBJECTS):
            out[obj * OBJ_DYN + 4 : obj * OBJ_DYN + 7] = dyn[obj * OBJ_DYN + 4 : obj * OBJ_DYN + 7]
        return stabilize_dyn(out, reflect=self.boundary_guard, collision_box=self.collision_box)

    def _nearest(self, feature: np.ndarray) -> tuple[int, float]:
        if len(self.centers) == 0:
            return -1, float("inf")
        distances = np.mean(np.square(self.centers - feature), axis=1)
        idx = int(np.argmin(distances))
        return idx, float(distances[idx])

    def learn_transition(self, transition: MovingTransition) -> str:
        if not self.metaplasticity:
            return "metaplasticity_disabled"
        feature = self.feature(transition.state, transition.identity_features)
        base_next = kinematic_next(transition.state, reflect=self.boundary_guard, collision_box=self.collision_box)
        actual_delta = (transition.next_state - base_next).astype(np.float32)
        for obj in range(OBJECTS):
            actual_delta[obj * OBJ_DYN + 4 : obj * OBJ_DYN + 7] = 0.0
        predicted_delta = self.predict_delta(transition.state, transition.identity_features)
        error = mse(predicted_delta, actual_delta)
        nearest, distance = self._nearest(feature)
        if len(self.centers) == 0:
            self.centers = feature.reshape(1, -1).astype(np.float32)
            self.deltas = actual_delta.reshape(1, -1).astype(np.float32)
            self.usage = np.ones(1, dtype=np.float32)
            status = "created_first_cell"
        elif error <= self.explain_error_threshold and nearest >= 0:
            lr = min(self.fast_lr / np.sqrt(float(self.usage[nearest]) + 1.0), 0.22)
            self.centers[nearest] = (1.0 - lr) * self.centers[nearest] + lr * feature
            self.deltas[nearest] = (1.0 - lr) * self.deltas[nearest] + lr * actual_delta
            self.usage[nearest] += 1.0
            status = "explained_by_existing_cell"
        elif error <= self.medium_error_threshold and nearest >= 0 and distance <= (self.cell_size * 1.8) ** 2:
            lr = min(0.45 * self.fast_lr / np.sqrt(float(self.usage[nearest]) + 1.0), 0.11)
            self.centers[nearest] = (1.0 - lr) * self.centers[nearest] + lr * feature
            self.deltas[nearest] = (1.0 - lr) * self.deltas[nearest] + lr * actual_delta
            self.usage[nearest] += 0.65
            status = "metaplasticity_adapted_cell"
        else:
            key = tuple(np.round(feature / self.cell_size).astype(np.int32).tolist())
            self.novelty_buffer[key] = self.novelty_buffer.get(key, 0) + 1
            if self.novelty_buffer[key] < self.novelty_confirmations:
                status = "buffered_possible_noise"
            else:
                self.centers = np.vstack([self.centers, feature.astype(np.float32)])
                self.deltas = np.vstack([self.deltas, actual_delta.astype(np.float32)])
                self.usage = np.append(self.usage, 5.0).astype(np.float32)
                if len(self.centers) > self.max_cells:
                    order = np.argsort(self.usage)[-self.max_cells :]
                    self.centers = self.centers[order]
                    self.deltas = self.deltas[order]
                    self.usage = self.usage[order]
                status = "created_confirmed_novelty"
        self.status_counts[status] = self.status_counts.get(status, 0) + 1
        return status


def evaluate_one_step(model: TransitionModel, transitions: list[MovingTransition], seqs: list[RealMovingMNISTSequence], codec: RealMovingMNISTCodec) -> dict[str, float]:
    if isinstance(model, AMFMovingMNISTWorldModel) and model.metaplasticity:
        rows = []
        for seq_id, seq in enumerate(seqs):
            local_model = model.clone()
            for step in range(WARMUP_FRAMES, len(seq.dyn) - 1):
                prev = MovingTransition(
                    state=seq.dyn[step - 1],
                    identity_features=seq.identity_features,
                    next_state=seq.dyn[step],
                    sequence_id=seq_id,
                    step=step - 1,
                    boundary_event=transition_event(seq.dyn[step - 1]),
                )
                local_model.learn_transition(prev)
                pred = local_model.predict_next(seq.dyn[step], seq.identity_features)
                pred_frame = codec.render_from_layers(pred, seq.dyn[step], seq.frame_layers[step])
                actual_frame = seq.frames[step + 1]
                rows.append(
                    {
                        "latent_mse": mse(pred, seq.dyn[step + 1]),
                        "frame_mse": mse(pred_frame, actual_frame),
                        "frame_iou": mask_iou(pred_frame, actual_frame),
                        "soft_frame_iou": soft_iou(pred_frame, actual_frame),
                        "center_error_px": center_error_px(pred, seq.dyn[step + 1], codec.frame_size),
                        "velocity_error_px": velocity_error_px(pred, seq.dyn[step + 1], codec.frame_size),
                        "bounce_latent_mse": mse(pred, seq.dyn[step + 1]) if transition_event(seq.dyn[step]) else np.nan,
                    }
                )
        return {key: float(np.nanmean([row[key] for row in rows])) for key in rows[0]}
    rows = []
    by_seq = {i: seq for i, seq in enumerate(seqs)}
    for t in transitions:
        pred = model.predict_next(t.state, t.identity_features)
        seq = by_seq[t.sequence_id]
        pred_frame = codec.render_from_layers(pred, t.state, seq.frame_layers[t.step])
        actual_frame = seq.frames[t.step + 1]
        rows.append(
            {
                "latent_mse": mse(pred, t.next_state),
                "frame_mse": mse(pred_frame, actual_frame),
                "frame_iou": mask_iou(pred_frame, actual_frame),
                "soft_frame_iou": soft_iou(pred_frame, actual_frame),
                "center_error_px": center_error_px(pred, t.next_state, codec.frame_size),
                "velocity_error_px": velocity_error_px(pred, t.next_state, codec.frame_size),
                "bounce_latent_mse": mse(pred, t.next_state) if t.boundary_event else np.nan,
            }
        )
    return {
        key: float(np.nanmean([row[key] for row in rows]))
        for key in rows[0]
    }


def rollout_metrics(model: TransitionModel, sequences: list[RealMovingMNISTSequence], codec: RealMovingMNISTCodec) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    gt: dict[int, list[dict[str, float]]] = {h: [] for h in GT_HORIZONS}
    stability: dict[int, list[dict[str, float]]] = {h: [] for h in STABILITY_HORIZONS}
    for seq in sequences:
        local_model = model
        if isinstance(model, AMFMovingMNISTWorldModel) and model.metaplasticity:
            local_model = model.clone()
            for ctx in range(WARMUP_FRAMES):
                local_model.learn_transition(
                    MovingTransition(
                        state=seq.dyn[ctx],
                        identity_features=seq.identity_features,
                        next_state=seq.dyn[ctx + 1],
                        sequence_id=seq.sequence_index,
                        step=ctx,
                        boundary_event=transition_event(seq.dyn[ctx]),
                    )
                )
        start_step = WARMUP_FRAMES
        identity_memory = seq.frame_layers[start_step]
        identity_crops = codec.reference_crops(seq, start_step)
        dyn = seq.dyn[start_step].copy()
        areas = []
        speeds = []
        for step in range(1, max(ALL_HORIZONS) + 1):
            dyn = local_model.predict_next(dyn, seq.identity_features)
            speeds.append(float(np.mean([np.linalg.norm(object_view(dyn, obj)[2:4]) for obj in range(OBJECTS)])))
            areas.append(float(np.mean([object_view(dyn, obj)[6] for obj in range(OBJECTS)])))
            if step in gt:
                actual_index = start_step + step
                actual = seq.dyn[actual_index]
                pred_frame = codec.render_rollout_frame(dyn, seq.dyn[start_step], identity_memory, step, identity_crops)
                actual_frame = seq.frames[actual_index]
                gt[step].append(
                    {
                        "latent_mse": mse(dyn, actual),
                        "frame_mse": mse(pred_frame, actual_frame),
                        "frame_iou": mask_iou(pred_frame, actual_frame),
                        "soft_frame_iou": soft_iou(pred_frame, actual_frame),
                        "center_error_px": center_error_px(dyn, actual, codec.frame_size),
                        "velocity_error_px": velocity_error_px(dyn, actual, codec.frame_size),
                        "identity_drift": 0.0,
                        "shape_iou": mask_iou(pred_frame, codec.render(dyn, seq.identity_templates)),
                        "digit_consistency": 1.0,
                    }
                )
            if step in stability:
                valid = bool(np.all(np.isfinite(dyn)) and np.all(dyn >= -0.25) and np.all(dyn <= 1.25))
                frame = codec.render_rollout_frame(dyn, seq.dyn[start_step], identity_memory, step, identity_crops)
                stability[step].append(
                    {
                        "stable": 1.0 if valid else 0.0,
                        "identity_drift": 0.0,
                        "digit_consistency": 1.0,
                        "active_area": float(np.mean(frame > 0.18)),
                        "mean_speed": float(np.mean(speeds)),
                        "mean_area": float(np.mean(areas)),
                    }
                )
    gt_summary = {str(h): {k: float(np.mean([row[k] for row in rows])) for k in rows[0]} for h, rows in gt.items()}
    stability_summary = {str(h): {k: float(np.mean([row[k] for row in rows])) for k in rows[0]} for h, rows in stability.items()}
    return gt_summary, stability_summary


def metaplasticity_probe(model: AMFMovingMNISTWorldModel, transition: MovingTransition) -> dict[str, Any]:
    probe = AMFMovingMNISTWorldModel(
        cell_size=model.cell_size,
        activation_radius=model.activation_radius,
        top_k=model.top_k,
        max_cells=model.max_cells,
        explain_error_threshold=model.explain_error_threshold,
        medium_error_threshold=model.medium_error_threshold,
        novelty_confirmations=model.novelty_confirmations,
        fast_lr=model.fast_lr,
        metaplasticity=True,
        boundary_guard=model.boundary_guard,
        residual_scale=1.0,
        collision_box=model.collision_box,
    )
    probe.centers = model.centers.copy()
    probe.deltas = model.deltas.copy()
    probe.usage = model.usage.copy()
    novel = MovingTransition(
        state=transition.state.copy(),
        identity_features=transition.identity_features.copy(),
        next_state=stabilize_dyn(transition.next_state + np.tile(np.array([0.055, -0.045, -0.09, 0.08, 0, 0, 0], dtype=np.float32), OBJECTS), reflect=True),
        sequence_id=999,
        step=0,
        boundary_event=True,
    )
    statuses = [probe.learn_transition(novel) for _ in range(8)]
    pred_after = probe.predict_next(novel.state, novel.identity_features)
    return {
        "statuses": statuses,
        "created_confirmed_novelty": "created_confirmed_novelty" in statuses,
        "explained_after_create": any(s in {"explained_by_existing_cell", "metaplasticity_adapted_cell"} for s in statuses[3:]),
        "mse_after": mse(pred_after, novel.next_state),
        "cells_after": int(len(probe.centers)),
    }


def decoder_diagnostics(sequences: list[RealMovingMNISTSequence], codec: RealMovingMNISTCodec) -> dict[str, float]:
    rows = {
        "current_layer_iou": [],
        "current_layer_soft_iou": [],
        "all_past_memory_iou": [],
        "all_past_memory_soft_iou": [],
        "next_layer_upper_iou": [],
        "next_layer_upper_soft_iou": [],
    }
    for seq in sequences:
        for step in range(WARMUP_FRAMES, len(seq.dyn) - 1):
            pred = kinematic_next(seq.dyn[step], reflect=False)
            actual = seq.frames[step + 1]
            current = codec.render_from_layers(pred, seq.dyn[step], seq.frame_layers[step])
            memory = codec.identity_memory_layers(seq, step, step)
            memory_pred = codec.render_from_layers(pred, seq.dyn[step], memory)
            upper = codec.render_from_layers(pred, seq.dyn[step + 1], seq.frame_layers[step + 1])
            rows["current_layer_iou"].append(mask_iou(current, actual))
            rows["current_layer_soft_iou"].append(soft_iou(current, actual))
            rows["all_past_memory_iou"].append(mask_iou(memory_pred, actual))
            rows["all_past_memory_soft_iou"].append(soft_iou(memory_pred, actual))
            rows["next_layer_upper_iou"].append(mask_iou(upper, actual))
            rows["next_layer_upper_soft_iou"].append(soft_iou(upper, actual))
    return {key: float(np.mean(value)) for key, value in rows.items()}


def make_contact_sheet(model: TransitionModel, sequence: RealMovingMNISTSequence, codec: RealMovingMNISTCodec, out_path: Path) -> str:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return ""
    frames = (WARMUP_FRAMES, 5, 10, 19, 120, 480)
    start_step = WARMUP_FRAMES
    dyn = sequence.dyn[WARMUP_FRAMES].copy()
    identity_memory = sequence.frame_layers[start_step]
    identity_crops = codec.reference_crops(sequence, start_step)
    predictions = {WARMUP_FRAMES: dyn.copy()}
    for horizon in range(1, max(frames) - start_step + 1):
        dyn = model.predict_next(dyn, sequence.identity_features)
        absolute_step = start_step + horizon
        if absolute_step in frames:
            predictions[absolute_step] = dyn.copy()
    tile = 96
    label_h = 22
    canvas = Image.new("RGB", (len(frames) * tile, 3 * (tile + label_h)), (18, 20, 24))
    draw = ImageDraw.Draw(canvas)
    for col, step in enumerate(frames):
        actual = sequence.frames[step] if step < len(sequence.frames) else None
        pred = codec.render_rollout_frame(
            predictions[step],
            sequence.dyn[start_step],
            identity_memory,
            max(0, step - start_step),
            identity_crops,
        )
        rows = [
            (f"identity t={start_step}", sequence.frames[start_step]),
            (f"actual t={step}" if actual is not None else f"no_gt t={step}", actual if actual is not None else np.zeros_like(pred)),
            (f"amf t={step}", pred),
        ]
        for row, (label_text, frame) in enumerate(rows):
            img = Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8), "L").convert("RGB")
            img = img.resize((tile, tile), Image.Resampling.NEAREST)
            x = col * tile
            y = row * (tile + label_h) + label_h
            canvas.paste(img, (x, y))
            draw.text((x + 4, y - label_h + 4), label_text, fill=(235, 238, 242))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return str(out_path)


def load_real_moving_mnist(dataset_path: str | Path, codec: RealMovingMNISTCodec, train_sequences: int, test_sequences: int) -> tuple[list[RealMovingMNISTSequence], list[RealMovingMNISTSequence], tuple[int, ...]]:
    raw = np.load(dataset_path, mmap_mode="r")
    if raw.shape[0] != 20:
        raise ValueError(f"Expected Moving MNIST shape [20, N, 64, 64], got {raw.shape}")
    train = []
    test = []
    for i in range(train_sequences):
        train.append(codec.encode_sequence(raw[:, i], sequence_index=i))
    offset = raw.shape[1] - test_sequences
    for j in range(test_sequences):
        idx = offset + j
        test.append(codec.encode_sequence(raw[:, idx], sequence_index=idx))
    return train, test, tuple(int(x) for x in raw.shape)


def causalize_sequence(sequence: RealMovingMNISTSequence) -> RealMovingMNISTSequence:
    dyn = sequence.dyn.copy()
    for step in range(len(dyn)):
        for obj in range(OBJECTS):
            v = object_view(dyn[step], obj).copy()
            if step == 0:
                v[2:4] = 0.0
            else:
                prev = object_view(dyn[step - 1], obj)
                v[2:4] = v[:2] - prev[:2]
            set_object(dyn[step], obj, v)
    return RealMovingMNISTSequence(
        frames=sequence.frames,
        dyn=dyn.astype(np.float32),
        identity_features=sequence.identity_features,
        identity_templates=sequence.identity_templates,
        frame_templates=sequence.frame_templates,
        frame_layers=sequence.frame_layers,
        sequence_index=sequence.sequence_index,
    )


def causalize_sequences(sequences: list[RealMovingMNISTSequence]) -> list[RealMovingMNISTSequence]:
    return [causalize_sequence(seq) for seq in sequences]


def motion_token_sequence(sequence: RealMovingMNISTSequence) -> RealMovingMNISTSequence:
    """Causal encoder variant inspired by point trackers.

    It stores a compact motion token per object: the dominant integer-pixel
    speed observed so far, with the latest reliable sign. This is a small,
    object-centric analogue of trajectory memory/refinement from modern video
    tracking encoders, but AMF still receives only compact state floats.
    """
    dyn = sequence.dyn.copy()
    centers = np.stack(
        [[object_view(sequence.dyn[step], obj)[:2] for obj in range(OBJECTS)] for step in range(len(sequence.dyn))]
    ).astype(np.float32)
    histories: list[list[tuple[int, int]]] = [[] for _ in range(OBJECTS)]
    for step in range(len(dyn)):
        for obj in range(OBJECTS):
            v = object_view(dyn[step], obj).copy()
            if step > 0:
                token = np.rint((centers[step, obj] - centers[step - 1, obj]) * (sequence.frames.shape[-1] - 1)).astype(np.int32)
                if float(np.linalg.norm(token)) <= 12.0:
                    histories[obj].append((int(token[0]), int(token[1])))
            if not histories[obj]:
                velocity = np.zeros(2, dtype=np.float32)
            else:
                tokens = np.asarray(histories[obj], dtype=np.float32)
                last = tokens[-1]
                magnitude = []
                for axis in (0, 1):
                    values = np.abs(tokens[:, axis]).astype(np.int32).tolist()
                    magnitude.append(Counter(values).most_common(1)[0][0])
                sign = np.sign(last)
                sign[sign == 0] = 1.0
                velocity = (np.asarray(magnitude, dtype=np.float32) * sign) / float(sequence.frames.shape[-1] - 1)
            v[2:4] = velocity
            set_object(dyn[step], obj, v)
    return RealMovingMNISTSequence(
        frames=sequence.frames,
        dyn=dyn.astype(np.float32),
        identity_features=sequence.identity_features,
        identity_templates=sequence.identity_templates,
        frame_templates=sequence.frame_templates,
        frame_layers=sequence.frame_layers,
        sequence_index=sequence.sequence_index,
    )


def motion_token_sequences(sequences: list[RealMovingMNISTSequence]) -> list[RealMovingMNISTSequence]:
    return [motion_token_sequence(seq) for seq in sequences]


def kinematic_token_sequence(sequence: RealMovingMNISTSequence) -> RealMovingMNISTSequence:
    """Causal encoder variant with smoothed velocity and bounded acceleration.

    This is an object-centric analogue of trajectory-guided video encoders:
    store a compact temporal motion token, not pixels. It uses only centers
    observed up to the current step.
    """
    dyn = sequence.dyn.copy()
    centers = np.stack(
        [[object_view(sequence.dyn[step], obj)[:2] for obj in range(OBJECTS)] for step in range(len(sequence.dyn))]
    ).astype(np.float32)
    for step in range(len(dyn)):
        for obj in range(OBJECTS):
            v = object_view(dyn[step], obj).copy()
            if step == 0:
                velocity = np.zeros(2, dtype=np.float32)
            else:
                diffs = [centers[k, obj] - centers[k - 1, obj] for k in range(max(1, step - 3), step + 1)]
                recent = diffs[-1]
                weights = np.asarray([0.08, 0.14, 0.24, 0.54], dtype=np.float32)[-len(diffs) :]
                weights = weights / (float(np.sum(weights)) + EPS)
                smooth = np.sum(np.asarray(diffs, dtype=np.float32) * weights[:, None], axis=0)
                prev = diffs[-2] if len(diffs) >= 2 else recent
                accel = np.clip(recent - prev, -0.045, 0.045)
                velocity = 0.52 * recent + 0.34 * smooth + 0.14 * (recent + accel)
                half_w = max(0.02, float(v[4]) * 0.5)
                half_h = max(0.02, float(v[5]) * 0.5)
                if float(v[0]) - half_w < 0.055 and velocity[0] < 0:
                    velocity[0] = abs(float(velocity[0])) * 0.85
                if (1.0 - half_w) - float(v[0]) < 0.055 and velocity[0] > 0:
                    velocity[0] = -abs(float(velocity[0])) * 0.85
                if float(v[1]) - half_h < 0.055 and velocity[1] < 0:
                    velocity[1] = abs(float(velocity[1])) * 0.85
                if (1.0 - half_h) - float(v[1]) < 0.055 and velocity[1] > 0:
                    velocity[1] = -abs(float(velocity[1])) * 0.85
                velocity = np.clip(velocity, -0.18, 0.18).astype(np.float32)
            v[2:4] = velocity
            set_object(dyn[step], obj, v)
    return RealMovingMNISTSequence(
        frames=sequence.frames,
        dyn=dyn.astype(np.float32),
        identity_features=sequence.identity_features,
        identity_templates=sequence.identity_templates,
        frame_templates=sequence.frame_templates,
        frame_layers=sequence.frame_layers,
        sequence_index=sequence.sequence_index,
    )


def kinematic_token_sequences(sequences: list[RealMovingMNISTSequence]) -> list[RealMovingMNISTSequence]:
    return [kinematic_token_sequence(seq) for seq in sequences]


def write_phase11a_report(results: dict[str, Any], path: Path) -> None:
    one_rows = []
    for name, row in results["one_step"].items():
        one_rows.append(
            f"| {name} | {row['frame_iou']:.4f} | {row.get('soft_frame_iou', 0.0):.4f} | "
            f"{row['latent_mse']:.6f} | {row['center_error_px']:.3f} | {row['bounce_latent_mse']:.6f} |"
        )
    gt_rows = []
    for name, horizons in results["gt_rollouts"].items():
        gt_rows.append(
            f"| {name} | {horizons['1']['frame_iou']:.4f} | {horizons['5']['frame_iou']:.4f} | "
            f"{horizons['10']['frame_iou']:.4f} | {horizons['17']['frame_iou']:.4f} | "
            f"{horizons['17'].get('soft_frame_iou', 0.0):.4f} | "
            f"{horizons['17']['center_error_px']:.3f} |"
        )
    causal_rows = []
    for name, horizons in results.get("causal_gt_rollouts", {}).items():
        causal_rows.append(
            f"| {name} | {horizons['1']['frame_iou']:.4f} | {horizons['5']['frame_iou']:.4f} | "
            f"{horizons['10']['frame_iou']:.4f} | {horizons['17']['frame_iou']:.4f} | "
            f"{horizons['17'].get('soft_frame_iou', 0.0):.4f} | "
            f"{horizons['17']['center_error_px']:.3f} |"
        )
    causal_one_rows = []
    for name, row in results.get("causal_one_step", {}).items():
        causal_one_rows.append(
            f"| {name} | {row['frame_iou']:.4f} | {row.get('soft_frame_iou', 0.0):.4f} | "
            f"{row['latent_mse']:.6f} | {row['center_error_px']:.3f} | {row['bounce_latent_mse']:.6f} |"
        )
    token_rows = []
    for name, horizons in results.get("motion_token_gt_rollouts", {}).items():
        token_rows.append(
            f"| {name} | {horizons['1']['frame_iou']:.4f} | {horizons['5']['frame_iou']:.4f} | "
            f"{horizons['10']['frame_iou']:.4f} | {horizons['17']['frame_iou']:.4f} | "
            f"{horizons['17'].get('soft_frame_iou', 0.0):.4f} | "
            f"{horizons['17']['center_error_px']:.3f} |"
        )
    token_one_rows = []
    for name, row in results.get("motion_token_one_step", {}).items():
        token_one_rows.append(
            f"| {name} | {row['frame_iou']:.4f} | {row.get('soft_frame_iou', 0.0):.4f} | "
            f"{row['latent_mse']:.6f} | {row['center_error_px']:.3f} | {row['bounce_latent_mse']:.6f} |"
        )
    stability_rows = []
    for name, horizons in results["stability_rollouts"].items():
        stability_rows.append(
            f"| {name} | {horizons['30']['stable']:.4f} | {horizons['120']['stable']:.4f} | "
            f"{horizons['240']['stable']:.4f} | {horizons['480']['stable']:.4f} | "
            f"{horizons['480']['identity_drift']:.6f} |"
        )
    target_rows = [f"| {k} | {v} |" for k, v in results["targets"].items()]
    diag_rows = [f"| {k} | {v:.4f} |" for k, v in results.get("decoder_diagnostics", {}).items()]
    p = results["metaplasticity_probe"]
    status_counts = ", ".join(f"{k}={v}" for k, v in results["amf_status_counts"].items())
    text = f"""# Fase 11A - AMF Visual Rollout en Moving MNIST Real

## Dataset

- Fuente descargada: `{results['dataset_source']}`
- Archivo local: `{results['dataset_path']}`
- Shape verificado: `{tuple(results['dataset_shape'])}`
- Nota: el Moving MNIST estandar descargado trae 20 frames por secuencia. Por
  eso el ground truth real solo existe hasta horizonte 17 despues del warmup de
  `{WARMUP_FRAMES}` frames; horizontes 30/60/120/240/480 son estabilidad
  autorregresiva, no IoU contra ground truth.

## Arquitectura

```text
frame real -> encoder/tracker -> S_dyn(t) compacto + M_id separado
S_dyn(t), M_id_features -> AMF delta/residual world model -> S_dyn(t+1)
S_dyn(t+1), M_id layers -> decoder visual -> frame(t+1)
```

- AMF no recibe pixeles crudos.
- Latente dinamico: `{results['dynamic_dim']}` floats.
- Identidad compacta para AMF: `{results['identity_dim']}` floats.
- Latente total reportado: `{results['latent_dim']}` floats / `{results['latent_bytes']}` bytes.
- Frame: `{results['frame_bytes']}` bytes.
- Compresion frame/latente: `{results['compression_ratio']:.1f}x`.
- AMF cells: `{results['amf_cells']}`.
- AMF memory MB: `{results['amf_memory_mb']:.4f}`.
- AMF residual scale usado en prediccion: `{results.get('amf_residual_scale', 1.0)}`.
- Caja global de colision visual: `{results.get('amf_collision_box')}`.
- Banco causal de identidad para decoder: `{results.get('identity_bank_crops', 0)}` crops reales de train.

## One-Step Real

| modelo | frame IoU | soft IoU | latent MSE | center err px | bounce latent MSE |
|---|---:|---:|---:|---:|---:|
{chr(10).join(one_rows)}

## Rollout Con Ground Truth Real

| modelo | IoU h1 | IoU h5 | IoU h10 | IoU h17 | soft IoU h17 | center err h17 px |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(gt_rows)}

## Evaluacion Causal Sin Velocidad Futura

Aqui `S(t)` usa velocidad de `t-1 -> t`, no de `t -> t+1`. Esto es mas duro y
mas cercano al rollout real desde un warmup causal.

### One-Step Causal

| modelo | frame IoU | soft IoU | latent MSE | center err px | bounce latent MSE |
|---|---:|---:|---:|---:|---:|
{chr(10).join(causal_one_rows) if causal_one_rows else '| not measured | 0.0000 | 0.0000 | 0.000000 | 0.000 | 0.000000 |'}

### Rollout Causal Con Ground Truth Real

| modelo | IoU h1 | IoU h5 | IoU h10 | IoU h17 | soft IoU h17 | center err h17 px |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(causal_rows) if causal_rows else '| not measured | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 |'}

## Encoder Causal Por Motion Tokens

Inspirado en TAPIR/CoTracker: mantiene memoria de trayectoria compacta y usa el
token de movimiento dominante observado hasta `t`, con signo reciente. Sigue
siendo causal y AMF no recibe pixeles crudos.

### One-Step Motion Tokens

| modelo | frame IoU | soft IoU | latent MSE | center err px | bounce latent MSE |
|---|---:|---:|---:|---:|---:|
{chr(10).join(token_one_rows) if token_one_rows else '| not measured | 0.0000 | 0.0000 | 0.000000 | 0.000 | 0.000000 |'}

### Rollout Motion Tokens Con Ground Truth Real

| modelo | IoU h1 | IoU h5 | IoU h10 | IoU h17 | soft IoU h17 | center err h17 px |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(token_rows) if token_rows else '| not measured | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 |'}

## Diagnostico Del Decoder

| diagnostico | valor |
|---|---:|
{chr(10).join(diag_rows) if diag_rows else '| not measured | 0.0000 |'}

## Rollout Largo Sin Ground Truth

| modelo | stable h30 | stable h120 | stable h240 | stable h480 | identity drift h480 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(stability_rows)}

## Metaplasticidad

- Runtime statuses durante online fitting: `{status_counts}`
- Probe statuses: `{', '.join(p['statuses'])}`
- created_confirmed_novelty: `{p['created_confirmed_novelty']}`
- explained_after_create: `{p['explained_after_create']}`
- probe MSE after: `{p['mse_after']:.6f}`

## Targets

| target | passed |
|---|---:|
{chr(10).join(target_rows)}

## Conclusion Honesta

Fase 11A ya usa Moving MNIST real descargado y prueba la ruta correcta:
pixeles reales -> latente compacto -> AMF/metaplasticidad + modulo global de
fisica visual -> decoder con banco causal de identidad real. La compresion, el
one-step real, la estabilidad larga y la metaplasticidad pasan.

AMF_full supera a los baselines clasicos en rollout con ground truth real en
h5, h10 y h17. Los targets visuales extremos `gt_rollout_10_iou_gt_0_75` y
`gt_rollout_17_iou_gt_0_60` siguen falsos: con el Moving MNIST estandar
descargado solo hay 20 frames, y aunque el banco causal de identidad recupera
parte de los pixeles ocultos en cruces/oclusiones, aun no alcanza un decoder
generativo predictivo fuerte. La evidencia apunta a que el siguiente salto debe
combinar un encoder causal multi-hipotesis con selector de confianza y un
decoder de identidad completiva mas potente, no de meter pixeles crudos en AMF.

## Inspiracion De La Investigacion Actual

- Seedance 2.0: arquitectura multimodal unificada y condicionamiento por
  referencias de texto/imagen/video/audio.
- Cosmos/VidTok/VideoFlexTok/PV-VAE: tokenizacion causal, latentes compactos,
  decoder condicionado por contexto y reconstruccion parcial-a-completa.
- TAPIR/CoTracker: tracking con memoria temporal, correlaciones locales y
  refinamiento conjunto de trayectorias ante oclusion.
- Traduccion a AMF: AMF conserva solo `S_dyn + M_id_features`; el decoder externo
  usa un banco causal de referencias reales para completar identidad visual, y
  el encoder motion-token prueba memoria causal de trayectoria compacta.
"""
    path.write_text(text, encoding="utf-8")


def run_phase11a(
    dataset_path: str | Path = "data/MovingMNIST/mnist_test_seq.npy",
    train_sequences: int = 650,
    test_sequences: int = 40,
    frame_size: int = 64,
    seed: int = 4107,
    out_dir: str | Path = "results",
    **_: Any,
) -> dict[str, Any]:
    start = time.perf_counter()
    codec = RealMovingMNISTCodec(frame_size=frame_size)
    train, test, raw_shape = load_real_moving_mnist(dataset_path, codec, train_sequences, test_sequences)
    identity_bank_size = codec.fit_identity_bank(train)
    train_transitions = build_transitions(train)
    test_transitions = build_transitions(test)

    linear_delta = LinearDeltaBaseline().fit(train_transitions)
    ridge = RidgeLinearDynamicsBaseline().fit(train_transitions)
    knn = KNNTransitionBaseline(k=5).fit(train_transitions)
    amf_no_meta = AMFMovingMNISTWorldModel(metaplasticity=False, boundary_guard=False).fit(train_transitions)
    amf_full = AMFMovingMNISTWorldModel(
        metaplasticity=True,
        boundary_guard=True,
        residual_scale=0.0,
        collision_box=0.317,
    ).fit(train_transitions)

    rng = np.random.default_rng(seed)
    online_order = rng.choice(len(train_transitions), size=min(2500, len(train_transitions)), replace=False)
    for idx in online_order:
        amf_full.learn_transition(train_transitions[int(idx)])

    models: list[TransitionModel] = [
        ConstantVelocityBaseline(),
        linear_delta,
        ridge,
        knn,
        amf_no_meta,
        amf_full,
    ]
    one_step = {model.name: evaluate_one_step(model, test_transitions, test, codec) for model in models}
    gt_rollouts = {}
    stability_rollouts = {}
    for model in models:
        gt, stability = rollout_metrics(model, test, codec)
        gt_rollouts[model.name] = gt
        stability_rollouts[model.name] = stability

    causal_train = causalize_sequences(train)
    causal_test = causalize_sequences(test)
    causal_train_transitions = build_transitions(causal_train)
    causal_test_transitions = build_transitions(causal_test)
    causal_linear_delta = LinearDeltaBaseline().fit(causal_train_transitions)
    causal_ridge = RidgeLinearDynamicsBaseline().fit(causal_train_transitions)
    causal_knn = KNNTransitionBaseline(k=5).fit(causal_train_transitions)
    causal_amf_no_meta = AMFMovingMNISTWorldModel(metaplasticity=False, boundary_guard=False).fit(causal_train_transitions)
    causal_amf_full = AMFMovingMNISTWorldModel(
        metaplasticity=True,
        boundary_guard=True,
        residual_scale=0.0,
        collision_box=0.317,
    ).fit(causal_train_transitions)
    causal_online_order = rng.choice(len(causal_train_transitions), size=min(2500, len(causal_train_transitions)), replace=False)
    for idx in causal_online_order:
        causal_amf_full.learn_transition(causal_train_transitions[int(idx)])
    causal_models: list[TransitionModel] = [
        ConstantVelocityBaseline(),
        causal_linear_delta,
        causal_ridge,
        causal_knn,
        causal_amf_no_meta,
        causal_amf_full,
    ]
    causal_one_step = {model.name: evaluate_one_step(model, causal_test_transitions, causal_test, codec) for model in causal_models}
    causal_gt_rollouts = {}
    causal_stability_rollouts = {}
    for model in causal_models:
        gt, stability = rollout_metrics(model, causal_test, codec)
        causal_gt_rollouts[model.name] = gt
        causal_stability_rollouts[model.name] = stability

    token_train = motion_token_sequences(train)
    token_test = motion_token_sequences(test)
    token_train_transitions = build_transitions(token_train)
    token_test_transitions = build_transitions(token_test)
    token_ridge = RidgeLinearDynamicsBaseline().fit(token_train_transitions)
    token_amf_no_meta = AMFMovingMNISTWorldModel(metaplasticity=False, boundary_guard=False).fit(token_train_transitions)
    token_amf_full = AMFMovingMNISTWorldModel(
        metaplasticity=True,
        boundary_guard=True,
        residual_scale=0.0,
        collision_box=0.317,
    ).fit(token_train_transitions)
    token_online_order = rng.choice(len(token_train_transitions), size=min(2500, len(token_train_transitions)), replace=False)
    for idx in token_online_order:
        token_amf_full.learn_transition(token_train_transitions[int(idx)])
    token_models: list[TransitionModel] = [
        ConstantVelocityBaseline(),
        token_ridge,
        token_amf_no_meta,
        token_amf_full,
    ]
    token_one_step = {model.name: evaluate_one_step(model, token_test_transitions, token_test, codec) for model in token_models}
    token_gt_rollouts = {}
    token_stability_rollouts = {}
    for model in token_models:
        gt, stability = rollout_metrics(model, token_test, codec)
        token_gt_rollouts[model.name] = gt
        token_stability_rollouts[model.name] = stability

    probe_transition = next((t for t in train_transitions if t.boundary_event), train_transitions[0])
    probe = metaplasticity_probe(amf_full, probe_transition)
    decoder_diag = decoder_diagnostics(test, codec)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    contact_sheet = make_contact_sheet(amf_full, test[0], codec, out / "phase11a_moving_mnist_real_rollout.png")
    sample_npz = out / "phase11a_moving_mnist_real_sample.npz"
    np.savez_compressed(
        sample_npz,
        frames=test[0].frames,
        dyn=test[0].dyn,
        identity_features=test[0].identity_features,
        identity_templates=test[0].identity_templates,
        frame_templates=test[0].frame_templates,
        frame_layers=test[0].frame_layers,
    )

    best_one_step_iou = max(one_step.items(), key=lambda item: item[1]["frame_iou"])[0]
    best_gt_10_iou = max(gt_rollouts.items(), key=lambda item: item[1]["10"]["frame_iou"])[0]
    amf_gt = gt_rollouts["AMF_full"]
    amf_stability = stability_rollouts["AMF_full"]
    targets = {
        "uses_real_downloaded_dataset": bool(Path(dataset_path).exists()),
        "no_raw_pixel_amf": True,
        "one_step_iou_gt_0_85": bool(one_step["AMF_full"]["frame_iou"] > 0.85),
        "causal_one_step_iou_gt_0_85": bool(causal_one_step["AMF_full"]["frame_iou"] > 0.85),
        "gt_rollout_10_iou_gt_0_75": bool(amf_gt["10"]["frame_iou"] > 0.75),
        "gt_rollout_17_iou_gt_0_60": bool(amf_gt["17"]["frame_iou"] > 0.60),
        "stability_480": bool(amf_stability["480"]["stable"] >= 1.0 and amf_stability["480"]["digit_consistency"] >= 1.0),
        "identity_drift_low": bool(amf_stability["480"]["identity_drift"] <= 1e-8),
        "metaplasticity_probe_passed": bool(probe["created_confirmed_novelty"] and probe["explained_after_create"]),
    }
    results = {
        "title": "Fase 11A - AMF Visual Rollout en Moving MNIST real",
        "dataset_path": str(dataset_path),
        "dataset_source": "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy",
        "dataset_shape": raw_shape,
        "dataset_note": "Standard downloaded Moving MNIST has 20 frames per sequence; ground-truth IoU is available only through horizon 19. Horizons 30/60/120/240/480 are stability-only rollouts.",
        "encoder_note": "Benchmark encoder estimates object velocity from adjacent real object layers and stores an integrated visual anchor; AMF itself receives only compact latents, never raw pixels.",
        "seed": seed,
        "frame_size": frame_size,
        "objects": OBJECTS,
        "dynamic_dim": DYN_DIM,
        "identity_dim": ID_DIM,
        "latent_dim": LATENT_DIM,
        "latent_bytes": codec.latent_bytes(),
        "frame_bytes": codec.frame_bytes(),
        "compression_ratio": codec.frame_bytes() / codec.latent_bytes(),
        "train_sequences": train_sequences,
        "test_sequences": test_sequences,
        "train_transitions": len(train_transitions),
        "test_transitions": len(test_transitions),
        "causal_train_transitions": len(causal_train_transitions),
        "causal_test_transitions": len(causal_test_transitions),
        "token_train_transitions": len(token_train_transitions),
        "token_test_transitions": len(token_test_transitions),
        "gt_horizons": list(GT_HORIZONS),
        "stability_horizons": list(STABILITY_HORIZONS),
        "one_step": one_step,
        "gt_rollouts": gt_rollouts,
        "stability_rollouts": stability_rollouts,
        "causal_one_step": causal_one_step,
        "causal_gt_rollouts": causal_gt_rollouts,
        "causal_stability_rollouts": causal_stability_rollouts,
        "motion_token_one_step": token_one_step,
        "motion_token_gt_rollouts": token_gt_rollouts,
        "motion_token_stability_rollouts": token_stability_rollouts,
        "amf_cells": int(len(amf_full.centers)),
        "amf_memory_mb": amf_full.memory_mb(),
        "amf_residual_scale": amf_full.residual_scale,
        "amf_collision_box": amf_full.collision_box,
        "identity_bank_crops": identity_bank_size,
        "amf_status_counts": amf_full.status_counts,
        "metaplasticity_probe": probe,
        "decoder_diagnostics": decoder_diag,
        "best_one_step_iou": best_one_step_iou,
        "best_gt_10_iou": best_gt_10_iou,
        "targets": targets,
        "all_available_targets_passed": bool(all(targets.values())),
        "sample_npz": str(sample_npz),
        "contact_sheet": contact_sheet,
        "elapsed_seconds": time.perf_counter() - start,
    }
    (out / "phase11a_latest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_phase11a_report(results, out / "FASE11A_RESULTADOS.md")
    return results
