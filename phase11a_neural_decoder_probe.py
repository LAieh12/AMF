from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from phase11a_moving_mnist import (
    AMFMovingMNISTWorldModel,
    GT_HORIZONS,
    WARMUP_FRAMES,
    MovingTransition,
    RealMovingMNISTCodec,
    build_transitions,
    load_real_moving_mnist,
    mask_iou,
    transition_event,
)


class TinyCompletionDecoder(nn.Module):
    def __init__(self, channels: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 24, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rough = x[:, :1]
        delta = 0.45 * torch.tanh(self.net(x))
        return torch.clamp(rough + delta, 0.0, 1.0)


def completion_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weight = 1.0 + 8.0 * (target > 0.05).float()
    bce = F.binary_cross_entropy(pred, target, weight=weight)
    l1 = F.l1_loss(pred, target)
    intersection = torch.sum(torch.minimum(pred, target), dim=(1, 2, 3))
    union = torch.sum(torch.maximum(pred, target), dim=(1, 2, 3)) + 1e-6
    soft_iou_loss = 1.0 - torch.mean(intersection / union)
    return bce + 0.15 * l1 + 0.35 * soft_iou_loss


def sample_input(codec: RealMovingMNISTCodec, seq, dyn: np.ndarray, horizon: int) -> np.ndarray:
    start = WARMUP_FRAMES
    rough = codec.render_from_layers(dyn, seq.dyn[start], seq.frame_layers[start])
    horizon_map = np.full_like(rough, float(horizon) / max(GT_HORIZONS), dtype=np.float32)
    return np.stack(
        [
            rough,
            seq.frames[start],
            seq.frame_layers[start, 0],
            seq.frame_layers[start, 1],
            horizon_map,
        ],
        axis=0,
    ).astype(np.float32)


def build_tensor_dataset(codec: RealMovingMNISTCodec, sequences, max_horizon: int) -> TensorDataset:
    xs = []
    ys = []
    for seq in sequences:
        start = WARMUP_FRAMES
        for horizon in range(1, max_horizon + 1):
            xs.append(sample_input(codec, seq, seq.dyn[start + horizon], horizon))
            ys.append(seq.frames[start + horizon][None, :, :].astype(np.float32))
    return TensorDataset(torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)))


def evaluate_actual_dyn(model: nn.Module, codec: RealMovingMNISTCodec, sequences) -> dict[str, dict[str, float]]:
    model.eval()
    out = {str(h): {"rough": [], "refined": [], "merged": []} for h in GT_HORIZONS}
    with torch.no_grad():
        for seq in sequences:
            start = WARMUP_FRAMES
            for horizon in GT_HORIZONS:
                dyn = seq.dyn[start + horizon]
                rough = codec.render_from_layers(dyn, seq.dyn[start], seq.frame_layers[start])
                x = torch.from_numpy(sample_input(codec, seq, dyn, horizon)[None, :, :, :])
                pred = model(x).squeeze().cpu().numpy()
                merged = np.maximum(rough, pred)
                actual = seq.frames[start + horizon]
                out[str(horizon)]["rough"].append(mask_iou(rough, actual))
                out[str(horizon)]["refined"].append(mask_iou(pred, actual))
                out[str(horizon)]["merged"].append(mask_iou(merged, actual))
    return {h: {k: float(np.mean(v)) for k, v in rows.items()} for h, rows in out.items()}


def evaluate_amf_dyn(model: nn.Module, codec: RealMovingMNISTCodec, amf: AMFMovingMNISTWorldModel, sequences) -> dict[str, dict[str, float]]:
    model.eval()
    out = {str(h): {"rough": [], "refined": [], "merged": []} for h in GT_HORIZONS}
    with torch.no_grad():
        for seq in sequences:
            local = amf.clone()
            for ctx in range(WARMUP_FRAMES):
                local.learn_transition(
                    MovingTransition(
                        state=seq.dyn[ctx],
                        identity_features=seq.identity_features,
                        next_state=seq.dyn[ctx + 1],
                        sequence_id=seq.sequence_index,
                        step=ctx,
                        boundary_event=transition_event(seq.dyn[ctx]),
                    )
                )
            start = WARMUP_FRAMES
            dyn = seq.dyn[start].copy()
            for horizon in range(1, max(GT_HORIZONS) + 1):
                dyn = local.predict_next(dyn, seq.identity_features)
                if horizon in GT_HORIZONS:
                    rough = codec.render_from_layers(dyn, seq.dyn[start], seq.frame_layers[start])
                    x = torch.from_numpy(sample_input(codec, seq, dyn, horizon)[None, :, :, :])
                    pred = model(x).squeeze().cpu().numpy()
                    merged = np.maximum(rough, pred)
                    actual = seq.frames[start + horizon]
                    out[str(horizon)]["rough"].append(mask_iou(rough, actual))
                    out[str(horizon)]["refined"].append(mask_iou(pred, actual))
                    out[str(horizon)]["merged"].append(mask_iou(merged, actual))
    return {h: {k: float(np.mean(v)) for k, v in rows.items()} for h, rows in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe a real-data neural completion decoder for Fase 11A.")
    parser.add_argument("--train-sequences", type=int, default=220)
    parser.add_argument("--test-sequences", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", default="results/phase11a_neural_decoder_probe.json")
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    start_time = time.perf_counter()
    codec = RealMovingMNISTCodec()
    train, test, shape = load_real_moving_mnist(
        "data/MovingMNIST/mnist_test_seq.npy",
        codec,
        args.train_sequences,
        args.test_sequences,
    )
    dataset = build_tensor_dataset(codec, train, max(GT_HORIZONS))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = TinyCompletionDecoder()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    losses = []
    for _ in range(args.epochs):
        model.train()
        epoch_losses = []
        for x, y in loader:
            pred = model(x)
            loss = completion_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_losses)))

    train_transitions = build_transitions(train)
    amf = AMFMovingMNISTWorldModel(
        metaplasticity=True,
        boundary_guard=True,
        residual_scale=0.0,
        collision_box=0.317,
    ).fit(train_transitions)
    rng = np.random.default_rng(4107)
    for idx in rng.choice(len(train_transitions), size=min(2500, len(train_transitions)), replace=False):
        amf.learn_transition(train_transitions[int(idx)])

    results = {
        "dataset_shape": shape,
        "train_sequences": args.train_sequences,
        "test_sequences": args.test_sequences,
        "epochs": args.epochs,
        "losses": losses,
        "actual_dyn_decoder": evaluate_actual_dyn(model, codec, test),
        "amf_dyn_decoder": evaluate_amf_dyn(model, codec, amf, test),
        "elapsed_seconds": time.perf_counter() - start_time,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
