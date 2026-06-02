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

from phase11a_moving_mnist import GT_HORIZONS, WARMUP_FRAMES, RealMovingMNISTCodec, load_real_moving_mnist, mask_iou


def build_input(frames: np.ndarray, start: int, horizon: int) -> np.ndarray:
    f0 = frames[start - 2]
    f1 = frames[start - 1]
    f2 = frames[start]
    d01 = f1 - f0
    d12 = f2 - f1
    hmap = np.full_like(f2, float(horizon) / max(GT_HORIZONS), dtype=np.float32)
    return np.stack([f0, f1, f2, d01, d12, hmap], axis=0).astype(np.float32)


def build_dataset(sequences, horizons=GT_HORIZONS, max_examples: int | None = None, seed: int = 4107) -> TensorDataset:
    xs = []
    ys = []
    for seq in sequences:
        frames = seq.frames.astype(np.float32)
        for horizon in horizons:
            for start in range(WARMUP_FRAMES, len(frames) - horizon):
                xs.append(build_input(frames, start, horizon))
                ys.append(frames[start + horizon][None, :, :].astype(np.float32))
    if max_examples is not None and len(xs) > int(max_examples):
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(xs), size=int(max_examples), replace=False)
        xs = [xs[int(i)] for i in keep]
        ys = [ys[int(i)] for i in keep]
    return TensorDataset(torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)))


class GatedConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.proj = nn.Conv2d(channels, channels * 2, 3, padding=dilation, dilation=dilation)
        self.mix = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.proj(self.norm(x)).chunk(2, dim=1)
        return x + self.mix(torch.tanh(a) * torch.sigmoid(b))


class MiniSimVP(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.enc1 = nn.Conv2d(6, channels, 3, padding=1)
        self.enc2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.temporal = nn.Sequential(
            GatedConvBlock(channels, dilation=1),
            GatedConvBlock(channels, dilation=2),
            GatedConvBlock(channels, dilation=4),
            GatedConvBlock(channels, dilation=1),
        )
        self.dec1 = nn.Conv2d(channels + 1, channels // 2, 3, padding=1)
        self.dec2 = nn.Conv2d(channels // 2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, 2:3]
        z = F.relu(self.enc1(x))
        skip = F.relu(self.enc2(z))
        z = self.temporal(skip)
        z = torch.cat([z, last], dim=1)
        logits = self.dec2(F.relu(self.dec1(z)))
        return torch.sigmoid(logits + 0.8 * (last - 0.5))


def prediction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy(pred.clamp(1e-4, 1.0 - 1e-4), target)
    mse = F.mse_loss(pred, target)
    inter = torch.sum(torch.minimum(pred, target), dim=(1, 2, 3))
    union = torch.sum(torch.maximum(pred, target), dim=(1, 2, 3)) + 1e-6
    soft_iou_loss = 1.0 - torch.mean(inter / union)
    return 0.45 * bce + 0.35 * mse + 0.20 * soft_iou_loss


def evaluate(model: nn.Module, sequences, device: torch.device) -> dict[str, dict[str, float]]:
    model.eval()
    rows = {str(h): {"simvp": [], "last_frame": []} for h in GT_HORIZONS}
    with torch.no_grad():
        for seq in sequences:
            frames = seq.frames.astype(np.float32)
            start = WARMUP_FRAMES
            for horizon in GT_HORIZONS:
                x = torch.from_numpy(build_input(frames, start, horizon)[None]).to(device)
                pred = model(x).cpu().numpy()[0, 0]
                actual = frames[start + horizon]
                rows[str(horizon)]["simvp"].append(mask_iou(pred, actual))
                rows[str(horizon)]["last_frame"].append(mask_iou(frames[start], actual))
    return {key: {metric: float(np.mean(vals)) for metric, vals in metrics.items()} for key, metrics in rows.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 11A mini-SimVP temporal predictor probe.")
    parser.add_argument("--train-sequences", type=int, default=220)
    parser.add_argument("--test-sequences", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--max-examples", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=4107)
    parser.add_argument("--out", default="results/phase11a_simvp_probe.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    start_time = time.perf_counter()
    codec = RealMovingMNISTCodec()
    train, test, raw_shape = load_real_moving_mnist(
        "data/MovingMNIST/mnist_test_seq.npy",
        codec,
        args.train_sequences,
        args.test_sequences,
    )
    dataset = build_dataset(train, max_examples=args.max_examples, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniSimVP(channels=args.channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    losses = []
    for _ in range(args.epochs):
        model.train()
        epoch_losses = []
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = prediction_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    metrics = evaluate(model, test, device)
    results = {
        "dataset_shape": raw_shape,
        "train_sequences": args.train_sequences,
        "test_sequences": args.test_sequences,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "channels": args.channels,
        "max_examples": args.max_examples,
        "device": str(device),
        "train_examples": int(len(dataset)),
        "losses": losses,
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - start_time,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
