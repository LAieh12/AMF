from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase11a_moving_mnist import run_phase11a


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fase 11A - AMF Visual Rollout en Moving MNIST.")
    parser.add_argument("--train-sequences", type=int, default=220)
    parser.add_argument("--test-sequences", type=int, default=18)
    parser.add_argument("--dataset-path", default="data/MovingMNIST/mnist_test_seq.npy")
    parser.add_argument("--frame-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=4107)
    parser.add_argument("--out-dir", default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_phase11a(
        dataset_path=args.dataset_path,
        train_sequences=args.train_sequences,
        test_sequences=args.test_sequences,
        frame_size=args.frame_size,
        seed=args.seed,
        out_dir=Path(args.out_dir),
    )
    amf = results["one_step"]["AMF_full"]
    r10 = results["gt_rollouts"]["AMF_full"]["10"]
    r17 = results["gt_rollouts"]["AMF_full"]["17"]
    s480 = results["stability_rollouts"]["AMF_full"]["480"]
    print("report:", str(Path(args.out_dir) / "phase11a_latest.json"))
    print("contact_sheet:", results["contact_sheet"])
    print("all_available_targets_passed:", results["all_available_targets_passed"])
    print(
        "AMF one_step_iou={:.4f} gt_rollout10_iou={:.4f} gt_rollout17_iou={:.4f} "
        "stability480={:.4f} compression={:.1f}x cells={} memory_mb={:.4f}".format(
            amf["frame_iou"],
            r10["frame_iou"],
            r17["frame_iou"],
            s480["stable"],
            results["compression_ratio"],
            results["amf_cells"],
            results["amf_memory_mb"],
        )
    )
    print("targets:", json.dumps(results["targets"], sort_keys=True))


if __name__ == "__main__":
    main()
