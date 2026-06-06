from __future__ import annotations

import argparse
from pathlib import Path

from phase13_amf_ltm_model import HORIZONS_13
from phase13_scene_loader import DEFAULT_CACHE_ROOT, select_scenes
from phase13b_regime_expert_selector import run_phase13b, write_phase13b_outputs


def parse_horizons(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip().lstrip("h")) for part in raw.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 13B Regime Expert Selector.")
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--include-tier2", action="store_true")
    parser.add_argument("--include-tier3", action="store_true")
    parser.add_argument("--previous-matrix-json", default="results/phase13_latest.json")
    parser.add_argument("--horizons", default="1,5,15,30,60,120")
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--memory-window", type=int, default=20)
    parser.add_argument("--max-cells", type=int, default=5000)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--model-radius", type=float, default=0.75)
    parser.add_argument("--model-top-k", type=int, default=24)
    parser.add_argument("--ltm-radius", type=float, default=1.25)
    parser.add_argument("--ltm-top-k", type=int, default=24)
    parser.add_argument("--tie-tolerance", type=float, default=0.10)
    parser.add_argument("--selector-step", type=float, default=0.5)
    parser.add_argument("--min-group", type=int, default=256)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    horizons = parse_horizons(args.horizons)
    unsupported = [h for h in horizons if h not in HORIZONS_13]
    if unsupported:
        raise ValueError(f"Unsupported horizons for Phase 13B: {unsupported}")

    scene_shards = select_scenes(
        args.scenes,
        include_tier2=args.include_tier2,
        include_tier3=args.include_tier3,
        cache_root=Path(args.cache_root),
    )
    if not scene_shards:
        raise FileNotFoundError("No cached PhysicalAI physics shards selected for Phase 13B.")

    result = run_phase13b(
        scene_shards,
        previous_matrix_path=Path(args.previous_matrix_json),
        train_fraction=args.train_fraction,
        split_seed=args.split_seed,
        stride=args.stride,
        memory_window=args.memory_window,
        max_cells=args.max_cells,
        ridge=args.ridge,
        model_radius=args.model_radius,
        model_top_k=args.model_top_k,
        ltm_radius=args.ltm_radius,
        ltm_top_k=args.ltm_top_k,
        tie_tolerance=args.tie_tolerance,
        selector_step=args.selector_step,
        min_group=args.min_group,
        horizons=horizons,
    )
    write_phase13b_outputs(result, Path(args.out_dir))
    print(
        {
            "out_json": str(Path(args.out_dir) / "phase13b_latest.json"),
            "scenes": result["scenes"],
            "passed": result["cross_scene"]["phase13b_passed"],
        }
    )


if __name__ == "__main__":
    main()
