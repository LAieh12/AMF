from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase14_formal_amf_world_model import run_formal_world_model


def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Phase 14 AMF World Model training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-scenes", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of scene shards to train in parallel. Use 1 for deterministic serial training.",
    )
    args = parser.parse_args()
    config = load_config(Path(args.config))
    result = run_formal_world_model(
        config,
        resume=args.resume,
        stop_after_scenes=args.stop_after_scenes,
        workers=max(1, args.workers),
    )
    print(
        {
            "completed_shards": result["completed_shards"],
            "resume_used": result["resume_used"],
            "workers": result["workers"],
            "latest_json": config["output_paths"]["latest_json"],
            "model_export_dir": result["model_export_dir"],
            "model_export_index": f"{result['model_export_dir']}/model_index.json",
        }
    )


if __name__ == "__main__":
    main()
