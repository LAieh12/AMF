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
    args = parser.parse_args()
    result = run_formal_world_model(load_config(Path(args.config)), resume=args.resume, stop_after_scenes=args.stop_after_scenes)
    print(
        {
            "completed_shards": result["completed_shards"],
            "resume_used": result["resume_used"],
            "latest_json": load_config(Path(args.config))["output_paths"]["latest_json"],
        }
    )


if __name__ == "__main__":
    main()
