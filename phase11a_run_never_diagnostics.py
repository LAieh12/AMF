from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_STEPS = [
    {
        "name": "compile",
        "cmd": [
            sys.executable,
            "-m",
            "py_compile",
            "phase11a_never_bottleneck_audit.py",
            "phase11a_never_world_codec_probe.py",
            "phase11a_never_definitive_codec_probe.py",
            "phase11a_frontier_slot_warp_probe.py",
        ],
        "timeout": 120,
    },
    {
        "name": "audit",
        "cmd": [
            sys.executable,
            "phase11a_never_bottleneck_audit.py",
            "--out",
            "results/FASE11A_NEVER_BOTTLENECK_AUDIT.md",
        ],
        "timeout": 180,
    },
    {
        "name": "world_codec_smoke",
        "cmd": [
            sys.executable,
            "phase11a_never_world_codec_probe.py",
            "--train-sequences",
            "80",
            "--test-sequences",
            "20",
            "--tune-sequences",
            "20",
            "--out",
            "results/phase11a_never_world_codec_probe_smoke.json",
        ],
        "timeout": 1200,
    },
    {
        "name": "definitive_codec_smoke",
        "cmd": [
            sys.executable,
            "phase11a_never_definitive_codec_probe.py",
            "--train-sequences",
            "80",
            "--test-sequences",
            "20",
            "--selector-sequences",
            "20",
            "--out",
            "results/phase11a_never_definitive_codec_probe_smoke.json",
        ],
        "timeout": 1200,
    },
]


FULL_STEPS = [
    {
        "name": "definitive_codec_220_40",
        "cmd": [
            sys.executable,
            "phase11a_never_definitive_codec_probe.py",
            "--train-sequences",
            "220",
            "--test-sequences",
            "40",
            "--selector-sequences",
            "60",
            "--out",
            "results/phase11a_never_definitive_codec_probe_220_40.json",
        ],
        "timeout": 3600,
    },
    {
        "name": "world_codec_220_40",
        "cmd": [
            sys.executable,
            "phase11a_never_world_codec_probe.py",
            "--train-sequences",
            "220",
            "--test-sequences",
            "40",
            "--tune-sequences",
            "60",
            "--out",
            "results/phase11a_never_world_codec_probe_220_40.json",
        ],
        "timeout": 3600,
    },
]


def run_step(step: dict[str, object], retries: int, delay: float) -> dict[str, object]:
    last: dict[str, object] | None = None
    for attempt in range(1, retries + 1):
        started = time.time()
        try:
            completed = subprocess.run(
                step["cmd"],
                cwd=Path(__file__).resolve().parent,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=float(step["timeout"]),
                check=False,
            )
            last = {
                "name": step["name"],
                "attempt": attempt,
                "returncode": completed.returncode,
                "elapsed_seconds": time.time() - started,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
                "cmd": step["cmd"],
            }
            if completed.returncode == 0:
                return last
        except Exception as exc:  # pragma: no cover - runtime diagnostics path
            last = {
                "name": step["name"],
                "attempt": attempt,
                "returncode": None,
                "elapsed_seconds": time.time() - started,
                "error": repr(exc),
                "cmd": step["cmd"],
            }
        if attempt < retries:
            time.sleep(delay)
    if last is None:
        raise RuntimeError(f"step did not run: {step['name']}")
    return last


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 11A Never diagnostics with retries.")
    parser.add_argument("--full", action="store_true", help="Run 220/40 probes after smoke diagnostics.")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--out", default="results/phase11a_never_diagnostics_run.json")
    args = parser.parse_args()

    steps = list(DEFAULT_STEPS)
    if args.full:
        steps.extend(FULL_STEPS)

    results = []
    for step in steps:
        print(f"running {step['name']}: {' '.join(step['cmd'])}", flush=True)
        result = run_step(step, args.retries, args.delay)
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
        if result.get("returncode") != 0:
            break

    summary = {
        "script": "phase11a_run_never_diagnostics.py",
        "full": args.full,
        "retries": args.retries,
        "delay": args.delay,
        "results": results,
        "all_passed": all(item.get("returncode") == 0 for item in results) and len(results) == len(steps),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
