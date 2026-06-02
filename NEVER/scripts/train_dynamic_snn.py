"""Retired SNN entrypoint.

Phase 10c removed the dynamic SNN branch from N.E.V.E.R.  Use
`phase10c_never_amf_orchestrator.py` instead.  The new path does not train a
spiking network; it loads the warmed AMF world model, keeps identity frozen,
maps prompt text to an action vector, predicts the next latent, and decodes a
frame.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Retired SNN wrapper; forwards to Phase 10c AMF orchestrator.")
    parser.add_argument("--prompt", default="Personaje saltando, camara rotando 45 grados")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--offline", action="store_true", default=True)
    args = parser.parse_args()

    orchestrator = Path(__file__).with_name("phase10c_never_amf_orchestrator.py")
    cmd = [sys.executable, str(orchestrator), "--prompt", args.prompt, "--steps", str(args.steps), "--offline"]
    print("[SNN retired] Forwarding to AMF orchestrator:")
    print(" ".join(cmd))
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
