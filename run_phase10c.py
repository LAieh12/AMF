from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    script = Path(__file__).resolve().parent / "NEVER" / "scripts" / "phase10c_never_amf_orchestrator.py"
    cmd = [sys.executable, str(script), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd, cwd=str(script.parents[1])))


if __name__ == "__main__":
    main()
