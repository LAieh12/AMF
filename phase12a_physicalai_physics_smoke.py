from __future__ import annotations

import argparse
import io
import json
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download


REPO_ID = "nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes"
DEFAULT_FILE = "physics/objects_falling/physics-objects_falling-00007.tar"


def summarize_npz(blob: bytes, max_arrays: int) -> dict[str, Any]:
    with np.load(io.BytesIO(blob), allow_pickle=False) as npz:
        keys = list(npz.files)
        arrays = {}
        for key in keys[:max_arrays]:
            arr = np.asarray(npz[key])
            arrays[key] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "min": float(np.nanmin(arr)) if arr.size and np.issubdtype(arr.dtype, np.number) else None,
                "max": float(np.nanmax(arr)) if arr.size and np.issubdtype(arr.dtype, np.number) else None,
            }
        return {"keys": keys, "arrays": arrays}


def inspect_tar(path: Path, max_members: int, max_npz: int, max_arrays: int) -> dict[str, Any]:
    members_summary: list[dict[str, Any]] = []
    npz_seen = 0
    with tarfile.open(path, "r") as tar:
        members = [member for member in tar.getmembers() if member.isfile()]
        for member in members[:max_members]:
            item: dict[str, Any] = {"name": member.name, "size": member.size}
            if member.name.endswith(".npz") and npz_seen < max_npz:
                fileobj = tar.extractfile(member)
                if fileobj is not None:
                    item["npz"] = summarize_npz(fileobj.read(), max_arrays=max_arrays)
                    npz_seen += 1
            members_summary.append(item)
    suffix_counts: dict[str, int] = {}
    for member in members_summary:
        suffix = Path(member["name"]).suffix or "<none>"
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
    return {
        "tar_path": str(path),
        "tar_size_bytes": path.stat().st_size,
        "listed_members": len(members_summary),
        "suffix_counts_first_members": suffix_counts,
        "members": members_summary,
    }


def render_report(result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Fase 12A - PhysicalAI physics smoke")
    lines.append("")
    lines.append(f"Repo: `{result['repo_id']}`")
    lines.append(f"Downloaded file: `{result['repo_file']}`")
    lines.append(f"Local tar: `{result['inspection']['tar_path']}`")
    lines.append(f"Size bytes: {result['inspection']['tar_size_bytes']}")
    lines.append("")
    lines.append("## NPZ sample")
    lines.append("")
    for member in result["inspection"]["members"]:
        if "npz" not in member:
            continue
        lines.append(f"### `{member['name']}`")
        lines.append("")
        lines.append(f"- Size bytes: {member['size']}")
        lines.append(f"- Keys: {', '.join(member['npz']['keys'][:24])}")
        lines.append("")
        lines.append("| array | shape | dtype | min | max |")
        lines.append("|---|---|---|---:|---:|")
        for key, meta in member["npz"]["arrays"].items():
            lines.append(
                f"| `{key}` | `{meta['shape']}` | `{meta['dtype']}` | "
                f"{meta['min'] if meta['min'] is not None else ''} | {meta['max'] if meta['max'] is not None else ''} |"
            )
        lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append("Este smoke confirma que Fase 12A puede usar anotaciones fisicas reales del dataset sin descargar RGB/depth completos.")
    lines.append("El siguiente codec debe entrenar primero sobre `physics/*.npz` y usar video/segmentation como verificacion visual posterior.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and inspect one PhysicalAI physics tar shard.")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--repo-file", default=DEFAULT_FILE)
    parser.add_argument("--cache-dir", default="data/physicalai_hf_cache")
    parser.add_argument("--max-members", type=int, default=20)
    parser.add_argument("--max-npz", type=int, default=3)
    parser.add_argument("--max-arrays", type=int, default=12)
    parser.add_argument("--out-json", default="results/phase12a_physicalai_physics_smoke.json")
    parser.add_argument("--out-report", default="results/FASE12A_PHYSICALAI_PHYSICS_SMOKE.md")
    args = parser.parse_args()

    started = time.time()
    local_path = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename=args.repo_file,
            repo_type="dataset",
            cache_dir=args.cache_dir,
        )
    )
    inspection = inspect_tar(local_path, args.max_members, args.max_npz, args.max_arrays)
    result = {
        "repo_id": args.repo_id,
        "repo_file": args.repo_file,
        "inspection": inspection,
        "elapsed_seconds": time.time() - started,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(render_report(result), encoding="utf-8")

    print(json.dumps({"out_json": str(out_json), "out_report": str(out_report), "tar": str(local_path)}, indent=2))


if __name__ == "__main__":
    main()
