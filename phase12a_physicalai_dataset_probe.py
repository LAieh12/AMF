from __future__ import annotations

import argparse
import json
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_url


REPO_ID = "nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes"
REPO_URL = f"https://huggingface.co/datasets/{REPO_ID}"

PHASE_LADDER = {
    "12A": {
        "name": "clean multi-object physics",
        "scenes": ["objects_falling", "billiards"],
        "reason": "gravity, bounce, settling, and clean elastic collisions with direct physics labels.",
    },
    "12B": {
        "name": "causality and structured collisions",
        "scenes": ["dominoes", "bowling", "rolling_ramp_objects", "rolling_ramp_obstruct", "obstruction"],
        "reason": "trigger chains, ramps, fixed obstacles, directed impact, and scatter.",
    },
    "12C": {
        "name": "chaos and collapse",
        "scenes": ["ball_mixer", "towers", "wrecking_ball"],
        "reason": "persistent mixing, structural collapse, pendulum constraints, and chaotic secondary motion.",
    },
}

MODALITY_PRIORITY = ("physics", "segmentation", "cameras", "captions", "videos", "depths", "scene")


def _head_size(repo_id: str, path: str, timeout: float = 20.0) -> int | None:
    url = hf_hub_url(repo_id=repo_id, filename=path, repo_type="dataset")
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            size = response.headers.get("content-length")
            return int(size) if size else None
    except Exception:
        return None


def build_manifest(repo_id: str, size_probe: bool) -> dict[str, Any]:
    started = time.time()
    api = HfApi()
    info = api.dataset_info(repo_id)
    by_scene: dict[str, Counter[str]] = defaultdict(Counter)
    first_files: dict[str, dict[str, str]] = defaultdict(dict)
    modality_counts: Counter[str] = Counter()

    for sibling in info.siblings or []:
        path = sibling.rfilename
        parts = path.split("/")
        if len(parts) < 3:
            continue
        modality, scene = parts[0], parts[1]
        by_scene[scene][modality] += 1
        modality_counts[modality] += 1
        first_files[scene].setdefault(modality, path)

    first_file_sizes: dict[str, dict[str, int | None]] = defaultdict(dict)
    if size_probe:
        for stage in ("12A", "12B", "12C"):
            for scene in PHASE_LADDER[stage]["scenes"]:
                for modality in ("physics", "videos", "cameras"):
                    path = first_files.get(scene, {}).get(modality)
                    if path:
                        first_file_sizes[scene][modality] = _head_size(repo_id, path)

    stages = {}
    for stage, spec in PHASE_LADDER.items():
        stages[stage] = {
            **spec,
            "scene_counts": {scene: dict(sorted(by_scene.get(scene, {}).items())) for scene in spec["scenes"]},
            "first_files": {scene: dict(sorted(first_files.get(scene, {}).items())) for scene in spec["scenes"]},
            "first_file_sizes_bytes": {
                scene: dict(sorted(first_file_sizes.get(scene, {}).items())) for scene in spec["scenes"]
            },
        }

    return {
        "repo_id": repo_id,
        "repo_url": REPO_URL,
        "sha": info.sha,
        "siblings": len(info.siblings or []),
        "modalities": dict(sorted(modality_counts.items())),
        "scenes": {scene: dict(sorted(counts.items())) for scene, counts in sorted(by_scene.items())},
        "phase_ladder": stages,
        "size_probe": size_probe,
        "elapsed_seconds": time.time() - started,
    }


def stream_smoke(repo_id: str, rows: int) -> list[dict[str, Any]]:
    if rows <= 0:
        return []
    from datasets import load_dataset

    out: list[dict[str, Any]] = []
    dataset = load_dataset(repo_id, split="train", streaming=True)
    for index, row in zip(range(rows), dataset):
        payload_key = next((key for key in row.keys() if key not in {"__key__", "__url__"}), None)
        payload = row.get(payload_key) if payload_key else None
        summary = {
            "index": index,
            "__key__": row.get("__key__"),
            "__url__": row.get("__url__"),
            "payload_key": payload_key,
            "payload_type": type(payload).__name__,
        }
        if isinstance(payload, dict):
            summary["payload_fields"] = sorted(payload.keys())[:24]
            summary["frame_count"] = payload.get("frame_count")
            summary["camera_name"] = payload.get("camera_name")
        out.append(summary)
    return out


def render_report(manifest: dict[str, Any], smoke_rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Fase 12A - NVIDIA PhysicalAI dataset probe")
    lines.append("")
    lines.append(f"Repo: {manifest['repo_url']}")
    lines.append(f"Commit/SHA: `{manifest['sha']}`")
    lines.append("")
    lines.append("## Veredicto")
    lines.append("")
    lines.append("Si, usar este dataset es el siguiente paso correcto despues de MovingMNIST.")
    lines.append("La razon es que introduce fisica multi-objeto con ground truth limpio antes de saltar a videos humanos reales.")
    lines.append("")
    lines.append("## Escalera")
    lines.append("")
    for stage, spec in manifest["phase_ladder"].items():
        lines.append(f"- `{stage}` - {spec['name']}: {', '.join(spec['scenes'])}.")
        lines.append(f"  Motivo: {spec['reason']}")
    lines.append("")
    lines.append("## Manifest")
    lines.append("")
    lines.append(f"- Archivos/shards listados por HF: {manifest['siblings']}")
    lines.append("- Modalidades:")
    for modality, count in manifest["modalities"].items():
        lines.append(f"  - `{modality}`: {count}")
    lines.append("")
    lines.append("## Escenas")
    lines.append("")
    lines.append("| escena | cameras | physics | segmentation | videos | depths | captions | scene |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for scene, counts in manifest["scenes"].items():
        lines.append(
            "| "
            + scene
            + " | "
            + " | ".join(str(counts.get(key, 0)) for key in ("cameras", "physics", "segmentation", "videos", "depths", "captions", "scene"))
            + " |"
        )
    lines.append("")
    if manifest["size_probe"]:
        lines.append("## Tamano de primeros shards")
        lines.append("")
        lines.append("Esto confirma que no conviene descargar todo a ciegas.")
        lines.append("")
        lines.append("| escena | modalidad | primer shard bytes |")
        lines.append("|---|---|---:|")
        for scene, sizes in manifest["phase_ladder"]["12A"]["first_file_sizes_bytes"].items():
            for modality, size in sizes.items():
                lines.append(f"| {scene} | {modality} | {size if size is not None else 'unknown'} |")
        lines.append("")
    lines.append("## Streaming smoke")
    lines.append("")
    if not smoke_rows:
        lines.append("No se solicitaron filas de streaming.")
    else:
        for row in smoke_rows:
            lines.append(
                f"- row {row['index']}: key `{row['__key__']}`, payload `{row['payload_key']}`, "
                f"camera `{row.get('camera_name')}`, frames `{row.get('frame_count')}`."
            )
    lines.append("")
    lines.append("## Decision tecnica")
    lines.append("")
    lines.append("- 12A debe empezar con `objects_falling` y `billiards` usando metadata/physics/segmentation antes de RGB pesado.")
    lines.append("- El objetivo del encoder cambia de blobs 2D a slots fisicos: posicion, velocidad, spin, CoM, identidad de mascara y contacto.")
    lines.append("- El decoder no debe inventar pixeles primero; debe predecir estados fisicos y luego render/warp/segmentar.")
    lines.append("- MovingMNIST sigue vivo como smoke test barato, pero ya no es suficiente para validar Never.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect NVIDIA PhysicalAI dataset for AMF/Never Phase 12A.")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--stream-rows", type=int, default=3)
    parser.add_argument("--no-size-probe", action="store_true")
    parser.add_argument("--out-json", default="results/phase12a_physicalai_manifest.json")
    parser.add_argument("--out-report", default="results/FASE12A_PHYSICALAI_DATASET.md")
    args = parser.parse_args()

    manifest = build_manifest(args.repo_id, size_probe=not args.no_size_probe)
    smoke_rows = stream_smoke(args.repo_id, args.stream_rows)
    manifest["streaming_smoke_rows"] = smoke_rows

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(render_report(manifest, smoke_rows), encoding="utf-8")

    print(json.dumps({"out_json": str(out_json), "out_report": str(out_report), "sha": manifest["sha"]}, indent=2))


if __name__ == "__main__":
    main()
