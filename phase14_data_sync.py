from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from phase13_scene_loader import SceneShard, scene_tier
from phase13c_readiness import build_manifest, build_splits, make_config


REPO_ID = "nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes"
PHYSICS_PATTERN = "physics/**/*.tar"
CAPTIONS_PATTERN = "captions/**/*.tar"
FORBIDDEN_PREFIXES = (
    "rgb/",
    "rgbs/",
    "depth/",
    "depths/",
    "segmentation/",
    "segmentations/",
    "segmentation_png/",
    "scene_usda/",
    "usd/",
    "camera/",
    "cameras/",
)


def remote_size_summary() -> dict[str, Any]:
    api = HfApi()
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    wanted = [f for f in files if f.startswith("physics/") and f.endswith(".tar")]
    captions = [f for f in files if f.startswith("captions/") and f.endswith(".tar")]
    total_physics = 0
    total_captions = 0
    by_scene: dict[str, int] = {}
    for batch in (wanted[i : i + 50] for i in range(0, len(wanted), 50)):
        for info in api.get_paths_info(repo_id=REPO_ID, repo_type="dataset", paths=batch):
            size = int(info.size or 0)
            total_physics += size
            scene = info.path.split("/")[1]
            by_scene[scene] = by_scene.get(scene, 0) + size
    for batch in (captions[i : i + 50] for i in range(0, len(captions), 50)):
        for info in api.get_paths_info(repo_id=REPO_ID, repo_type="dataset", paths=batch):
            total_captions += int(info.size or 0)
    return {
        "repo": REPO_ID,
        "physics_shards_remote": len(wanted),
        "captions_shards_remote": len(captions),
        "physics_total_gb": round(total_physics / 1024**3, 2),
        "captions_total_gb": round(total_captions / 1024**3, 2),
        "physics_by_scene_gb": {scene: round(size / 1024**3, 2) for scene, size in sorted(by_scene.items())},
    }


def remote_allowed_files(include_captions: bool) -> list[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    wanted = [f for f in files if f.startswith("physics/") and f.endswith(".tar")]
    if include_captions:
        wanted.extend(f for f in files if f.startswith("captions/") and f.endswith(".tar"))
    return sorted(wanted)


def relative_allowed_path(path: Path) -> str | None:
    parts = path.parts
    lowered = [part.lower() for part in parts]
    for prefix in ("physics", "captions"):
        if prefix not in lowered:
            continue
        idx = lowered.index(prefix)
        if len(parts) - idx != 3:
            continue
        if not parts[-1].endswith(".tar"):
            continue
        return Path(*parts[idx:]).as_posix()
    return None


def reuse_existing_allowed_files(
    local_dir: Path,
    search_roots: list[Path],
    wanted: set[str],
) -> dict[str, Any]:
    reused = []
    for root in search_roots:
        if not root.exists():
            continue
        for source in root.rglob("*.tar"):
            rel = relative_allowed_path(source)
            if rel is None or rel not in wanted:
                continue
            target = local_dir / Path(rel)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                if source.resolve() == target.resolve():
                    continue
            except FileNotFoundError:
                pass
            try:
                os.link(source, target)
                method = "hardlink"
            except OSError:
                shutil.copy2(source, target)
                method = "copy"
            reused.append({"path": rel, "source": str(source), "method": method})
    return {
        "search_roots": [str(root) for root in search_roots],
        "reused_count": len(reused),
        "reused": reused[:200],
    }


def download_missing_allowed_files(local_dir: Path, wanted: list[str], workers: int) -> dict[str, Any]:
    missing = [rel for rel in wanted if not (local_dir / Path(rel)).exists()]
    if not missing:
        return {"mode": "missing_only", "missing_before_download": 0, "downloaded_count": 0, "downloaded": []}

    downloaded = []

    def fetch(rel: str) -> str:
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=rel,
            local_dir=str(local_dir),
        )
        return rel

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(fetch, rel): rel for rel in missing}
        for done, future in enumerate(as_completed(futures), start=1):
            rel = future.result()
            downloaded.append(rel)
            print(
                json.dumps(
                    {
                        "downloaded": done,
                        "total_missing": len(missing),
                        "path": rel,
                    }
                ),
                flush=True,
            )
    return {
        "mode": "missing_only",
        "missing_before_download": len(missing),
        "downloaded_count": len(downloaded),
        "downloaded": downloaded,
    }


def download_physics_and_captions(
    local_dir: Path,
    include_captions: bool,
    missing_only: bool,
    workers: int,
    reuse_roots: list[Path],
) -> dict[str, Any]:
    wanted = remote_allowed_files(include_captions)
    reuse_report = reuse_existing_allowed_files(local_dir, reuse_roots, set(wanted)) if reuse_roots else None
    if missing_only:
        download_report = download_missing_allowed_files(local_dir, wanted, workers)
    else:
        patterns = [PHYSICS_PATTERN]
        if include_captions:
            patterns.append(CAPTIONS_PATTERN)
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=str(local_dir),
            allow_patterns=patterns,
            max_workers=max(1, workers),
        )
        download_report = {
            "mode": "snapshot_download",
            "wanted_count": len(wanted),
            "downloaded_count": None,
        }
    missing_after = [rel for rel in wanted if not (local_dir / Path(rel)).exists()]
    return {
        "wanted_count": len(wanted),
        "workers": max(1, workers),
        "reuse_report": reuse_report,
        "download": download_report,
        "missing_after_count": len(missing_after),
        "missing_after": missing_after[:200],
    }


def legacy_snapshot_download(local_dir: Path, include_captions: bool) -> Path:
    patterns = [PHYSICS_PATTERN]
    if include_captions:
        patterns.append(CAPTIONS_PATTERN)
    return Path(
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=str(local_dir),
            allow_patterns=patterns,
            max_workers=8,
        )
    )


def discover_local_physics(local_dir: Path) -> list[SceneShard]:
    shards = []
    for tar_path in sorted((local_dir / "physics").glob("*/*.tar")):
        scene = tar_path.parent.name
        shards.append(SceneShard(scene=scene, tar_path=tar_path, tier=scene_tier(scene)))
    return shards


def discover_local_captions(local_dir: Path) -> list[dict[str, Any]]:
    records = []
    for tar_path in sorted((local_dir / "captions").glob("*/*.tar")):
        records.append(
            {
                "scene": tar_path.parent.name,
                "caption_shard": str(tar_path),
                "size_bytes": tar_path.stat().st_size,
            }
        )
    return records


def verify_forbidden_assets(local_dir: Path) -> dict[str, Any]:
    violations = []
    for path in local_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix().lower()
        if rel.startswith(FORBIDDEN_PREFIXES):
            violations.append(rel)
    return {
        "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
        "violations": violations[:100],
        "violation_count": len(violations),
        "passed": len(violations) == 0,
    }


def update_phase14_config(config: dict[str, Any], shards: list[SceneShard], splits_path: Path) -> dict[str, Any]:
    updated = make_config(shards, splits_path)
    updated["remote_data_sync"] = {
        "source": REPO_ID,
        "physics_complete": True,
        "captions_downloaded": True,
        "forbidden_assets_downloaded": False,
        "synced_at": time.time(),
    }
    # Preserve frozen protocol settings from the existing config shape.
    for key in ("model", "selector", "training", "checkpoint", "output_paths"):
        if key in config:
            updated[key] = config[key]
    updated.setdefault("output_paths", {})["model_export_dir"] = config.get("output_paths", {}).get(
        "model_export_dir", "models/phase14"
    )
    return updated


def run_data_sync(args: argparse.Namespace) -> dict[str, Any]:
    local_dir = Path(args.local_dir)
    results_dir = Path(args.results_dir)
    config_path = Path(args.config)
    results_dir.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)

    summary = (
        {
            "repo": REPO_ID,
            "skipped": True,
            "reason": "--skip-remote-summary",
        }
        if args.skip_remote_summary
        else remote_size_summary()
    )
    download_report = None
    if args.download:
        download_report = download_physics_and_captions(
            local_dir,
            include_captions=not args.no_captions,
            missing_only=args.missing_only,
            workers=args.download_workers,
            reuse_roots=[Path(root) for root in args.reuse_existing_from],
        )

    shards = discover_local_physics(local_dir)
    if args.scenes:
        wanted = set(args.scenes)
        shards = [shard for shard in shards if shard.scene in wanted]
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    if not shards:
        raise FileNotFoundError(f"No local physics shards found under {local_dir / 'physics'}")
    used = {str(shard.tar_path) for shard in shards}
    manifest = build_manifest(shards, used)
    manifest["remote_size_summary"] = summary
    manifest["caption_records"] = discover_local_captions(local_dir)
    manifest["forbidden_asset_check"] = verify_forbidden_assets(local_dir)
    manifest_path = results_dir / "phase13c_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    splits = build_splits(shards, train_fraction=args.train_fraction, seed=args.seed)
    splits_path = results_dir / "phase13c_splits.json"
    splits_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")

    existing_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    config = update_phase14_config(existing_config, shards, splits_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    report = {
        "phase": "phase14_data_sync",
        "local_dir": str(local_dir),
        "physics_shards_local": len(shards),
        "caption_shards_local": len(manifest["caption_records"]),
        "remote_size_summary": summary,
        "download_report": download_report,
        "manifest_path": str(manifest_path),
        "splits_path": str(splits_path),
        "config_path": str(config_path),
        "forbidden_asset_check": manifest["forbidden_asset_check"],
        "phase14_command": f"python run_phase14.py --config {config_path.as_posix()}",
    }
    report_path = results_dir / "phase14_data_sync_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/sync full PhysicalAI physics+captions for frozen Phase 14.")
    parser.add_argument("--local-dir", default="data/physicalai_physics_captions_full")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--config", default="configs/phase14_world_model_train.yaml")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--missing-only", action="store_true")
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--reuse-existing-from", nargs="*", default=[])
    parser.add_argument("--no-captions", action="store_true")
    parser.add_argument("--skip-remote-summary", action="store_true")
    parser.add_argument("--limit-shards", type=int, default=0)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    print(json.dumps(run_data_sync(args), indent=2))


if __name__ == "__main__":
    main()
