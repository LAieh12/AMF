from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from phase12a_physicalai_world_probe import PhysicsTrack, load_tracks, split_train_validation


DEFAULT_CACHE_ROOT = Path("data/physicalai_hf_cache")
TIER1_SCENES = ("objects_falling", "dominoes", "wrecking_ball")
TIER2_SCENES = ("billiards", "rolling_ramp_objects", "rolling_ramp_obstruct", "bowling")
TIER3_SCENES = ("ball_mixer", "towers", "obstruction")


@dataclass(frozen=True)
class SceneShard:
    scene: str
    tar_path: Path
    tier: int


@dataclass
class SceneData:
    shard: SceneShard
    tracks: list[PhysicsTrack]
    sequences: list[str]
    fit_sequences: set[str]
    validation_sequences: set[str]
    train_sequences: set[str]
    test_sequences: set[str]


def scene_tier(scene: str) -> int:
    if scene in TIER1_SCENES:
        return 1
    if scene in TIER2_SCENES:
        return 2
    return 3


def discover_scene_shards(cache_root: Path = DEFAULT_CACHE_ROOT) -> dict[str, SceneShard]:
    shards: dict[str, SceneShard] = {}
    if not cache_root.exists():
        return shards
    for tar_path in sorted(cache_root.rglob("physics-*.tar")):
        scene = tar_path.parent.name
        if scene not in shards:
            shards[scene] = SceneShard(scene=scene, tar_path=tar_path, tier=scene_tier(scene))
    return shards


def select_scenes(
    requested: list[str] | None,
    include_tier2: bool,
    include_tier3: bool,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> list[SceneShard]:
    available = discover_scene_shards(cache_root)
    if requested:
        missing = [scene for scene in requested if scene not in available]
        if missing:
            raise FileNotFoundError(f"Missing cached scene shards: {', '.join(missing)}")
        return [available[scene] for scene in requested]

    scenes = list(TIER1_SCENES)
    if include_tier2:
        scenes.extend(TIER2_SCENES)
    if include_tier3:
        scenes.extend(TIER3_SCENES)
    return [available[scene] for scene in scenes if scene in available]


def load_scene_data(shard: SceneShard, train_fraction: float, split_seed: int) -> SceneData:
    tracks = load_tracks(shard.tar_path)
    sequences = sorted({track.sequence for track in tracks})
    fit_sequences, validation_sequences, train_sequences, test_sequences = split_train_validation(
        sequences, train_fraction, split_seed
    )
    return SceneData(
        shard=shard,
        tracks=tracks,
        sequences=sequences,
        fit_sequences=fit_sequences,
        validation_sequences=validation_sequences,
        train_sequences=train_sequences,
        test_sequences=test_sequences,
    )
