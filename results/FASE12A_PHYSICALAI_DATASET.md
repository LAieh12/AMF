# Fase 12A - NVIDIA PhysicalAI dataset probe

Repo: https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes
Commit/SHA: `ff9b2b0a93243d84925c6c7474ee8a5bc02886cb`

## Veredicto

Si, usar este dataset es el siguiente paso correcto despues de MovingMNIST.
La razon es que introduce fisica multi-objeto con ground truth limpio antes de saltar a videos humanos reales.

## Escalera

- `12A` - clean multi-object physics: objects_falling, billiards.
  Motivo: gravity, bounce, settling, and clean elastic collisions with direct physics labels.
- `12B` - causality and structured collisions: dominoes, bowling, rolling_ramp_objects, rolling_ramp_obstruct, obstruction.
  Motivo: trigger chains, ramps, fixed obstacles, directed impact, and scatter.
- `12C` - chaos and collapse: ball_mixer, towers, wrecking_ball.
  Motivo: persistent mixing, structural collapse, pendulum constraints, and chaotic secondary motion.

## Manifest

- Archivos/shards listados por HF: 9520
- Modalidades:
  - `cameras`: 81
  - `captions`: 81
  - `depths`: 7653
  - `physics`: 81
  - `scene`: 81
  - `segmentation`: 770
  - `videos`: 770

## Escenas

| escena | cameras | physics | segmentation | videos | depths | captions | scene |
|---|---:|---:|---:|---:|---:|---:|---:|
| ball_mixer | 2 | 2 | 16 | 16 | 158 | 2 | 2 |
| billiards | 9 | 9 | 88 | 88 | 876 | 9 | 9 |
| bowling | 9 | 9 | 85 | 85 | 846 | 9 | 9 |
| dominoes | 9 | 9 | 88 | 88 | 878 | 9 | 9 |
| objects_falling | 8 | 8 | 72 | 72 | 715 | 8 | 8 |
| obstruction | 9 | 9 | 82 | 82 | 811 | 9 | 9 |
| rolling_ramp_objects | 7 | 7 | 67 | 67 | 670 | 7 | 7 |
| rolling_ramp_obstruct | 9 | 9 | 89 | 89 | 881 | 9 | 9 |
| towers | 8 | 8 | 77 | 77 | 767 | 8 | 8 |
| wrecking_ball | 11 | 11 | 106 | 106 | 1051 | 11 | 11 |

## Tamano de primeros shards

Esto confirma que no conviene descargar todo a ciegas.

| escena | modalidad | primer shard bytes |
|---|---|---:|
| objects_falling | cameras | 411658240 |
| objects_falling | physics | 477818880 |
| objects_falling | videos | 2991759360 |
| billiards | cameras | 419471360 |
| billiards | physics | 176343040 |
| billiards | videos | 1459087360 |

## Streaming smoke

- row 0: key `ball_mixer_9af0e375_0/camera_e`, payload `json`, camera `camera_e`, frames `240`.
- row 1: key `ball_mixer_9af0e375_0/camera_n`, payload `json`, camera `camera_n`, frames `240`.
- row 2: key `ball_mixer_9af0e375_0/camera_s`, payload `json`, camera `camera_s`, frames `240`.

## Decision tecnica

- 12A debe empezar con `objects_falling` y `billiards` usando metadata/physics/segmentation antes de RGB pesado.
- El objetivo del encoder cambia de blobs 2D a slots fisicos: posicion, velocidad, spin, CoM, identidad de mascara y contacto.
- El decoder no debe inventar pixeles primero; debe predecir estados fisicos y luego render/warp/segmentar.
- MovingMNIST sigue vivo como smoke test barato, pero ya no es suficiente para validar Never.
