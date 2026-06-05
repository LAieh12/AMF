# Fase 12B - PhysicalAI physics smoke

Repo: `nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes`
Downloaded file: `physics/bowling/physics-bowling-00000.tar`
Local tar: `data\physicalai_hf_cache\datasets--nvidia--PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes\snapshots\ff9b2b0a93243d84925c6c7474ee8a5bc02886cb\physics\bowling\physics-bowling-00000.tar`
Size bytes: 194949120

## NPZ sample

### `bowling_e787b7fd_0/Bowler_com.npz`

- Size bytes: 5556
- Keys: frame_count, segmentation_colors, data, global_min, global_max

| array | shape | dtype | min | max |
|---|---|---|---:|---:|
| `frame_count` | `[]` | `int64` | 150.0 | 150.0 |
| `segmentation_colors` | `[11, 4]` | `uint8` | 3.0 | 255.0 |
| `data` | `[11, 150, 3]` | `float32` | -9.14690113067627 | 1.0461963415145874 |
| `global_min` | `[3]` | `float32` | -9.14690113067627 | -0.9153060913085938 |
| `global_max` | `[3]` | `float32` | 0.0 | 1.0461963415145874 |

### `bowling_e787b7fd_0/Bowler_rot.npz`

- Size bytes: 6627
- Keys: frame_count, segmentation_colors, data

| array | shape | dtype | min | max |
|---|---|---|---:|---:|
| `frame_count` | `[]` | `int64` | 150.0 | 150.0 |
| `segmentation_colors` | `[11, 4]` | `uint8` | 3.0 | 255.0 |
| `data` | `[11, 150, 4]` | `float32` | -0.8763408660888672 | 0.9440765976905823 |

### `bowling_e787b7fd_0/Bowler_spin.npz`

- Size bytes: 6231
- Keys: frame_count, segmentation_colors, data, global_min, global_max

| array | shape | dtype | min | max |
|---|---|---|---:|---:|
| `frame_count` | `[]` | `int64` | 150.0 | 150.0 |
| `segmentation_colors` | `[11, 4]` | `uint8` | 3.0 | 255.0 |
| `data` | `[11, 150, 3]` | `float32` | -3320.45556640625 | 1020.291748046875 |
| `global_min` | `[3]` | `float32` | -3320.45556640625 | -344.8894348144531 |
| `global_max` | `[3]` | `float32` | 331.6059875488281 | 1020.291748046875 |

### `bowling_e787b7fd_0/Bowler_velocity.npz`

- Size bytes: 11514
- Keys: frame_count, segmentation_colors, velocities, data, global_min, global_max

| array | shape | dtype | min | max |
|---|---|---|---:|---:|
| `frame_count` | `[]` | `int64` | 150.0 | 150.0 |
| `segmentation_colors` | `[11, 4]` | `uint8` | 3.0 | 255.0 |
| `velocities` | `[11, 150, 3]` | `float32` | -9.625945091247559 | 1.814766764640808 |
| `data` | `[11, 150, 3]` | `float32` | -9.625945091247559 | 1.814766764640808 |
| `global_min` | `[3]` | `float32` | -9.625945091247559 | -0.6915184259414673 |
| `global_max` | `[3]` | `float32` | 0.42898914217948914 | 1.814766764640808 |

### `bowling_e787b7fd_0/Overhead_com.npz`

- Size bytes: 5090
- Keys: frame_count, segmentation_colors, data, global_min, global_max

| array | shape | dtype | min | max |
|---|---|---|---:|---:|
| `frame_count` | `[]` | `int64` | 150.0 | 150.0 |
| `segmentation_colors` | `[11, 4]` | `uint8` | 3.0 | 255.0 |
| `data` | `[11, 150, 3]` | `float32` | -5.076499938964844 | 3.752500057220459 |
| `global_min` | `[3]` | `float32` | -5.076499938964844 | -1.025001049041748 |
| `global_max` | `[3]` | `float32` | 0.0 | 3.0637311935424805 |

## Decision

Este smoke confirma que Fase 12A puede usar anotaciones fisicas reales del dataset sin descargar RGB/depth completos.
El siguiente codec debe entrenar primero sobre `physics/*.npz` y usar video/segmentation como verificacion visual posterior.
