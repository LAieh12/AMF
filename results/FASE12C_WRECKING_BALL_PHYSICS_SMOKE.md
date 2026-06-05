# Fase 12C - PhysicalAI physics smoke

Repo: `nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes`
Downloaded file: `physics/wrecking_ball/physics-wrecking_ball-00000.tar`
Local tar: `data\physicalai_hf_cache\datasets--nvidia--PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes\snapshots\ff9b2b0a93243d84925c6c7474ee8a5bc02886cb\physics\wrecking_ball\physics-wrecking_ball-00000.tar`
Size bytes: 350085120

## NPZ sample

### `wrecking_ball_cad96f3f_0/Corner_com.npz`

- Size bytes: 18103
- Keys: frame_count, segmentation_colors, data, global_min, global_max

| array | shape | dtype | min | max |
|---|---|---|---:|---:|
| `frame_count` | `[]` | `int64` | 150.0 | 150.0 |
| `segmentation_colors` | `[22, 4]` | `uint8` | 3.0 | 255.0 |
| `data` | `[22, 150, 3]` | `float32` | -15.869248390197754 | 5.195318222045898 |
| `global_min` | `[3]` | `float32` | -15.769497871398926 | -2.381088972091675 |
| `global_max` | `[3]` | `float32` | 0.0 | 4.191408157348633 |

### `wrecking_ball_cad96f3f_0/Corner_rot.npz`

- Size bytes: 18977
- Keys: frame_count, segmentation_colors, data

| array | shape | dtype | min | max |
|---|---|---|---:|---:|
| `frame_count` | `[]` | `int64` | 150.0 | 150.0 |
| `segmentation_colors` | `[22, 4]` | `uint8` | 3.0 | 255.0 |
| `data` | `[22, 150, 4]` | `float32` | -0.9717042446136475 | 0.9946826100349426 |

### `wrecking_ball_cad96f3f_0/Corner_spin.npz`

- Size bytes: 24005
- Keys: frame_count, segmentation_colors, data, global_min, global_max

| array | shape | dtype | min | max |
|---|---|---|---:|---:|
| `frame_count` | `[]` | `int64` | 150.0 | 150.0 |
| `segmentation_colors` | `[22, 4]` | `uint8` | 3.0 | 255.0 |
| `data` | `[22, 150, 3]` | `float32` | -960.0807495117188 | 1345.6773681640625 |
| `global_min` | `[3]` | `float32` | -960.0807495117188 | -503.1648254394531 |
| `global_max` | `[3]` | `float32` | 531.3043212890625 | 1345.6773681640625 |

### `wrecking_ball_cad96f3f_0/Corner_velocity.npz`

- Size bytes: 46058
- Keys: frame_count, segmentation_colors, velocities, data, global_min, global_max

| array | shape | dtype | min | max |
|---|---|---|---:|---:|
| `frame_count` | `[]` | `int64` | 150.0 | 150.0 |
| `segmentation_colors` | `[22, 4]` | `uint8` | 3.0 | 255.0 |
| `velocities` | `[22, 150, 3]` | `float32` | -9.946737289428711 | 6.079237461090088 |
| `data` | `[22, 150, 3]` | `float32` | -9.946737289428711 | 6.079237461090088 |
| `global_min` | `[3]` | `float32` | -9.946737289428711 | -2.851130962371826 |
| `global_max` | `[3]` | `float32` | 2.5634193420410156 | 6.079237461090088 |

### `wrecking_ball_cad96f3f_0/Front_com.npz`

- Size bytes: 17884
- Keys: frame_count, segmentation_colors, data, global_min, global_max

| array | shape | dtype | min | max |
|---|---|---|---:|---:|
| `frame_count` | `[]` | `int64` | 150.0 | 150.0 |
| `segmentation_colors` | `[22, 4]` | `uint8` | 3.0 | 255.0 |
| `data` | `[22, 150, 3]` | `float32` | -15.702860832214355 | 5.268680095672607 |
| `global_min` | `[3]` | `float32` | -15.702860832214355 | -2.107199192047119 |
| `global_max` | `[3]` | `float32` | 0.0 | 4.753413200378418 |

## Decision

Este smoke confirma que Fase 12C puede usar anotaciones fisicas reales del dataset sin descargar RGB/depth completos.
El siguiente codec debe entrenar primero sobre `physics/*.npz` y usar video/segmentation como verificacion visual posterior.
