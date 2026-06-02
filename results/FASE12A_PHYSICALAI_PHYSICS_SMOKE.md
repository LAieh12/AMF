# Fase 12A - PhysicalAI physics smoke

Repo: `nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes`
Downloaded file: `physics/objects_falling/physics-objects_falling-00007.tar`
Local tar: `data\physicalai_hf_cache\datasets--nvidia--PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes\snapshots\ff9b2b0a93243d84925c6c7474ee8a5bc02886cb\physics\objects_falling\physics-objects_falling-00007.tar`
Size bytes: 69283840

## NPZ sample

### `objects_falling_c8917ca2_871/Corner_com.npz`

- Size bytes: 12294
- Keys: frame_count, segmentation_colors, data, global_min, global_max

| array | shape | dtype | min | max |
|---|---|---|---:|---:|
| `frame_count` | `[]` | `int64` | 150.0 | 150.0 |
| `segmentation_colors` | `[19, 4]` | `uint8` | 3.0 | 255.0 |
| `data` | `[19, 150, 3]` | `float32` | -12.850132942199707 | 5.966682434082031 |
| `global_min` | `[3]` | `float32` | -12.850132942199707 | -0.9649797081947327 |
| `global_max` | `[3]` | `float32` | 0.0 | 2.0490121841430664 |

### `objects_falling_c8917ca2_871/Corner_rot.npz`

- Size bytes: 16213
- Keys: frame_count, segmentation_colors, data

| array | shape | dtype | min | max |
|---|---|---|---:|---:|
| `frame_count` | `[]` | `int64` | 150.0 | 150.0 |
| `segmentation_colors` | `[19, 4]` | `uint8` | 3.0 | 255.0 |
| `data` | `[19, 150, 4]` | `float32` | -0.9272323846817017 | 0.9444342255592346 |

### `objects_falling_c8917ca2_871/Corner_spin.npz`

- Size bytes: 13868
- Keys: frame_count, segmentation_colors, data, global_min, global_max

| array | shape | dtype | min | max |
|---|---|---|---:|---:|
| `frame_count` | `[]` | `int64` | 150.0 | 150.0 |
| `segmentation_colors` | `[19, 4]` | `uint8` | 3.0 | 255.0 |
| `data` | `[19, 150, 3]` | `float32` | -2131.604248046875 | 2632.371826171875 |
| `global_min` | `[3]` | `float32` | -2131.604248046875 | -766.3021850585938 |
| `global_max` | `[3]` | `float32` | 883.4943237304688 | 2632.371826171875 |

## Decision

Este smoke confirma que Fase 12A puede usar anotaciones fisicas reales del dataset sin descargar RGB/depth completos.
El siguiente codec debe entrenar primero sobre `physics/*.npz` y usar video/segmentation como verificacion visual posterior.
