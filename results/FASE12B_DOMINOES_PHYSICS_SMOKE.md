# Fase 12B - PhysicalAI physics smoke

Repo: `nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes`
Downloaded file: `physics/dominoes/physics-dominoes-00000.tar`
Local tar: `data\physicalai_hf_cache\datasets--nvidia--PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes\snapshots\ff9b2b0a93243d84925c6c7474ee8a5bc02886cb\physics\dominoes\physics-dominoes-00000.tar`
Size bytes: 449157120

## NPZ sample

### `dominoes_21b8d802_0/EndGround_com.npz`

- Size bytes: 13006
- Keys: frame_count, segmentation_colors, data, global_min, global_max

| array | shape | dtype | min | max |
|---|---|---|---:|---:|
| `frame_count` | `[]` | `int64` | 150.0 | 150.0 |
| `segmentation_colors` | `[32, 4]` | `uint8` | 2.0 | 255.0 |
| `data` | `[32, 150, 3]` | `float32` | -0.6982359290122986 | 0.2202133685350418 |
| `global_min` | `[3]` | `float32` | -0.6982359290122986 | -0.02520778588950634 |
| `global_max` | `[3]` | `float32` | 0.0 | 0.2202133685350418 |

### `dominoes_21b8d802_0/EndGround_rot.npz`

- Size bytes: 17123
- Keys: frame_count, segmentation_colors, data

| array | shape | dtype | min | max |
|---|---|---|---:|---:|
| `frame_count` | `[]` | `int64` | 150.0 | 150.0 |
| `segmentation_colors` | `[32, 4]` | `uint8` | 2.0 | 255.0 |
| `data` | `[32, 150, 4]` | `float32` | -0.7432217597961426 | 0.7329170107841492 |

### `dominoes_21b8d802_0/EndGround_spin.npz`

- Size bytes: 17422
- Keys: frame_count, segmentation_colors, data, global_min, global_max

| array | shape | dtype | min | max |
|---|---|---|---:|---:|
| `frame_count` | `[]` | `int64` | 150.0 | 150.0 |
| `segmentation_colors` | `[32, 4]` | `uint8` | 2.0 | 255.0 |
| `data` | `[32, 150, 3]` | `float32` | -423.8628234863281 | 1047.1810302734375 |
| `global_min` | `[3]` | `float32` | -423.8628234863281 | -200.04129028320312 |
| `global_max` | `[3]` | `float32` | 359.7142333984375 | 1047.1810302734375 |

### `dominoes_21b8d802_0/EndGround_velocity.npz`

- Size bytes: 33891
- Keys: frame_count, segmentation_colors, velocities, data, global_min, global_max

| array | shape | dtype | min | max |
|---|---|---|---:|---:|
| `frame_count` | `[]` | `int64` | 150.0 | 150.0 |
| `segmentation_colors` | `[32, 4]` | `uint8` | 2.0 | 255.0 |
| `velocities` | `[32, 150, 3]` | `float32` | -0.5468375086784363 | 0.514182448387146 |
| `data` | `[32, 150, 3]` | `float32` | -0.5468375086784363 | 0.514182448387146 |
| `global_min` | `[3]` | `float32` | -0.5468375086784363 | -0.04503101110458374 |
| `global_max` | `[3]` | `float32` | 0.10560987889766693 | 0.514182448387146 |

### `dominoes_21b8d802_0/StartGround_com.npz`

- Size bytes: 13217
- Keys: frame_count, segmentation_colors, data, global_min, global_max

| array | shape | dtype | min | max |
|---|---|---|---:|---:|
| `frame_count` | `[]` | `int64` | 150.0 | 150.0 |
| `segmentation_colors` | `[32, 4]` | `uint8` | 2.0 | 255.0 |
| `data` | `[32, 150, 3]` | `float32` | -0.45978251099586487 | 0.319227010011673 |
| `global_min` | `[3]` | `float32` | -0.45978251099586487 | -0.02000436559319496 |
| `global_max` | `[3]` | `float32` | 0.0 | 0.319227010011673 |

## Decision

Este smoke confirma que Fase 12A puede usar anotaciones fisicas reales del dataset sin descargar RGB/depth completos.
El siguiente codec debe entrenar primero sobre `physics/*.npz` y usar video/segmentation como verificacion visual posterior.
