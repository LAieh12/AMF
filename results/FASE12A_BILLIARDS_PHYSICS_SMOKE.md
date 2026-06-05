# Fase 12A - PhysicalAI physics smoke

Repo: `nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes`
Downloaded file: `physics/billiards/physics-billiards-00000.tar`
Local tar: `data\physicalai_hf_cache\datasets--nvidia--PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes\snapshots\ff9b2b0a93243d84925c6c7474ee8a5bc02886cb\physics\billiards\physics-billiards-00000.tar`
Size bytes: 176343040

## NPZ sample

### `billiards_452790ee_0/BehindCue_com.npz`

- Size bytes: 6205
- Keys: frame_count, segmentation_colors, data, global_min, global_max

| array | shape | dtype | min | max |
|---|---|---|---:|---:|
| `frame_count` | `[]` | `int64` | 150.0 | 150.0 |
| `segmentation_colors` | `[11, 4]` | `uint8` | 3.0 | 255.0 |
| `data` | `[11, 150, 3]` | `float32` | -4.588768005371094 | 1.400251030921936 |
| `global_min` | `[3]` | `float32` | -4.588768005371094 | -0.30610892176628113 |
| `global_max` | `[3]` | `float32` | 0.0 | 1.400251030921936 |

### `billiards_452790ee_0/BehindCue_rot.npz`

- Size bytes: 7403
- Keys: frame_count, segmentation_colors, data

| array | shape | dtype | min | max |
|---|---|---|---:|---:|
| `frame_count` | `[]` | `int64` | 150.0 | 150.0 |
| `segmentation_colors` | `[11, 4]` | `uint8` | 3.0 | 255.0 |
| `data` | `[11, 150, 4]` | `float32` | -0.8802263736724854 | 0.8434653878211975 |

### `billiards_452790ee_0/BehindCue_spin.npz`

- Size bytes: 6119
- Keys: frame_count, segmentation_colors, data, global_min, global_max

| array | shape | dtype | min | max |
|---|---|---|---:|---:|
| `frame_count` | `[]` | `int64` | 150.0 | 150.0 |
| `segmentation_colors` | `[11, 4]` | `uint8` | 3.0 | 255.0 |
| `data` | `[11, 150, 3]` | `float32` | -3059.329345703125 | 722.9945068359375 |
| `global_min` | `[3]` | `float32` | -3059.329345703125 | -28.997148513793945 |
| `global_max` | `[3]` | `float32` | 127.12030029296875 | 722.9945068359375 |

### `billiards_452790ee_0/BehindCue_velocity.npz`

- Size bytes: 11544
- Keys: frame_count, segmentation_colors, velocities, data, global_min, global_max

| array | shape | dtype | min | max |
|---|---|---|---:|---:|
| `frame_count` | `[]` | `int64` | 150.0 | 150.0 |
| `segmentation_colors` | `[11, 4]` | `uint8` | 3.0 | 255.0 |
| `velocities` | `[11, 150, 3]` | `float32` | -2.515937089920044 | 0.3373613953590393 |
| `data` | `[11, 150, 3]` | `float32` | -2.515937089920044 | 0.3373613953590393 |
| `global_min` | `[3]` | `float32` | -2.515937089920044 | -0.12695571780204773 |
| `global_max` | `[3]` | `float32` | 0.007285023108124733 | 0.3373613953590393 |

## Decision

Este smoke confirma que Fase 12A puede usar anotaciones fisicas reales del dataset sin descargar RGB/depth completos.
El siguiente codec debe entrenar primero sobre `physics/*.npz` y usar video/segmentation como verificacion visual posterior.
