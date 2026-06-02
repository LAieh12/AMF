# Fase 12A - PhysicalAI world probe

Tar: `data\physicalai_hf_cache\datasets--nvidia--PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes\snapshots\ff9b2b0a93243d84925c6c7474ee8a5bc02886cb\physics\objects_falling\physics-objects_falling-00007.tar`
Tracks: 19036
Sequences: 142 (106 train / 36 test)

## Metrics

| horizon | candidate | MSE | MAE | last MSE | skill vs last |
|---|---|---:|---:|---:|---:|
| h1 | amf_residual | 0.000109 | 0.002280 | 0.001903 | 0.942582 |
| h5 | amf_residual | 0.003526 | 0.016348 | 0.045364 | 0.922268 |
| h15 | amf_residual | 0.069023 | 0.089128 | 0.345907 | 0.800456 |
| h30 | ridge | 0.314095 | 0.263486 | 0.863386 | 0.636205 |

## Lectura

Este probe usa ground truth fisico real (`com` y `velocity`) del dataset NVIDIA PhysicalAI.
La primera meta de 12A no es renderizar RGB: es que Never prediga estados fisicos multi-slot con memoria AMF y baselines claros.
