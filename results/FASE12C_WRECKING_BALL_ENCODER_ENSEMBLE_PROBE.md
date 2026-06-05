# Fase 12C - Encoder ensemble world probe

Tar: `data\physicalai_hf_cache\datasets--nvidia--PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes\snapshots\ff9b2b0a93243d84925c6c7474ee8a5bc02886cb\physics\wrecking_ball\physics-wrecking_ball-00000.tar`
Tracks: 63712
Sequences: 1000 (750 train / 250 test)
Fit/validation/test: 600 / 150 / 250 (seed 123)
Encoders: base, identity, orientation
Ensemble step: 0.25

## Metrics

| horizon | weights | MSE | MAE | best Ridge MSE | gain vs Ridge | skill vs last |
|---|---|---:|---:|---:|---:|---:|
| h1 | base:0.50, orientation:0.50 | 0.000045 | 0.001850 | 0.000072 | 0.375427 | 0.957910 |
| h5 | base:0.75, orientation:0.25 | 0.003237 | 0.023878 | 0.003577 | 0.095049 | 0.874016 |
| h15 | base:0.75, orientation:0.25 | 0.065687 | 0.128875 | 0.072202 | 0.090236 | 0.699606 |
| h30 | base:0.50, orientation:0.50 | 0.321837 | 0.333005 | 0.398862 | 0.193113 | 0.593050 |

## Lectura

Los pesos del ensemble se eligen solo en validacion y despues se aplican al test separado.
La comparacion contra Ridge usa el mejor Ridge disponible entre los encoders para evitar inflar el gain.
