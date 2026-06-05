# Fase 12B - Encoder ensemble world probe

Tar: `data\physicalai_hf_cache\datasets--nvidia--PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes\snapshots\ff9b2b0a93243d84925c6c7474ee8a5bc02886cb\physics\dominoes\physics-dominoes-00000.tar`
Tracks: 138720
Sequences: 1000 (750 train / 250 test)
Fit/validation/test: 600 / 150 / 250 (seed 123)
Encoders: base, identity, orientation
Ensemble step: 0.25

## Metrics

| horizon | weights | MSE | MAE | best Ridge MSE | gain vs Ridge | skill vs last |
|---|---|---:|---:|---:|---:|---:|
| h1 | identity:0.25, orientation:0.75 | 0.000000 | 0.000087 | 0.000000 | -0.000127 | 0.999247 |
| h5 | identity:0.25, orientation:0.75 | 0.000011 | 0.000929 | 0.000011 | -0.000340 | 0.996797 |
| h15 | identity:0.25, orientation:0.75 | 0.000330 | 0.004111 | 0.000385 | 0.144735 | 0.987565 |
| h30 | identity:0.50, orientation:0.50 | 0.004697 | 0.012474 | 0.005586 | 0.159212 | 0.931880 |

## Lectura

Los pesos del ensemble se eligen solo en validacion y despues se aplican al test separado.
La comparacion contra Ridge usa el mejor Ridge disponible entre los encoders para evitar inflar el gain.
