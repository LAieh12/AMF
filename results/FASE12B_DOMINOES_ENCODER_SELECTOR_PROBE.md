# Fase 12B - Encoder selector world probe

Tar: `data\physicalai_hf_cache\datasets--nvidia--PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes\snapshots\ff9b2b0a93243d84925c6c7474ee8a5bc02886cb\physics\dominoes\physics-dominoes-00000.tar`
Tracks: 138720
Sequences: 1000 (750 train / 250 test)
Fit/validation/test: 600 / 150 / 250 (seed 123)
Encoders: base, identity, orientation

## Metrics

| horizon | encoder | candidate | MSE | MAE | Ridge MSE | gain vs Ridge | skill vs last |
|---|---|---|---:|---:|---:|---:|---:|
| h1 | orientation | ridge | 0.000000 | 0.000087 | 0.000000 | 0.000000 | 0.999247 |
| h5 | orientation | ridge | 0.000011 | 0.000931 | 0.000011 | 0.000000 | 0.996798 |
| h15 | orientation | ridge_amf_1.0 | 0.000335 | 0.004126 | 0.000385 | 0.129610 | 0.987345 |
| h30 | orientation | ridge_amf_1.0 | 0.004839 | 0.011646 | 0.005586 | 0.133776 | 0.929819 |

## Lectura

El encoder activo se elige en validacion entre base, identidad de slot y orientacion.
Despues se reentrena el encoder elegido sobre todo train y se evalua en test separado.
