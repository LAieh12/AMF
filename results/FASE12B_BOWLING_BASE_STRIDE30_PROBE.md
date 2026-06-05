# Fase 12B - PhysicalAI world probe

Tar: `data\physicalai_hf_cache\datasets--nvidia--PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes\snapshots\ff9b2b0a93243d84925c6c7474ee8a5bc02886cb\physics\bowling\physics-bowling-00000.tar`
Tracks: 61312
Sequences: 1000 (750 train / 250 test)
Fit/validation/test: 600 / 150 / 250 (seed 123)

## Metrics

| horizon | selected | MSE | MAE | last MSE | Ridge MSE | gain vs Ridge | skill vs last |
|---|---|---:|---:|---:|---:|---:|---:|
| h1 | cv_amf | 0.000042 | 0.000659 | 0.001531 | 0.000300 | 0.859485 | 0.972510 |
| h5 | ridge_amf_1.0 | 0.001276 | 0.007229 | 0.038532 | 0.007378 | 0.827111 | 0.966897 |
| h15 | ridge_amf_1.0 | 0.015288 | 0.024682 | 0.359850 | 0.063440 | 0.759009 | 0.957514 |
| h30 | ridge_amf_1.0 | 0.096619 | 0.090986 | 0.919237 | 0.213070 | 0.546538 | 0.894892 |

## Selector

El candidato activo se elige en validacion y luego se reentrena sobre todo el bloque train antes del test.
Si velocidad constante gana validacion pero Ridge o Ridge+AMF quedan cerca, el selector prefiere el candidato global/interpolado mas estable.
Esto evita elegir el mejor metodo mirando el test y hace mas justa la comparacion contra Ridge.

## Lectura

Este probe usa ground truth fisico real (`com` y `velocity`) del dataset NVIDIA PhysicalAI.
La arquitectura activa combina encoder fisico enriquecido, Ridge global y memorias AMF locales normalizadas para corregir residuales.
