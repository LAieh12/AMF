# Fase 12B - PhysicalAI world probe

Tar: `data\physicalai_hf_cache\datasets--nvidia--PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes\snapshots\ff9b2b0a93243d84925c6c7474ee8a5bc02886cb\physics\dominoes\physics-dominoes-00000.tar`
Tracks: 138720
Sequences: 1000 (750 train / 250 test)
Fit/validation/test: 600 / 150 / 250 (seed 123)

## Metrics

| horizon | selected | MSE | MAE | last MSE | Ridge MSE | gain vs Ridge | skill vs last |
|---|---|---:|---:|---:|---:|---:|---:|
| h1 | ridge | 0.000000 | 0.000087 | 0.000125 | 0.000000 | 0.000000 | 0.999246 |
| h5 | ridge | 0.000011 | 0.000928 | 0.003395 | 0.000011 | 0.000000 | 0.996785 |
| h15 | ridge_amf_0.5 | 0.000339 | 0.004564 | 0.026507 | 0.000387 | 0.125099 | 0.987228 |
| h30 | ridge_amf_0.5 | 0.004809 | 0.014274 | 0.068945 | 0.005605 | 0.142018 | 0.930254 |

## Selector

El candidato activo se elige en validacion y luego se reentrena sobre todo el bloque train antes del test.
Si velocidad constante gana validacion pero Ridge o Ridge+AMF quedan cerca, el selector prefiere el candidato global/interpolado mas estable.
Esto evita elegir el mejor metodo mirando el test y hace mas justa la comparacion contra Ridge.

## Lectura

Este probe usa ground truth fisico real (`com` y `velocity`) del dataset NVIDIA PhysicalAI.
La arquitectura activa combina encoder fisico enriquecido, identidad de slot desde segmentation color, Ridge global y memorias AMF locales normalizadas para corregir residuales.
