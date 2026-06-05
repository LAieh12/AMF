# Fase 12B - PhysicalAI world probe

Tar: `data\physicalai_hf_cache\datasets--nvidia--PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes\snapshots\ff9b2b0a93243d84925c6c7474ee8a5bc02886cb\physics\bowling\physics-bowling-00000.tar`
Tracks: 61312
Sequences: 1000 (750 train / 250 test)
Fit/validation/test: 600 / 150 / 250 (seed 123)

## Metrics

| horizon | selected | MSE | MAE | last MSE | Ridge MSE | gain vs Ridge | skill vs last |
|---|---|---:|---:|---:|---:|---:|---:|
| h1 | ridge_amf_0.5 | 0.000063 | 0.002204 | 0.001735 | 0.000102 | 0.381367 | 0.963797 |
| h5 | ridge_amf_0.25 | 0.002284 | 0.013533 | 0.041012 | 0.002838 | 0.195091 | 0.944300 |
| h15 | ridge_amf_0.5 | 0.024849 | 0.059195 | 0.332220 | 0.030110 | 0.174738 | 0.925204 |
| h30 | ridge_amf_1.0 | 0.097678 | 0.085598 | 1.141135 | 0.135595 | 0.279631 | 0.914402 |

## Selector

El candidato activo se elige en validacion y luego se reentrena sobre todo el bloque train antes del test.
Si velocidad constante gana validacion pero Ridge o Ridge+AMF quedan cerca, el selector prefiere el candidato global/interpolado mas estable.
Esto evita elegir el mejor metodo mirando el test y hace mas justa la comparacion contra Ridge.

## Lectura

Este probe usa ground truth fisico real (`com` y `velocity`) del dataset NVIDIA PhysicalAI.
La arquitectura activa combina encoder fisico enriquecido, Ridge global y memorias AMF locales normalizadas para corregir residuales.
