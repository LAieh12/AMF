# Fase 12B - PhysicalAI world probe

Tar: `data\physicalai_hf_cache\datasets--nvidia--PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes\snapshots\ff9b2b0a93243d84925c6c7474ee8a5bc02886cb\physics\bowling\physics-bowling-00000.tar`
Tracks: 61312
Sequences: 1000 (750 train / 250 test)
Fit/validation/test: 600 / 150 / 250 (seed 123)

## Metrics

| horizon | selected | MSE | MAE | last MSE | Ridge MSE | gain vs Ridge | skill vs last |
|---|---|---:|---:|---:|---:|---:|---:|
| h1 | cv_amf | 0.000053 | 0.000565 | 0.001531 | 0.000269 | 0.803817 | 0.965537 |
| h5 | cv_amf | 0.001630 | 0.005103 | 0.038532 | 0.006579 | 0.752241 | 0.957699 |
| h15 | cv_amf | 0.022583 | 0.023781 | 0.359850 | 0.056979 | 0.603660 | 0.937243 |
| h30 | ridge_amf_1.0 | 0.092665 | 0.107283 | 0.919237 | 0.182750 | 0.492941 | 0.899193 |

## Selector

El candidato activo se elige en validacion y luego se reentrena sobre todo el bloque train antes del test.
Si velocidad constante gana validacion pero Ridge o Ridge+AMF quedan cerca, el selector prefiere el candidato global/interpolado mas estable.
Esto evita elegir el mejor metodo mirando el test y hace mas justa la comparacion contra Ridge.

## Lectura

Este probe usa ground truth fisico real (`com` y `velocity`) del dataset NVIDIA PhysicalAI.
La arquitectura activa combina encoder fisico enriquecido, contexto de contacto multi-objeto, Ridge global y memorias AMF locales normalizadas para corregir residuales.
