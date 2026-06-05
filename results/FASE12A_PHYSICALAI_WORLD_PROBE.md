# Fase 12A - PhysicalAI world probe

Tar: `data\physicalai_hf_cache\datasets--nvidia--PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes\snapshots\ff9b2b0a93243d84925c6c7474ee8a5bc02886cb\physics\objects_falling\physics-objects_falling-00007.tar`
Tracks: 19036
Sequences: 142 (106 train / 36 test)

## Metrics

| horizon | selected | MSE | MAE | last MSE | Ridge MSE | gain vs Ridge | skill vs last |
|---|---|---:|---:|---:|---:|---:|---:|
| h1 | ridge_amf_0.25 | 0.000118 | 0.002897 | 0.001903 | 0.000120 | 0.012644 | 0.937993 |
| h5 | ridge_amf_0.5 | 0.005272 | 0.025094 | 0.045364 | 0.006663 | 0.208659 | 0.883776 |
| h15 | cv_amf | 0.037346 | 0.064790 | 0.345907 | 0.094127 | 0.603245 | 0.892036 |
| h30 | ridge_amf_0.5 | 0.184980 | 0.190139 | 0.863386 | 0.285415 | 0.351889 | 0.785750 |

## Selector

El candidato activo se elige en validacion y luego se reentrena sobre todo el bloque train antes del test.
Esto evita elegir el mejor metodo mirando el test y hace mas justa la comparacion contra Ridge.

## Lectura

Este probe usa ground truth fisico real (`com` y `velocity`) del dataset NVIDIA PhysicalAI.
La arquitectura activa combina encoder fisico enriquecido, Ridge global y memorias AMF locales normalizadas para corregir residuales.
