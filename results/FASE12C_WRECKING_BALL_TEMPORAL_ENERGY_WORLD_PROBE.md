# Fase 12C - PhysicalAI world probe

Tar: `data\physicalai_hf_cache\datasets--nvidia--PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes\snapshots\ff9b2b0a93243d84925c6c7474ee8a5bc02886cb\physics\wrecking_ball\physics-wrecking_ball-00000.tar`
Tracks: 63712
Sequences: 1000 (750 train / 250 test)
Fit/validation/test: 600 / 150 / 250 (seed 123)

## Metrics

| horizon | selected | MSE | MAE | last MSE | Ridge MSE | gain vs Ridge | skill vs last |
|---|---|---:|---:|---:|---:|---:|---:|
| h1 | ridge_amf_1.0 | 0.000029 | 0.001483 | 0.001069 | 0.000044 | 0.350149 | 0.973066 |
| h5 | ridge_amf_0.25 | 0.002655 | 0.022465 | 0.025693 | 0.002822 | 0.059193 | 0.896658 |
| h15 | ridge_amf_0.25 | 0.060144 | 0.130443 | 0.218669 | 0.062119 | 0.031791 | 0.724952 |
| h30 | ridge_amf_0.25 | 0.306422 | 0.340186 | 0.790851 | 0.340345 | 0.099673 | 0.612542 |

## Selector

El candidato activo se elige en validacion y luego se reentrena sobre todo el bloque train antes del test.
Si velocidad constante gana validacion pero Ridge o Ridge+AMF quedan cerca, el selector prefiere el candidato global/interpolado mas estable.
Esto evita elegir el mejor metodo mirando el test y hace mas justa la comparacion contra Ridge.

## Lectura

Este probe usa ground truth fisico real (`com` y `velocity`) del dataset NVIDIA PhysicalAI.
La arquitectura activa combina energia/constraint, historia temporal corta, aceleracion, Ridge global y memorias AMF locales normalizadas para corregir residuales.
