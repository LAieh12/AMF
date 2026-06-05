# Fase 12C - PhysicalAI world probe

Tar: `data\physicalai_hf_cache\datasets--nvidia--PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes\snapshots\ff9b2b0a93243d84925c6c7474ee8a5bc02886cb\physics\wrecking_ball\physics-wrecking_ball-00000.tar`
Tracks: 63712
Sequences: 1000 (750 train / 250 test)
Fit/validation/test: 600 / 150 / 250 (seed 123)

## Metrics

| horizon | selected | MSE | MAE | last MSE | Ridge MSE | gain vs Ridge | skill vs last |
|---|---|---:|---:|---:|---:|---:|---:|
| h1 | cv_amf | 0.000049 | 0.001311 | 0.001069 | 0.000069 | 0.283189 | 0.953801 |
| h5 | ridge_amf_0.25 | 0.003164 | 0.024955 | 0.025693 | 0.003457 | 0.084819 | 0.876852 |
| h15 | ridge_amf_0.25 | 0.064402 | 0.135618 | 0.218669 | 0.068098 | 0.054281 | 0.705484 |
| h30 | ridge_amf_0.25 | 0.307913 | 0.341232 | 0.790851 | 0.361978 | 0.149360 | 0.610656 |

## Selector

El candidato activo se elige en validacion y luego se reentrena sobre todo el bloque train antes del test.
Si velocidad constante gana validacion pero Ridge o Ridge+AMF quedan cerca, el selector prefiere el candidato global/interpolado mas estable.
Esto evita elegir el mejor metodo mirando el test y hace mas justa la comparacion contra Ridge.

## Lectura

Este probe usa ground truth fisico real (`com` y `velocity`) del dataset NVIDIA PhysicalAI.
La arquitectura activa combina encoder fisico enriquecido, orientacion, identidad, energia aproximada, constraint radial y memorias AMF locales normalizadas para corregir residuales.
