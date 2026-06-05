# Fase 12A - PhysicalAI world probe

Tar: `data\physicalai_hf_cache\datasets--nvidia--PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes\snapshots\ff9b2b0a93243d84925c6c7474ee8a5bc02886cb\physics\billiards\physics-billiards-00000.tar`
Tracks: 40028
Sequences: 1000 (750 train / 250 test)
Fit/validation/test: 600 / 150 / 250 (seed 123)

## Metrics

| horizon | selected | MSE | MAE | last MSE | Ridge MSE | gain vs Ridge | skill vs last |
|---|---|---:|---:|---:|---:|---:|---:|
| h1 | constant_velocity | 0.000003 | 0.000119 | 0.002442 | 0.000006 | 0.519399 | 0.998893 |
| h5 | ridge | 0.000347 | 0.005801 | 0.058515 | 0.000347 | 0.000000 | 0.994063 |
| h15 | ridge_amf_0.25 | 0.006750 | 0.030308 | 0.494746 | 0.007350 | 0.081616 | 0.986357 |
| h30 | ridge_amf_0.5 | 0.051179 | 0.066987 | 1.832920 | 0.061607 | 0.169266 | 0.972078 |

## Selector

El candidato activo se elige en validacion y luego se reentrena sobre todo el bloque train antes del test.
Si velocidad constante gana validacion pero Ridge o Ridge+AMF quedan cerca, el selector prefiere el candidato global/interpolado mas estable.
Esto evita elegir el mejor metodo mirando el test y hace mas justa la comparacion contra Ridge.

## Lectura

Este probe usa ground truth fisico real (`com` y `velocity`) del dataset NVIDIA PhysicalAI.
La arquitectura activa combina encoder fisico enriquecido, Ridge global y memorias AMF locales normalizadas para corregir residuales.
