# Fase 12C - AMF-LTM diagnostic

Tar: `data\physicalai_hf_cache\datasets--nvidia--PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes\snapshots\ff9b2b0a93243d84925c6c7474ee8a5bc02886cb\physics\wrecking_ball\physics-wrecking_ball-00000.tar`
Tracks: 63712
Sequences: 1000 (750 train / 250 test)
Fit/validation/test: 600 / 150 / 250 (seed 123)

## Metrics

| horizon | selected | MSE | MAE | last MSE | Ridge MSE | gain vs Ridge | skill vs last |
|---|---|---:|---:|---:|---:|---:|---:|
| h1 | ridge_amf_1.0 | 0.000034 | 0.001799 | 0.001022 | 0.000049 | 0.294215 | 0.966274 |
| h5 | ridge_amf_0.25 | 0.003241 | 0.027620 | 0.022493 | 0.003367 | 0.037399 | 0.855923 |
| h15 | ridge_amf_0.25 | 0.066412 | 0.149247 | 0.180429 | 0.069185 | 0.040085 | 0.631920 |
| h30 | ridge_amf_0.25 | 0.358970 | 0.390712 | 0.798582 | 0.385675 | 0.069242 | 0.550491 |

## Selector

El candidato activo se elige en validacion y luego se reentrena sobre todo el bloque train antes del test.
Si velocidad constante gana validacion pero Ridge o Ridge+AMF quedan cerca, el selector prefiere el candidato global/interpolado mas estable.
Esto evita elegir el mejor metodo mirando el test y hace mas justa la comparacion contra Ridge.

## Lectura

Este probe usa ground truth fisico real (`com` y `velocity`) del dataset NVIDIA PhysicalAI.
La arquitectura activa combina AMF-LTM con H_fast, H_event, H_regime, H_workspace y memorias AMF locales normalizadas para corregir residuales.

## Nota

Este diagnostico usa `stride=30` y `max_cells=8000`.
AMF-LTM esta implementado con cuatro niveles:

- `H_fast`: historia corta, aceleracion, jerk, tendencia de energia.
- `H_event`: contacto cercano, closing speed, rebote/impacto.
- `H_regime`: pendulo, caida, impacto, reposo, energia.
- `H_workspace`: identidad, color de segmentacion, slots, centro/spread local.

Resultado: LTM con compuertas mejora al LTM crudo, pero no supera todavia al encoder temporal-energy.
La leccion es que los niveles largos deben recuperar/escribir memorias de forma selectiva, no concatenarse como features densas equivalentes.
