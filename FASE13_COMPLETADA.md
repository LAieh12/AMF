# Fase 13 completada - Formal AMF World Model Training

Fecha: 2026-06-05

## Estado

Phase 13 quedo implementada y ejecutada como entrenamiento formal multi-escena sobre shards reales de NVIDIA PhysicalAI, pero el gate estricto queda como:

```text
phase13_passed = false
```

No se maquilla el resultado: AMF-LTM selected mejora consistentemente al champion temporal-energy en largo plazo, pero no supera al mejor AMF previo en la mayoria de escenas/horizontes largos.

## Escenas reales usadas

Tier 1 obligatorio:

- `objects_falling`
- `dominoes`
- `wrecking_ball`

Configuracion:

- horizontes: h1/h5/h15/h30/h60/h120
- stride: 60
- ventana LTM: 20 frames
- max cells: 5000
- split por secuencia, seed 123
- calibracion solo con fit/validation

## Resultado agregado

| horizonte | mean temporal-energy | mean best previous AMF | mean AMF-LTM selected | W/T/L vs temporal | W/T/L vs previous |
|---|---:|---:|---:|---|---|
| h1 | 0.000024 | 0.000022 | 0.000024 | 2/1/0 | 2/0/1 |
| h5 | 0.001134 | 0.001096 | 0.001114 | 2/1/0 | 2/0/1 |
| h15 | 0.012833 | 0.012382 | 0.012373 | 2/1/0 | 1/2/0 |
| h30 | 0.075521 | 0.066106 | 0.070193 | 3/0/0 | 0/1/2 |
| h60 | 0.431312 | 0.370184 | 0.396456 | 3/0/0 | 2/0/1 |
| h120 | 0.889331 | 0.875799 | 0.851361 | 3/0/0 | 1/1/1 |

Long-horizon audit:

```text
vs temporal-energy: 9/9 wins-or-ties
vs best previous AMF: 3/9 wins-or-ties
```

## Lectura

AMF-LTM selected si aporta memoria larga: mejora h30/h60/h120 contra temporal-energy en todas las escenas Tier 1. Esto prueba que el residual episodico no dependia solo de `wrecking_ball`.

El problema es que el mejor AMF previo por escena/horizonte muchas veces es `amf_ensemble_12c` o `amf_residual`, y ese rival ya captura bastante del sesgo fisico. Phase 13 no lo supera de forma mayoritaria.

## Fallos identificados

- `objects_falling`: `amf_residual` y `amf_ensemble_12c` siguen siendo mas fuertes en h30/h60; LTM ayuda sobre temporal-energy, pero el residual no sabe cuando competir contra el ensemble previo.
- `dominoes`: LTM mejora temporal-energy, pero h30/h120 siguen por debajo de `amf_ensemble_12c`; falta relacion objeto-objeto/cadena causal explicita.
- `wrecking_ball`: LTM ya gana h60/h120 contra el mejor previo y empata casi h30, pero h30 aun queda apenas peor que `amf_ensemble_12c`.
- Seguridad residual: validation tiende a elegir correccion activa casi siempre (`off=0`), asi que la confianza todavia no apaga suficiente en casos malos.
- Regime detection: los labels son heuristica local; falta detector de contacto/relacion objeto-objeto real para cadenas causales y colisiones.

## Archivos

- `phase13_scene_loader.py`
- `phase13_amf_ltm_model.py`
- `phase13_world_model_train.py`
- `phase13_eval_horizons.py`
- `phase13_cross_scene_eval.py`
- `phase13_ltm_diagnostics.py`
- `run_phase13.py`
- `results/phase13_latest.json`
- `results/FASE13_WORLD_MODEL_TRAINING.md`
- `results/FASE13_CROSS_SCENE_EVAL.md`
- `results/FASE13_LTM_DIAGNOSTIC.md`

## Siguiente cuello

Phase 14 deberia atacar el punto que Phase 13 expuso:

1. selector residual que pueda elegir entre `amf_ensemble_12c` y AMF-LTM, no solo corregir temporal-energy;
2. contacto objeto-objeto explicito;
3. detector de regimen entrenado/calibrado, no solo heuristico;
4. confidence gating con objetivo de no corregir cuando el ensemble previo ya es mejor.
