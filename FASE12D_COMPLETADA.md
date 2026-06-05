# Fase 12D completada - AMF-LTM router/retriever

Fecha: 2026-06-05

## Objetivo

Temporal-energy quedo congelado como baseline fuerte y AMF-LTM dejo de ser una concatenacion densa. La fase 12D usa memoria larga como router/retriever episodico:

- `H_fast`: velocidad, tendencia corta, aceleracion, jerk y volatilidad de energia.
- `H_event`: impacto, rebote/sign flips, cambio brusco de energia/orientacion.
- `H_regime`: constraint radial, velocidades radial/tangencial, regimen pendular/freefall/impact/rest.
- `H_workspace`: identidad de objeto, slot, color de segmentacion, ancla y orientacion.

Las memorias se escriben con ventanas de 20 frames y guardan regimen fisico, tendencia de energia, constraint radial, velocidades radial/tangencial, orientacion, impacto/cambio, objeto/slot/color, residual de sorpresa y predictor ganador local en validation.

## Disciplina anti-leakage

- Split por secuencia: 600 fit / 150 validation / 250 test.
- Predictores, ensemble estatico, alphas de residual y memorias episodicas se calibran solo con train/validation.
- `oracle_selector_test_only_invalid` usa test para elegir predictor y queda marcado solo como techo diagnostico.
- `H_event` y `H_workspace` no entran como features densas del predictor; se usan en claves de retrieval, routing y confianza.

## Resultado

Dataset real:

```text
nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes
physics/wrecking_ball/physics-wrecking_ball-00000.tar
```

Corrida 12D matched:

- stride: 30
- max cells por memoria AMF local: 8000
- router top-k: 32
- ventana episodica: 20 frames
- horizontes: h1/h5/h15/h30/h60

| horizonte | temporal-energy MSE | AMF-LTM residual MSE | gain vs temporal | oracle invalid MSE |
|---|---:|---:|---:|---:|
| h1 | 0.000029 | 0.000024 | 0.175261 | 0.000018 |
| h5 | 0.003250 | 0.002925 | 0.099936 | 0.002616 |
| h15 | 0.073433 | 0.066685 | 0.091894 | 0.060559 |
| h30 | 0.392941 | 0.347153 | 0.116527 | 0.318632 |
| h60 | 1.068345 | 0.967422 | 0.094467 | 0.725167 |

## Lectura

12D supera a temporal-energy en todos los horizontes medidos, incluyendo el objetivo principal h30+:

```text
h30: 0.392941 -> 0.347153  (+11.65%)
h60: 1.068345 -> 0.967422  (+9.45%)
```

El router puro de predictores quedo calibrado de forma conservadora: validation apago la mezcla cruda y mantuvo temporal-energy cuando el selector no era confiable. El avance real vino del residual episodico recuperado por AMF-LTM. Esto es importante: la memoria larga si esta aportando, pero como correccion local sobre el champion, no como reemplazo global del predictor.

El oracle invalido todavia muestra techo disponible en h30/h60, asi que el siguiente cuello no es falta de informacion: es aprender un router de predictores mas estable, probablemente con estimacion de incertidumbre por regimen y clipping/residualizacion de candidatos antes de mezclar.
