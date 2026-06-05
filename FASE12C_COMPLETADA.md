# Fase 12C completada - PhysicalAI wrecking ball world probe

Fecha: 2026-06-05

## Dataset

Se amplio el entrenamiento del AMF world model a una escena 12C real:

```text
nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes
physics/wrecking_ball/physics-wrecking_ball-00000.tar
```

Uso validado:

- descarga real acotada del shard fisico `wrecking_ball`;
- inspeccion de `com`, `velocity`, `spin`, `rot` y `segmentation_colors`;
- 63,712 tracks fisicos;
- 1,000 secuencias con split por secuencia;
- fit/validation/test: 600 / 150 / 250;
- ensemble de encoders elegido solo en validacion.

## Resultado

| horizonte | pesos seleccionados | MSE | best Ridge MSE | gain vs Ridge | skill vs last |
|---|---|---:|---:|---:|---:|
| h1 | base 0.50 + orientation 0.50 | 0.000045 | 0.000072 | 0.375427 | 0.957910 |
| h5 | base 0.75 + orientation 0.25 | 0.003237 | 0.003577 | 0.095049 | 0.874016 |
| h15 | base 0.75 + orientation 0.25 | 0.065687 | 0.072202 | 0.090236 | 0.699606 |
| h30 | base 0.50 + orientation 0.50 | 0.321837 | 0.398862 | 0.193113 | 0.593050 |

## Energia

Se agrego `phase12c_energy_world_probe.py`, con desplazamiento desde ancla inicial, radio, velocidad radial/tangencial, energia cinetica/potencial aproximada y energia de spin.

Comparacion con el ensemble anterior:

| horizonte | ensemble MSE | energy MSE | delta vs ensemble |
|---|---:|---:|---:|
| h1 | 0.000045 | 0.000049 | -0.000004 |
| h5 | 0.003237 | 0.003164 | 0.000073 |
| h15 | 0.065687 | 0.064402 | 0.001285 |
| h30 | 0.321837 | 0.307913 | 0.013924 |

## Temporal energy y AMF-LTM

Se agregaron:

- `phase12c_temporal_energy_world_probe.py`: historia corta, aceleracion, jerk y tendencia de energia.
- `phase12c_ltm_world_probe.py`: AMF-LTM con `H_fast`, `H_event`, `H_regime`, `H_workspace`.

Comparacion:

| modelo | h1 MSE | h5 MSE | h15 MSE | h30 MSE |
|---|---:|---:|---:|---:|
| energy | 0.000049 | 0.003164 | 0.064402 | 0.307913 |
| temporal-energy | 0.000029 | 0.002655 | 0.060144 | 0.306422 |
| AMF-LTM diagnostic | 0.000034 | 0.003241 | 0.066412 | 0.358970 |

## Lectura

`wrecking_ball` introduce una dinamica 12C mas dificil: pendulo, impacto y movimiento secundario fuerte. El error absoluto sube mucho frente a 12A/12B, pero el AMF ensemble sigue superando al mejor Ridge disponible en todos los horizontes.

La senal de orientacion ya no es solo diagnostica: el ensemble la usa en todos los horizontes. La energia mejora el largo plazo en MSE absoluto. El nuevo cuello es largo plazo caotico:

```text
h30: 0.306422 AMF temporal-energy vs 0.340345 Ridge
```

La siguiente mejora debe hacer que AMF-LTM recupere/escriba memorias de evento y regimen de forma selectiva. La concatenacion densa de los cuatro niveles ya no basta.
