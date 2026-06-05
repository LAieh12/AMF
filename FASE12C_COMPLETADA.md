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

## Lectura

`wrecking_ball` introduce una dinamica 12C mas dificil: pendulo, impacto y movimiento secundario fuerte. El error absoluto sube mucho frente a 12A/12B, pero el AMF ensemble sigue superando al mejor Ridge disponible en todos los horizontes.

La senal de orientacion ya no es solo diagnostica: el ensemble la usa en todos los horizontes. El nuevo cuello es largo plazo caotico:

```text
h30: 0.321837 AMF ensemble vs 0.398862 Ridge
```

La siguiente mejora debe modelar energia/constraint pendular o memoria temporal mas larga, porque la extrapolacion local empieza a sufrir en h15/h30.
