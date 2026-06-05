# Fase 12C - Energy diagnostic

Escena: `wrecking_ball`

Comparacion con split fijo, `stride=10`, `max_cells=20000`, `radius=0.75`, `top_k=32`.
El encoder de energia agrega desplazamiento desde ancla inicial, radio, velocidad radial/tangencial, energia cinetica/potencial aproximada y energia de spin.

| horizon | ensemble MSE | energy MSE | delta vs ensemble | energy gain vs Ridge |
|---|---:|---:|---:|---:|
| h1 | 0.000045 | 0.000049 | -0.000004 | 0.283189 |
| h5 | 0.003237 | 0.003164 | 0.000073 | 0.084819 |
| h15 | 0.065687 | 0.064402 | 0.001285 | 0.054281 |
| h30 | 0.321837 | 0.307913 | 0.013924 | 0.149360 |

## Lectura

La energia empeora ligeramente h1, pero mejora h5/h15/h30 en MSE absoluto.
En 12C el cuello ya es dinamica de largo plazo: senales de energia/constraint ayudan mas que identidad pura.
