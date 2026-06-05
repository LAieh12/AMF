# Fase 12C - Temporal and AMF-LTM diagnostic

Escena: `wrecking_ball`

Comparacion principal con split fijo. `Temporal-energy` usa `stride=10`, `max_cells=20000`.
`AMF-LTM diagnostic` usa `stride=30`, `max_cells=8000`, asi que no se reporta como reemplazo directo del mejor modelo.

| model | h1 MSE | h5 MSE | h15 MSE | h30 MSE | notes |
|---|---:|---:|---:|---:|---|
| energy | 0.000049 | 0.003164 | 0.064402 | 0.307913 | energia/constraint radial |
| temporal-energy | 0.000029 | 0.002655 | 0.060144 | 0.306422 | mejor 12C actual |
| AMF-LTM diagnostic | 0.000034 | 0.003241 | 0.066412 | 0.358970 | cuatro niveles, compuertas fijas |

## Lectura

La historia corta mejora todos los horizontes frente a energy, incluyendo h30.
AMF-LTM ya existe como estructura de cuatro niveles, pero la version actual necesita recuperacion/escritura selectiva de memoria; con compuertas fijas aun mete ruido en h15/h30.
