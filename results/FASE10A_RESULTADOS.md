# Fase 10a - AMF world model sintetico

Objetivo: pretraining sintetico en Python puro para calentar un world model AMF.

Flujo:

```text
simulador juguete -> (S_t, accion, S_t+1) -> celdas AMF de dinamica -> export caliente
```

Simulador: estado `[x, y, vx, vy]`, accion `[ax, ay]`, gravedad, drag, viento
suave y rebote contra paredes.

Datos:

- trayectorias: 900
- pasos por trayectoria: 70
- train transitions: 53550
- test transitions: 9450

Reglas: no LLM = True, no decoder denso =
True, no backprop = True.

## Resultados

| system | one-step MSE | pos RMSE | vel RMSE | bounce MSE | rollout MSE | rollout final MSE | ms/trans | fit s | memory/cells |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| amf_dynamics_world_model | 0.017430 | 0.014594 | 0.186139 | 0.085049 | 0.287559 | 0.558269 | 0.347663 | 5.596762 | 0.686646 MB / 9000 |
| static_state | 0.062986 | 0.058774 | 0.350024 | 0.295767 | 0.512606 | 0.487352 | 0.000002 | 0.000001 | - |
| constant_velocity | 0.061372 | 0.015090 | 0.350024 | 0.294774 | 0.697889 | 0.816304 | 0.005381 | 0.000001 | - |
| ridge_linear_dynamics | 0.047347 | 0.015037 | 0.307358 | 0.196999 | 0.348361 | 0.604533 | 0.008925 | 0.610923 | - |

## Export AMF calentado

- NPZ: `data\phase10a_warm_amf.npz`
- metadata: `data\phase10a_warm_amf.json`
- cells: 9000
- arrays memory MB: 0.686646
- reload max abs diff: 0.0000000000

## Metaplasticidad

- guarda delta, no estado completo: True
- raw cells antes de regular: 26375
- celdas podadas por bajo uso: 16090
- celdas fusionadas por similitud: 37
- celdas finales: 9000
- identidad congelada: True
- identidad learning rate: 0.0
- probe celda existente: `explained_by_existing_cell`
- probe ruido inicial: `buffered_possible_noise`
- probe novedad confirmada: `created_confirmed_novelty`
- probe passed: True

## Lectura

El AMF aprende deltas locales `S_t+1 - S_t` como celdas de dinamica sobre el
espacio `(estado, accion)`. En prediccion activa las celdas cercanas y mezcla
sus deltas por resonancia local. La metaplasticidad evita aprendizaje infinito
bruto: no crea celda si una existente explica bien, fusiona celdas parecidas,
poda celdas poco usadas, congela identidad y exige confirmacion antes de tratar
ruido como novedad. Esto lo deja listo para Fase 10b: cargar el NPZ caliente y
usarlo como world model inicial.
