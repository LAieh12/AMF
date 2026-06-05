# Fase 12B - Encoder diagnostic

Escena: `dominoes`

Comparacion con split fijo, `stride=10`, `max_cells=20000`, `radius=0.75`, `top_k=32`.

| encoder | h15 MSE | h30 MSE | h15 gain vs Ridge | h30 gain vs Ridge |
|---|---:|---:|---:|---:|
| base | 0.000342 | 0.004851 | 0.124300 | 0.141610 |
| identity | 0.000339 | 0.004809 | 0.125099 | 0.142018 |
| orientation | 0.000335 | 0.004839 | 0.129610 | 0.133776 |
| validation selector | 0.000335 | 0.004839 | 0.129610 | 0.133776 |
| validation ensemble | 0.000330 | 0.004697 | 0.144735 | 0.159212 |

## Lectura

La orientacion (`rot` + delta de quaternion) es la mejor senal validada para h15 y fue seleccionada tambien en h30 por validacion.
El ensemble validado entre identidad y orientacion resuelve el desacuerdo: mejora h15 y h30 frente a todos los encoders individuales sin elegir pesos mirando test.
