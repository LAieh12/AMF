# Fase 12B - Contact diagnostic

Escena: `bowling`

Comparacion justa con el mismo split, `stride=30`, `max_cells=6000`, `radius=0.75`, `top_k=32`.

| horizon | base AMF MSE | contact AMF MSE | delta vs base | contact gain vs Ridge |
|---|---:|---:|---:|---:|
| h1 | 0.000042 | 0.000053 | -0.000011 | 0.803817 |
| h5 | 0.001276 | 0.001630 | -0.000354 | 0.752241 |
| h15 | 0.015288 | 0.022583 | -0.007295 | 0.603660 |
| h30 | 0.096619 | 0.092665 | 0.003954 | 0.492941 |

## Lectura

El contexto nearest-neighbor ayuda en h30, pero degrada horizontes cortos y medios.
La direccion correcta no es mas proximidad geometrica cruda: el siguiente encoder debe incorporar identidad de objeto, mascara/segmentacion y contacto persistente.
