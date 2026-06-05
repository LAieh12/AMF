# Fase 12B - Slot identity diagnostic

Escena: `dominoes`

Comparacion con el mismo split, `stride=10`, `max_cells=20000`, `radius=0.75`, `top_k=32`.
El encoder de identidad agrega `segmentation_colors`, codigo estable de objeto y slot index al estado fisico.

| horizon | base AMF MSE | identity AMF MSE | delta vs base | identity gain vs Ridge |
|---|---:|---:|---:|---:|
| h1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| h5 | 0.000011 | 0.000011 | 0.000000 | 0.000000 |
| h15 | 0.000342 | 0.000339 | 0.000003 | 0.125099 |
| h30 | 0.004851 | 0.004809 | 0.000042 | 0.142018 |

## Lectura

La identidad de slot ayuda poco pero de forma consistente en la cadena causal de `dominoes`, especialmente h15/h30.
La senal de `segmentation_colors` es util, pero todavia no reemplaza un encoder visual de mascara/contacto persistente.
