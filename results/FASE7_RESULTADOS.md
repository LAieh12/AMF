# Fase 7 - AMF7 SuperField

Objetivo: modificar la arquitectura hasta que supere a modelos clasicos fuertes.

Seeds: [0, 1, 2]
Modelos de millones de parametros incluidos: True
scikit-learn: 1.8.0
Tiempo total: 2041.9 s

## Tabla principal

| Dataset | n | d | C | AMF7 acc | Best classic | Gap | AMF7 F1 | fit s | pred s | MB | cells | experts |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| iris | 150 | 4 | 3 | 0.956 +- 0.031 | sk_gaussian_nb 0.944 | +0.011 | 0.956 | 16.77 | 0.0587 | 0.00 | 29.0 | 19.0 |
| wine | 178 | 13 | 3 | 1.000 +- 0.000 | sk_extra_trees 1.000 | +0.000 | 1.000 | 15.97 | 0.0452 | 0.00 | 51.7 | 19.0 |
| wdbc | 569 | 30 | 2 | 0.988 +- 0.011 | sk_logistic 0.988 | +0.000 | 0.988 | 24.30 | 0.0573 | 0.00 | 94.0 | 19.0 |
| madelon | 1800 | 500 | 2 | 0.833 +- 0.010 | sk_hist_gradient_boosting 0.787 | +0.046 | 0.833 | 153.82 | 0.6668 | 0.00 | 472.0 | 22.0 |
| spambase | 4601 | 57 | 2 | 0.960 +- 0.003 | sk_extra_trees 0.957 | +0.003 | 0.958 | 213.56 | 0.7645 | 0.00 | 808.3 | 19.0 |
| ionosphere | 351 | 34 | 2 | 0.952 +- 0.018 | sk_extra_trees 0.952 | +0.000 | 0.947 | 19.59 | 0.0502 | 0.00 | 63.0 | 19.0 |
| sonar | 208 | 60 | 2 | 0.881 +- 0.034 | sk_extra_trees 0.873 | +0.008 | 0.879 | 21.20 | 0.0681 | 0.00 | 83.3 | 19.0 |

## Score global

- Wins AMF7 vs mejor clasico por dataset: 4/7
- Gap promedio AMF7 - mejor clasico: +0.0097

## Lectura

AMF7 es un supercampo hibrido: conserva memorias morfogenicas locales AMF,
agrega expertos globales fuertes y aprende pesos/estrategia en validation. El
test permanece sellado hasta la evaluacion final. Si el gap sigue negativo,
todavia no se debe declarar la fase ganada.
