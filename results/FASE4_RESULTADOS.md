# Fase 4: arquitectura morfogenica atencional

Hipotesis principal: fase 3 era fuerte aprendiendo celulas, pero fragil al
decidir con una sola celula ganadora. Fase 4 conserva el sustrato morfogenico y
cambia la inferencia a un campo local blando: selecciona las dimensiones mas
informativas con Fisher, busca en ese subespacio y deja votar a varias celulas
por distancia, pureza, importancia y radio.

## 1. Stress adversarial grande

Dataset: 4200 train, 1800 test,
320 dimensiones, 8 clases.
La columna adversarial usa epsilon 0.12.

| Modelo | clean | adv | fit s | pred s | MB | celulas | candidatos |
|---|---:|---:|---:|---:|---:|---:|---:|
| phase3_nearest_cell | 0.994 | 0.802 | 21.487 | 0.315 | 1.080 | 280 | 48.300 |
| phase4_attention_field | 1.000 | 0.988 | 24.101 | 0.315 | 0.832 | 280 | 280.000 |
| phase4_robust_field | 1.000 | 0.985 | 27.294 | 0.175 | 0.832 | 280 | 280.000 |
| random_forest_reference | 0.998 | 0.945 | 51.347 | 0.108 | 0.812 | n/a | n/a |

Sweep:

| epsilon | fase3 | fase4 | fase4 robust | random forest |
|---:|---:|---:|---:|---:|
| 0.030 | 0.991 | 1.000 | 1.000 | 0.996 |
| 0.060 | 0.978 | 1.000 | 0.999 | 0.991 |
| 0.090 | 0.937 | 0.998 | 0.996 | 0.977 |
| 0.120 | 0.802 | 0.988 | 0.985 | 0.945 |
| 0.160 | 0.515 | 0.891 | 0.910 | 0.850 |
| 0.200 | 0.285 | 0.554 | 0.697 | 0.664 |

## 2. Clases nuevas despues del entrenamiento

| Modelo | old antes | old despues | nuevas | olvido | MB |
|---|---:|---:|---:|---:|---:|
| fase3 | 0.999 | 0.991 | 0.936 | 0.008 | 0.477 |
| fase4 | 1.000 | 1.000 | 1.000 | 0.000 | 0.328 |

## 3. Drift temporal

| Modelo | mean acc | ultimo chunk | update s | MB |
|---|---:|---:|---:|---:|
| fase3 | 1.000 | 1.000 | 0.020 | 0.105 |
| fase4 | 1.000 | 1.000 | 0.019 | 0.014 |

## Resultado

La mejora principal es grande: en el benchmark adversarial epsilon 0.12, fase 4
sube de 0.802 a 0.988, y en modo robusto mantiene 0.985. En clases nuevas sube
de 0.936 a 1.000 en nuevas clases y elimina el olvido medible. Ademas, fase 4
mantiene clean 1.000 y supera al random forest en el sweep hasta epsilon 0.16,
sin dejar de ser una arquitectura de celulas locales.
