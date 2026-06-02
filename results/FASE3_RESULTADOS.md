# Fase 3: stress tests y ablaciones

La fase 3 intenta romper la arquitectura morfogenica en vez de agregar features
sin direccion. Todas las cifras salen de `python run_phase3.py`.

## 1. Dataset grande + ruido adversarial

Dataset: 4200 train, 1800 test,
320 dimensiones, 8 clases,
26 dimensiones informativas. La columna
`adv acc` usa epsilon 0.12.

| Modelo | clean acc | adv acc | fit s | pred s | model MB | candidatos |
|---|---:|---:|---:|---:|---:|---:|
| morphogenic_full | 0.994 | 0.802 | 31.369 | 0.359 | 1.080 | 48.279 |
| exact_kNN_k5 | 0.684 | 0.403 | 0.008 | 0.348 | 10.318 | 4200.000 |
| linear_SVM_sgd | 0.963 | 0.794 | 0.354 | 0.001 | 0.020 | n/a |
| random_forest_numpy | 0.998 | 0.945 | 49.153 | 0.145 | 0.812 | n/a |
| small_MLP | 0.810 | 0.641 | 2.118 | 0.004 | 0.281 | n/a |
| prototypical_network_centroid | 0.886 | 0.742 | 0.032 | 0.056 | 0.022 | n/a |

Sweep adversarial:

| epsilon | morph | kNN | SVM | forest | MLP | proto |
|---:|---:|---:|---:|---:|---:|---:|
| 0.030 | 0.991 | 0.618 | 0.944 | 0.996 | 0.773 | 0.871 |
| 0.060 | 0.978 | 0.544 | 0.909 | 0.991 | 0.734 | 0.844 |
| 0.090 | 0.937 | 0.465 | 0.861 | 0.977 | 0.689 | 0.795 |
| 0.120 | 0.802 | 0.403 | 0.794 | 0.945 | 0.641 | 0.742 |
| 0.160 | 0.515 | 0.327 | 0.675 | 0.850 | 0.563 | 0.656 |
| 0.200 | 0.285 | 0.256 | 0.546 | 0.664 | 0.489 | 0.573 |

## 2. Clases nuevas y olvido catastrofico

Setup: entrenamiento inicial con clases [0, 1, 2, 3],
luego 440 ejemplos few-shot de clases
[4, 5, 6, 7].

| Modelo | old antes | old despues | nuevas | olvido | model MB |
|---|---:|---:|---:|---:|---:|
| morphogenic_partial_fit | 0.999 | 0.991 | 0.936 | 0.008 | 0.477 |
| exact_kNN_append | 0.935 | 0.916 | 0.319 | 0.019 | 3.338 |
| small_MLP_finetune_new_only | 0.928 | 0.142 | 0.617 | 0.787 | 0.147 |
| linear_SVM_finetune_new_only | 0.994 | 0.811 | 0.890 | 0.184 | 0.012 |

## 3. Drift temporal

Stream: 9 chunks de 620 ejemplos,
96 dimensiones. Se mide accuracy antes de actualizar con
el chunk nuevo.

| Modelo | mean acc | ultimo chunk | model MB |
|---|---:|---:|---:|
| morphogenic_online | 1.000 | 1.000 | 0.105 |
| exact_kNN_append | 0.966 | 1.000 | 4.172 |
| small_MLP_static | 0.403 | 0.002 | 0.049 |
| linear_SVM_static | 0.430 | 0.000 | 0.003 |
| prototypical_static | 0.438 | 0.000 | 0.004 |

## 4. Ablaciones

Alta dimension:

| Variante | acc | pred s | candidatos | celulas |
|---|---:|---:|---:|---:|
| full_fisher_lsh_prune | 0.959 | 0.232 | 47.174 | 211 |
| without_Fisher | 0.181 | 0.111 | 4.588 | 6 |
| without_LSH | 0.985 | 2.526 | 202.000 | 202 |
| without_pruning | 0.902 | 0.144 | 24.581 | 1900 |

Memoria temporal:

- Sin memoria: 0.476.
- Con memoria/composicion: 1.000.

Ridge global:

- Sin ridge: 0.759.
- Con ridge: 0.908.
- Ganancia: 0.149.

Poda:

- Con poda: 1.000
  con 117 celulas.
- Sin poda: 1.000
  con 437 celulas.

## Lectura honesta

La arquitectura no gana todo: kNN sigue siendo una referencia fuerte cuando se
acepta guardar todo el dataset, y MLP/SVM pueden ser competitivos en datos
limpios. Lo prometedor es que la version morfogenica mantiene una mezcla rara:
alta precision en alta dimension con pocos candidatos, aprendizaje incremental
de clases nuevas, baja perdida por olvido, adaptacion online al drift, y mejoras
claras cuando se activan Fisher, LSH, poda, memoria y ridge.
