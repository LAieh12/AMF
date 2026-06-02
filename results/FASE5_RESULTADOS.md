# Fase 5 - Generalizacion y anatomia del campo morfogenico atencional

Hipotesis: AMF5 should not win every accuracy contest, but should occupy a useful research niche: strong high-dimensional robustness, compact local memory, few-shot incremental learning, low forgetting, and interpretable anatomy.

Seeds: [0, 1, 2, 3, 4]
Tiempo total: 530.8 s

## Datasets reales

- iris: AMF5 acc 0.920 +- 0.030, cells 7.600 +- 2.332; top modelos: rbf_svm_like 0.982, extra_trees 0.978, gradient_boosting_stumps 0.973, hist_gradient_boosting_stumps 0.973
- wine: AMF5 acc 0.951 +- 0.009, cells 11.800 +- 1.470; top modelos: gaussian_nb 0.981, rbf_svm_like 0.977, extra_trees 0.977, online_passive_aggressive 0.970
- wdbc: AMF5 acc 0.936 +- 0.022, cells 24.200 +- 3.429; top modelos: sgd_classifier 0.981, rbf_svm_like 0.971, weighted_kNN 0.958, extra_trees 0.954
- optdigits: AMF5 acc 0.947 +- 0.008, cells 221.600 +- 6.468; top modelos: weighted_kNN 0.977, sgd_classifier 0.963, extra_trees 0.959, rbf_svm_like 0.953
- madelon: AMF5 acc 0.631 +- 0.021, cells 133.800 +- 7.139; top modelos: AMF5_full 0.631, hist_gradient_boosting_stumps 0.602, gradient_boosting_stumps 0.596, gaussian_nb 0.580

## Ataques no basados en prototipos

- madelon:
  - clean: acc 0.631 +- 0.021
  - feature_dropout: acc 0.620 +- 0.013
  - feature_swap_top: acc 0.486 +- 0.015
  - gaussian_noise: acc 0.624 +- 0.025
  - generic_boundary_blackbox_sample: acc 0.478 +- 0.021
  - random_direction: acc 0.607 +- 0.025
  - top_fisher_perturbation: acc 0.601 +- 0.024
- optdigits:
  - clean: acc 0.947 +- 0.008
  - feature_dropout: acc 0.889 +- 0.017
  - feature_swap_top: acc 0.340 +- 0.017
  - gaussian_noise: acc 0.921 +- 0.014
  - generic_boundary_blackbox_sample: acc 0.369 +- 0.040
  - random_direction: acc 0.878 +- 0.018
  - top_fisher_perturbation: acc 0.867 +- 0.022

## Anatomia del campo atencional en Madelon

| Variante | clean acc | top-Fisher attack acc | votos | MB |
|---|---:|---:|---:|---:|
| AMF5_full | 0.631 +- 0.021 | 0.603 +- 0.025 | 8.000 +- 0.000 | 0.569 +- 0.029 |
| vote_k=1 | 0.614 +- 0.006 | 0.590 +- 0.026 | 1.000 +- 0.000 | 0.569 +- 0.029 |
| vote_k=3 | 0.626 +- 0.021 | 0.605 +- 0.022 | 3.000 +- 0.000 | 0.569 +- 0.029 |
| vote_k=5 | 0.633 +- 0.022 | 0.602 +- 0.021 | 5.000 +- 0.000 | 0.569 +- 0.029 |
| vote_k=8 | 0.631 +- 0.021 | 0.603 +- 0.025 | 8.000 +- 0.000 | 0.569 +- 0.029 |
| vote_k=16 | 0.629 +- 0.026 | 0.602 +- 0.024 | 16.000 +- 0.000 | 0.569 +- 0.029 |
| vote_k=32 | 0.629 +- 0.026 | 0.603 +- 0.025 | 32.000 +- 0.000 | 0.569 +- 0.029 |
| no_distance_weight | 0.629 +- 0.021 | 0.612 +- 0.015 | 8.000 +- 0.000 | 0.569 +- 0.029 |
| no_radius | 0.629 +- 0.019 | 0.600 +- 0.021 | 8.000 +- 0.000 | 0.569 +- 0.029 |
| no_importance | 0.632 +- 0.020 | 0.603 +- 0.025 | 8.000 +- 0.000 | 0.569 +- 0.029 |
| no_purity | 0.624 +- 0.021 | 0.594 +- 0.017 | 8.000 +- 0.000 | 0.569 +- 0.029 |
| uniform_vote | 0.629 +- 0.017 | 0.617 +- 0.012 | 8.000 +- 0.000 | 0.569 +- 0.029 |
| class_normalized | 0.627 +- 0.032 | 0.601 +- 0.026 | 8.000 +- 0.000 | 0.569 +- 0.029 |
| no_Fisher | 0.503 +- 0.001 | 0.504 +- 0.004 | 2.000 +- 0.000 | 0.009 +- 0.000 |

## Drift mas duro

- gradual: before 0.999 +- 0.001, after 1.000 +- 0.000, recovery chunk 1.000 +- 0.000
- sudden: before 0.849 +- 0.014, after 0.996 +- 0.002, recovery chunk 1.000 +- 0.000
- recurring: before 0.997 +- 0.002, after 1.000 +- 0.000, recovery chunk 1.000 +- 0.000

## Clases nuevas few-shot crueles

- 1 shots: old_after 0.981 +- 0.006, new 0.337 +- 0.074, forgetting 0.002 +- 0.002, cells_added 4.800 +- 0.400
- 5 shots: old_after 0.978 +- 0.007, new 0.421 +- 0.059, forgetting 0.006 +- 0.003, cells_added 4.800 +- 0.400
- 10 shots: old_after 0.969 +- 0.009, new 0.584 +- 0.020, forgetting 0.015 +- 0.011, cells_added 7.000 +- 0.894
- 50 shots: old_after 0.953 +- 0.011, new 0.836 +- 0.020, forgetting 0.030 +- 0.012, cells_added 32.600 +- 1.744

## Label noise

- AMF5_full: acc 0.944 +- 0.008, macroF1 0.940 +- 0.008
- extra_trees: acc 0.947 +- 0.019, macroF1 0.944 +- 0.020
- rbf_svm_like: acc 0.942 +- 0.018, macroF1 0.938 +- 0.018

## Lectura corta

AMF5 no gana todos los datasets reales en accuracy puro, y eso es justo lo que
queriamos medir. Su zona fuerte aparece en alta dimension, anatomia robusta del
campo, memoria compacta, ataques a features Fisher y adaptacion online. La
ablacion muestra que k=1 y no_Fisher son los cortes mas peligrosos, mientras que
el campo completo mantiene mejor equilibrio entre clean, ataque y costo.
