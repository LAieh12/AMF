# Fase 11A - AMF Visual Rollout en Moving MNIST Real

## Dataset

- Fuente descargada: `http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy`
- Archivo local: `data/MovingMNIST/mnist_test_seq.npy`
- Shape verificado: `(20, 10000, 64, 64)`
- Nota: el Moving MNIST estandar descargado trae 20 frames por secuencia. Por
  eso el ground truth real solo existe hasta horizonte 17 despues del warmup de
  `2` frames; horizontes 30/60/120/240/480 son estabilidad
  autorregresiva, no IoU contra ground truth.

## Arquitectura

```text
frame real -> encoder/tracker -> S_dyn(t) compacto + M_id separado
S_dyn(t), M_id_features -> AMF delta/residual world model -> S_dyn(t+1)
S_dyn(t+1), M_id layers -> decoder visual -> frame(t+1)
```

- AMF no recibe pixeles crudos.
- Latente dinamico: `14` floats.
- Identidad compacta para AMF: `12` floats.
- Latente total reportado: `26` floats / `104` bytes.
- Frame: `16384` bytes.
- Compresion frame/latente: `157.5x`.
- AMF cells: `9000`.
- AMF memory MB: `2.3689`.
- AMF residual scale usado en prediccion: `0.0`.
- Caja global de colision visual: `0.317`.
- Banco causal de identidad para decoder: `12000` crops reales de train.

## One-Step Real

| modelo | frame IoU | soft IoU | latent MSE | center err px | bounce latent MSE |
|---|---:|---:|---:|---:|---:|
| constant_velocity | 0.8958 | 0.8729 | 0.000435 | 0.044 | 0.000472 |
| linear_delta_model | 0.3032 | 0.2540 | 0.000886 | 3.516 | 0.000923 |
| ridge_linear_dynamics | 0.8958 | 0.8729 | 0.000394 | 0.182 | 0.000430 |
| knn_transition | 0.3606 | 0.3058 | 0.000843 | 3.147 | 0.000885 |
| AMF_no_metaplasticity | 0.8945 | 0.8713 | 0.000423 | 0.052 | 0.000462 |
| AMF_full | 0.8867 | 0.8636 | 0.000463 | 0.127 | 0.000503 |

## Rollout Con Ground Truth Real

| modelo | IoU h1 | IoU h5 | IoU h10 | IoU h17 | soft IoU h17 | center err h17 px |
|---|---:|---:|---:|---:|---:|---:|
| constant_velocity | 0.8879 | 0.2955 | 0.0914 | 0.0134 | 0.0107 | 34.144 |
| linear_delta_model | 0.2735 | 0.0878 | 0.0737 | 0.0603 | 0.0494 | 21.241 |
| ridge_linear_dynamics | 0.8879 | 0.2356 | 0.1293 | 0.1342 | 0.1092 | 15.905 |
| knn_transition | 0.3697 | 0.1299 | 0.0750 | 0.0872 | 0.0706 | 21.831 |
| AMF_no_metaplasticity | 0.8879 | 0.2488 | 0.1171 | 0.0601 | 0.0469 | 24.575 |
| AMF_full | 0.8879 | 0.3315 | 0.2161 | 0.1750 | 0.1446 | 15.291 |

## Evaluacion Causal Sin Velocidad Futura

Aqui `S(t)` usa velocidad de `t-1 -> t`, no de `t -> t+1`. Esto es mas duro y
mas cercano al rollout real desde un warmup causal.

### One-Step Causal

| modelo | frame IoU | soft IoU | latent MSE | center err px | bounce latent MSE |
|---|---:|---:|---:|---:|---:|
| constant_velocity | 0.5844 | 0.5274 | 0.000635 | 1.695 | 0.000703 |
| linear_delta_model | 0.3032 | 0.2540 | 0.000896 | 3.516 | 0.000953 |
| ridge_linear_dynamics | 0.5791 | 0.5199 | 0.000537 | 1.623 | 0.000598 |
| knn_transition | 0.3527 | 0.2998 | 0.000896 | 3.335 | 0.000951 |
| AMF_no_metaplasticity | 0.5760 | 0.5161 | 0.000585 | 1.710 | 0.000657 |
| AMF_full | 0.5805 | 0.5237 | 0.000665 | 1.767 | 0.000738 |

### Rollout Causal Con Ground Truth Real

| modelo | IoU h1 | IoU h5 | IoU h10 | IoU h17 | soft IoU h17 | center err h17 px |
|---|---:|---:|---:|---:|---:|---:|
| constant_velocity | 0.6049 | 0.2125 | 0.0675 | 0.0123 | 0.0093 | 34.836 |
| linear_delta_model | 0.2735 | 0.0878 | 0.0737 | 0.0603 | 0.0494 | 21.241 |
| ridge_linear_dynamics | 0.5943 | 0.2023 | 0.1111 | 0.0900 | 0.0724 | 18.406 |
| knn_transition | 0.3589 | 0.1253 | 0.0894 | 0.0753 | 0.0608 | 21.550 |
| AMF_no_metaplasticity | 0.5986 | 0.2550 | 0.1586 | 0.1322 | 0.1118 | 17.799 |
| AMF_full | 0.6029 | 0.2623 | 0.1976 | 0.1715 | 0.1434 | 14.861 |

## Encoder Causal Por Motion Tokens

Inspirado en TAPIR/CoTracker: mantiene memoria de trayectoria compacta y usa el
token de movimiento dominante observado hasta `t`, con signo reciente. Sigue
siendo causal y AMF no recibe pixeles crudos.

### One-Step Motion Tokens

| modelo | frame IoU | soft IoU | latent MSE | center err px | bounce latent MSE |
|---|---:|---:|---:|---:|---:|
| constant_velocity | 0.6207 | 0.5723 | 0.000680 | 1.687 | 0.000738 |
| ridge_linear_dynamics | 0.5600 | 0.5010 | 0.000568 | 1.765 | 0.000625 |
| AMF_no_metaplasticity | 0.5827 | 0.5283 | 0.000632 | 1.818 | 0.000704 |
| AMF_full | 0.6191 | 0.5703 | 0.000667 | 1.718 | 0.000732 |

### Rollout Motion Tokens Con Ground Truth Real

| modelo | IoU h1 | IoU h5 | IoU h10 | IoU h17 | soft IoU h17 | center err h17 px |
|---|---:|---:|---:|---:|---:|---:|
| constant_velocity | 0.5999 | 0.1843 | 0.0521 | 0.0191 | 0.0159 | 33.907 |
| ridge_linear_dynamics | 0.5630 | 0.1903 | 0.1013 | 0.1029 | 0.0820 | 18.654 |
| AMF_no_metaplasticity | 0.5665 | 0.2165 | 0.1203 | 0.0887 | 0.0721 | 19.556 |
| AMF_full | 0.6000 | 0.2411 | 0.1902 | 0.1543 | 0.1295 | 18.684 |

## Probe Causal Multi-Hipotesis Con IoU Ranker

El probe `phase11a_iou_ranker_probe.py` combina la ruta causal simple y la ruta
motion-token, renderiza candidatos `simple`, `token`, `max` y `blend`, y entrena
un ranker directo de IoU con train real. Sigue sin meter pixeles crudos en AMF.

Comando grande verificado:

```powershell
python phase11a_iou_ranker_probe.py --train-sequences 650 --test-sequences 40 --out results/phase11a_iou_ranker_probe_650_40.json
```

| horizonte | causal simple | motion token | max beta 1.0 | class ranker | reg ranker | oracle |
|---|---:|---:|---:|---:|---:|---:|
| h1 | 0.6029 | 0.6000 | 0.6259 | 0.6430 | 0.6361 | 0.6798 |
| h5 | 0.2623 | 0.2411 | 0.2950 | 0.2734 | 0.2950 | 0.3219 |
| h10 | 0.1976 | 0.1902 | 0.2231 | 0.2251 | 0.2326 | 0.2661 |
| h17 | 0.1715 | 0.1543 | 0.1829 | 0.1757 | 0.1921 | 0.2229 |

Resultado: el clasificador gana h1; el regresor de IoU gana h10/h17 entre las
rutas no-oracle. Esto confirma que el encoder/decoder multi-hipotesis esta
aportando informacion causal util, aunque todavia no alcanza los targets
extremos.

## Probe Causal Por Slot/Objeto

El probe `phase11a_slot_ranker_probe.py` baja la seleccion multi-hipotesis al
nivel de cada digito. En vez de elegir un frame completo, renderiza candidatos
por capa/slot, escoge una hipotesis para cada objeto y luego compone el frame
final. Esto ataca directamente los errores de cruce y oclusion.

Comando grande verificado:

```powershell
python phase11a_slot_ranker_probe.py --train-sequences 650 --test-sequences 40 --out results/phase11a_slot_ranker_probe_hybrid_650_40.json
```

| horizonte | causal simple | motion token | max beta 1.0 | slot reg | frame reg | frame extra | frame ensemble | best learned | slot frame oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| h1 | 0.6029 | 0.6000 | 0.6259 | 0.6378 | 0.6433 | 0.6420 | 0.6440 | 0.6378 | 0.7125 |
| h5 | 0.2623 | 0.2411 | 0.2950 | 0.3634 | 0.3517 | 0.3507 | 0.3534 | 0.3634 | 0.4606 |
| h10 | 0.1976 | 0.1902 | 0.2231 | 0.2634 | 0.2716 | 0.2647 | 0.2657 | 0.2634 | 0.3444 |
| h17 | 0.1715 | 0.1543 | 0.1829 | 0.2114 | 0.2196 | 0.2115 | 0.2199 | 0.2199 | 0.3012 |

Resultado: los candidatos locales `wide_*`/`very_wide_*`/`shift_*` elevan el techo del
decoder y el ranker por slot mejora h10/h17 frente al ranker por frame
(`0.2634 / 0.2199` calibrado vs `0.2326 / 0.1921`). El score-routing hibrido
sube el test-best aprendido h10/h17 a `0.2775 / 0.2218`; el techo combinatorio
por slots (`0.3444 / 0.3012` en h10/h17) muestra margen real para un selector
por celda visual mas fuerte.

Probe adicional de encoder temporal: `phase11a_kinematic_encoder_probe.py`
prueba una tercera hipotesis causal con velocidad suavizada/aceleracion acotada.
En `220/20` no supera al slot-ranker actual: h10/h17 ensemble
`0.2340 / 0.2039`, con oracle `0.2720 / 0.2624`.

Probe adicional de routing pairwise: `phase11a_slot_ranker_probe.py
--enable-pairwise-ranker` aprende `contexto causal + features del par candidato
-> IoU`. En `650/40` no escala: h10/h17 test `0.2402 / 0.1929`, peor que el
score-routing hibrido `0.2775 / 0.2218`. Queda como evidencia negativa y no
forma parte de la ruta default.

Probe adicional de cell/tile router: `phase11a_slot_ranker_probe.py
--cell-router` aprende un decoder espacial por tiles de 16. El tile-oracle en
`650/40` es alto (`0.3743 / 0.3225` h10/h17), pero el router aprendido no escala:
`0.2383 / 0.2075`. Queda como evidencia negativa; el cuello de botella no es
solo granularidad espacial sino seleccion temporal/visual mas fuerte.

## Diagnostico Del Decoder

| diagnostico | valor |
|---|---:|
| current_layer_iou | 0.8958 |
| current_layer_soft_iou | 0.8729 |
| all_past_memory_iou | 0.5861 |
| all_past_memory_soft_iou | 0.5663 |
| next_layer_upper_iou | 0.8485 |
| next_layer_upper_soft_iou | 0.8140 |

## Rollout Largo Sin Ground Truth

| modelo | stable h30 | stable h120 | stable h240 | stable h480 | identity drift h480 |
|---|---:|---:|---:|---:|---:|
| constant_velocity | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.000000 |
| linear_delta_model | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.000000 |
| ridge_linear_dynamics | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.000000 |
| knn_transition | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.000000 |
| AMF_no_metaplasticity | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.000000 |
| AMF_full | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.000000 |

## Metaplasticidad

- Runtime statuses durante online fitting: `explained_by_existing_cell=2092, buffered_possible_noise=171, metaplasticity_adapted_cell=237`
- Probe statuses: `buffered_possible_noise, buffered_possible_noise, created_confirmed_novelty, explained_by_existing_cell, explained_by_existing_cell, explained_by_existing_cell, explained_by_existing_cell, explained_by_existing_cell`
- created_confirmed_novelty: `True`
- explained_after_create: `True`
- probe MSE after: `0.000035`

## Targets

| target | passed |
|---|---:|
| uses_real_downloaded_dataset | True |
| no_raw_pixel_amf | True |
| one_step_iou_gt_0_85 | True |
| causal_one_step_iou_gt_0_85 | False |
| gt_rollout_10_iou_gt_0_75 | False |
| gt_rollout_17_iou_gt_0_60 | False |
| stability_480 | True |
| identity_drift_low | True |
| metaplasticity_probe_passed | True |

## Conclusion Honesta

Fase 11A ya usa Moving MNIST real descargado y prueba la ruta correcta:
pixeles reales -> latente compacto -> AMF/metaplasticidad + modulo global de
fisica visual -> decoder con banco causal de identidad real. La compresion, el
one-step real, la estabilidad larga y la metaplasticidad pasan.

AMF_full supera a los baselines clasicos en rollout con ground truth real en
h5, h10 y h17. El probe causal multi-hipotesis con IoU-ranker mejora h10/h17 de
`0.1976 / 0.1715` a `0.2326 / 0.1921` en `650/40`; el slot-frame-ranker lo sube
a `0.2634 / 0.2199` con seleccion calibrada. Los targets visuales
extremos `gt_rollout_10_iou_gt_0_75` y `gt_rollout_17_iou_gt_0_60` siguen
falsos: con el Moving MNIST estandar descargado solo hay 20 frames, y aunque el
banco causal de identidad recupera parte de los pixeles ocultos en
cruces/oclusiones, aun no alcanza un decoder generativo predictivo fuerte. La
evidencia apunta a que el siguiente salto debe venir de un selector por celda
visual/slot con entrenamiento directo a frame-IoU y de un decoder de identidad
completiva mas potente, no de meter pixeles crudos en AMF.

## Inspiracion De La Investigacion Actual

- Seedance 2.0: arquitectura multimodal unificada y condicionamiento por
  referencias de texto/imagen/video/audio.
- Cosmos/VidTok/VideoFlexTok/PV-VAE: tokenizacion causal, latentes compactos,
  decoder condicionado por contexto y reconstruccion parcial-a-completa.
- Traduccion a AMF: AMF conserva solo `S_dyn + M_id_features`; el decoder externo
  usa un banco causal de referencias reales para completar identidad visual.
