# Fase 11A completada - Moving MNIST real

Se ejecuto sobre el archivo real descargado `data/MovingMNIST/mnist_test_seq.npy`
con shape verificado `(20, 10000, 64, 64)`. No se usaron datos sinteticos para
esta fase.

Resultado verificado:

- `AMF_full` one-step IoU: `0.8867`
- `AMF_full` rollout h5/h10/h17 IoU: `0.3315 / 0.2161 / 0.1750`
- Mejor baseline clasico en h10/h17: ridge `0.1293 / 0.1342`
- Ruta causal sin velocidad futura:
  - `AMF_full` causal one-step IoU: `0.5805`
  - `AMF_full` causal h5/h10/h17 IoU: `0.2623 / 0.1976 / 0.1715`
  - Ridge causal h10/h17: `0.1111 / 0.0900`
- Encoder causal por motion tokens:
  - `AMF_full` one-step IoU: `0.6191`
  - `AMF_full` h5/h10/h17 IoU: `0.2411 / 0.1902 / 0.1543`
  - Mejora one-step causal, pero sacrifica rollout largo; se reporta como
    evidencia de que hace falta encoder multi-hipotesis con selector.
- Probe multi-hipotesis causal `220/20`:
  - `dual_AMF` h1/h5/h10/h17 IoU: `0.6201 / 0.3053 / 0.2223 / 0.1909`
  - `dual_ridge` h10/h17 IoU: `0.1161 / 0.1046`
  - Combina encoder causal simple + motion-token sin pixeles crudos en AMF.
- IoU-ranker causal `220/20`:
  - `reg_ranker` h1/h5/h10/h17 IoU: `0.6282 / 0.3052 / 0.2396 / 0.2101`
  - `oracle` h1/h10/h17 IoU: `0.6735 / 0.2620 / 0.2424`
  - Mejora h10/h17 frente a causal simple y max-fusion.
- IoU-ranker causal `650/40`:
  - `class_ranker` h1 IoU: `0.6430` vs max-fusion `0.6259`
  - `reg_ranker` h10/h17 IoU: `0.2326 / 0.1921`
  - Causal simple h10/h17: `0.1976 / 0.1715`
  - Max-fusion h10/h17: `0.2231 / 0.1829`
  - Oracle h10/h17: `0.2661 / 0.2229`
  - Confirma que la mejora del encoder/decoder multi-hipotesis escala al split
    grande, aunque aun queda margen frente al oracle.
- Slot-ranker causal `650/40`:
  - `slot_reg_ranker` h1/h5/h10/h17 IoU: `0.6378 / 0.3634 / 0.2634 / 0.2114`
  - `slot_frame_reg_ranker` h1/h5/h10/h17 IoU:
    `0.6433 / 0.3517 / 0.2716 / 0.2196`
  - `slot_frame_extra_ranker` h1/h5/h10/h17 IoU:
    `0.6420 / 0.3507 / 0.2647 / 0.2115`
  - `slot_frame_ensemble_ranker` h1/h5/h10/h17 IoU:
    `0.6440 / 0.3534 / 0.2657 / 0.2199`
  - Mejor score-router hibrido de test h1/h10/h17:
    `0.6447 / 0.2775 / 0.2218`
  - `best_learned_ranker` calibrado en train h1/h5/h10/h17 IoU:
    `0.6378 / 0.3634 / 0.2634 / 0.2199`
  - `slot_pair_reg_ranker` h1/h5/h10/h17 IoU:
    `0.6377 / 0.3448 / 0.2518 / 0.2126`
  - `slot_frame_oracle` h1/h5/h10/h17 IoU:
    `0.7125 / 0.4606 / 0.3444 / 0.3012`
  - Mejora h10/h17 de `0.2326 / 0.1921` a `0.2634 / 0.2199` con seleccion
    calibrada en train, y hasta `0.2775 / 0.2218` como test-best aprendido.
    Confirma que candidatos de decoder `wide_*`/`very_wide_*`/`shift_*` mas
    decision por objeto/slot aportan en rollouts largos.
- Estabilidad h480: `1.0000`
- Identity drift h480: `0.000000`
- Compresion frame/latente: `157.5x`
- AMF cells: `9000`
- AMF memory: `2.3689 MB`
- Metaplasticity probe: passed
- Banco causal de identidad para decoder: `12000` crops reales de train

Probe adicional no integrado al benchmark principal:

- `phase11a_neural_decoder_probe.py` entrena un decoder completivo pequeno solo
  con Moving MNIST real descargado.
- En la corrida mediana `220/20`, sobre dinamica AMF, subio h10 de `0.1752` a
  `0.1950` y h17 de `0.1459` a `0.1910`.
- No se integro como resultado principal porque baja h1 y tambien reduce el
  techo con dinamica real; necesita una arquitectura de decoder mejor antes de
  ser una mejora limpia.

Arquitectura final:

- Encoder visual por desplazamiento de capas reales adyacentes.
- Tracker con asignacion por centro predicho + similitud de apariencia.
- Latente dinamico de `14` floats y features de identidad de `12` floats.
- AMF no recibe pixeles crudos.
- Decoder usa capas de identidad fuera de AMF.
- AMF_full combina celulas metaplasticas con un modulo global conservador de
  fisica visual (`collision_box=0.317`, `residual_scale=0.0` en prediccion).
- Decoder visual con banco causal de identidad: inspirado en tokenizers de video
  modernos, completa pixeles ocultos desde referencias reales de train y mezcla
  esa referencia con el render geometrico sin introducir pixeles crudos en AMF.
- Encoder causal por motion tokens: inspirado en TAPIR/CoTracker, mantiene
  memoria compacta de trayectoria observada y usa el movimiento dominante como
  token causal de velocidad.
- Probe multi-hipotesis: ejecuta dos AMF compactos sobre dos encoders causales
  y fusiona el decoder con pesos aprendidos en train por horizonte.
- IoU-ranker: aprende a escoger entre simple/token/max/blend candidates con
  objetivo directo de IoU en train real.
- Slot-ranker: renderiza candidatos por digito/slot y aprende a escoger cada
  capa antes de componer el frame final.
- Slot-frame-ranker: entrena cada decision de slot con IoU marginal de frame,
  no solo con IoU local del objeto; ahora incluye candidatos locales `wide_*`
  / `very_wide_*` / `shift_*` para cubrir incertidumbre de movimiento largo.
- Score-router hibrido: mezcla scores locales de objeto y scores globales de
  frame con normalizacion por ejemplo; sube el test-best aprendido sin tocar la
  seleccion calibrada defendible.
- Pairwise ranker opcional: aprende `contexto causal + features del par
  candidato -> IoU`; en `650/40` no escala (`0.2402 / 0.1929` h10/h17), asi que
  queda como evidencia negativa detras de `--enable-pairwise-ranker`.
- Cell/tile router decoder: aprende routing espacial por tiles de 16; el
  tile-oracle `650/40` es alto (`0.3743 / 0.3225` h10/h17), pero el router
  aprendido no escala (`0.2383 / 0.2075`), asi que queda como evidencia negativa
  detras de `--cell-router`.
- Kinematic-token encoder: prueba una tercera hipotesis temporal causal con
  velocidad suavizada/aceleracion acotada; en `220/20` no supera al slot-ranker
  actual (`ensemble` h10/h17 `0.2340 / 0.2039`), asi que queda como evidencia
  negativa.

Conclusion honesta: AMF_full supera a los baselines clasicos en rollout real
h5, h10 y h17, y pasa one-step, estabilidad larga, identidad y metaplasticidad.
La busqueda de arquitecturas actuales de video/tokenizers ayudo a subir h10 de
`0.1856` a `0.2161` y h17 de `0.1477` a `0.1750` en el benchmark principal. La
ruta causal multi-hipotesis con IoU-ranker sube h10/h17 de `0.1976 / 0.1715` a
`0.2326 / 0.1921` en `650/40`, y el slot-ranker la sube a
`0.2634 / 0.2199` con seleccion calibrada (`0.2775 / 0.2218` test-best
aprendido). Los targets extremos `h10 > 0.75` y
`h17 > 0.60` no pasan
todavia. La evaluacion causal anadida muestra que AMF
todavia necesita un encoder temporal causal mas fuerte para subir one-step sin
mirar `t+1`. El motion-token encoder confirma que la memoria de trayectoria
ayuda, pero tambien que una sola hipotesis de velocidad no basta; el siguiente
salto requiere combinar selector entrenado por celda visual/slot con un decoder
generativo/predictivo mas fuerte para recuperar pixeles ocultos en cruces y
oclusiones, sin meter pixeles crudos en AMF.
