# Fase 11A - Encoder/Decoder Research Notes

Objetivo: traducir ideas de video frontier a AMF sin romper la regla principal:
AMF no recibe pixeles crudos; recibe latentes compactos y el decoder externo
reconstruye visualmente.

## Fuentes Revisadas

- Seedance 2.0: arquitectura unificada multi-modal audio/video, con entradas de
  texto, imagen, audio y video como referencias.
  - https://arxiv.org/abs/2604.14148
- Sora 2 / Sora 2 Pro: generacion con referencias, continuidad de escena,
  personajes reutilizables y extensiones de video; la documentacion publica no
  expone detalles completos de encoder/decoder.
  - https://developers.openai.com/api/docs/guides/video-generation
  - https://openai.com/cs-CZ/index/sora-2-system-card/
- TAPIR: tracking de puntos en dos etapas: matching por frame y refinamiento
  temporal por correlaciones locales.
  - https://arxiv.org/abs/2306.08637
- CoTracker: tracking conjunto de muchos puntos, no independiente, con memoria
  causal en ventanas y robustez ante oclusiones.
  - https://arxiv.org/abs/2307.07635
- Slot Attention: representaciones objeto-centricas intercambiables.
  - https://arxiv.org/abs/2006.15055
- Ca2-VDM: generacion autoregresiva causal con cache compartido para evitar
  recomputar contexto.
  - https://proceedings.mlr.press/v267/gao25m.html

## Traduccion A AMF

### Encoder

Antes: un estado por objeto con centro, velocidad, bbox y area. El benchmark
alto usa velocidad `t -> t+1`, que sirve como techo diagnostico pero no como
encoder causal.

Ahora se reportan tres rutas:

- `benchmark`: velocidad desde capas adyacentes reales; mide decoder/world model
  bajo un latente muy informativo.
- `causal`: velocidad desde `t-1 -> t`; no mira el futuro.
- `motion_token`: memoria causal de trayectoria compacta, inspirada en
  TAPIR/CoTracker; guarda el token de movimiento dominante observado hasta `t`
  y el signo reciente.
- `multi_hypothesis_probe`: combina causal simple + motion-token con pesos de
  fusion aprendidos en train por horizonte.

Resultado: motion-token mejora el one-step causal de AMF_full de `0.5805` a
`0.6191`, pero baja h10/h17 frente a la ruta causal simple. Esto demuestra que
el siguiente salto debe ser multi-hipotesis con selector/confianza, no una sola
velocidad comprimida.

El probe multi-hipotesis `220/20` confirma esa direccion:

- dual_AMF h1 `0.6201`
- dual_AMF h10 `0.2223`
- dual_AMF h17 `0.1909`
- dual_ridge h10/h17 `0.1161 / 0.1046`

Esto no cierra los targets extremos, pero ya evita elegir una sola trayectoria
cuando hay incertidumbre de cruce/occlusion.

Se probo tambien un selector causal de confianza:

- selected h10 `0.2238` vs max-fusion h10 `0.2223`
- selected h17 `0.1956` vs oracle h17 `0.2421`
- No se integra como resultado principal porque no domina todos los horizontes.

La conclusion nueva es que si hay informacion causal para seleccionar hipotesis,
pero el selector debe entrenarse con objetivo de IoU/ranking, no solo con una
etiqueta binaria de `token > simple`.

Ese paso se implemento despues como `phase11a_iou_ranker_probe.py`:

- h1: max-fusion `0.6201`, reg-ranker `0.6282`
- h10: causal simple `0.2168`, max-fusion `0.2223`, reg-ranker `0.2396`
- h17: causal simple `0.1981`, max-fusion `0.1909`, reg-ranker `0.2101`
- oracle h10/h17: `0.2620 / 0.2424`

Esto confirma que el selector/ranker alineado con IoU es superior al selector
binario y que el encoder multi-hipotesis ya esta aportando informacion usable.

La corrida grande `650/40` mantiene la tendencia:

- h1: max-fusion `0.6259`, class-ranker `0.6430`, reg-ranker `0.6361`
- h10: causal simple `0.1976`, max-fusion `0.2231`, reg-ranker `0.2326`
- h17: causal simple `0.1715`, max-fusion `0.1829`, reg-ranker `0.1921`
- oracle h10/h17: `0.2661 / 0.2229`

Esto ya es una mejora reproducible del encoder/decoder causal: el encoder no
colapsa a una sola velocidad, el decoder produce varias reconstrucciones
geometricas, y el ranker escoge la hipotesis visual mas probable con objetivo
directo de IoU.

Despues se bajo la decision a nivel de slot/objeto en
`phase11a_slot_ranker_probe.py`. Este paso sigue la intuicion de Slot Attention:
cada digito puede necesitar una hipotesis distinta durante cruces u oclusiones.
El decoder renderiza capas por objeto, el ranker predice IoU por candidato/slot,
y el frame final se compone con `max` entre capas.

Resultado grande `650/40`:

- h1: hybrid-extra `0.6447`, slot-frame-oracle `0.7125`
- h10: causal simple `0.1976`, frame-ranker `0.2326`,
  selection calibrada `0.2634` (`slot_hybrid` test-best `0.2775`)
- h17: causal simple `0.1715`, frame-ranker `0.1921`,
  frame-ensemble calibrado `0.2199` (`w75` test-best `0.2218`)
- slot-frame-oracle h10/h17: `0.3444 / 0.3012`

Esto confirma que la representacion objeto-centrica no solo es elegante: aporta
IoU real en rollout largo y deja un techo medible para un selector mejor. El
paso clave fue cambiar el objetivo de entrenamiento desde IoU por objeto hacia
IoU marginal de frame: para cada slot se aprende que tan bueno puede ser ese
candidato una vez compuesto con el otro slot.

### Decoder

Antes: warping de la capa visible del warmup.

Ahora: decoder externo con banco causal de identidad de `12000` crops reales de
train. Completa pixeles ocultos usando referencias reales y mezcla con el render
geometrico. Esto subio benchmark h10/h17 de `0.1856 / 0.1477` a
`0.2161 / 0.1750`.

## Estado Honesto

- AMF_full supera a baselines clasicos en h5/h10/h17.
- One-step benchmark pasa `>0.85`.
- Estabilidad h480 e identity drift pasan.
- Causal one-step todavia no pasa; motion-token mejora pero no basta.
- El probe multi-hipotesis mejora h1/h10 causal, pero todavia necesita selector
  de confianza por objeto/frame.
- El selector RandomForest confirma margen, pero requiere perdida directa de IoU
  o ranking por hipotesis.
- El IoU-ranker ya mejora h1/h10/h17 frente a las fusiones simples en `220/20`
  y mejora h10/h17 en `650/40`, pero todavia queda margen oracle.
- El slot-frame-ranker ya mejora h10/h17 frente al frame-ranker en `650/40`,
  pero el techo combinatorio muestra que el selector todavia deja informacion
  sin usar.
- Se implemento y ejecuto `phase11a_patch_decoder_probe.py` como siguiente
  traduccion de los video-tokenizers/VAEs modernos: decoder local por patches
  condicionado por hipotesis visuales, coordenadas, horizonte, distancia a slots
  y latentes dinamicos. Esta variante sigue dejando a AMF sin pixeles crudos,
  pero el output puro por patches no gano en `220/20`: h10/h17
  `0.1864 / 0.1639`, debajo del slot-frame-ranker. Se mantiene como evidencia
  negativa: el decoder local necesita actuar como refinador condicionado, no
  reemplazar las hipotesis geometricas.
- Se implemento y ejecuto `phase11a_kinematic_encoder_probe.py` como siguiente
  traduccion de encoders temporales/tracking: velocidad suavizada, aceleracion
  acotada y amortiguacion cerca de paredes. En `220/20`, la hipotesis
  kinematica no supero al slot-ranker actual: h10/h17 ensemble
  `0.2340 / 0.2039`, con oracle `0.2720 / 0.2624`. Queda como evidencia
  negativa: una sola velocidad suavizada no basta; el siguiente encoder debe
  representar incertidumbre multi-trayectoria por slot/celda.
- Se implemento una extension de score-routing hibrido: mezcla scores locales de
  objeto y scores globales de frame con normalizacion por ejemplo. Esto sigue la
  idea de routing/conditioning de decoders modernos sin convertir AMF en un
  transformer pesado. La seleccion calibrada se mantiene en `0.2634 / 0.2199`,
  pero el test-best aprendido sube a `0.2775 / 0.2218`.
- Se implemento y escalo un ranker pairwise opcional inspirado por token routing:
  `contexto causal + features del par candidato -> IoU`. En `650/40`, no escalo:
  h10/h17 test `0.2402 / 0.1929`, peor que el score-routing hibrido. Queda como
  evidencia negativa y se deja detras de `--enable-pairwise-ranker`.
- Se implemento y escalo un cell/tile router inspirado por decoders/tokenizers
  espaciales: selecciona candidatos por tile de 16. El tile-oracle en `650/40`
  es alto (`0.3743 / 0.3225` h10/h17), pero el router aprendido no escalo:
  `0.2383 / 0.2075`. Esto falsifica la hipotesis de que mas granularidad
  espacial basta; falta memoria temporal/seleccion secuencial mas fuerte.
- Siguiente arquitectura necesaria: selector por celda visual/slot con objetivo
  directo de frame-IoU y decoder completivo generativo mas fuerte.
