# Fase 11A - diagnostico Never/AMF antes de seguir con encoder/decoder

Fecha: 2026-06-02

## Pregunta central

Antes de construir otro encoder/decoder, hay que responder:

- cual es el mejor resultado real;
- por que es el mejor;
- cual es el bottleneck principal;
- si se esta usando Never completo con AMF World Model o solo piezas;
- si el dataset actual todavia sirve;
- si encoder/decoder son el problema real o si la arquitectura esta mirando el sitio equivocado.

## Mejor resultado actual

El mejor camino medido sigue siendo el slot-hybrid/ranker:

```text
phase11a_slot_ranker_probe_hybrid_650_40.json
test-best learned h10/h17: 0.2775 / 0.2218
oracle h10/h17: 0.3444 / 0.3012
```

La lectura importante no es solo el numero. La brecha entre learned y oracle indica que todavia existe capacidad latente si el sistema escoge mejor entre candidatos.

## Por que es el mejor

El slot-hybrid gana porque combina tres propiedades que los probes negativos no tienen al mismo tiempo:

1. Representa objetos/slots, no solo frames globales.
2. Usa multiples candidatos/ramas y selecciona por horizonte.
3. Conserva una forma simple de dinamica local, evitando reconstruir todo desde un embedding demasiado comprimido.

Los probes que intentaron mejorar solo una pieza fallaron:

- patch-decoder: mejora local insuficiente;
- kinematic encoder: dinamica manual insuficiente;
- cell/tile router: el oracle de tile era bueno, pero el router aprendido no escogia bien;
- multi-encoder: el oracle de ramas era prometedor, el ensemble aprendido no alcanzo;
- patch-attention simple: temporal modeling demasiado debil;
- Mini-SimVP rapido: CPU/small-run insuficiente y no resolvio h10/h17;
- AMF wide: subir `max_cells` no aumento celulas efectivas ni metrica.

Conclusion: el problema no es "faltan encoders" o "faltan decoders" en abstracto. El problema es el ciclo completo de world model: estado -> transicion -> seleccion -> render.

## Se esta usando Never completo?

No de forma suficiente.

Las pruebas actuales usan partes:

- AMF local;
- slots/tiles;
- rankers;
- decoders/probes aislados;
- algunos mecanismos globales ligeros.

Pero no esta probado como un Never World Model completo donde el estado latente, la memoria de transiciones, la seleccion global, el rollout temporal y el decoder sean una sola ruta entrenada/evaluada.

Esto importa porque un decoder mejor no arregla una transicion equivocada. Y un encoder mejor no arregla un selector que elige la rama incorrecta.

## Mayor bottleneck

El bottleneck principal parece ser seleccion/modelado temporal, no reconstruccion visual pura.

Evidencia:

- Los oracles suelen superar claramente a las versiones aprendidas.
- El tile oracle fue alto, pero el router/regresor aprendido fue bajo.
- Multi-encoder tuvo branch_oracle mejor que ensemble.
- Aumentar capacidad nominal de celulas no mejoro si no aumento cobertura efectiva.

Si el decoder fuera el bottleneck dominante, los probes de patch/copy/tile tendrian que acercarse mas al oracle. No lo hicieron.

## Como mejorar el mejor resultado

La mejora correcta debe atacar la brecha learned-vs-oracle:

1. Encoder de estado:
   slots/tiles con posicion, masa, bbox, velocidad, aceleracion, identidad corta y confianza.

2. AMF World Model:
   memoria de transiciones por celulas que aprende residuales de movimiento, no solo clases o scores.

3. Selector global:
   ranker por horizonte que elige candidato/branch usando features de incertidumbre y cobertura.

4. Decoder copy-skip:
   reconstruir moviendo detalle visual real desde frames observados, no renderizando desde un vector comprimido.

5. Control de crecimiento:
   medir celulas efectivas, no solo `max_cells`; fusionar por importancia y error de transicion.

## Vale seguir usando MovingMNIST?

Si, pero ya no como unico dataset.

MovingMNIST todavia sirve porque:

- es real descargado, barato y reproducible;
- aisla el problema de dinamica visual;
- permite medir h1/h5/h10/h17 rapidamente;
- deja ver si el world model aprende transiciones, rebotes y composicion simple.

Pero no basta para declarar un encoder/decoder "definitivo" para Never:

- es demasiado simple;
- casi no evalua semantica;
- no prueba lenguaje, memoria larga ni razonamiento;
- puede favorecer heuristicas cinematicas manuales.

Decision:

- Mantener MovingMNIST como test de humo y laboratorio de dinamica.
- Agregar al menos un dataset secuencial mas dificil antes de claims fuertes: KTH/BAIR/TaxiBJ para video/dinamica, o una tarea simbolica/secuencial si Never apunta a razonamiento.

## Nueva ruta implementada

Se agregaron tres scripts para forzar esta disciplina:

```text
phase11a_never_bottleneck_audit.py
phase11a_never_world_codec_probe.py
phase11a_never_definitive_codec_probe.py
```

El definitivo implementa la ruta que faltaba:

```text
encoder de slots -> candidatos dinamicos -> memoria AMF -> politica por horizonte -> decoder copy-skip
```

## Resultado medido

El definitive codec ya fue medido en 220/40 y 650/40 con MovingMNIST real descargado.

650/40:

```text
h1  MSE 0.022883  skill_vs_last 0.518941  rama slot_amf_mean
h5  MSE 0.049480  skill_vs_last 0.294730  rama slot_amf_mean
h10 MSE 0.051417  skill_vs_last 0.281871  rama slot_amf_mean
h17 MSE 0.054018  skill_vs_last 0.268249  rama slot_amf_mean
```

Las celulas efectivas subieron a 2596 en 650/40. Esto prueba crecimiento real de memoria.

## Estado actual

La ejecucion local ya fue recuperada y los JSON principales existen:

```text
results/phase11a_never_definitive_codec_probe_220_40.json
results/phase11a_never_definitive_codec_probe_650_40.json
results/FASE11A_NEVER_BOTTLENECK_AUDIT.md
```

Conclusion: el bottleneck ya no es "falta de decoder". El limite restante es dinamica de colisiones/oclusiones y validacion fuera de MovingMNIST. La mezcla `slot_amf_mean` cerro casi toda la brecha contra `candidate_oracle_mse` en h10/h17.
