# Fase 12A completada - PhysicalAI hybrid world probe

Fecha: 2026-06-05

## Dataset

Se adopto el dataset:

```text
nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes
```

Uso validado:

- manifest de 9,520 shards;
- escalera 12A/12B/12C por dificultad fisica;
- streaming smoke sin descargar todo;
- descarga real acotada de `physics/objects_falling/physics-objects_falling-00007.tar`;
- descarga real acotada de `physics/billiards/physics-billiards-00000.tar`;
- benchmark fisico con `com` y `velocity` reales.

## Resultado 12A

Shards reales:

```text
physics/objects_falling/physics-objects_falling-00007.tar
physics/billiards/physics-billiards-00000.tar
```

Resumen `objects_falling`:

```text
tracks: 19036
sequences: 142
fit/validation/test: 85 / 21 / 36
```

| horizonte | candidato seleccionado | MSE | Ridge MSE | gain vs Ridge | skill vs last |
|---|---|---:|---:|---:|---:|
| h1 | ridge_amf_0.25 | 0.000194 | 0.000197 | 0.018836 | 0.893101 |
| h5 | ridge_amf_0.5 | 0.006390 | 0.008028 | 0.204043 | 0.850660 |
| h15 | cv_amf | 0.054260 | 0.095757 | 0.433353 | 0.828436 |
| h30 | cv_amf | 0.225712 | 0.265542 | 0.149995 | 0.697530 |

Resumen `billiards`:

```text
tracks: 40028
sequences: 1000
fit/validation/test: 600 / 150 / 250
```

| horizonte | candidato seleccionado | MSE | Ridge MSE | gain vs Ridge | skill vs last |
|---|---|---:|---:|---:|---:|
| h1 | constant_velocity | 0.000003 | 0.000006 | 0.519399 | 0.998893 |
| h5 | ridge | 0.000347 | 0.000347 | 0.000000 | 0.994063 |
| h15 | ridge_amf_0.25 | 0.006750 | 0.007350 | 0.081616 | 0.986357 |
| h30 | ridge_amf_0.5 | 0.051179 | 0.061607 | 0.169266 | 0.972078 |

Disciplina:

- el split de secuencias se aleatoriza con seed fija `123`;
- el candidato activo se elige en un bloque de validacion;
- luego se reentrena con todas las secuencias train completas;
- el test queda separado por secuencia;
- no se elige el mejor candidato mirando el test.

## Lectura

El salto desde 11A fue correcto: PhysicalAI permite evaluar world model fisico con ground truth limpio, sin depender primero de render RGB.

El cuello h30 anterior ya no queda como derrota frente a Ridge en ninguna de las dos escenas 12A:

```text
objects_falling h30: 0.225712 AMF hibrido vs 0.265542 Ridge
billiards       h30: 0.051179 AMF hibrido vs 0.061607 Ridge
```

La mejora viene de combinar un encoder fisico mas rico, un componente global Ridge y memorias AMF locales normalizadas que corrigen residuales. La siguiente fase debe ampliar a causalidad estructurada (`dominoes`, `bowling`, rampas) y agregar senales de contacto/mascara para que el decoder visual pueda apoyarse en estados fisicos reales.
