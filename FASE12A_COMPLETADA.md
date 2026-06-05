# Fase 12A completada - PhysicalAI hybrid world probe

Fecha: 2026-06-04

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
- benchmark fisico con `com` y `velocity` reales.

## Resultado 12A

Shard real:

```text
physics/objects_falling/physics-objects_falling-00007.tar
```

Resumen:

```text
tracks: 19036
sequences: 142
train/test: 106 / 36
```

Metricas actuales:

| horizonte | candidato seleccionado en validacion | MSE | Ridge MSE | gain vs Ridge | skill vs last |
|---|---|---:|---:|---:|---:|
| h1 | ridge_amf_0.25 | 0.000118 | 0.000120 | 0.012644 | 0.937993 |
| h5 | ridge_amf_0.5 | 0.005272 | 0.006663 | 0.208659 | 0.883776 |
| h15 | cv_amf | 0.037346 | 0.094127 | 0.603245 | 0.892036 |
| h30 | ridge_amf_0.5 | 0.184980 | 0.285415 | 0.351889 | 0.785750 |

Disciplina:

- el candidato activo se elige en un bloque de validacion de 21 secuencias;
- luego se reentrena con las 106 secuencias train completas;
- el test queda separado en 36 secuencias;
- no se elige el mejor candidato mirando el test.

## Lectura

El salto desde 11A fue correcto: PhysicalAI permite evaluar world model fisico con ground truth limpio, sin depender primero de render RGB.

El cuello h30 anterior ya no queda como derrota frente a Ridge:

```text
h30: 0.184980 AMF hibrido vs 0.285415 Ridge
```

La mejora viene de combinar un encoder fisico mas rico, un componente global Ridge y memorias AMF locales normalizadas que corrigen residuales. La siguiente fase debe ampliar de `objects_falling` a colisiones/causalidad (`billiards`, `dominoes`) y agregar senales de contacto/mascara para que el decoder visual pueda apoyarse en estados fisicos reales.
