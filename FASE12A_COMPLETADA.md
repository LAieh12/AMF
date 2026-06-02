# Fase 12A completada - PhysicalAI physics smoke

Fecha: 2026-06-02

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

Metricas:

| horizonte | mejor candidato | MSE | skill vs last |
|---|---|---:|---:|
| h1 | amf_residual | 0.000109 | 0.942582 |
| h5 | amf_residual | 0.003526 | 0.922268 |
| h15 | amf_residual | 0.069023 | 0.800456 |
| h30 | ridge | 0.314095 | 0.636205 |

## Lectura

El salto desde 11A fue correcto: PhysicalAI permite evaluar world model fisico con ground truth limpio, sin depender primero de render RGB.

Nuevo bottleneck:

```text
h30 / largo plazo fisico
```

AMF residual gana corto/medio plazo; a h30 la memoria local pierde y Ridge generaliza mejor. La siguiente mejora debe combinar AMF local con un modelo global de gravedad/contacto/colision.
