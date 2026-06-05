# Fase 12B completada - PhysicalAI bowling world probe

Fecha: 2026-06-05

## Dataset

Se amplio el entrenamiento del AMF world model a una escena 12B real:

```text
nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes
physics/bowling/physics-bowling-00000.tar
```

Uso validado:

- descarga real acotada del shard fisico `bowling`;
- inspeccion de `com`, `velocity`, `spin` y `rot`;
- 61,312 tracks fisicos;
- 1,000 secuencias con split por secuencia;
- fit/validation/test: 600 / 150 / 250;
- selector elegido en validacion, sin mirar test.

## Resultado

| horizonte | candidato seleccionado | MSE | Ridge MSE | gain vs Ridge | skill vs last |
|---|---|---:|---:|---:|---:|
| h1 | ridge_amf_0.5 | 0.000063 | 0.000102 | 0.381367 | 0.963797 |
| h5 | ridge_amf_0.25 | 0.002284 | 0.002838 | 0.195091 | 0.944300 |
| h15 | ridge_amf_0.5 | 0.024849 | 0.030110 | 0.174738 | 0.925204 |
| h30 | ridge_amf_1.0 | 0.097678 | 0.135595 | 0.279631 | 0.914402 |

## Lectura

`bowling` introduce impacto dirigido y dispersion de pines. En esta primera escena 12B, el AMF hibrido supera a Ridge en todos los horizontes y conserva buen largo plazo:

```text
h30: 0.097678 AMF hibrido vs 0.135595 Ridge
```

Esto confirma que la memoria local residual no solo ayuda en caida limpia o billar simple; tambien corrige dinamica de impacto estructurado. El siguiente cuello ya no es un baseline lineal, sino representar contactos explicitos, identidad de objeto y mascara/segmentacion para conectar el estado fisico con decoder visual.
