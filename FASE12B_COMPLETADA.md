# Fase 12B completada - PhysicalAI impact and causal world probes

Fecha: 2026-06-05

## Dataset

Se amplio el entrenamiento del AMF world model a escenas 12B reales:

```text
nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes
physics/bowling/physics-bowling-00000.tar
physics/dominoes/physics-dominoes-00000.tar
```

Uso validado:

- descarga real acotada de shards fisicos `bowling` y `dominoes`;
- inspeccion de `com`, `velocity`, `spin` y `rot`;
- `bowling`: 61,312 tracks fisicos;
- `dominoes`: 138,720 tracks fisicos;
- 1,000 secuencias por escena con split por secuencia;
- fit/validation/test por escena: 600 / 150 / 250;
- selector elegido en validacion, sin mirar test.

## Resultado Bowling

| horizonte | candidato seleccionado | MSE | Ridge MSE | gain vs Ridge | skill vs last |
|---|---|---:|---:|---:|---:|
| h1 | ridge_amf_0.5 | 0.000063 | 0.000102 | 0.381367 | 0.963797 |
| h5 | ridge_amf_0.25 | 0.002284 | 0.002838 | 0.195091 | 0.944300 |
| h15 | ridge_amf_0.5 | 0.024849 | 0.030110 | 0.174738 | 0.925204 |
| h30 | ridge_amf_1.0 | 0.097678 | 0.135595 | 0.279631 | 0.914402 |

## Resultado Dominoes

| horizonte | candidato seleccionado | MSE | Ridge MSE | gain vs Ridge | skill vs last |
|---|---|---:|---:|---:|---:|
| h1 | ridge | 0.000000 | 0.000000 | 0.000000 | 0.999245 |
| h5 | ridge | 0.000011 | 0.000011 | 0.000000 | 0.996770 |
| h15 | ridge_amf_0.5 | 0.000342 | 0.000391 | 0.124300 | 0.987091 |
| h30 | ridge_amf_0.5 | 0.004851 | 0.005652 | 0.141610 | 0.929635 |

## Contacto multi-objeto

Se agrego `phase12b_contact_world_probe.py` con features de vecino cercano, velocidad relativa, closing speed y densidad local. El probe compila y corre sobre datos reales.

Resultado diagnostico en `bowling`:

```text
h30 contacto stride30: 0.092665 AMF hibrido vs 0.182750 Ridge
h30 base     stride30: 0.096619 AMF hibrido vs 0.213070 Ridge
```

Lectura: el contexto nearest-neighbor ayuda en largo plazo (`h30`) pero mete ruido en h1/h5/h15. Esto apunta a que el siguiente encoder necesita identidad de objeto y mascara/segmentacion real, no solo proximidad geometrica.

## Identidad de slot

Se agrego `phase12b_identity_world_probe.py`, usando `segmentation_colors`, codigo estable de objeto y slot index.

Comparacion en `dominoes`, mismo split y parametros del run principal:

```text
h15 base:      0.000342
h15 identidad: 0.000339

h30 base:      0.004851
h30 identidad: 0.004809
```

Lectura: la identidad de slot mejora poco pero consistentemente h15/h30. Es la primera senal directa de mascara/identidad sin descargar RGB ni segmentacion pesada.

## Lectura

`bowling` introduce impacto dirigido y dispersion de pines. `dominoes` introduce cadena causal de contactos. En ambas escenas, el AMF hibrido supera a Ridge en largo plazo:

```text
bowling  h30: 0.097678 AMF hibrido vs 0.135595 Ridge
dominoes h30: 0.004809 AMF hibrido con identidad vs 0.005605 Ridge
```

Esto confirma que la memoria local residual no solo ayuda en caida limpia o billar simple; tambien corrige dinamica de impacto estructurado y cadenas de contacto. El siguiente cuello ya no es un baseline lineal, sino representar contactos explicitos, identidad de objeto y mascara/segmentacion para conectar el estado fisico con decoder visual.
