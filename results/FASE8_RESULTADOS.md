# Fase 8 - Decoder morfogenetico

Objetivo: agregar capacidad de output a un sistema morfogenetico sin usar LLMs
ni pegar un decoder denso clasico.

Arquitectura probada:

```text
Input -> morphogenic input field -> active latent state -> morphogenic decoder -> output
```

Reglas: no LLM = True, no decoder denso =
True, backprop = False.

Training examples: 90
Test prompts: 15
Tiempo total: 0.09 s

## Comparacion de decoders

| Decoder | talk score | complete | verb | keyword | relevance | unique | repetition | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| resonant_morphogenic_decoder | 0.905 | 1.000 | 0.933 | 1.000 | 0.544 | 1.000 | 0.000 | 13.600 |
| frame_slot_cells | 0.876 | 1.000 | 0.867 | 0.933 | 0.561 | 0.933 | 0.000 | 13.467 |
| nearest_response_cells | 0.759 | 1.000 | 0.600 | 0.800 | 0.394 | 0.867 | 0.000 | 13.400 |
| transition_cells | 0.161 | 0.000 | 0.200 | 0.333 | 0.111 | 0.467 | 0.000 | 2.000 |

## Mejor decoder

`resonant_morphogenic_decoder` fue el mejor por `talk_score`.

## Ejemplos de habla

- Prompt: `hola, puedes generar palabras`
  Output: Hola, el campo activo esta despierto y puede responder.
- Prompt: `quien eres exactamente`
  Output: Soy un campo de celulas que recuerda patrones y produce respuestas.
- Prompt: `describe input campo latente decoder output`
  Output: La entrada activa el campo, el estado latente selecciona celulas y el decoder produce decoder morfogenetico.
- Prompt: `como aprende el sistema sin una red densa`
  Output: Aprendo cuando una experiencia refuerza celulas y mejora futuras palabras.
- Prompt: `quiero compararte con modelos clasicos`
  Output: La mejora se mide contra baselines clasicos y contra salidas generadas por mis celulas.
- Prompt: `explica una relacion con pasos claros`
  Output: Primero activo conceptos cercanos, luego compongo relaciones y finalmente doy una respuesta.
- Prompt: `inventa una idea para mejorar el decoder`
  Output: Una idea nueva combina decoder morfogenetico, memoria activa y una regla de composicion.
- Prompt: `que pasa dentro del estado activo`
  Output: El campo activo contiene memorias cercanas, intenciones probables y rutas de salida.
- Prompt: `ayudame a seguir con la prueba`
  Output: Puedo darte una respuesta util si el campo encuentra memorias relevantes.
- Prompt: `nos vemos luego`
  Output: Hasta luego, la memoria queda lista para la siguiente conversacion.
- Prompt: `puedes hablar conmigo`
  Output: Hola, el campo activo responde con una salida util.
- Prompt: `como conviertes memoria en frase`
  Output: El flujo es entrada, campo morfogenetico, estado activo, decoder morfogenetico y respuesta.
- Prompt: `que celulas recuerdan la experiencia`
  Output: Las celulas guardan memoria local y la usan para generar una respuesta estable.
- Prompt: `dame una salida creativa y util`
  Output: Una idea nueva es combinar memoria local, resonancia y una regla de composicion.
- Prompt: `orientame con el siguiente paso`
  Output: El siguiente paso es convertir la idea en una prueba pequena y medible.

## Lectura

El mejor resultado no sale de una capa densa. Sale de un campo de entrada que
activa intenciones y memorias locales, seguido por decoders de celulas: vecinos
de respuesta, transiciones token-a-token, frames con slots y resonancia entre
candidatos. El sistema ya puede producir frases completas y relevantes desde
prompts no vistos.
