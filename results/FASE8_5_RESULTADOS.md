# Fase 8.5 - Benchmark justo de generacion

Este benchmark compara generacion de respuestas desde prompts. No usa
perplexity, HellaSwag ni tareas que no correspondan al objetivo de AMF8.
Pythia-70M se evalua como lo que es: un modelo de completion puro, con
templates `Human/Assistant`, `Pregunta/Respuesta` y few-shot local.

## Tabla principal

| Metrica | AMF8 | Pythia-70M prompted |
|---|---:|---:|
| Relevancia/talk_score | 0.9047 | 0.1893 |
| Diversidad unique score | 1.0000 | 1.0000 |
| Inferencia total | 0.2455 s / 90 prompts | 40.2022 |
| Segundos por prompt | 0.0027 | 0.4467 |
| Entrenamiento local | 0.0945 s / 90 ejemplos | pretraining externo |
| Memoria modelo | 0.8897 MB | 267.0288 MB (cpu, fp32 estimate) |
| Necesita GPU practicamente | no | yes |
| Dispositivo medido | CPU | cpu |
| Usa LLM denso | no | yes |

## Estado de Pythia-70M

`measured`.

70M parametros implican aproximadamente 133.5 MB en fp16 y
267.0 MB en fp32, antes de overhead de runtime. En esta corrida,
el dispositivo reportado fue `cpu`.

Template principal: `raw_usuario_respuesta` con decoding
`greedy`.

Nota de justicia: la fila principal de Pythia usa el mejor resultado entre
templates en la suite final, es decir un upper bound optimista para Pythia. El
protocolo mas estricto, seleccionado solo con prompts de calibracion, fue
`raw_usuario_respuesta` / `top_p_seeded`
con score final `0.1800`.

## Plantillas Pythia probadas

| Template / decoding | score calibracion | score final | diversidad final |
|---|---:|---:|---:|
| raw_usuario_respuesta / greedy | 0.2430 | 0.1893 | 1.0000 |
| fewshot_qa / top_p_seeded | 0.2540 | 0.1804 | 1.0000 |
| raw_usuario_respuesta / top_p_seeded | 0.2820 | 0.1800 | 1.0000 |
| human_assistant / top_p_seeded | 0.1964 | 0.1752 | 1.0000 |
| human_assistant / greedy | 0.1528 | 0.1635 | 1.0000 |
| fewshot_usuario_asistente / top_p_seeded | 0.2040 | 0.1594 | 1.0000 |
| spanish_instruction / top_p_seeded | 0.1264 | 0.1544 | 1.0000 |
| fewshot_usuario_asistente / greedy | 0.1300 | 0.1386 | 1.0000 |
| fewshot_qa / greedy | 0.1383 | 0.1173 | 1.0000 |
| spanish_instruction / greedy | 0.1622 | 0.1128 | 1.0000 |

## Definicion exacta del scorer

`talk_score` no es una metrica universal de calidad linguistica. Es un scorer
local y auditable para esta suite controlada. Para cada output:

```text
complete = 1 si hay al menos 6 tokens y termina en ., ! o ?
has_verb = 1 si contiene un verbo de la lista cerrada de run_phase8.py
keyword_hit = 1 si aparece alguna keyword esperada del intent
relevance = overlap(output_tokens, intent_keywords) / numero_de_keywords
length_ok = 1 si la salida tiene entre 6 y 26 tokens
repetition_penalty = pares consecutivos repetidos / pares posibles
unique_output_rate = outputs unicos / prompts

talk_score =
  0.24 * complete_rate
+ 0.20 * verb_rate
+ 0.22 * keyword_hit_rate
+ 0.18 * relevance
+ 0.10 * length_ok_rate
+ 0.06 * unique_output_rate
- 0.12 * repetition_penalty
```

Sesgo conocido: este scorer favorece frases breves, bien cerradas, con verbos y
palabras clave esperadas. Eso es adecuado para medir si AMF8 adquirio capacidad
basica de respuesta en este dominio pequeno, pero no prueba calidad linguistica
general, conocimiento abierto, razonamiento largo, fluidez multilingue ni
preferencia humana. Pythia-70M puede producir texto mas largo o fuera del
vocabulario esperado y ser penalizado aunque ese texto sea linguisticamente mas
natural en otra evaluacion.

Por eso el claim de Fase 8.5 queda limitado: AMF8 gana esta prueba local de
respuesta controlada y eficiencia, no una competencia general de generacion de
lenguaje.


## Ventaja documentada

AMF8 aprende de 90 ejemplos locales en 0.0945 s,
corre en CPU, no usa LLM y no usa decoder denso. El claim valido no es
"Pythia falla porque se le dio un mal prompt"; eso quedo corregido. El claim
valido es que, en este dominio pequeno y con scorer de respuesta local, AMF8 es
mucho mas eficiente y se compara contra un Pythia configurado con templates
razonables para completion.

## Ejemplos AMF8

- Prompt: `hola, puedes generar palabras`
  AMF8: Hola, el campo activo esta despierto y puede responder.
- Prompt: `quien eres exactamente`
  AMF8: Soy un campo de celulas que recuerda patrones y produce respuestas.
- Prompt: `describe input campo latente decoder output`
  AMF8: La entrada activa el campo, el estado latente selecciona celulas y el decoder produce decoder morfogenetico.
- Prompt: `como aprende el sistema sin una red densa`
  AMF8: Aprendo cuando una experiencia refuerza celulas y mejora futuras palabras.
- Prompt: `quiero compararte con modelos clasicos`
  AMF8: La mejora se mide contra baselines clasicos y contra salidas generadas por mis celulas.
- Prompt: `explica una relacion con pasos claros`
  AMF8: Primero activo conceptos cercanos, luego compongo relaciones y finalmente doy una respuesta.
- Prompt: `inventa una idea para mejorar el decoder`
  AMF8: Una idea nueva combina decoder morfogenetico, memoria activa y una regla de composicion.
- Prompt: `que pasa dentro del estado activo`
  AMF8: El campo activo contiene memorias cercanas, intenciones probables y rutas de salida.

## Ejemplos Pythia-70M

- Prompt: `hola, puedes generar palabras`
  Pythia-70M: se ha visto a la tabla de las cárceles. En el caso del mundo y en los que no es un lugar donde está todo lo contraste con una mujer o
- Prompt: `quien eres exactamente`
  Pythia-70M: No, no. No." (preg_name = 'A' )
- Prompt: `describe input campo latente decoder output`
  Pythia-70M: A" is a bit-streaming encoding of the word, and it's an encoder. The decoding can be done in two ways (1) by using one or more bits to encode each
- Prompt: `como aprende el sistema sin una red densa`
  Pythia-70M: Ahora, no se es la de las que tienen unos cuentas". Por ejemplo. Se puede decirle a los ojos y pues en su lugar para entenderlo
- Prompt: `quiero compararte con modelos clasicos`
  Pythia-70M: se acepta la página de los que no es un tipo. Ahora, el mismo modo y las cosidas del mundo en este caso está bien para ser una cosa como lo h
