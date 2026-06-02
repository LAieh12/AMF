# Fase 9 - AMF8 Domain Expansion

Objetivo: convertir la demo de habla controlada en un asistente morfogenetico
servible, escalando por dominios y memorias, no por capas densas.

Arquitectura:

```text
prompt -> domain router -> domain memory -> resonant morphogenic assistant -> structured output
```

Reglas: no LLM = True, no decoder denso =
True, no backprop = True.

Dominios: conversation, architecture, research, code, structured, safety.

## Escala AMF9

| examples | service | talk | domain | format | comp | relevance | rep | diversity | fit s | ms/prompt | memory MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 90 | 0.8267 | 0.8022 | 0.8583 | 1.0000 | 0.7500 | 0.2476 | 0.0000 | 0.9500 | 0.6115 | 0.8901 | 0.2653 |
| 300 | 0.9088 | 0.8686 | 0.9833 | 1.0000 | 1.0000 | 0.2938 | 0.0000 | 0.9500 | 3.3483 | 1.2374 | 0.8830 |
| 1000 | 0.9137 | 0.8728 | 1.0000 | 1.0000 | 1.0000 | 0.3101 | 0.0000 | 0.9500 | 13.3113 | 0.9566 | 2.8442 |
| 3000 | 0.9137 | 0.8728 | 1.0000 | 1.0000 | 1.0000 | 0.3098 | 0.0000 | 0.9500 | 41.3510 | 1.1707 | 8.5203 |
| 10000 | 0.9132 | 0.8720 | 1.0000 | 1.0000 | 1.0000 | 0.3057 | 0.0000 | 0.9500 | 137.4498 | 5.4907 | 28.4242 |

## Baselines

| examples | system | service | talk | domain | format | comp | ms/prompt | memory MB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 90 | domain_template_router | 0.6593 | 0.6532 | 0.6917 | 0.7500 | 0.6000 | 0.1783 | 0.0367 |
| 90 | global_nearest_memory | 0.4260 | 0.4869 | 0.2583 | 0.2500 | 0.3500 | 0.1683 | 0.1243 |
| 300 | domain_template_router | 0.7443 | 0.7042 | 0.9167 | 0.7500 | 0.9000 | 0.2564 | 0.0757 |
| 300 | global_nearest_memory | 0.5823 | 0.6165 | 0.3667 | 0.6500 | 0.3500 | 0.3832 | 0.4659 |
| 1000 | domain_template_router | 0.7443 | 0.7042 | 0.9167 | 0.7500 | 0.9000 | 0.2982 | 0.1058 |
| 1000 | global_nearest_memory | 0.6261 | 0.6577 | 0.4667 | 0.7000 | 0.3500 | 1.1062 | 1.6062 |
| 10000 | domain_template_router | 0.7443 | 0.7042 | 0.9167 | 0.7500 | 0.9000 | 0.2049 | 0.8240 |
| 10000 | global_nearest_memory | 0.6261 | 0.6577 | 0.4667 | 0.7000 | 0.3500 | 9.1672 | 16.3005 |

## Aprendizaje online

| examples | relevance antes | relevance despues | learn seconds | improved |
|---:|---:|---:|---:|---|
| 1000 | 0.0000 | 1.0000 | 0.0005 | True |
| 3000 | 0.2500 | 1.0000 | 0.0005 | True |
| 10000 | 0.2500 | 1.0000 | 0.0015 | True |

## Salidas de ejemplo en el tamano mayor

- Prompt: `hola, ayudame a seguir con una prueba`
  Domains: conversation, code, research
  Output: Hola, tomo la solicitud y mantengo el seguimiento. La evaluacion compara metrica, baseline y ablacion. El codigo se revisa con reproduccion, prueba y refactor.
- Prompt: `explica como el campo activa celulas de memoria`
  Domains: architecture, conversation, research
  Output: La memoria por dominio reduce interferencia entre preguntas distintas. En decoder resonante, la respuesta prioriza celulas y evidencia local.
- Prompt: `dame pasos para probar el decoder resonante`
  Domains: architecture, structured, research
  Output: 1. Identificar decoder resonante en el dominio research.
2. Activar memoria relevante para comparacion.
3. Generar una salida breve y verificable.
4. Medir relevancia, formato y repeticion.
- Prompt: `devuelve json simple para una prueba de memoria online`
  Domains: structured, architecture, conversation
  Output: {"domain": ["structured", "architecture", "conversation"], "intent": "tabla", "next_step": "activar memoria por dominio y medir formato", "evidence": "router, celulas activas y decoder resonante"}
- Prompt: `haz una tabla que compare memoria global y memoria por dominio`
  Domains: structured, architecture, research
  Output: | Elemento | Accion |
|---|---|
| Dominio | usar memoria structured |
| Tema | evaluar memoria por dominio |
| Intent | responder diagnostico |
- Prompt: `propone un experimento con metrica y ablacion`
  Domains: research, structured, conversation
  Output: Hipotesis: composicion multi-intencion mejora ablacion en research.
Metrica: relevancia, formato, latencia y repeticion.
Baseline: memoria global y template fijo.
Ablacion: quitar router, composicion y aprendizaje online.
- Prompt: `compara AMF8 con Pythia y dame el siguiente paso`
  Domains: research, conversation, architecture
  Output: 1. Activar memoria de research, conversation y architecture.
2. Seleccionar celulas relevantes para comparacion.
3. Componer una respuesta con formato verificable.
4. Medir relevancia, latencia, diversidad y repeticion.
- Prompt: `escribe pseudocodigo para enrutar un prompt al dominio correcto`
  Domains: code, architecture, safety
  Output: funcion responder(prompt):
    dominios = enrutar(prompt, candidatos=['code', 'architecture', 'safety'])
    memoria = activar_celulas(dominios, intent='pseudocodigo')
    salida = decoder_resonante(memoria)
    retornar validar_formato(salida)

## Lectura

Fase 9 separa memorias por dominio: conversation, architecture, research, code,
structured y safety. El prompt activa un router de dominio, luego solo consulta
las memorias relevantes y finalmente compone una salida normal o estructurada.
Esto reduce interferencia entre saludos, arquitectura, research, codigo y
formatos utiles.

El score sigue siendo local y auditable, no una metrica universal de calidad de
lenguaje. La evidencia que importa aqui es la curva de escala: score, dominio,
formato, composicion, repeticion, latencia, memoria y aprendizaje online.
