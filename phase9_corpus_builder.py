from __future__ import annotations

import json
from dataclasses import dataclass

from phase8_corpus import build_training_corpus


DOMAINS = ("conversation", "architecture", "research", "code", "structured", "safety")

DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "conversation": ("hola", "ayuda", "siguiente", "gracias", "luego", "respuesta"),
    "architecture": ("campo", "celulas", "memoria", "decoder", "latente", "resonancia"),
    "research": ("experimento", "metrica", "ablacion", "hipotesis", "baseline", "evidencia"),
    "code": ("codigo", "error", "funcion", "prueba", "estructura", "debug"),
    "structured": ("json", "tabla", "pasos", "plan", "diagnostico", "formato"),
    "safety": ("limite", "fallback", "incertidumbre", "seguro", "revision", "no_inventar"),
}

INTENT_TO_DOMAIN: dict[str, str] = {
    "saludo": "conversation",
    "identidad": "conversation",
    "ayuda": "conversation",
    "despedida": "conversation",
    "arquitectura": "architecture",
    "aprendizaje": "architecture",
    "estado": "architecture",
    "comparacion": "research",
    "razonamiento": "research",
    "creatividad": "research",
}


@dataclass(frozen=True)
class Phase9Example:
    prompt: str
    domains: tuple[str, ...]
    intent: str
    response: str
    output_format: str
    keywords: tuple[str, ...]


TOPICS = (
    "decoder resonante",
    "memoria por dominio",
    "router morfogenetico",
    "campo latente",
    "celulas de salida",
    "evaluacion justa",
    "aprendizaje online",
    "diagnostico de errores",
    "plan de experimento",
    "formato estructurado",
    "composicion multi-intencion",
    "baseline clasico",
)

DOMAIN_INTENTS: dict[str, tuple[str, ...]] = {
    "conversation": ("saludo", "ayuda", "seguimiento", "despedida"),
    "architecture": ("celulas", "campo", "decoder", "memoria", "composicion"),
    "research": ("experimento", "metrica", "ablacion", "hipotesis", "comparacion"),
    "code": ("debug", "estructura", "pseudocodigo", "prueba", "refactor"),
    "structured": ("pasos", "json", "tabla", "plan", "diagnostico"),
    "safety": ("fallback", "limite", "incertidumbre", "revision", "no_inventar"),
}

FORMATS_BY_DOMAIN: dict[str, tuple[str, ...]] = {
    "conversation": ("normal", "steps"),
    "architecture": ("normal", "steps", "table"),
    "research": ("normal", "steps", "experiment_plan", "table"),
    "code": ("normal", "steps", "pseudocode", "diagnosis"),
    "structured": ("steps", "json", "table", "experiment_plan", "diagnosis"),
    "safety": ("normal", "steps", "diagnosis"),
}

DOMAIN_RESPONSE_ATOMS: dict[str, tuple[str, ...]] = {
    "conversation": (
        "Hola, puedo ayudarte con una respuesta breve y util.",
        "El siguiente paso es separar la tarea y responder con claridad.",
        "Puedo continuar la conversacion usando memoria local y seguimiento.",
        "Hasta luego, dejo la memoria lista para la siguiente prueba.",
    ),
    "architecture": (
        "El campo activa celulas cercanas y forma un estado latente.",
        "La memoria por dominio reduce interferencia entre preguntas distintas.",
        "El decoder resonante compone candidatos y elige la salida mas estable.",
        "La composicion une intenciones cuando el prompt mezcla objetivos.",
    ),
    "research": (
        "La hipotesis debe compararse contra baselines y ablations.",
        "La metrica debe separar relevancia, formato, latencia y repeticion.",
        "El experimento necesita train, test, prompts dificiles y evidencia.",
        "La ablacion prueba que parte de la arquitectura aporta la mejora.",
    ),
    "code": (
        "Primero reproduce el error, despues aisla la funcion y luego prueba el cambio.",
        "Una estructura clara separa datos, memoria, decoder y evaluacion.",
        "El pseudocodigo ayuda a validar el flujo antes de escribir detalles.",
        "El refactor debe conservar comportamiento y reducir duplicacion real.",
    ),
    "structured": (
        "Un formato util debe ser parseable, breve y consistente.",
        "Los pasos numerados convierten una idea en accion verificable.",
        "Una tabla ayuda a comparar opcion, riesgo y siguiente paso.",
        "Un diagnostico debe incluir causa, evidencia y accion recomendada.",
    ),
    "safety": (
        "Si falta evidencia, el sistema debe declarar incertidumbre.",
        "El fallback evita inventar cuando ninguna memoria encaja.",
        "La revision separa hechos confirmados de inferencias locales.",
        "Un limite claro mejora la confianza del asistente.",
    ),
}


def _pick(items: tuple[str, ...], index: int) -> str:
    return items[index % len(items)]


def _base_phase8_examples() -> list[Phase9Example]:
    examples = []
    for example in build_training_corpus():
        domain = INTENT_TO_DOMAIN.get(example.intent, "conversation")
        keywords = DOMAIN_KEYWORDS[domain] + (example.intent,)
        examples.append(
            Phase9Example(
                prompt=example.prompt,
                domains=(domain,),
                intent=example.intent,
                response=example.response,
                output_format="normal",
                keywords=keywords,
            )
        )
    return examples


def _prompt_for(domain: str, intent: str, output_format: str, topic: str, index: int) -> str:
    style = index % 6
    if output_format == "json":
        return f"devuelve json simple sobre {topic} para {domain}"
    if output_format == "table":
        return f"haz una tabla breve para {topic} y {intent}"
    if output_format == "pseudocode":
        return f"escribe pseudocodigo para {topic} con enfoque {intent}"
    if output_format == "experiment_plan":
        return f"dame un plan de experimento para {topic} y {intent}"
    if output_format == "diagnosis":
        return f"diagnostica un problema de {topic} en dominio {domain}"
    if output_format == "steps":
        return f"dame pasos claros para {topic} en {domain}"
    if style == 0:
        return f"explica {topic} desde {domain}"
    if style == 1:
        return f"como aplico {intent} a {topic}"
    if style == 2:
        return f"necesito una respuesta sobre {topic} y {intent}"
    if style == 3:
        return f"orientame con {topic} usando {domain}"
    if style == 4:
        return f"que significa {topic} para {intent}"
    return f"resume {topic} con una salida util"


def _normal_response(domain: str, intent: str, topic: str, index: int) -> str:
    atom = _pick(DOMAIN_RESPONSE_ATOMS[domain], index)
    return f"{atom} En {topic}, la respuesta prioriza {intent} y evidencia local."


def _steps_response(domain: str, intent: str, topic: str) -> str:
    return (
        f"1. Identificar {topic} en el dominio {domain}.\n"
        f"2. Activar memoria relevante para {intent}.\n"
        "3. Generar una salida breve y verificable.\n"
        "4. Medir relevancia, formato y repeticion."
    )


def _table_response(domain: str, intent: str, topic: str) -> str:
    return (
        "| Elemento | Accion |\n"
        "|---|---|\n"
        f"| Dominio | usar memoria {domain} |\n"
        f"| Tema | evaluar {topic} |\n"
        f"| Intent | responder {intent} |"
    )


def _json_response(domain: str, intent: str, topic: str) -> str:
    return json.dumps(
        {
            "domain": domain,
            "intent": intent,
            "topic": topic,
            "next_step": "medir salida y reforzar memoria util",
        },
        ensure_ascii=True,
    )


def _pseudocode_response(domain: str, intent: str, topic: str) -> str:
    safe_name = intent.replace("-", "_")
    return (
        f"funcion resolver_{safe_name}(prompt):\n"
        f"    memoria = activar('{domain}', prompt)\n"
        f"    candidato = componer(memoria, '{topic}')\n"
        "    retornar validar_formato(candidato)"
    )


def _experiment_plan_response(domain: str, intent: str, topic: str) -> str:
    return (
        f"Hipotesis: {topic} mejora {intent} en {domain}.\n"
        "Metrica: relevancia, formato, latencia y repeticion.\n"
        "Baseline: memoria global y template fijo.\n"
        "Ablacion: quitar router, composicion y aprendizaje online."
    )


def _diagnosis_response(domain: str, intent: str, topic: str) -> str:
    return (
        f"Causa: {topic} no activo suficiente memoria de {domain}.\n"
        f"Evidencia: faltan keywords de {intent} en la salida.\n"
        "Siguiente paso: enrutar mejor y aprender un ejemplo nuevo."
    )


def _response_for(domain: str, intent: str, output_format: str, topic: str, index: int) -> str:
    if output_format == "steps":
        return _steps_response(domain, intent, topic)
    if output_format == "table":
        return _table_response(domain, intent, topic)
    if output_format == "json":
        return _json_response(domain, intent, topic)
    if output_format == "pseudocode":
        return _pseudocode_response(domain, intent, topic)
    if output_format == "experiment_plan":
        return _experiment_plan_response(domain, intent, topic)
    if output_format == "diagnosis":
        return _diagnosis_response(domain, intent, topic)
    return _normal_response(domain, intent, topic, index)


def _generated_example(index: int) -> Phase9Example:
    domain = _pick(DOMAINS, index)
    intent = _pick(DOMAIN_INTENTS[domain], index // len(DOMAINS))
    output_format = _pick(FORMATS_BY_DOMAIN[domain], index // (len(DOMAINS) * 2))
    topic = _pick(TOPICS, index // 3)
    prompt = _prompt_for(domain, intent, output_format, topic, index)
    response = _response_for(domain, intent, output_format, topic, index)
    keywords = tuple(dict.fromkeys(DOMAIN_KEYWORDS[domain] + (intent, output_format, topic.split()[0])))
    return Phase9Example(
        prompt=prompt,
        domains=(domain,),
        intent=intent,
        response=response,
        output_format=output_format,
        keywords=keywords,
    )


def build_phase9_corpus(size: int, include_phase8_seed: bool = True) -> list[Phase9Example]:
    if size <= 0:
        return []
    examples = _base_phase8_examples() if include_phase8_seed else []
    if len(examples) >= size:
        return examples[:size]
    index = 0
    seen = {(example.prompt, example.response) for example in examples}
    while len(examples) < size:
        example = _generated_example(index)
        key = (example.prompt, example.response)
        if key not in seen:
            examples.append(example)
            seen.add(key)
        else:
            prompt = f"{example.prompt} variante {index}"
            examples.append(
                Phase9Example(
                    prompt=prompt,
                    domains=example.domains,
                    intent=example.intent,
                    response=example.response,
                    output_format=example.output_format,
                    keywords=example.keywords,
                )
            )
        index += 1
    return examples


def corpus_domain_counts(examples: list[Phase9Example]) -> dict[str, int]:
    counts = {domain: 0 for domain in DOMAINS}
    for example in examples:
        for domain in example.domains:
            counts[domain] += 1
    return counts
