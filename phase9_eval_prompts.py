from __future__ import annotations

from dataclasses import dataclass

from phase9_corpus_builder import DOMAIN_KEYWORDS


@dataclass(frozen=True)
class Phase9EvalPrompt:
    prompt: str
    domains: tuple[str, ...]
    intents: tuple[str, ...]
    output_format: str
    keywords: tuple[str, ...]
    requires_composition: bool = False


def _kw(*domains: str, extra: tuple[str, ...] = ()) -> tuple[str, ...]:
    values: list[str] = []
    for domain in domains:
        values.extend(DOMAIN_KEYWORDS[domain])
    values.extend(extra)
    return tuple(dict.fromkeys(values))


def build_phase9_eval_prompts() -> list[Phase9EvalPrompt]:
    return [
        Phase9EvalPrompt(
            "hola, ayudame a seguir con una prueba",
            ("conversation",),
            ("saludo", "ayuda"),
            "normal",
            _kw("conversation", extra=("hola", "ayuda")),
            True,
        ),
        Phase9EvalPrompt(
            "explica como el campo activa celulas de memoria",
            ("architecture",),
            ("campo", "memoria"),
            "normal",
            _kw("architecture", extra=("campo", "celulas", "memoria")),
        ),
        Phase9EvalPrompt(
            "dame pasos para probar el decoder resonante",
            ("architecture", "structured"),
            ("decoder", "pasos"),
            "steps",
            _kw("architecture", "structured", extra=("decoder", "pasos")),
            True,
        ),
        Phase9EvalPrompt(
            "devuelve json simple para una prueba de memoria online",
            ("structured", "architecture"),
            ("json", "memoria"),
            "json",
            _kw("structured", "architecture", extra=("json", "memoria")),
            True,
        ),
        Phase9EvalPrompt(
            "haz una tabla que compare memoria global y memoria por dominio",
            ("structured", "architecture"),
            ("tabla", "memoria"),
            "table",
            _kw("structured", "architecture", extra=("tabla", "memoria")),
            True,
        ),
        Phase9EvalPrompt(
            "propone un experimento con metrica y ablacion",
            ("research",),
            ("experimento", "ablacion"),
            "experiment_plan",
            _kw("research", extra=("experimento", "metrica", "ablacion")),
        ),
        Phase9EvalPrompt(
            "compara AMF8 con Pythia y dame el siguiente paso",
            ("research", "conversation"),
            ("comparacion", "seguimiento"),
            "steps",
            _kw("research", "conversation", extra=("baseline", "siguiente")),
            True,
        ),
        Phase9EvalPrompt(
            "escribe pseudocodigo para enrutar un prompt al dominio correcto",
            ("code", "architecture"),
            ("pseudocodigo", "router"),
            "pseudocode",
            _kw("code", "architecture", extra=("pseudocodigo", "router")),
            True,
        ),
        Phase9EvalPrompt(
            "diagnostica por que una respuesta mezcla saludo con arquitectura rara",
            ("code", "safety", "architecture"),
            ("diagnostico", "fallback"),
            "diagnosis",
            _kw("code", "safety", "architecture", extra=("diagnostico", "saludo")),
            True,
        ),
        Phase9EvalPrompt(
            "necesito estructura de archivos para fase 9",
            ("code",),
            ("estructura",),
            "steps",
            _kw("code", extra=("estructura", "fase")),
        ),
        Phase9EvalPrompt(
            "que metrica evita que el sistema repita palabras",
            ("research",),
            ("metrica",),
            "normal",
            _kw("research", extra=("metrica", "repeticion")),
        ),
        Phase9EvalPrompt(
            "orientame con un plan de experimento para evaluar composicion",
            ("research", "structured"),
            ("plan", "composicion"),
            "experiment_plan",
            _kw("research", "structured", extra=("plan", "composicion")),
            True,
        ),
        Phase9EvalPrompt(
            "si no sabes la respuesta, que deberia hacer el fallback",
            ("safety",),
            ("fallback", "incertidumbre"),
            "normal",
            _kw("safety", extra=("fallback", "incertidumbre")),
        ),
        Phase9EvalPrompt(
            "dame una salida normal sobre resonancia y celulas",
            ("architecture",),
            ("celulas", "resonancia"),
            "normal",
            _kw("architecture", extra=("resonancia", "celulas")),
        ),
        Phase9EvalPrompt(
            "crea pasos para debuggear un decoder que no produce json",
            ("code", "structured"),
            ("debug", "json"),
            "steps",
            _kw("code", "structured", extra=("debug", "json")),
            True,
        ),
        Phase9EvalPrompt(
            "haz json con domain intent y next_step para una tarea de research",
            ("structured", "research"),
            ("json", "research"),
            "json",
            _kw("structured", "research", extra=("json", "next_step")),
            True,
        ),
        Phase9EvalPrompt(
            "necesito una tabla de riesgos para aprendizaje online",
            ("structured", "research"),
            ("tabla", "online"),
            "table",
            _kw("structured", "research", extra=("tabla", "online")),
            True,
        ),
        Phase9EvalPrompt(
            "hola, explica el decoder y termina con un siguiente paso",
            ("conversation", "architecture"),
            ("saludo", "decoder", "seguimiento"),
            "steps",
            _kw("conversation", "architecture", extra=("hola", "decoder", "siguiente")),
            True,
        ),
        Phase9EvalPrompt(
            "propone una ablacion para quitar el router de dominio",
            ("research", "architecture"),
            ("ablacion", "router"),
            "experiment_plan",
            _kw("research", "architecture", extra=("ablacion", "router")),
            True,
        ),
        Phase9EvalPrompt(
            "diagnostica una latencia alta al crecer a diez mil memorias",
            ("code", "research", "safety"),
            ("diagnostico", "latencia"),
            "diagnosis",
            _kw("code", "research", "safety", extra=("latencia", "memoria")),
            True,
        ),
    ]
