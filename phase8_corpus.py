from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConversationExample:
    prompt: str
    intent: str
    response: str


@dataclass(frozen=True)
class PromptTest:
    prompt: str
    intent: str
    keywords: tuple[str, ...]


INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "saludo": ("hola", "listo", "ayudar"),
    "identidad": ("soy", "campo", "morfogenetico"),
    "arquitectura": ("entrada", "campo", "decoder", "salida"),
    "aprendizaje": ("aprendo", "celulas", "experiencia"),
    "comparacion": ("clasicos", "comparar", "pruebas"),
    "razonamiento": ("relacion", "paso", "respuesta"),
    "creatividad": ("idea", "nueva", "combinar"),
    "estado": ("activo", "latente", "memoria"),
    "ayuda": ("puedo", "ayudarte", "paso"),
    "despedida": ("hasta", "luego", "aprendido"),
}


PROMPTS: dict[str, tuple[str, ...]] = {
    "saludo": (
        "hola",
        "buenos dias",
        "hey sistema",
        "hola puedes hablar",
        "saludame con una frase",
    ),
    "identidad": (
        "que eres",
        "quien eres",
        "define tu identidad",
        "eres una red normal",
        "explica que tipo de sistema eres",
    ),
    "arquitectura": (
        "explica la arquitectura",
        "como fluye la informacion",
        "describe el decoder morfogenetico",
        "como conviertes entrada en salida",
        "que pasa entre input y output",
    ),
    "aprendizaje": (
        "como aprendes",
        "como recuerdas ejemplos",
        "que hacen tus celulas",
        "como usas la experiencia",
        "puedes aprender sin backprop",
    ),
    "comparacion": (
        "comparame con modelos clasicos",
        "como pruebas si eres mejor",
        "que mides contra los baselines",
        "puedes competir con modelos clasicos",
        "como sabes si mejoras",
    ),
    "razonamiento": (
        "razona paso a paso",
        "como conectas ideas",
        "puedes explicar una relacion",
        "dame una respuesta razonada",
        "como compones conceptos",
    ),
    "creatividad": (
        "dame una idea nueva",
        "crea una propuesta",
        "imagina otra opcion",
        "combina dos conceptos",
        "quiero una salida creativa",
    ),
    "estado": (
        "que ocurre en tu estado latente",
        "que celulas estan activas",
        "describe el campo activo",
        "como se ve tu memoria activa",
        "que significa resonancia",
    ),
    "ayuda": (
        "ayudame con esto",
        "que puedes hacer por mi",
        "necesito una respuesta util",
        "dame el siguiente paso",
        "orientame con una tarea",
    ),
    "despedida": (
        "adios",
        "hasta luego",
        "terminamos por ahora",
        "cierra la conversacion",
        "nos vemos despues",
    ),
}


RESPONSES: dict[str, tuple[str, ...]] = {
    "saludo": (
        "Hola, estoy listo para ayudarte con una idea concreta.",
        "Hola, el campo activo esta despierto y puede responder.",
        "Hola, puedo hablar con frases simples y utiles.",
    ),
    "identidad": (
        "Soy un sistema morfogenetico que activa memorias locales y genera salida.",
        "Soy un campo de celulas que recuerda patrones y produce respuestas.",
        "Soy una arquitectura local, no un decoder denso pegado al final.",
    ),
    "arquitectura": (
        "La entrada activa un campo latente, el campo selecciona celulas y el decoder produce salida.",
        "El flujo es entrada, campo morfogenetico, estado activo, decoder morfogenetico y respuesta.",
        "La arquitectura convierte senales de entrada en una frase mediante celulas de salida.",
    ),
    "aprendizaje": (
        "Aprendo creando celulas, ajustando memorias locales y reforzando transiciones utiles.",
        "La experiencia modifica celulas cercanas y deja rutas para futuras respuestas.",
        "No uso backprop denso; uso memoria local, activacion y composicion.",
    ),
    "comparacion": (
        "Me comparo con modelos clasicos usando pruebas, metricas y ejemplos no vistos.",
        "La mejora se mide contra baselines clasicos y contra salidas generadas por mis celulas.",
        "Si supero los modelos clasicos, la evidencia aparece en accuracy, relevancia y robustez.",
    ),
    "razonamiento": (
        "Primero activo conceptos cercanos, luego compongo relaciones y finalmente doy una respuesta.",
        "Una relacion aparece cuando varias celulas coinciden y refuerzan la misma conclusion.",
        "Puedo razonar de forma local conectando pasos cortos en una secuencia estable.",
    ),
    "creatividad": (
        "Una idea nueva es combinar memoria local, resonancia y una regla de composicion.",
        "Puedo crear una propuesta si mezclo celulas cercanas con una salida controlada.",
        "La creatividad aparece cuando dos patrones activos producen una ruta nueva.",
    ),
    "estado": (
        "El estado latente es un mapa de celulas activas con pesos de resonancia.",
        "El campo activo contiene memorias cercanas, intenciones probables y rutas de salida.",
        "La resonancia significa que varias celulas apoyan la misma direccion de respuesta.",
    ),
    "ayuda": (
        "Puedo ayudarte separando la tarea, activando ejemplos y proponiendo el siguiente paso.",
        "El siguiente paso es convertir la idea en una prueba pequena y medible.",
        "Puedo darte una respuesta util si el campo encuentra memorias relevantes.",
    ),
    "despedida": (
        "Hasta luego, el campo conserva lo aprendido para la siguiente prueba.",
        "Nos vemos despues, la memoria queda lista para otra conversacion.",
        "Cierro la conversacion con una respuesta breve y estable.",
    ),
}


def build_training_corpus() -> list[ConversationExample]:
    examples: list[ConversationExample] = []
    for intent, prompts in PROMPTS.items():
        responses = RESPONSES[intent]
        for i, prompt in enumerate(prompts):
            for j, response in enumerate(responses):
                if (i + j) % 2 == 0 or i == 0:
                    examples.append(ConversationExample(prompt=prompt, intent=intent, response=response))
    return examples


def build_prompt_suite() -> list[PromptTest]:
    return [
        PromptTest("hola, puedes generar palabras", "saludo", INTENT_KEYWORDS["saludo"]),
        PromptTest("quien eres exactamente", "identidad", INTENT_KEYWORDS["identidad"]),
        PromptTest("describe input campo latente decoder output", "arquitectura", INTENT_KEYWORDS["arquitectura"]),
        PromptTest("como aprende el sistema sin una red densa", "aprendizaje", INTENT_KEYWORDS["aprendizaje"]),
        PromptTest("quiero compararte con modelos clasicos", "comparacion", INTENT_KEYWORDS["comparacion"]),
        PromptTest("explica una relacion con pasos claros", "razonamiento", INTENT_KEYWORDS["razonamiento"]),
        PromptTest("inventa una idea para mejorar el decoder", "creatividad", INTENT_KEYWORDS["creatividad"]),
        PromptTest("que pasa dentro del estado activo", "estado", INTENT_KEYWORDS["estado"]),
        PromptTest("ayudame a seguir con la prueba", "ayuda", INTENT_KEYWORDS["ayuda"]),
        PromptTest("nos vemos luego", "despedida", INTENT_KEYWORDS["despedida"]),
        PromptTest("puedes hablar conmigo", "saludo", INTENT_KEYWORDS["saludo"]),
        PromptTest("como conviertes memoria en frase", "arquitectura", INTENT_KEYWORDS["arquitectura"]),
        PromptTest("que celulas recuerdan la experiencia", "aprendizaje", INTENT_KEYWORDS["aprendizaje"]),
        PromptTest("dame una salida creativa y util", "creatividad", INTENT_KEYWORDS["creatividad"]),
        PromptTest("orientame con el siguiente paso", "ayuda", INTENT_KEYWORDS["ayuda"]),
    ]
