from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass

from phase8_morphogenic_decoder import tokenize
from phase9_corpus_builder import DOMAIN_KEYWORDS, Phase9Example
from phase9_domain_memory import Phase9ActiveState, Phase9DomainField


FORMAT_HINTS: dict[str, tuple[str, ...]] = {
    "json": ("json",),
    "table": ("tabla", "cuadro"),
    "steps": ("pasos", "siguiente", "lista", "plan"),
    "pseudocode": ("pseudocodigo", "funcion", "algoritmo"),
    "experiment_plan": ("experimento", "hipotesis", "ablacion", "metrica"),
    "diagnosis": ("diagnostica", "diagnostico", "causa", "error", "latencia"),
}


@dataclass
class GenerationResult:
    output: str
    domains: tuple[str, ...]
    output_format: str
    source: str


class Phase9MorphogenicAssistant:
    name = "phase9_domain_resonant_assistant"

    def __init__(self, dims: int = 256, top_k: int = 9, radius: float = 0.42):
        self.field = Phase9DomainField(dims=dims, top_k=top_k, radius=radius)
        self.examples: list[Phase9Example] = []

    def fit(self, examples: list[Phase9Example]) -> "Phase9MorphogenicAssistant":
        self.examples = list(examples)
        self.field.fit(self.examples)
        self.intent_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.format_counts: Counter[str] = Counter()
        for example in self.examples:
            self.intent_counts[example.domains[0]][example.intent] += 1
            self.format_counts[example.output_format] += 1
        return self

    def learn(self, example: Phase9Example) -> None:
        self.examples.append(example)
        self.field.learn(example)
        self.intent_counts[example.domains[0]][example.intent] += 1
        self.format_counts[example.output_format] += 1

    def detect_format(self, prompt: str, state: Phase9ActiveState) -> str:
        tokens = set(tokenize(prompt))
        if "estructura" in tokens and ("archivo" in tokens or "archivos" in tokens):
            return "steps"
        if tokens & {"pasos", "lista", "siguiente"}:
            if "json" in tokens and tokens & {"devuelve", "haz"} and "pasos" not in tokens:
                return "json"
            return "steps"
        if "json" in tokens:
            return "json"
        for output_format, hints in FORMAT_HINTS.items():
            if tokens & set(hints):
                if output_format == "steps" and {"experimento", "hipotesis", "ablacion"} & tokens:
                    return "experiment_plan"
                return output_format
        return "normal"

    def _best_intent(self, state: Phase9ActiveState) -> str:
        votes: Counter[str] = Counter()
        for activation in state.activations:
            for cell, weight in activation.cells:
                votes[cell.example.intent] += weight * activation.score
        if votes:
            return votes.most_common(1)[0][0]
        return "fallback"

    def _nearest_response(self, state: Phase9ActiveState, output_format: str) -> str:
        scores: dict[str, float] = defaultdict(float)
        prompt_tokens = set(tokenize(state.prompt))
        for activation in state.activations:
            for cell, weight in activation.cells:
                score = weight * activation.score
                cell_prompt_tokens = set(tokenize(cell.example.prompt))
                overlap = len(prompt_tokens & cell_prompt_tokens) / max(1, len(prompt_tokens))
                if cell.example.prompt == state.prompt:
                    score *= 6.0
                elif overlap >= 0.70:
                    score *= 2.5
                if cell.example.output_format == output_format:
                    score *= 1.45
                scores[cell.example.response] += score
        if not scores:
            return "No tengo suficiente memoria para responder con seguridad."
        return max(scores.items(), key=lambda item: item[1])[0]

    def _domain_phrase(self, domains: tuple[str, ...]) -> str:
        if len(domains) == 1:
            return domains[0]
        return ", ".join(domains[:-1]) + " y " + domains[-1]

    def _normal_frame(self, prompt: str, domains: tuple[str, ...], intent: str) -> str:
        domain = domains[0] if domains else "safety"
        keywords = ", ".join(DOMAIN_KEYWORDS.get(domain, ())[:3])
        if "conversation" in domains and len(domains) > 1:
            return (
                f"Hola, activo {self._domain_phrase(domains)} y respondo sobre {intent} "
                f"con memoria local, {keywords} y un siguiente paso claro."
            )
        if domain == "architecture":
            return f"El campo usa celulas, memoria y decoder para responder sobre {intent} con estado latente."
        if domain == "research":
            return f"La respuesta propone experimento, metrica y baseline para validar {intent} con evidencia."
        if domain == "code":
            return f"Primero reviso codigo, error y prueba; despues propongo una estructura para {intent}."
        if domain == "structured":
            return f"El formato debe ser parseable, breve y verificable para {intent}."
        if domain == "safety":
            return f"Si falta evidencia, activo fallback, declaro incertidumbre y pido revision."
        return f"Puedo responder sobre {intent} usando memoria local y salida breve."

    def _steps_frame(self, domains: tuple[str, ...], intent: str) -> str:
        return (
            f"1. Activar memoria de {self._domain_phrase(domains)}.\n"
            f"2. Seleccionar celulas relevantes para {intent}.\n"
            "3. Componer una respuesta con formato verificable.\n"
            "4. Medir relevancia, latencia, diversidad y repeticion."
        )

    def _table_frame(self, domains: tuple[str, ...], intent: str) -> str:
        return (
            "| Elemento | Resultado |\n"
            "|---|---|\n"
            f"| Dominio | {self._domain_phrase(domains)} |\n"
            f"| Intent | {intent} |\n"
            "| Accion | activar memoria, componer salida y medir evidencia |"
        )

    def _json_frame(self, domains: tuple[str, ...], intent: str) -> str:
        return json.dumps(
            {
                "domain": list(domains),
                "intent": intent,
                "next_step": "activar memoria por dominio y medir formato",
                "evidence": "router, celulas activas y decoder resonante",
            },
            ensure_ascii=True,
        )

    def _pseudocode_frame(self, domains: tuple[str, ...], intent: str) -> str:
        return (
            "funcion responder(prompt):\n"
            f"    dominios = enrutar(prompt, candidatos={list(domains)})\n"
            f"    memoria = activar_celulas(dominios, intent='{intent}')\n"
            "    salida = decoder_resonante(memoria)\n"
            "    retornar validar_formato(salida)"
        )

    def _experiment_frame(self, domains: tuple[str, ...], intent: str) -> str:
        return (
            f"Hipotesis: {intent} mejora cuando se enruta por {self._domain_phrase(domains)}.\n"
            "Metrica: relevancia, formato, composicion, latencia y memoria.\n"
            "Baseline: memoria global y template fijo.\n"
            "Ablacion: quitar router, quitar composicion y quitar aprendizaje online."
        )

    def _diagnosis_frame(self, domains: tuple[str, ...], intent: str) -> str:
        return (
            f"Causa: el prompt mezcla {self._domain_phrase(domains)} y puede activar memorias rivales.\n"
            f"Evidencia: revisar dominio, intent {intent}, latencia y keywords ausentes.\n"
            "Siguiente paso: reforzar router, aprender ejemplo online y repetir la prueba."
        )

    def _frame_response(self, prompt: str, domains: tuple[str, ...], intent: str, output_format: str) -> str:
        if output_format == "steps":
            return self._steps_frame(domains, intent)
        if output_format == "table":
            return self._table_frame(domains, intent)
        if output_format == "json":
            return self._json_frame(domains, intent)
        if output_format == "pseudocode":
            return self._pseudocode_frame(domains, intent)
        if output_format == "experiment_plan":
            return self._experiment_frame(domains, intent)
        if output_format == "diagnosis":
            return self._diagnosis_frame(domains, intent)
        return self._normal_frame(prompt, domains, intent)

    def _composition_response(self, prompt: str, state: Phase9ActiveState, output_format: str) -> str:
        domains = tuple(domain for domain, _ in state.domains[:3])
        intent = self._best_intent(state)
        if len(domains) <= 1:
            return self._frame_response(prompt, domains, intent, output_format)
        if output_format == "normal":
            parts = []
            if "conversation" in domains:
                parts.append("Hola, tomo la solicitud y mantengo el seguimiento.")
            if "architecture" in domains:
                parts.append("La arquitectura activa campo, celulas, memoria y decoder.")
            if "research" in domains:
                parts.append("La evaluacion compara metrica, baseline y ablacion.")
            if "code" in domains:
                parts.append("El codigo se revisa con reproduccion, prueba y refactor.")
            if "structured" in domains:
                parts.append("El formato final debe ser parseable y verificable.")
            if "safety" in domains:
                parts.append("Si falta evidencia, uso fallback y declaro limite.")
            return " ".join(parts[:3])
        return self._frame_response(prompt, domains, intent, output_format)

    def _format_bonus(self, output: str, output_format: str) -> float:
        stripped = output.strip()
        if output_format == "json":
            try:
                json.loads(stripped)
                return 3.0
            except json.JSONDecodeError:
                return -2.0
        if output_format == "table":
            return 2.5 if "|" in output and "---" in output else -1.0
        if output_format == "steps":
            return 2.5 if "1." in output and "2." in output else -1.0
        if output_format == "pseudocode":
            return 2.5 if "funcion" in output and "retornar" in output else -1.0
        if output_format == "experiment_plan":
            return 2.5 if "Hipotesis:" in output and "Metrica:" in output and "Baseline:" in output else -1.0
        if output_format == "diagnosis":
            return 2.5 if "Causa:" in output and "Siguiente paso:" in output else -1.0
        return 1.0 if stripped.endswith((".", "!", "?")) else -0.5

    def _score_candidate(self, prompt: str, output: str, state: Phase9ActiveState, output_format: str) -> float:
        tokens = tokenize(output)
        prompt_tokens = set(tokenize(prompt))
        token_set = set(tokens)
        domain_tokens = set()
        for domain, _ in state.domains:
            domain_tokens.update(DOMAIN_KEYWORDS.get(domain, ()))
        overlap = len((token_set & prompt_tokens) | (token_set & domain_tokens))
        repeated = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
        return (
            self._format_bonus(output, output_format)
            + 0.22 * overlap
            + 0.05 * min(len(token_set), 32)
            - 0.8 * repeated
            - 0.02 * max(0, len(tokens) - 80)
        )

    def generate_result(self, prompt: str) -> GenerationResult:
        state = self.field.activate(prompt)
        domains = tuple(domain for domain, _ in state.domains)
        output_format = self.detect_format(prompt, state)
        intent = self._best_intent(state)
        candidates = [
            ("frame", self._frame_response(prompt, domains, intent, output_format)),
            ("composition", self._composition_response(prompt, state, output_format)),
            ("nearest", self._nearest_response(state, output_format)),
        ]
        prompt_tokens = set(tokenize(prompt))
        scored = []
        for source, output in candidates:
            score = self._score_candidate(prompt, output, state, output_format)
            if source == "nearest":
                overlap = len(prompt_tokens & set(tokenize(output)))
                if overlap >= max(2, len(prompt_tokens) // 3):
                    score += 2.0
            scored.append((score, source, output))
        scored.sort(key=lambda item: item[0], reverse=True)
        _, source, output = scored[0]
        return GenerationResult(output=output, domains=domains, output_format=output_format, source=source)

    def generate(self, prompt: str) -> str:
        return self.generate_result(prompt).output
