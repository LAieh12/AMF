from __future__ import annotations

import numpy as np

from phase9_corpus_builder import DOMAINS, Phase9Example
from phase9_decoder_scaling import GenerationResult
from phase9_domain_memory import DomainRouter, Phase9SparseEncoder


class GlobalNearestBaseline:
    name = "global_nearest_memory"

    def __init__(self, dims: int = 256):
        self.encoder = Phase9SparseEncoder(dims=dims)
        self.examples: list[Phase9Example] = []
        self.matrix = np.zeros((0, dims), dtype=np.float32)

    def fit(self, examples: list[Phase9Example]) -> "GlobalNearestBaseline":
        self.examples = list(examples)
        vectors = [
            self.encoder.encode(
                f"{example.prompt} {example.intent} {example.output_format}",
                extra=example.domains + (example.intent, example.output_format),
            )
            for example in self.examples
        ]
        self.matrix = np.vstack(vectors).astype(np.float32) if vectors else self.matrix
        return self

    def generate_result(self, prompt: str) -> GenerationResult:
        if not self.examples:
            return GenerationResult("No tengo memoria disponible.", ("safety",), "normal", self.name)
        x = self.encoder.encode(prompt)
        distances = np.mean(np.square(self.matrix - x), axis=1)
        idx = int(np.argmin(distances))
        example = self.examples[idx]
        return GenerationResult(
            output=example.response,
            domains=example.domains,
            output_format=example.output_format,
            source=self.name,
        )

    def generate(self, prompt: str) -> str:
        return self.generate_result(prompt).output


class DomainTemplateBaseline:
    name = "domain_template_router"

    def __init__(self, dims: int = 256):
        self.encoder = Phase9SparseEncoder(dims=dims)
        self.router = DomainRouter(self.encoder)

    def fit(self, examples: list[Phase9Example]) -> "DomainTemplateBaseline":
        self.router.fit(examples)
        return self

    def generate_result(self, prompt: str) -> GenerationResult:
        routed = self.router.route(prompt, top_n=2)
        domains = tuple(domain for domain, _ in routed) or ("safety",)
        domain = domains[0]
        if "json" in prompt:
            output = '{"domain": "%s", "next_step": "revisar memoria"}' % domain
            output_format = "json"
        elif "tabla" in prompt:
            output = "| Dominio | Accion |\n|---|---|\n| %s | revisar memoria |" % domain
            output_format = "table"
        elif "pseudocodigo" in prompt:
            output = "funcion responder(prompt):\n    retornar revisar_memoria(prompt)"
            output_format = "pseudocode"
        elif "diagnostica" in prompt or "diagnostico" in prompt:
            output = "Causa: memoria insuficiente.\nSiguiente paso: revisar el dominio."
            output_format = "diagnosis"
        elif "experimento" in prompt or "ablacion" in prompt:
            output = "Hipotesis: mejorar dominio.\nMetrica: score local.\nBaseline: template."
            output_format = "experiment_plan"
        else:
            output = f"Respuesta del dominio {domain}: activar memoria y dar el siguiente paso."
            output_format = "normal"
        return GenerationResult(output=output, domains=domains, output_format=output_format, source=self.name)

    def generate(self, prompt: str) -> str:
        return self.generate_result(prompt).output


def build_phase9_baselines() -> list[object]:
    return [
        DomainTemplateBaseline(),
        GlobalNearestBaseline(),
    ]
