from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from phase8_morphogenic_decoder import stable_hash, tokenize
from phase9_corpus_builder import DOMAINS, DOMAIN_KEYWORDS, Phase9Example


EPS = 1e-9


class Phase9SparseEncoder:
    def __init__(self, dims: int = 256):
        self.dims = dims

    def encode(self, text: str, extra: tuple[str, ...] = ()) -> np.ndarray:
        vec = np.zeros(self.dims, dtype=np.float32)
        tokens = tokenize(text)
        features = []
        features.extend(f"tok:{token}" for token in tokens)
        features.extend(f"extra:{item}" for item in extra)
        joined = " ".join(tokens)
        for n in (3, 4):
            for i in range(0, max(0, len(joined) - n + 1)):
                features.append(f"ch{n}:{joined[i:i+n]}")
        for feature in features:
            h = stable_hash(feature)
            idx = h % self.dims
            sign = 1.0 if (h >> 31) == 0 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        return vec / (norm + EPS)


@dataclass
class DomainCell:
    vector: np.ndarray
    example: Phase9Example
    usage: int = 1


@dataclass
class DomainActivation:
    domain: str
    score: float
    cells: list[tuple[DomainCell, float]]


@dataclass
class Phase9ActiveState:
    prompt: str
    domains: list[tuple[str, float]]
    activations: list[DomainActivation]

    @property
    def best_domain(self) -> str:
        return self.domains[0][0] if self.domains else "safety"


class DomainMemory:
    def __init__(self, domain: str, encoder: Phase9SparseEncoder, top_k: int = 9, radius: float = 0.42):
        self.domain = domain
        self.encoder = encoder
        self.top_k = top_k
        self.radius = radius
        self.cells: list[DomainCell] = []
        self.matrix = np.zeros((0, self.encoder.dims), dtype=np.float32)

    def fit(self, examples: list[Phase9Example]) -> "DomainMemory":
        self.cells = []
        vectors = []
        for example in examples:
            vector = self.encoder.encode(
                f"{example.prompt} {example.intent} {example.output_format}",
                extra=example.domains + (example.intent, example.output_format),
            )
            self.cells.append(DomainCell(vector=vector, example=example))
            vectors.append(vector)
        self.matrix = np.vstack(vectors).astype(np.float32) if vectors else np.zeros((0, self.encoder.dims), dtype=np.float32)
        return self

    def add(self, example: Phase9Example) -> None:
        vector = self.encoder.encode(
            f"{example.prompt} {example.intent} {example.output_format}",
            extra=example.domains + (example.intent, example.output_format),
        )
        self.cells.append(DomainCell(vector=vector, example=example))
        if len(self.matrix) == 0:
            self.matrix = vector.reshape(1, -1).astype(np.float32)
        else:
            self.matrix = np.vstack([self.matrix, vector.astype(np.float32)])

    def activate(self, prompt: str, extra: tuple[str, ...] = ()) -> list[tuple[DomainCell, float]]:
        if not self.cells:
            return []
        x = self.encoder.encode(prompt, extra=extra)
        distances = np.mean(np.square(self.matrix - x), axis=1)
        k = min(self.top_k, len(self.cells))
        if k == len(self.cells):
            order = np.argsort(distances)
        else:
            order = np.argpartition(distances, k - 1)[:k]
            order = order[np.argsort(distances[order])]
        active = []
        for idx in order:
            distance = float(distances[int(idx)])
            weight = math.exp(-distance / (2.0 * self.radius * self.radius))
            active.append((self.cells[int(idx)], weight))
        return active


class DomainRouter:
    DIRECT_HINTS: dict[str, tuple[str, ...]] = {
        "conversation": ("hola", "ayuda", "gracias", "luego", "siguiente", "orientame"),
        "architecture": ("campo", "celula", "celulas", "decoder", "memoria", "latente", "resonancia", "router", "amf8"),
        "research": ("experimento", "metrica", "ablacion", "hipotesis", "baseline", "comparar", "compara", "pythia", "amf8", "evidencia", "latencia"),
        "code": ("codigo", "debug", "debuggear", "funcion", "pseudocodigo", "archivo", "archivos", "estructura", "error", "refactor", "latencia", "diagnostica", "mezcla", "rara"),
        "structured": ("json", "tabla", "pasos", "plan", "diagnostico", "formato"),
        "safety": ("fallback", "incertidumbre", "seguro", "limite", "revision", "inventar", "diagnostica", "diagnostico", "mezcla", "rara", "latencia"),
    }

    def __init__(self, encoder: Phase9SparseEncoder):
        self.encoder = encoder
        self.lexicon: dict[str, set[str]] = {domain: set(DOMAIN_KEYWORDS[domain]) for domain in DOMAINS}
        self.centroids: dict[str, np.ndarray] = {
            domain: np.zeros(self.encoder.dims, dtype=np.float32) for domain in DOMAINS
        }

    def fit(self, examples: list[Phase9Example]) -> "DomainRouter":
        vectors: dict[str, list[np.ndarray]] = {domain: [] for domain in DOMAINS}
        for example in examples:
            text = f"{example.prompt} {example.response} {example.intent} {example.output_format}"
            tokens = {token for token in tokenize(text) if len(token) > 2}
            for domain in example.domains:
                self.lexicon.setdefault(domain, set()).update(tokens)
                vectors.setdefault(domain, []).append(
                    self.encoder.encode(text, extra=example.domains + (example.intent, example.output_format))
                )
        for domain in DOMAINS:
            self.lexicon.setdefault(domain, set()).update(self.DIRECT_HINTS[domain])
            if vectors.get(domain):
                centroid = np.mean(np.vstack(vectors[domain]), axis=0).astype(np.float32)
                norm = float(np.linalg.norm(centroid))
                self.centroids[domain] = centroid / (norm + EPS)
        return self

    def route(self, prompt: str, top_n: int = 3) -> list[tuple[str, float]]:
        tokens = {token for token in tokenize(prompt) if len(token) > 2}
        x = self.encoder.encode(prompt)
        scores = {}
        for domain in DOMAINS:
            overlap = len(tokens & self.lexicon.get(domain, set()))
            direct = len(tokens & set(self.DIRECT_HINTS[domain]))
            centroid = float(np.dot(x, self.centroids[domain]))
            scores[domain] = max(0.0, centroid) + 0.35 * overlap + 0.85 * direct
        if all(value <= 0.0 for value in scores.values()):
            scores["safety"] = 1.0
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best = ordered[0][1] + EPS
        selected = [(domain, score) for domain, score in ordered[:top_n] if score >= 0.18 * best or domain == ordered[0][0]]
        total = sum(score for _, score in selected) + EPS
        return [(domain, score / total) for domain, score in selected]


class Phase9DomainField:
    def __init__(self, dims: int = 256, top_k: int = 9, radius: float = 0.42):
        self.encoder = Phase9SparseEncoder(dims=dims)
        self.top_k = top_k
        self.radius = radius
        self.memories = {
            domain: DomainMemory(domain=domain, encoder=self.encoder, top_k=top_k, radius=radius)
            for domain in DOMAINS
        }
        self.router = DomainRouter(self.encoder)

    def fit(self, examples: list[Phase9Example]) -> "Phase9DomainField":
        self.examples = list(examples)
        self.router.fit(examples)
        for domain in DOMAINS:
            domain_examples = [example for example in examples if domain in example.domains]
            self.memories[domain].fit(domain_examples)
        return self

    def activate(self, prompt: str, max_domains: int = 3) -> Phase9ActiveState:
        routed = self.router.route(prompt, top_n=max_domains)
        activations = []
        for domain, score in routed:
            cells = self.memories[domain].activate(prompt, extra=(domain,))
            activations.append(DomainActivation(domain=domain, score=score, cells=cells))
        return Phase9ActiveState(prompt=prompt, domains=routed, activations=activations)

    def learn(self, example: Phase9Example) -> None:
        self.examples.append(example)
        for domain in example.domains:
            self.memories[domain].add(example)
            self.router.lexicon.setdefault(domain, set()).update(tokenize(f"{example.prompt} {example.response}"))

    def memory_counts(self) -> dict[str, int]:
        return {domain: len(memory.cells) for domain, memory in self.memories.items()}
