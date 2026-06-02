from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import math
import re

import numpy as np

from phase8_corpus import ConversationExample


EPS = 1e-9
TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[.,;!?]")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def detokenize(tokens: list[str]) -> str:
    out = []
    for token in tokens:
        if token in {".", ",", ";", "!", "?"} and out:
            out[-1] = out[-1] + token
        else:
            out.append(token)
    text = " ".join(out).strip()
    return text[:1].upper() + text[1:] if text else text


def stable_hash(text: str) -> int:
    value = 2166136261
    for ch in text:
        value ^= ord(ch)
        value = (value * 16777619) & 0xFFFFFFFF
    return value


class SparseTextEncoder:
    def __init__(self, dims: int = 384):
        self.dims = dims

    def encode(self, text: str, extra: tuple[str, ...] = ()) -> np.ndarray:
        vec = np.zeros(self.dims, dtype=np.float64)
        tokens = tokenize(text)
        features = []
        features.extend(f"tok:{token}" for token in tokens)
        features.extend(f"extra:{item}" for item in extra)
        joined = " ".join(tokens)
        for n in (3, 4):
            for i in range(0, max(0, len(joined) - n + 1)):
                features.append(f"ch{n}:{joined[i:i+n]}")
        for feat in features:
            h = stable_hash(feat)
            idx = h % self.dims
            sign = 1.0 if (h >> 31) == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        return vec / (norm + EPS)


@dataclass
class InputCell:
    center: np.ndarray
    prompt: str
    intent: str
    response: str
    usage: int = 1


@dataclass
class ActiveState:
    prompt: str
    vector: np.ndarray
    cells: list[tuple[InputCell, float]]
    intent_scores: dict[str, float]

    @property
    def best_intent(self) -> str:
        return max(self.intent_scores.items(), key=lambda item: item[1])[0]


class MorphogenicInputField:
    def __init__(self, dims: int = 384, top_k: int = 7, radius: float = 0.55):
        self.encoder = SparseTextEncoder(dims=dims)
        self.top_k = top_k
        self.radius = radius
        self.cells: list[InputCell] = []
        self.intents: list[str] = []

    def fit(self, examples: list[ConversationExample]) -> "MorphogenicInputField":
        self.cells = [
            InputCell(
                center=self.encoder.encode(example.prompt),
                prompt=example.prompt,
                intent=example.intent,
                response=example.response,
            )
            for example in examples
        ]
        self.intents = sorted({example.intent for example in examples})
        self.intent_lexicon: dict[str, set[str]] = {intent: set() for intent in self.intents}
        for example in examples:
            self.intent_lexicon[example.intent].update(token for token in tokenize(example.prompt) if len(token) > 2)
            self.intent_lexicon[example.intent].update(token for token in tokenize(example.response) if len(token) > 4)
        self.intent_lexicon.get("saludo", set()).update({"hablar", "palabras", "conmigo"})
        self.intent_lexicon.get("arquitectura", set()).update({"convierte", "conviertes", "frase", "input", "output", "decoder"})
        self.intent_lexicon.get("despedida", set()).update({"vemos", "luego", "adios"})
        return self

    def activate(self, prompt: str) -> ActiveState:
        x = self.encoder.encode(prompt)
        distances = []
        for cell in self.cells:
            d2 = float(np.mean(np.square(x - cell.center)))
            distances.append(d2)
        order = np.argsort(distances)[: min(self.top_k, len(distances))]
        active: list[tuple[InputCell, float]] = []
        intent_scores = {intent: 0.0 for intent in self.intents}
        for idx in order:
            cell = self.cells[int(idx)]
            weight = math.exp(-distances[int(idx)] / (2.0 * self.radius * self.radius))
            active.append((cell, weight))
            intent_scores[cell.intent] += weight
        prompt_tokens = {token for token in tokenize(prompt) if len(token) > 2}
        for intent, lexicon in self.intent_lexicon.items():
            overlap = len(prompt_tokens & lexicon)
            if overlap:
                intent_scores[intent] += 0.45 * overlap + 0.45 * overlap * overlap
        total = sum(intent_scores.values()) + EPS
        intent_scores = {key: value / total for key, value in intent_scores.items()}
        latent = np.zeros(len(self.intents), dtype=np.float64)
        for i, intent in enumerate(self.intents):
            latent[i] = intent_scores[intent]
        return ActiveState(prompt=prompt, vector=latent, cells=active, intent_scores=intent_scores)


class BaseMorphogenicDecoder:
    name = "base"

    def fit(self, examples: list[ConversationExample], field: MorphogenicInputField) -> "BaseMorphogenicDecoder":
        self.examples = examples
        self.field = field
        return self

    def generate(self, prompt: str, max_tokens: int = 24) -> str:
        raise NotImplementedError


class NearestResponseDecoder(BaseMorphogenicDecoder):
    name = "nearest_response_cells"

    def generate(self, prompt: str, max_tokens: int = 24) -> str:
        state = self.field.activate(prompt)
        response_scores: dict[str, float] = defaultdict(float)
        for cell, weight in state.cells:
            response_scores[cell.response] += weight
        return max(response_scores.items(), key=lambda item: item[1])[0]


class TransitionCellDecoder(BaseMorphogenicDecoder):
    name = "transition_cells"

    def fit(self, examples: list[ConversationExample], field: MorphogenicInputField) -> "TransitionCellDecoder":
        super().fit(examples, field)
        self.intent_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.bigram_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.trigram_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
        self.global_counts: Counter[str] = Counter()
        for example in examples:
            tokens = ["<bos>", "<bos>"] + tokenize(example.response) + ["<eos>"]
            for i in range(2, len(tokens)):
                prev2, prev1, nxt = tokens[i - 2], tokens[i - 1], tokens[i]
                self.intent_counts[example.intent][nxt] += 1
                self.bigram_counts[(example.intent, prev1)][nxt] += 1
                self.trigram_counts[(example.intent, prev2, prev1)][nxt] += 1
                self.global_counts[nxt] += 1
        return self

    @staticmethod
    def _sample_from_counter(counter: Counter[str], banned: set[str]) -> str:
        if not counter:
            return "<eos>"
        ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        for token, _ in ranked:
            if token not in banned:
                return token
        return ranked[0][0]

    def generate(self, prompt: str, max_tokens: int = 24) -> str:
        state = self.field.activate(prompt)
        intent = state.best_intent
        tokens = ["<bos>", "<bos>"]
        out: list[str] = []
        for _ in range(max_tokens):
            prev2, prev1 = tokens[-2], tokens[-1]
            scores: Counter[str] = Counter()
            scores.update({k: v * 4 for k, v in self.trigram_counts[(intent, prev2, prev1)].items()})
            scores.update({k: v * 2 for k, v in self.bigram_counts[(intent, prev1)].items()})
            scores.update(self.intent_counts[intent])
            scores.update({k: max(1, v // 4) for k, v in self.global_counts.items()})
            banned = {"<bos>"}
            if len(out) >= 2 and out[-1] == out[-2]:
                banned.add(out[-1])
            nxt = self._sample_from_counter(scores, banned)
            if nxt == "<eos>":
                break
            out.append(nxt)
            tokens.append(nxt)
            if nxt == "." and len(out) >= 6:
                break
        if not out or out[-1] not in {".", "!", "?"}:
            out.append(".")
        return detokenize(out)


class FrameSlotDecoder(BaseMorphogenicDecoder):
    name = "frame_slot_cells"

    SLOT_WORDS = {
        "tema": {
            "decoder": "decoder morfogenetico",
            "salida": "salida",
            "output": "salida",
            "celulas": "celulas",
            "memoria": "memoria",
            "clasicos": "modelos clasicos",
            "prueba": "prueba",
            "idea": "idea",
        }
    }

    FRAMES = {
        "saludo": (
            "Hola, puedo hablar con una frase clara sobre {tema}.",
            "Hola, el campo activo responde con una salida util.",
        ),
        "identidad": (
            "Soy un sistema morfogenetico que convierte memoria local en palabras.",
            "Soy un campo activo con celulas de entrada, estado latente y decoder de salida.",
        ),
        "arquitectura": (
            "La entrada activa el campo, el estado latente selecciona celulas y el decoder produce {tema}.",
            "El flujo usa entrada, campo morfogenetico, estado activo, decoder morfogenetico y salida.",
        ),
        "aprendizaje": (
            "Aprendo cuando una experiencia refuerza celulas y mejora futuras palabras.",
            "Las celulas guardan memoria local y la usan para generar una respuesta estable.",
        ),
        "comparacion": (
            "La comparacion exige pruebas contra modelos clasicos y salidas no vistas.",
            "El sistema mejora si sus celulas producen respuestas mas utiles que los baselines.",
        ),
        "razonamiento": (
            "Primero activo una relacion, despues conecto pasos y al final genero una respuesta.",
            "El razonamiento aparece cuando varias celulas sostienen la misma conclusion.",
        ),
        "creatividad": (
            "Una idea nueva combina {tema}, memoria activa y una regla de composicion.",
            "Puedo crear una propuesta mezclando celulas cercanas sin usar un decoder denso.",
        ),
        "estado": (
            "El estado latente contiene celulas activas, pesos locales y rutas de salida.",
            "La resonancia indica que varias memorias apoyan la misma respuesta.",
        ),
        "ayuda": (
            "Puedo ayudarte si convierto la tarea en pasos pequenos y medibles.",
            "El siguiente paso es probar una salida, medirla y reforzar las celulas utiles.",
        ),
        "despedida": (
            "Hasta luego, la memoria queda lista para la siguiente conversacion.",
            "Nos vemos despues, el campo conserva la experiencia activa.",
        ),
    }

    def _slot_topic(self, prompt: str) -> str:
        tokens = set(tokenize(prompt))
        for token, value in self.SLOT_WORDS["tema"].items():
            if token in tokens:
                return value
        return "la idea"

    def generate(self, prompt: str, max_tokens: int = 24) -> str:
        state = self.field.activate(prompt)
        intent = state.best_intent
        frames = self.FRAMES.get(intent, self.FRAMES["ayuda"])
        idx = stable_hash(prompt + intent) % len(frames)
        return frames[idx].format(tema=self._slot_topic(prompt))


class ResonantMorphogenicDecoder(BaseMorphogenicDecoder):
    name = "resonant_morphogenic_decoder"

    VERBS = {
        "soy",
        "es",
        "esta",
        "puedo",
        "convierte",
        "activa",
        "selecciona",
        "produce",
        "aprendo",
        "guardan",
        "exige",
        "mejora",
        "conecto",
        "genero",
        "combina",
        "contiene",
        "indica",
        "queda",
        "conserva",
    }

    def fit(self, examples: list[ConversationExample], field: MorphogenicInputField) -> "ResonantMorphogenicDecoder":
        super().fit(examples, field)
        self.nearest = NearestResponseDecoder().fit(examples, field)
        self.transition = TransitionCellDecoder().fit(examples, field)
        self.frame = FrameSlotDecoder().fit(examples, field)
        return self

    def _score(self, prompt: str, candidate: str, state: ActiveState) -> float:
        tokens = tokenize(candidate)
        prompt_tokens = set(tokenize(prompt))
        intent = state.best_intent
        intent_tokens = set()
        for example in self.examples:
            if example.intent == intent:
                intent_tokens.update(tokenize(example.response))
        has_verb = any(token in self.VERBS for token in tokens)
        complete = bool(tokens and tokens[-1] in {".", "!", "?"} and len(tokens) >= 6)
        repeated = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
        relevance = len((set(tokens) & intent_tokens) | (set(tokens) & prompt_tokens))
        return (
            2.5 * float(complete)
            + 2.0 * float(has_verb)
            + 0.18 * relevance
            + 0.08 * min(len(set(tokens)), 14)
            - 1.2 * repeated
        )

    def generate(self, prompt: str, max_tokens: int = 24) -> str:
        state = self.field.activate(prompt)
        candidates = [
            self.frame.generate(prompt, max_tokens=max_tokens),
            self.transition.generate(prompt, max_tokens=max_tokens),
            self.nearest.generate(prompt, max_tokens=max_tokens),
        ]
        scored = [(self._score(prompt, candidate, state), candidate) for candidate in candidates]
        scored.sort(key=lambda item: (item[0], len(set(tokenize(item[1])))), reverse=True)
        return scored[0][1]


def build_decoders() -> list[BaseMorphogenicDecoder]:
    return [
        NearestResponseDecoder(),
        TransitionCellDecoder(),
        FrameSlotDecoder(),
        ResonantMorphogenicDecoder(),
    ]
