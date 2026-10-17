"""Embeddings.

Default is a deterministic, dependency-free hashing embedder that runs fully
offline (reproducible evals). Swap to a real model with WA_EMBEDDER=openai.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from waagent.config import settings

_WORD = re.compile(r"[a-z0-9']+")


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...


def _l2(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class MockEmbedder:
    """Hashed bag-of-unigrams+bigrams -> fixed-dim unit vector.

    Deterministic: identical text always maps to the identical vector, so the
    eval harness is reproducible and needs no model download or API key.
    Cosine similarity approximates shared-token overlap, which is enough to
    demonstrate real retrieval behaviour.
    """

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or settings.embed_dim

    def _feature(self, vec: list[float], token: str) -> None:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % self.dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[idx] += sign

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        words = _WORD.findall(text.lower())
        for w in words:
            self._feature(vec, w)
        for a, b in zip(words, words[1:]):
            self._feature(vec, f"{a}_{b}")
        return _l2(vec)


class OpenAIEmbedder:  # pragma: no cover - needs network + key
    """Thin adapter; only used when WA_EMBEDDER=openai."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        from openai import OpenAI  # type: ignore

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = model
        self.dim = settings.embed_dim

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(model=self._model, input=text)
        return _l2(list(resp.data[0].embedding))


def get_embedder() -> Embedder:
    if settings.embedder == "openai" and settings.openai_api_key:
        return OpenAIEmbedder()
    return MockEmbedder()
