"""Test doubles shared by the test modules (importable via ``pythonpath = ["tests"]``)."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

from markdown_memory.db import DEFAULT_EMBEDDING_DIM
from markdown_memory.models import SectionDraft, SectionVectors

_WORD = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Hashed bag-of-words vectors: deterministic, offline, and lexically meaningful.

    Texts sharing words get a high cosine similarity, which is all the storage,
    indexing and fusion tests need. Semantic behaviour is tested with the real model.
    """

    def __init__(self, dimension: int = DEFAULT_EMBEDDING_DIM, model_name: str = "fake") -> None:
        self._dimension = dimension
        self._model_name = model_name
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def weights_revision(self) -> str | None:
        return None

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for word in _WORD.findall(text.lower()):
            digest = hashlib.sha256(word.encode()).digest()
            vector[int.from_bytes(digest[:4], "big") % self._dimension] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]


def vectors_for(embedder: FakeEmbedder, sections: Sequence[SectionDraft]) -> list[SectionVectors]:
    """Section + passage vectors exactly as the indexer would produce them."""
    return [
        SectionVectors(
            section=embedder.embed_documents([s.embedding_text])[0] if s.units else None,
            units=tuple(embedder.embed_documents(list(s.unit_texts))),
        )
        for s in sections
    ]
