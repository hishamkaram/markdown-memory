"""Shared fixtures: a deterministic fake embedder, a temp database, the real ONNX model."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fakes import FakeEmbedder

from markdown_memory.config import ServerConfig
from markdown_memory.db import Database
from markdown_memory.embedders import DEFAULT_EMBEDDER, Embedder, create_embedder
from markdown_memory.exceptions import EmbeddingError

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def clean_doc() -> str:
    return (FIXTURES / "clean_doc.md").read_text(encoding="utf-8")


@pytest.fixture
def messy_doc() -> str:
    return (FIXTURES / "messy_doc.md").read_text(encoding="utf-8")


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    with Database(tmp_path / "index.db") as database:
        yield database


@pytest.fixture(scope="session")
def real_embedder() -> Embedder:
    """The default ONNX model. Skips (rather than fails) when it cannot be fetched.

    Pinned to ``DEFAULT_EMBEDDER``: with ``MARKDOWN_MEMORY_EMBEDDER`` set to the light
    preset these tests would score a model the assertions were never calibrated for -
    green for a model nobody ships. Only the cache location comes from the environment
    (``MARKDOWN_MEMORY_MODEL_CACHE``, else ``XDG_CACHE_HOME``).
    """
    embedder = create_embedder(DEFAULT_EMBEDDER, cache_dir=ServerConfig.from_env().model_cache_dir)
    try:
        embedder.embed_query("warm up")
    except EmbeddingError as exc:  # offline machine without a cached model
        pytest.skip(f"embedding model unavailable: {exc}")
    return embedder
