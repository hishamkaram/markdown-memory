"""Shared fixtures: a deterministic fake embedder, a temp database, the real ONNX model."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fakes import FakeEmbedder

from markdown_memory.db import Database
from markdown_memory.exceptions import EmbeddingError
from markdown_memory.indexer import Embedder, create_embedder
from markdown_memory.server import ServerConfig

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
    """The production ONNX model. Skips (rather than fails) when it cannot be fetched."""
    # Same resolution as the server: MARKDOWN_MEMORY_MODEL_CACHE, else XDG_CACHE_HOME.
    config = ServerConfig.from_env()
    embedder = create_embedder(config.embedder, cache_dir=config.model_cache_dir)
    try:
        embedder.embed_query("warm up")
    except EmbeddingError as exc:  # offline machine without a cached model
        pytest.skip(f"embedding model unavailable: {exc}")
    return embedder
