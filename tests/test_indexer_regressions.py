"""Regressions in the embedders: how the ONNX sessions they build are configured.

One test per defect found in code review; each fails when its fix is reverted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from markdown_memory.indexer import (
    BGE_SMALL_MODEL_NAME,
    GEMMA_FILES,
    EmbeddingGemmaEmbedder,
    FastEmbedEmbedder,
)

_SPIN_KEY = "session.intra_op.allow_spinning"


class _FakeTokenizer:
    """Stands in for the real tokenizer, which would reject the empty stub file."""

    @staticmethod
    def from_file(path: str) -> _FakeTokenizer:
        return _FakeTokenizer()

    def enable_truncation(self, max_length: int) -> None:
        pass

    def enable_padding(self) -> None:
        pass


def _config_entry(options: Any, key: str) -> str | None:
    """The session config value, or None: onnxruntime raises when the key was never set."""
    try:
        value = options.get_session_config_entry(key)
    except RuntimeError:
        return None
    return str(value)


def _session_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Load a Gemma embedder against stub files; return the SessionOptions it built."""
    import onnxruntime
    import tokenizers

    for name in GEMMA_FILES:
        stub = tmp_path / "embeddinggemma-300m-onnx" / name
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_bytes(b"")
    monkeypatch.setattr(tokenizers, "Tokenizer", _FakeTokenizer)

    captured: list[Any] = []

    def fake_session(path: str, options: Any, **kwargs: Any) -> object:
        captured.append(options)
        return object()

    monkeypatch.setattr(onnxruntime, "InferenceSession", fake_session)
    EmbeddingGemmaEmbedder(cache_dir=tmp_path).warm_up()
    return captured[0]


def test_gemma_session_disables_intra_op_spinning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spinning cost ~7s of CPU for a 0.6s query, and kept burning after it returned.

    Queries here arrive seconds apart, so the wake-up latency spinning buys back is
    never recovered.
    """
    monkeypatch.delenv("MARKDOWN_MEMORY_THREADS", raising=False)
    options = _session_options(tmp_path, monkeypatch)
    assert _config_entry(options, _SPIN_KEY) == "0"
    # Unset override: the count stays onnxruntime's business, which is 0 in its terms.
    assert options.intra_op_num_threads == 0


def test_gemma_session_honours_the_thread_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MARKDOWN_MEMORY_THREADS", "3")
    assert _session_options(tmp_path, monkeypatch).intra_op_num_threads == 3


class _FakeTextEmbedding:
    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        type(self).calls.append(kwargs)


def _fastembed_kwargs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Build a bge-small embedder against a stub fastembed; return its constructor kwargs."""
    import fastembed

    _FakeTextEmbedding.calls = []
    monkeypatch.setattr(fastembed, "TextEmbedding", _FakeTextEmbedding)
    FastEmbedEmbedder(BGE_SMALL_MODEL_NAME)._load()
    return _FakeTextEmbedding.calls[0]


def test_fastembed_passes_the_thread_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """fastembed builds its own session, so the override reaches it only as an argument.

    Without it MARKDOWN_MEMORY_THREADS was documented but ignored for this preset, and
    bge-small spent 718 ms of CPU on a query that costs 95 ms at four threads.
    """
    monkeypatch.setenv("MARKDOWN_MEMORY_THREADS", "4")
    # .get, not [...]: dropping the argument must fail the assertion, not error.
    assert _fastembed_kwargs(monkeypatch).get("threads") == 4


def test_fastembed_leaves_the_count_to_fastembed_when_the_override_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MARKDOWN_MEMORY_THREADS", raising=False)
    kwargs = _fastembed_kwargs(monkeypatch)
    assert "threads" in kwargs and kwargs["threads"] is None
