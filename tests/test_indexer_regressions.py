"""Regressions in the embedders: how the ONNX sessions they build are configured.

One test per defect found in code review; each fails when its fix is reverted.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeEmbedder

from markdown_memory.db import Database
from markdown_memory.exceptions import ModelLoadError
from markdown_memory.indexer import (
    BGE_SMALL_MODEL_NAME,
    GEMMA_FILES,
    EmbeddingGemmaEmbedder,
    FastEmbedEmbedder,
    Indexer,
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


def _session_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, Any]:
    """Load a Gemma embedder against stub files; return what it asked onnxruntime for."""
    import onnxruntime
    import tokenizers

    for name in GEMMA_FILES:
        stub = tmp_path / "embeddinggemma-300m-onnx" / name
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_bytes(b"")
    monkeypatch.setattr(tokenizers, "Tokenizer", _FakeTokenizer)

    captured: list[tuple[str, Any]] = []

    def fake_session(path: str, options: Any, **kwargs: Any) -> object:
        captured.append((path, options))
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
    _, options = _session_call(tmp_path, monkeypatch)
    assert _config_entry(options, _SPIN_KEY) == "0"
    # Unset override: the count stays onnxruntime's business, which is 0 in its terms.
    assert options.intra_op_num_threads == 0


def test_gemma_session_honours_the_thread_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MARKDOWN_MEMORY_THREADS", "3")
    assert _session_call(tmp_path, monkeypatch)[1].intra_op_num_threads == 3


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


# --- The model cache: keyed by revision, verified before it is loaded -------------------

_FILES = {"model.onnx": b"graph bytes", "nested/weights.bin": b"weights" * 100}
#: What the stubbed rewrite produces from _FILES["model.onnx"].
_DERIVED = b"the same graph, gathering first"


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Cache:
    """A Gemma embedder whose manifest is two tiny files and whose download is a stub."""
    import huggingface_hub

    from markdown_memory import indexer

    manifest = {
        name: (len(data), hashlib.sha256(data).hexdigest()) for name, data in _FILES.items()
    }
    monkeypatch.setattr(indexer, "GEMMA_MANIFEST", manifest)
    monkeypatch.setattr(indexer, "GEMMA_FILES", tuple(manifest))
    monkeypatch.setattr(indexer, "GEMMA_MODEL_FILE", "model.onnx")
    monkeypatch.setattr(indexer, "DERIVED_GRAPH_FILE", "derived.onnx")
    monkeypatch.setattr(indexer, "DERIVED_GRAPH_SHA256", hashlib.sha256(_DERIVED).hexdigest())
    harness = _Cache(tmp_path)
    monkeypatch.setattr(indexer, "gather_before_dequantize", harness.rewrite)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", harness.download)
    monkeypatch.setattr(EmbeddingGemmaEmbedder, "_open", lambda self: harness.open())
    return harness


class _Cache:
    """The scratch model cache, its download stub, and what each of them was asked to do."""

    def __init__(self, root: Path) -> None:
        from markdown_memory.indexer import gemma_model_dir

        self.root = root
        self.model_dir = gemma_model_dir(root)
        self.legacy = root / "embeddinggemma-300m-onnx"
        self.downloads: list[Path] = []
        self.opens = 0
        self.corrupt = False
        self.download_seconds = 0.0
        self.rewrites = 0
        self.refuse_rewrite = False

    def embedder(self) -> EmbeddingGemmaEmbedder:
        return EmbeddingGemmaEmbedder(cache_dir=self.root)

    def download(self, repository: str, **kwargs: Any) -> str:
        target = Path(kwargs["local_dir"])
        self.downloads.append(target)
        # Also on disk, so a forked child's downloads are counted by the parent.
        markers = self.root / "downloads"
        markers.mkdir(exist_ok=True)
        (markers / f"{os.getpid()}-{len(self.downloads)}").write_text("")
        time.sleep(self.download_seconds)
        self.write(target, corrupt=self.corrupt)
        return str(target)

    def rewrite(self, graph: bytes) -> bytes | None:
        self.rewrites += 1
        return None if self.refuse_rewrite else _DERIVED

    def download_count(self) -> int:
        return len(list((self.root / "downloads").glob("*")))

    def open(self) -> tuple[Any, Any]:
        self.opens += 1
        return object(), object()

    def write(self, directory: Path, *, corrupt: bool = False) -> None:
        for name, data in _FILES.items():
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x" * len(data) if corrupt else data)


def test_a_verified_cache_is_neither_hashed_nor_re_fetched_on_every_start(
    cache: _Cache, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hashing 330 MB costs most of a second of a core, and an editor starts a server

    per session. The stamp turns the steady state into a handful of stat calls.
    """
    from markdown_memory import indexer

    cache.embedder().warm_up()
    assert len(cache.downloads) == 1

    monkeypatch.setattr(
        indexer, "_hash_file", lambda path: pytest.fail(f"hashed {path} on a verified cache")
    )
    cache.embedder().warm_up()
    assert len(cache.downloads) == 1


def test_a_file_rewritten_with_its_old_mtime_restored_is_still_caught(cache: _Cache) -> None:
    """`cp -p`, `tar x` and `rsync --inplace` restore an overwritten file's mtime, so size,

    mtime and inode can all agree across different bytes. ctime cannot be set back.
    """
    cache.embedder().warm_up()
    victim = cache.model_dir / "model.onnx"
    before = victim.lstat()
    victim.write_bytes(b"different!!")  # same length, different bytes
    os.utime(victim, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert victim.lstat().st_mtime_ns == before.st_mtime_ns

    cache.embedder().warm_up()
    assert len(cache.downloads) == 2
    assert victim.read_bytes() == _FILES["model.onnx"]


def test_a_model_file_replaced_by_a_symlink_is_not_trusted(cache: _Cache) -> None:
    cache.embedder().warm_up()
    victim = cache.model_dir / "model.onnx"
    # The link's own size is the length of its target, so the name is chosen to match the
    # manifest's size exactly: nothing but "this is not a regular file" can catch it.
    target = "elsewhere.o"
    assert len(target) == len(_FILES["model.onnx"])
    (cache.model_dir / target).write_bytes(_FILES["model.onnx"])
    victim.unlink()
    victim.symlink_to(target)

    cache.embedder().warm_up()
    assert len(cache.downloads) == 2
    assert not victim.is_symlink()


def test_a_file_that_changes_while_it_is_hashed_is_not_trusted(
    cache: _Cache, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whatever the hash describes, it is not what is on disk now."""
    from markdown_memory import indexer

    cache.write(cache.model_dir)
    real_hash = indexer._hash_file

    sabotaged: list[Path] = []

    def hash_then_touch(path: Path) -> str:
        digest = real_hash(path)
        if not sabotaged:  # once: the writer got there between the read and the stat
            sabotaged.append(path)
            os.utime(path, ns=(0, 0))
        return digest

    monkeypatch.setattr(indexer, "_hash_file", hash_then_touch)
    cache.embedder().warm_up()
    assert len(cache.downloads) == 1


def test_files_of_the_right_name_but_the_wrong_size_are_replaced(cache: _Cache) -> None:
    cache.model_dir.mkdir(parents=True)
    (cache.model_dir / "model.onnx").write_bytes(b"truncated")
    (cache.model_dir / "nested").mkdir()
    (cache.model_dir / "nested" / "weights.bin").write_bytes(_FILES["nested/weights.bin"])

    cache.embedder().warm_up()
    assert len(cache.downloads) == 1


def test_another_revisions_files_do_not_stand_in_for_this_ones(cache: _Cache) -> None:
    """The revision is in the folder name precisely so that moving it fetches new weights."""
    cache.write(cache.root / "embeddinggemma-300m-onnx-0123456789ab")

    cache.embedder().warm_up()
    assert cache.downloads == [cache.model_dir]


def test_the_unversioned_folder_is_moved_rather_than_downloaded_again(cache: _Cache) -> None:
    cache.write(cache.legacy)
    inode = (cache.legacy / "model.onnx").lstat().st_ino
    with (cache.legacy / "model.onnx").open("rb") as still_open:
        cache.embedder().warm_up()
        # A descriptor opened before the rename keeps reading the same inode.
        assert still_open.read() == _FILES["model.onnx"]

    assert cache.downloads == []
    assert not cache.legacy.exists()
    assert (cache.model_dir / "model.onnx").lstat().st_ino == inode


def test_an_unversioned_folder_that_does_not_match_is_left_where_it_is(cache: _Cache) -> None:
    """Its bytes have to be fetched anyway, and they may be another checkout's."""
    cache.write(cache.legacy)
    (cache.legacy / "model.onnx").write_bytes(b"not the same")

    cache.embedder().warm_up()
    assert cache.legacy.is_dir()
    assert cache.downloads == [cache.model_dir]
    assert (cache.legacy / "model.onnx").read_bytes() == b"not the same"


def test_a_download_that_does_not_match_the_manifest_fails_without_downloading_again(
    cache: _Cache,
) -> None:
    cache.corrupt = True
    with pytest.raises(ModelLoadError, match="model.onnx"):
        cache.embedder().warm_up()
    assert len(cache.downloads) == 1


def test_a_load_failure_on_verified_files_is_not_treated_as_corruption(
    cache: _Cache, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verified bytes plus a failing load means onnxruntime, permissions or memory - none

    of which 330 MB of fresh download would fix.
    """
    cache.write(cache.model_dir)
    boom = RuntimeError("onnxruntime ABI mismatch")

    def fail(self: EmbeddingGemmaEmbedder) -> tuple[Any, Any]:
        raise ModelLoadError("Cannot load embedding model") from boom

    monkeypatch.setattr(EmbeddingGemmaEmbedder, "_open", fail)
    with pytest.raises(ModelLoadError) as raised:
        cache.embedder().warm_up()
    assert raised.value.__cause__ is boom
    assert cache.downloads == []


def test_older_versions_are_reported_and_never_deleted(
    cache: _Cache, caplog: pytest.LogCaptureFixture
) -> None:
    older = cache.root / "embeddinggemma-300m-onnx-0123456789ab"
    cache.write(older)
    with caplog.at_level("INFO", logger="markdown_memory.indexer"):
        cache.embedder().warm_up()
    assert str(older) in caplog.text
    assert (older / "model.onnx").exists()


def test_two_processes_starting_at_once_download_once_between_them(cache: _Cache) -> None:
    """Without a cross-process lock both fetch, and one writes over what the other verified."""
    cache.download_seconds = 0.5
    context = multiprocessing.get_context("fork")
    workers = [context.Process(target=_warm_up, args=(cache,)) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=60)
    assert [worker.exitcode for worker in workers] == [0, 0]
    assert cache.download_count() == 1


def _warm_up(cache: _Cache) -> None:
    cache.embedder().warm_up()


# --- Which weights the stored vectors came from ----------------------------------------


class _PinnedWeights(FakeEmbedder):
    """A FakeEmbedder that also says which snapshot of the weights produced its vectors."""

    def __init__(self, revision: str) -> None:
        super().__init__()
        self._revision = revision

    @property
    def weights_revision(self) -> str | None:
        return self._revision


@pytest.fixture
def one_document(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "README.md").write_text("# Readme\n\nhello\n")
    return root


def test_the_weights_behind_an_unchanged_model_name_are_recorded_and_compared(
    db: Database, one_document: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """fastembed pins no revision, so a re-download can bring different weights under the

    same model name. Nothing else in the index would notice: the stored passage vectors
    and the new query vectors would simply come from different models.
    """
    Indexer(db, _PinnedWeights("a" * 40)).index_directory(one_document)
    assert db.get_meta("embedding_weights_revision") == "a" * 40

    with caplog.at_level("WARNING", logger="markdown_memory.indexer"):
        report = Indexer(db, _PinnedWeights("b" * 40)).index_directory(one_document)
    assert any("changed since this index was built" in note for note in report.notes)
    assert "changed since this index was built" in caplog.text
    # Nothing is discarded, and the recorded revision still describes the stored vectors.
    assert db.count_rows("documents") == 1
    assert db.get_meta("embedding_weights_revision") == "a" * 40


def test_an_index_with_nothing_to_lose_records_the_weights_it_is_built_with(
    db: Database, one_document: Path
) -> None:
    Indexer(db, _PinnedWeights("a" * 40)).index_directory(one_document)
    db.clear()
    Indexer(db, _PinnedWeights("b" * 40)).index_directory(one_document)
    assert db.get_meta("embedding_weights_revision") == "b" * 40


@pytest.mark.embedding
def test_the_bge_small_folder_is_the_one_fastembed_really_creates() -> None:
    """Derived from fastembed's registry, not from the model name: it downloads its own

    re-export (`qdrant/bge-small-en-v1.5-onnx-q`), and guessing `models--BAAI--...` left
    the eval cache keyed on an empty string for that preset.
    """
    from markdown_memory.indexer import fastembed_model_dir
    from markdown_memory.server import ServerConfig

    cache_dir = ServerConfig.from_env().model_cache_dir
    assert cache_dir is not None
    derived = fastembed_model_dir(cache_dir)
    assert derived is not None
    if not derived.is_dir():
        pytest.skip(f"bge-small has not been downloaded into {cache_dir}")
    revision = FastEmbedEmbedder(BGE_SMALL_MODEL_NAME, cache_dir=cache_dir).weights_revision
    assert revision is not None and len(revision) == 40


# --- The derived graph: generated here, never downloaded --------------------------------


def test_the_derived_graph_is_written_once_and_then_left_alone(cache: _Cache) -> None:
    derived = cache.model_dir / "derived.onnx"
    cache.embedder().warm_up()
    assert derived.read_bytes() == _DERIVED
    assert cache.rewrites == 1

    cache.embedder().warm_up()
    assert cache.rewrites == 1  # the stamp covers it, so nothing is hashed or rebuilt


def test_a_missing_or_tampered_derived_graph_is_rebuilt_rather_than_downloaded(
    cache: _Cache,
) -> None:
    """It never came off the Hub, so re-fetching 330 MB would not produce it."""
    derived = cache.model_dir / "derived.onnx"
    cache.embedder().warm_up()

    derived.unlink()
    cache.embedder().warm_up()
    assert derived.read_bytes() == _DERIVED

    derived.write_bytes(b"something else entirely")
    cache.embedder().warm_up()
    assert derived.read_bytes() == _DERIVED
    assert len(cache.downloads) == 1


def test_only_the_three_published_files_are_ever_asked_of_the_hub(cache: _Cache) -> None:
    from markdown_memory.indexer import GEMMA_FILES

    assert "derived.onnx" not in GEMMA_FILES


def test_a_refused_rewrite_leaves_the_published_graph_running(cache: _Cache) -> None:
    """A refusal costs the memory saving and nothing else: the server still starts."""
    cache.refuse_rewrite = True
    cache.embedder().warm_up()
    assert not (cache.model_dir / "derived.onnx").exists()
    assert cache.opens == 1


def test_the_embedder_runs_the_derived_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of deriving it: ~1 GB per query lives in this one path."""
    from markdown_memory.indexer import DERIVED_GRAPH_FILE, gemma_model_dir

    derived = gemma_model_dir(tmp_path) / DERIVED_GRAPH_FILE
    derived.parent.mkdir(parents=True, exist_ok=True)
    derived.write_bytes(b"")
    path, _ = _session_call(tmp_path, monkeypatch)
    assert path == str(derived)
