"""Regressions in the embedders: how the ONNX sessions they build are configured.

One test per defect found in code review; each fails when its fix is reverted.
"""

from __future__ import annotations

import dataclasses
import hashlib
import multiprocessing
import os
import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeEmbedder
from helpers import draft

from markdown_memory.db import WEIGHTS_REVOKED, Database
from markdown_memory.embedders import EmbeddingGemmaEmbedder, FastEmbedEmbedder
from markdown_memory.exceptions import DatabaseError, IndexingError, ModelLoadError
from markdown_memory.indexer import Indexer
from markdown_memory.model_cache import BGE_SMALL_MODEL_NAME
from markdown_memory.models import SectionVectors

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
    """Load a Gemma embedder against stub files; return what it asked onnxruntime for.

    The stubs are written where the code now looks - the revision-keyed folder - and the
    manifest is stubbed to match them, so verification passes offline. `snapshot_download`
    is replaced by a failure: a test about session options has no business fetching
    218 MB, and before this was pinned these tests only passed on a machine whose
    Hugging Face cache happened to be warm.
    """
    import huggingface_hub
    import onnxruntime
    import tokenizers

    from markdown_memory import model_cache

    manifest = {
        name: (len(data), hashlib.sha256(data).hexdigest()) for name, data in _FILES.items()
    }
    monkeypatch.setattr(model_cache, "GEMMA_MANIFEST", manifest)
    monkeypatch.setattr(model_cache, "GEMMA_FILES", tuple(manifest))
    monkeypatch.setattr(model_cache, "GEMMA_MODEL_FILE", "model.onnx")
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *a, **k: pytest.fail("the model cache was verified, so nothing may be fetched"),
    )
    model_dir = model_cache.gemma_model_dir(tmp_path)
    for name, data in _FILES.items():
        stub = model_dir / name
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_bytes(data)
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


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Cache:
    """A Gemma embedder whose manifest is two tiny files and whose download is a stub."""
    import huggingface_hub

    from markdown_memory import model_cache

    manifest = {
        name: (len(data), hashlib.sha256(data).hexdigest()) for name, data in _FILES.items()
    }
    monkeypatch.setattr(model_cache, "GEMMA_MANIFEST", manifest)
    monkeypatch.setattr(model_cache, "GEMMA_FILES", tuple(manifest))
    monkeypatch.setattr(model_cache, "GEMMA_MODEL_FILE", "model.onnx")
    harness = _Cache(tmp_path)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", harness.download)
    monkeypatch.setattr(EmbeddingGemmaEmbedder, "_open", lambda self: harness.open())
    return harness


class _Cache:
    """The scratch model cache, its download stub, and what each of them was asked to do."""

    def __init__(self, root: Path) -> None:
        from markdown_memory.model_cache import gemma_model_dir

        self.root = root
        self.model_dir = gemma_model_dir(root)
        self.downloads: list[Path] = []
        self.opens = 0
        self.corrupt = False
        self.download_seconds = 0.0

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
    """Hashing 218 MB costs most of a second of a core, and an editor starts a server

    per session. The stamp turns the steady state into a handful of stat calls.
    """

    from markdown_memory import model_cache

    cache.embedder().warm_up()
    assert len(cache.downloads) == 1

    monkeypatch.setattr(
        model_cache, "_hash_file", lambda path: pytest.fail(f"hashed {path} on a verified cache")
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
    from markdown_memory import model_cache

    cache.write(cache.model_dir)
    real_hash = model_cache._hash_file

    sabotaged: list[Path] = []

    def hash_then_touch(path: Path) -> str:
        digest = real_hash(path)
        if not sabotaged:  # once: the writer got there between the read and the stat
            sabotaged.append(path)
            os.utime(path, ns=(0, 0))
        return digest

    monkeypatch.setattr(model_cache, "_hash_file", hash_then_touch)
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

    of which 218 MB of fresh download would fix.
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
    with caplog.at_level("INFO", logger="markdown_memory.embedders"):
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


def test_a_cache_that_cannot_be_repaired_fails_as_a_domain_error(
    cache: _Cache, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Downloading, hashing and stamping raise OSError and whatever the Hub raises. The

    SDK hides anything that is not a MarkdownMemoryError behind "Error executing tool",
    and the server's warm-up only catches that hierarchy, so a full disk or a read-only
    cache used to escape as an opaque crash.
    """

    def refuse(*_arguments: object, **_keywords: object) -> str:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("huggingface_hub.snapshot_download", refuse)

    with pytest.raises(ModelLoadError, match="No space left on device"):
        cache.embedder().warm_up()


def test_a_symlink_planted_at_a_temporary_path_is_not_written_through(
    cache: _Cache, tmp_path: Path
) -> None:
    """The temporary names are the pid, so they are guessable. A plain write follows a

    symlink left at one of them and overwrites whatever it points at, anywhere the user
    can write; creating the file exclusively, without following, makes that the kernel's
    refusal.
    """
    treasure = tmp_path / "treasure.txt"
    treasure.write_text("somebody else's file")
    cache.model_dir.mkdir(parents=True)
    (cache.model_dir / f".verified.{os.getpid()}").symlink_to(treasure)

    cache.embedder().warm_up()

    assert treasure.read_text() == "somebody else's file"
    assert (cache.model_dir / ".verified").is_file()


def test_the_helper_that_writes_those_files_refuses_a_symlink_outright(tmp_path: Path) -> None:
    """Clearing the path first closes the window only until the next process opens one.

    The refusal that holds is the kernel's, so it is asserted on the helper itself.
    """
    from markdown_memory.model_cache import _write_new_file

    treasure = tmp_path / "treasure.txt"
    treasure.write_text("somebody else's file")
    planted = tmp_path / "temporary"
    planted.symlink_to(treasure)

    with pytest.raises(OSError):
        _write_new_file(planted, b"overwritten")

    assert treasure.read_text() == "somebody else's file"


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


def _ranks_by_vector(db: Database, embedder: FakeEmbedder, query: str = "hello") -> bool:
    """Whether a search with ``embedder`` ranked any hit by vector."""
    from markdown_memory.search import HybridSearcher

    searcher = HybridSearcher(db, embedder)
    try:
        results = searcher.search(query)
    finally:
        searcher.close()
    assert results, "keyword search answers whatever state the vectors are in"
    return any(result.vec_rank is not None for result in results)


def test_a_model_whose_weights_changed_re_embeds_the_index_in_place(
    db: Database, one_document: Path
) -> None:
    """Weights that change - a new EmbeddingGemma pin, a fastembed re-download - used to

    stop the run and send the user to delete the database, and keyword search went with
    it. Each document carries the weights that embedded it, so the run re-embeds every
    document stamped by others, file by file, and vouches for the index once none is left.
    """
    Indexer(db, _PinnedWeights("a" * 40)).index_directory(one_document)
    assert db.get_meta("embedding_weights_revision") == "a" * 40
    (one_document / "GUIDE.md").write_text("# Guide\n\nnew file\n")

    report = Indexer(db, _PinnedWeights("b" * 40)).index_directory(one_document)

    assert (report.files_indexed, report.files_unchanged) == (2, 0)  # README re-embedded
    assert db.get_meta("embedding_weights_revision") == "b" * 40
    assert db.get_meta("embedding_weights_mismatch") is None
    assert db.index_status(str(one_document)).verified
    assert _ranks_by_vector(db, _PinnedWeights("b" * 40))
    assert not _ranks_by_vector(db, _PinnedWeights("a" * 40))

    again = Indexer(db, _PinnedWeights("b" * 40)).index_directory(one_document)
    assert (again.files_indexed, again.files_unchanged) == (0, 2), "repaired twice"


def test_vectors_from_other_weights_left_in_another_root_keep_the_index_revoked(
    db: Database, tmp_path: Path
) -> None:
    """The certificate speaks for every vector in the database, not for one root.

    Re-certifying once the root just walked was repaired would have ranked a query
    against the other root's vectors from the old model. It stays revoked, every search -
    old weights or new - ranks by keyword alone, the status names the directory still to
    be done, and indexing that directory restores the certificate.
    """
    first, second = tmp_path / "first", tmp_path / "second"
    for root in (first, second):
        root.mkdir()
        (root / "README.md").write_text("# Readme\n\nhello\n")
        Indexer(db, _PinnedWeights("a" * 40)).index_directory(root)

    Indexer(db, _PinnedWeights("b" * 40)).index_directory(first)

    assert db.get_meta("embedding_weights_revision") == WEIGHTS_REVOKED
    status = db.index_status(str(first))
    assert not status.verified
    assert str(second) in (status.message() or "")
    assert str(first) not in (status.message() or "")
    assert not _ranks_by_vector(db, _PinnedWeights("a" * 40))
    assert not _ranks_by_vector(db, _PinnedWeights("b" * 40))

    Indexer(db, _PinnedWeights("b" * 40)).index_directory(second)

    assert db.get_meta("embedding_weights_revision") == "b" * 40
    assert db.index_status(str(first)).verified
    assert _ranks_by_vector(db, _PinnedWeights("b" * 40))
    assert not _ranks_by_vector(db, _PinnedWeights("a" * 40))


class _UnreadableWeights(FakeEmbedder):
    """An embedder that loads, and still cannot say which weights it loaded."""

    @property
    def weights_revision(self) -> str | None:
        return None


class _UnloadableModel(FakeEmbedder):
    """An embedder whose model cannot be loaded at all."""

    def warm_up(self) -> None:
        raise ModelLoadError("the model will not load")


def test_weights_that_cannot_be_identified_are_not_assumed_to_be_the_right_ones(
    db: Database, one_document: Path
) -> None:
    """The model loaded, so something answered - it just cannot say which weights it is.

    Treating that as "nothing to compare" wrote unlabelled vectors beside labelled ones
    and cleared the very flag that said they might not match, which is the state this
    guard exists to prevent.
    """
    Indexer(db, _PinnedWeights("a" * 40)).index_directory(one_document)
    (one_document / "GUIDE.md").write_text("# Guide\n\nnew file\n")

    with pytest.raises(IndexingError, match="could not be read"):
        Indexer(db, _UnreadableWeights()).index_directory(one_document)

    assert db.count_rows("documents") == 1
    status = db.index_status(str(one_document))
    assert not status.verified
    assert "could not be read" in (status.message() or "")


def test_a_model_that_will_not_load_does_not_fail_a_run_that_needs_no_embedding(
    db: Database, one_document: Path
) -> None:
    """The check loads the model to ask it which weights it is. When loading fails, this

    run cannot write a vector whatever it does, so the index is in no danger - and an
    incremental run over unchanged files has nothing to embed anyway. Failing here turned
    a working no-op into an error on the one path where nothing could go wrong.
    """
    Indexer(db, _PinnedWeights("a" * 40)).index_directory(one_document)

    report = Indexer(db, _UnloadableModel()).index_directory(one_document)

    assert report.files_unchanged == 1
    assert db.index_status(str(one_document)).verified


class _RevisionAfterLoading(FakeEmbedder):
    """A real embedder shape: it only knows its weights once it has loaded them."""

    def __init__(self, revision: str) -> None:
        super().__init__()
        self._revision = revision
        self._loaded = False

    @property
    def weights_revision(self) -> str | None:
        return self._revision if self._loaded else None

    def warm_up(self) -> None:
        self._loaded = True


def test_only_the_model_whose_cache_it_can_find_reports_a_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`fastembed_model_dir` resolves one model's folder. Reading it for a different

    model would report a revision belonging to weights that are not the ones answering,
    which is worse than reporting none: none is merely unknown.
    """
    from markdown_memory import model_cache
    from markdown_memory.embedders import FastEmbedEmbedder
    from markdown_memory.model_cache import BGE_SMALL_MODEL_NAME

    snapshot = tmp_path / "models--qdrant--bge-small-en-v1.5-onnx-q"
    (snapshot / "refs").mkdir(parents=True)
    (snapshot / "refs" / "main").write_text("5239827812345678\n")
    monkeypatch.setattr(model_cache, "fastembed_model_dir", lambda _cache_dir: snapshot)

    assert FastEmbedEmbedder(BGE_SMALL_MODEL_NAME)._read_weights_revision() == "5239827812345678"
    other = FastEmbedEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    assert other._read_weights_revision() is None


def test_the_model_is_loaded_before_it_is_asked_which_weights_it_is(
    db: Database, one_document: Path
) -> None:
    """`weights_revision` is None until the model loads, which is indistinguishable from

    a revision that cannot be read. Asking before loading turns every ordinary change of
    weights into "could not be read" - true of nothing, and it sends the user to repair a
    cache that is perfectly healthy.

    Such a model is not loaded at the start of a run that has nothing pending, so the
    unchanged README is skipped before the change is known. The run that finds it
    therefore leaves the index revoked, and the next one - which loads first, because a
    repair is pending - finishes the job.
    """
    first = _RevisionAfterLoading("a" * 40)
    first.warm_up()
    Indexer(db, first).index_directory(one_document)
    (one_document / "GUIDE.md").write_text("# Guide\n\nnew file\n")

    Indexer(db, _RevisionAfterLoading("b" * 40)).index_directory(one_document)
    assert db.get_meta("embedding_weights_revision") == WEIGHTS_REVOKED
    message = db.get_meta("embedding_weights_mismatch") or ""
    assert "could not be read" not in message
    assert str(one_document) in message

    report = Indexer(db, _RevisionAfterLoading("b" * 40)).index_directory(one_document)
    assert (report.files_indexed, report.files_unchanged) == (1, 1)
    assert db.get_meta("embedding_weights_revision") == "b" * 40
    assert db.get_meta("embedding_weights_mismatch") is None


def test_a_document_that_embeds_nothing_records_no_provenance(db: Database, tmp_path: Path) -> None:
    """A file of nothing but headings is stored and produces no vector at all. Counting

    files rather than vectors put a revision on an index that holds none, and the next
    model was then refused over vectors that do not exist.
    """
    root = tmp_path / "headings"
    root.mkdir()
    (root / "TOC.md").write_text("# One\n\n## Two\n\n### Three\n")

    report = Indexer(db, _PinnedWeights("a" * 40)).index_directory(root)

    assert report.files_indexed == 1
    assert db.count_rows("units_vec") == 0
    assert db.get_meta("embedding_weights_revision") is None
    # And so the next model is not turned away from an index with nothing to protect.
    Indexer(db, _PinnedWeights("b" * 40)).index_directory(root)


def test_a_revision_left_over_a_vectorless_index_does_not_refuse_the_next_model(
    db: Database, tmp_path: Path
) -> None:
    """Documents without vectors - a file of nothing but headings - and a revision that

    outlived whatever it described, from an older version or an interrupted rebuild.
    There is nothing here for the guard to protect, and refusing would lock the database
    against every model for good.
    """
    root = tmp_path / "headings"
    root.mkdir()
    (root / "TOC.md").write_text("# One\n\n## Two\n")
    Indexer(db, _PinnedWeights("a" * 40)).index_directory(root)
    db.set_meta("embedding_weights_revision", "a" * 40)  # as an older version would leave it
    assert db.count_rows("documents") == 1 and db.count_rows("units_vec") == 0

    (root / "GUIDE.md").write_text("# Guide\n\nreal prose that embeds\n")
    Indexer(db, _PinnedWeights("b" * 40)).index_directory(root)

    # And the vectors it wrote are its own: an index that held none had nothing for this
    # run not to have built, whatever the document rows above them said.
    assert db.count_rows("units_vec") > 0
    assert db.get_meta("embedding_weights_revision") == "b" * 40


def test_a_revision_left_over_a_vectorless_index_does_not_refuse_unnamed_weights(
    db: Database, tmp_path: Path
) -> None:
    """The same leftover revision, met by a model that cannot say which weights it is.

    Unnamed weights are refused only where named vectors exist to be mixed with; here
    there are none, so the run writes, and the revision that described nothing is gone.
    """
    root = tmp_path / "headings"
    root.mkdir()
    (root / "TOC.md").write_text("# One\n\n## Two\n")
    Indexer(db, _PinnedWeights("a" * 40)).index_directory(root)
    db.set_meta("embedding_weights_revision", "a" * 40)  # as an older version would leave it

    (root / "GUIDE.md").write_text("# Guide\n\nreal prose that embeds\n")
    Indexer(db, _UnreadableWeights()).index_directory(root)

    assert db.count_rows("units_vec") > 0
    assert db.get_meta("embedding_weights_revision") is None


def test_a_revision_claimed_for_vectors_that_never_arrived_is_dropped_by_the_next_run(
    db: Database, tmp_path: Path
) -> None:
    """The revision is claimed before the first vector is written, so a run that dies in

    between leaves it over an empty index. A later run with the same weights and nothing
    to embed returned early - the revision agreed with it - and kept it for good.
    """
    root = tmp_path / "headings"
    root.mkdir()
    (root / "TOC.md").write_text("# One\n\n## Two\n")
    Indexer(db, _PinnedWeights("a" * 40)).index_directory(root)
    db.set_meta("embedding_weights_revision", "a" * 40)  # the claim, and then the crash

    report = Indexer(db, _PinnedWeights("a" * 40)).index_directory(root)

    assert report.files_unchanged == 1, "a run that writes nothing"
    assert db.count_rows("units_vec") == 0
    assert db.get_meta("embedding_weights_revision") is None


def test_the_revision_is_written_before_the_vectors_it_describes(
    db: Database, one_document: Path
) -> None:
    """Recorded at the end of the run, the provenance did not exist while the run was

    writing: another process reading those vectors in between found nothing saying where
    they came from, and ranked its own model's query against them. It is claimed before
    the first vector instead - and a run that dies in between leaves a revision over no
    vectors, which the next one clears.
    """
    seen: list[str | None] = []
    embedder = _PinnedWeights("a" * 40)
    original = db.replace_document

    def watch(**kwargs: object) -> Any:
        seen.append(db.get_meta("embedding_weights_revision"))
        return original(**kwargs)  # type: ignore[arg-type]

    db.replace_document = watch  # type: ignore[method-assign]
    try:
        Indexer(db, embedder).index_directory(one_document)
    finally:
        del db.replace_document  # type: ignore[attr-defined]

    assert seen == ["a" * 40]  # already true when the first vector reached the database


def test_a_file_that_embeds_nothing_does_not_need_a_model_that_loads(
    db: Database, tmp_path: Path
) -> None:
    """`embed_documents` returns without loading the model for a document of headings

    alone. Asking which weights are running before it made that file need a model it
    never uses, and the run failed where nothing could have gone wrong.
    """
    root = tmp_path / "headings"
    root.mkdir()
    (root / "TOC.md").write_text("# One\n\n## Two\n")

    report = Indexer(db, _UnloadableModel()).index_directory(root)

    assert report.files_indexed == 1
    assert db.count_rows("documents") == 1


def test_a_document_whose_prose_becomes_headings_leaves_no_provenance_behind(
    db: Database, tmp_path: Path
) -> None:
    """Replacing a document removes its old sections, and with them its vectors. When it

    held the last of them, the index is left with none - and a revision describing them
    would have search fall back to keywords over an index with nothing in it to compare.
    """
    root = tmp_path / "docs"
    root.mkdir()
    page = root / "README.md"
    page.write_text("# Readme\n\nprose that embeds\n")
    Indexer(db, _PinnedWeights("a" * 40)).index_directory(root)
    assert db.get_meta("embedding_weights_revision") == "a" * 40

    page.write_text("# Readme\n\n## Only headings\n")
    Indexer(db, _PinnedWeights("a" * 40)).index_directory(root)

    assert db.count_rows("units_vec") == 0
    assert db.get_meta("embedding_weights_revision") is None

    # In the write itself, not only when the run ends: a search in between would find a
    # revision describing vectors that are gone.
    page.write_text("# Readme\n\nprose that embeds\n")
    Indexer(db, _PinnedWeights("a" * 40)).index_directory(root)
    db.replace_document(
        file_path=str(page),
        title="Readme",
        content_hash="headings",
        last_modified=1,
        mtime_ns=1,
        sections=[dataclasses.replace(draft("Only headings", "## Only headings"), units=())],
        vectors=[SectionVectors(section=None, units=())],
    )
    assert db.count_rows("units_vec") == 0
    assert db.get_meta("embedding_weights_revision") is None


def test_purging_the_last_document_forgets_what_its_vectors_came_from(
    db: Database, one_document: Path
) -> None:
    """`delete_documents` and the rebuild for a new vector size empty the index without

    going through `clear()`. The revision they left behind described vectors that were
    gone, and search - which asks the database, not an indexing run - then ranked on
    keywords alone over an index with nothing wrong with it.
    """
    Indexer(db, _PinnedWeights("a" * 40)).index_directory(one_document)
    assert db.get_meta("embedding_weights_revision") == "a" * 40

    db.delete_documents([str(one_document / "README.md")])

    assert db.get_meta("embedding_weights_revision") is None


def test_a_cache_that_changes_while_no_document_does_still_stops_semantic_ranking(
    db: Database, one_document: Path
) -> None:
    """The hole every earlier version of this guard had. fastembed re-downloads the

    weights; no Markdown file has changed, so indexing is a clean no-op and nothing marks
    the index. Every query was then embedded by the new model and ranked against the old
    model's vectors, and the answer came back looking semantic. The search that notices
    records it, and that record is what makes the next run load the model and repair.
    """
    Indexer(db, _PinnedWeights("a" * 40)).index_directory(one_document)
    Indexer(db, _RevisionAfterLoading("b" * 40)).index_directory(one_document)  # no-op

    assert not _ranks_by_vector(db, _PinnedWeights("b" * 40))

    report = Indexer(db, _RevisionAfterLoading("b" * 40)).index_directory(one_document)
    assert report.files_indexed == 1
    assert _ranks_by_vector(db, _PinnedWeights("b" * 40))


def test_a_run_with_nothing_to_embed_does_not_refuse_and_does_not_load_the_model(
    db: Database, one_document: Path
) -> None:
    """With nothing pending and weights that cannot be named without loading, a run over

    unchanged files never loads a model it does not need. Search is what protects the
    reader in that case, by asking for itself rather than trusting a flag this run would
    have had to write.
    """
    Indexer(db, _PinnedWeights("a" * 40)).index_directory(one_document)

    embedder = _CountingWarmUp("b" * 40)
    report = Indexer(db, embedder).index_directory(one_document)

    assert report.files_unchanged == 1
    assert embedder.warm_ups == 0
    assert db.get_meta("embedding_weights_revision") == "a" * 40


def test_a_pending_repair_loads_the_model_at_the_start_of_the_run(
    db: Database, one_document: Path
) -> None:
    """Only a run that knows its weights can tell a stamped document from a stale one,

    and a run with a repair pending has embedding to do anyway - so it loads first,
    rather than skipping every unchanged document on content alone and repairing nothing.
    """
    Indexer(db, _PinnedWeights("a" * 40)).index_directory(one_document)
    db.record_weights_mismatch("a search found other weights")

    embedder = _CountingWarmUp("b" * 40)
    report = Indexer(db, embedder).index_directory(one_document)

    assert embedder.warm_ups >= 1
    assert report.files_indexed == 1
    assert db.get_meta("embedding_weights_revision") == "b" * 40


def test_vectors_no_revision_vouches_for_are_a_pending_repair(
    db: Database, one_document: Path
) -> None:
    """Vectors stored while the weights could not be read leave no revision and no

    mismatch behind. A run whose model names itself only once loaded skipped every
    unchanged document on content alone, found nothing to say, and left the index
    unvouched-for until some search happened to notice.
    """
    Indexer(db, FakeEmbedder()).index_directory(one_document)
    assert db.get_meta("embedding_weights_revision") is None

    embedder = _CountingWarmUp("b" * 40)
    report = Indexer(db, embedder).index_directory(one_document)

    assert embedder.warm_ups >= 1
    assert report.files_indexed == 1
    assert db.get_meta("embedding_weights_revision") == "b" * 40


def test_a_pending_repair_whose_model_will_not_load_does_not_fail_the_run(
    db: Database, one_document: Path
) -> None:
    """The load at the start is an optimisation of the repair, not a condition of the

    run: a model that will not load leaves a run with nothing to embed to finish as it
    always could, writing nothing about the weights and keeping the message that says
    what is wrong.
    """
    Indexer(db, _PinnedWeights("a" * 40)).index_directory(one_document)
    db.record_weights_mismatch("a search found other weights")

    report = Indexer(db, _UnloadableModel()).index_directory(one_document)

    assert report.files_unchanged == 1
    assert db.get_meta("embedding_weights_revision") == "a" * 40
    assert db.get_meta("embedding_weights_mismatch") == "a search found other weights"


class _CountingWarmUp(_RevisionAfterLoading):
    """An embedder that names its weights once loaded, and counts how often it loaded."""

    def __init__(self, revision: str) -> None:
        super().__init__(revision)
        self.warm_ups = 0

    def warm_up(self) -> None:
        self.warm_ups += 1
        super().warm_up()


def test_emptying_the_index_clears_a_mismatch_recorded_against_what_was_in_it(
    db: Database, one_document: Path
) -> None:
    """A mismatch describes the vectors that were stored. Discard them - a new embedding

    size, a model change, a purge - and the flag describes nothing, while still marking
    the index unverified and holding semantic ranking off for good.
    """
    Indexer(db, _PinnedWeights("a" * 40)).index_directory(one_document)
    assert not _ranks_by_vector(db, _PinnedWeights("b" * 40))
    assert db.get_meta("embedding_weights_mismatch") is not None

    # Every path that empties the index without going through `clear()` - a purge of the
    # last document, a rebuild for a new vector size, the old-format discard - used to
    # leave the flag behind, describing vectors that no longer exist, so the rebuilt index
    # stayed unverified and keyword-only for good.
    db.delete_documents([str(one_document / "README.md")])
    Indexer(db, _PinnedWeights("b" * 40)).index_directory(one_document)

    assert db.get_meta("embedding_weights_mismatch") is None
    assert db.index_status(str(one_document)).verified


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
    from markdown_memory.config import ServerConfig
    from markdown_memory.model_cache import fastembed_model_dir

    cache_dir = ServerConfig.from_env().model_cache_dir
    assert cache_dir is not None
    derived = fastembed_model_dir(cache_dir)
    assert derived is not None
    if not derived.is_dir():
        pytest.skip(f"bge-small has not been downloaded into {cache_dir}")
    embedder = FastEmbedEmbedder(BGE_SMALL_MODEL_NAME, cache_dir=cache_dir)
    # Read when the weights load, not on demand: the answer describes the model in
    # memory, so before anything is loaded there is honestly nothing to say.
    assert embedder.weights_revision is None
    embedder.embed_query("what revision is this")
    revision = embedder.weights_revision
    assert revision is not None and len(revision) == 40


# --- The published graph: what is fetched, and what is run ------------------------------


def test_only_the_files_in_the_manifest_are_asked_of_the_hub(
    cache: _Cache, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What is fetched is exactly what the stamp then vouches for, and nothing beside it.

    Observed at the Hub call rather than compared between two constants: an earlier
    version derived a fourth file on this machine that had to be kept out of the request,
    and a test that only reads the manifest would not notice the request drifting from it.
    """
    import huggingface_hub

    from markdown_memory import model_cache

    asked: list[list[str]] = []

    def record(repository: str, **kwargs: Any) -> str:
        asked.append(list(kwargs["allow_patterns"]))
        return cache.download(repository, **kwargs)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", record)
    cache.embedder().warm_up()

    assert asked == [list(model_cache.GEMMA_FILES)]
    assert set(asked[0]) == set(model_cache.GEMMA_MANIFEST)


def test_a_cache_holding_another_graph_is_refetched_rather_than_trusted(
    cache: _Cache,
) -> None:
    """Two graphs share one revision, so they share one cache directory.

    Upgrading from one to the other finds a directory whose every file is a perfectly good
    regular file and whose stamp is honestly signed - for the *other* graph. Only the
    recorded file set says so. Were that check to go, the stamp would be believed, this
    graph would never be fetched, and the embedder would open whatever is there.
    """
    import json

    from markdown_memory import model_cache

    cache.embedder().warm_up()
    assert len(cache.downloads) == 1

    # What the previous graph left behind: its file, and a stamp that is *honest* about
    # it - right revision, right identity, every recorded file present and unmodified.
    # Only the set of names it records is wrong, so only that check can reject it.
    graph = cache.model_dir / "model.onnx"
    graph.unlink()
    other = cache.model_dir / "other_graph.onnx"
    other.write_bytes(b"the graph that was here before")
    stamp = cache.model_dir / model_cache._VERIFIED_STAMP
    previous = json.loads(stamp.read_text())
    previous["files"] = {"other_graph.onnx": model_cache._file_identity(other)}
    stamp.write_text(json.dumps(previous))

    cache.embedder().warm_up()

    assert len(cache.downloads) == 2, "the other graph's stamp was trusted"
    assert graph.read_bytes() == _FILES["model.onnx"]
    assert json.loads(stamp.read_text())["files"].keys() == set(model_cache.GEMMA_FILES)


def test_the_embedder_runs_the_graph_the_manifest_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No derived file, no fallback: the graph that was verified is the graph that runs."""
    from markdown_memory.model_cache import gemma_model_dir

    path, _ = _session_call(tmp_path, monkeypatch)
    assert path == str(gemma_model_dir(tmp_path) / "model.onnx")


def test_a_directory_where_the_stamp_belongs_is_repaired(cache: _Cache) -> None:
    """`os.replace` will not put a file where a directory is, and the stamp reads back as

    simply invalid, so the verification it ends would run again on every start and end the
    same way - a cache holding perfectly good model files that can never be loaded.
    """
    cache.model_dir.mkdir(parents=True)
    cache.write(cache.model_dir)
    (cache.model_dir / ".verified").mkdir()
    (cache.model_dir / ".verified" / "stray.txt").write_text("")

    cache.embedder().warm_up()

    assert (cache.model_dir / ".verified").is_file()


def test_a_directory_where_a_model_file_belongs_is_repaired(cache: _Cache) -> None:
    """`unlink` cannot clear a directory, and the exception left the cache unusable."""
    cache.model_dir.mkdir(parents=True)
    (cache.model_dir / "model.onnx").mkdir()
    (cache.model_dir / "model.onnx" / "stray.txt").write_text("")

    cache.embedder().warm_up()
    assert (cache.model_dir / "model.onnx").read_bytes() == _FILES["model.onnx"]


def test_the_weights_are_recorded_only_for_vectors_every_one_of_which_they_embedded(
    db: Database, one_document: Path
) -> None:
    """The embedder loads lazily, so the first run of a fresh install starts with no

    revision to report. Recording one later is right only once every vector-bearing
    document carries it: vectors embedded blind stay unvouched-for until re-embedded.
    """
    Indexer(db, _LazyWeights(None)).index_directory(one_document)
    assert db.get_meta("embedding_weights_revision") is None

    # Vectors nobody vouched for, joined by named weights: the index is revoked before
    # the two are mixed, and the README - its only document - is re-embedded by them.
    (one_document / "GUIDE.md").write_text("# Guide\n\nnew file\n")
    (one_document / "README.md").write_text("# Readme\n\nedited\n")
    Indexer(db, _LazyWeights("b" * 40)).index_directory(one_document)
    assert db.get_meta("embedding_weights_revision") == "b" * 40

    # A revision left over an index with no vectors - by an older version, or a run that
    # died between claiming it and writing them - and a model that cannot say what it is.
    # Nothing replaces it, so it has to go rather than be left describing the next model's
    # vectors.
    db.clear()
    db.set_meta("embedding_weights_revision", "c" * 40)
    Indexer(db, _LazyWeights(None)).index_directory(one_document)
    assert db.get_meta("embedding_weights_revision") is None


def test_emptying_the_index_any_other_way_also_forgets_the_revision(
    db: Database, one_document: Path
) -> None:
    """`clear()` is not the only way the rows go: a purge of the last document, a rebuild

    for a new vector size and the old-format discard all empty it too. A revision left
    behind by any of them makes every later run report a mismatch that is not real.
    """
    Indexer(db, _PinnedWeights("a" * 40)).index_directory(one_document)
    assert db.get_meta("embedding_weights_revision") == "a" * 40

    (one_document / "README.md").unlink()
    Indexer(db, _PinnedWeights("a" * 40)).index_directory(one_document)  # purges the last one
    assert db.count_rows("documents") == 0

    (one_document / "GUIDE.md").write_text("# Guide\n\nnew\n")
    report = Indexer(db, _PinnedWeights("b" * 40)).index_directory(one_document)
    assert not any("changed since this index was built" in note for note in report.notes)
    assert db.get_meta("embedding_weights_revision") == "b" * 40


def test_discarding_every_document_discards_the_revision_that_described_them(
    db: Database, one_document: Path
) -> None:
    """Otherwise a switch to a model whose cache is not downloaded yet leaves the old

    revision behind, and every later run reports a mismatch that is not real.
    """
    Indexer(db, _PinnedWeights("a" * 40)).index_directory(one_document)
    db.clear()
    assert db.get_meta("embedding_weights_revision") is None


class _LazyWeights(FakeEmbedder):
    """An embedder that only knows its weights once it has been loaded."""

    def __init__(self, revision: str | None) -> None:
        super().__init__()
        self._revision = revision

    @property
    def weights_revision(self) -> str | None:
        return self._revision


def test_a_symlinked_directory_on_the_way_is_never_written_through(
    cache: _Cache, tmp_path: Path
) -> None:
    """`_file_identity` looks at the last component only, so a symlinked parent used to

    be trusted - and then "repaired", which meant deleting and overwriting files that
    were never in the cache at all.
    """
    outside = tmp_path / "outside"
    (outside / "nested").mkdir(parents=True)
    treasure = outside / "nested" / "weights.bin"
    treasure.write_bytes(b"somebody else's file")

    cache.model_dir.mkdir(parents=True)
    (cache.model_dir / "model.onnx").write_bytes(_FILES["model.onnx"])
    (cache.model_dir / "nested").symlink_to(outside / "nested")

    with pytest.raises(ModelLoadError, match="symlinked"):
        cache.embedder().warm_up()
    assert treasure.read_bytes() == b"somebody else's file"
    assert cache.downloads == []


def test_a_file_that_cannot_be_cleared_says_so_where_the_path_is_known(
    cache: _Cache, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A removal that quietly does nothing leaves the next step to fail without being

    able to explain itself - or, worse, leaves the damaged file in place.
    """
    from markdown_memory import model_cache

    cache.model_dir.mkdir(parents=True)
    (cache.model_dir / "model.onnx").write_bytes(b"damaged")
    monkeypatch.setattr(model_cache.shutil, "rmtree", lambda *a, **k: None)
    monkeypatch.setattr(Path, "unlink", lambda self, missing_ok=False: None)

    with pytest.raises(ModelLoadError, match="Cannot clear"):
        cache.embedder().warm_up()


# ------------------------------------------------------------------ parallel indexing


def _corpus(root: Path, files: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for n in range(files):
        (root / f"doc{n:02d}.md").write_text(
            f"# Doc {n}\n\nbody of document {n}\n\n## Detail {n}\n\nmore about {n}\n"
        )
    return root


def _index_contents(db: Database) -> list[tuple[int, str, str]]:
    conn = db.connection()
    return [
        (int(row[0]), str(row[1]), str(row[2]))
        for row in conn.execute(
            "SELECT s.id, d.file_path, s.heading_path FROM sections s "
            "JOIN documents d ON d.id = s.doc_id ORDER BY s.id"
        )
    ]


def test_several_workers_build_exactly_the_index_one_worker_builds(
    tmp_path: Path, fake_embedder: FakeEmbedder
) -> None:
    """Workers embed; the driver writes, in the order the tree was walked.

    Section ids are handed out as documents are stored, and search breaks a scoring tie
    by section id - so an index whose ids depend on which worker finished first would
    quietly answer the same query two ways on two machines.
    """
    root = _corpus(tmp_path / "docs", 12)
    built: list[list[tuple[int, str, str]]] = []
    for workers in (1, 4):
        with Database(tmp_path / f"w{workers}.db") as db:
            report = Indexer(db, fake_embedder, workers=workers).index_directory(root)
            assert (report.files_indexed, report.files_unchanged) == (12, 0)
            built.append(_index_contents(db))
    assert built[0] == built[1]


def test_no_worker_thread_ever_writes_to_the_database(
    db: Database, tmp_path: Path, fake_embedder: FakeEmbedder
) -> None:
    """One writer, whatever the worker count.

    SQLite takes one writer at a time anyway, and the ids, the certificate and the
    provenance metadata all have to be handed out in one order by one thread.
    """
    root = _corpus(tmp_path / "docs", 8)
    writers: set[int] = set()
    original = db.transaction

    def watch() -> Any:
        writers.add(threading.get_ident())
        return original()

    db.transaction = watch  # type: ignore[method-assign]
    try:
        Indexer(db, fake_embedder, workers=4).index_directory(root)
    finally:
        del db.transaction  # type: ignore[attr-defined]
    assert writers == {threading.get_ident()}


def test_one_file_failing_under_concurrency_does_not_take_the_run_with_it(
    db: Database, tmp_path: Path, fake_embedder: FakeEmbedder
) -> None:
    root = _corpus(tmp_path / "docs", 6)
    os.mkfifo(root / "fifo.md")  # reading it would block forever
    report = Indexer(db, fake_embedder, workers=4).index_directory(root)
    assert report.files_indexed == 6
    assert [failure.message for failure in report.errors] == ["Not a regular file; skipped"]
    assert report.errors[0].file_path.endswith("fifo.md")


def test_the_weights_of_a_run_are_settled_once_however_many_workers_embed(
    db: Database, tmp_path: Path
) -> None:
    """The check writes - it claims an empty index, or records a mismatch - so it runs on

    the thread that writes, once, rather than per file where it was a read-modify-write
    of the same two facts racing this run's own writes.
    """
    root = _corpus(tmp_path / "docs", 10)
    embedder = _PinnedWeights("a" * 40)
    loads = 0
    original = embedder.warm_up

    def count() -> None:
        nonlocal loads
        loads += 1
        original()

    embedder.warm_up = count  # type: ignore[method-assign]
    Indexer(db, embedder, workers=4).index_directory(root)
    assert loads == 1
    assert db.get_meta("embedding_weights_revision") == "a" * 40


def test_a_run_with_nothing_to_embed_loads_no_model_however_many_workers(
    db: Database, tmp_path: Path
) -> None:
    """A tree of headings alone produces no vector, so no model is needed to store it -

    and asking the embedder which weights it is running would load one.
    """

    class NoModel(FakeEmbedder):
        def warm_up(self) -> None:
            raise ModelLoadError("Cannot load embedding model: offline")

    root = tmp_path / "docs"
    root.mkdir()
    for n in range(6):
        (root / f"h{n}.md").write_text(f"# Heading {n}\n")
    report = Indexer(db, NoModel(), workers=4).index_directory(root)
    assert (report.files_indexed, report.passages_indexed) == (6, 0)


def test_the_driver_reads_ahead_by_a_bounded_window_not_by_the_whole_tree(
    db: Database, tmp_path: Path, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prepared file holds every vector of every passage it has - ~1.5 MiB for a

    64-passage section. Submitting the whole walk and collecting it afterwards would hold
    the corpus in memory; the window is what keeps the pool fed without doing that. What
    it bounds is how many files are handed to the pool ahead of the writer, not how much
    is embedded: embedding is serial per worker either way, so counting embeddings would
    score an unbounded submission green. (The walk itself runs to the end first - a list
    of paths costs nothing - so it is the paths *taken* from it that are counted.)
    """
    from markdown_memory import discovery

    root = _corpus(tmp_path / "docs", 12)
    taken = 0
    dedupe = discovery.without_aliases

    class _Counted(list[Path]):
        def __iter__(self) -> Iterator[Path]:
            nonlocal taken
            for path in super().__iter__():
                taken += 1
                yield path

    monkeypatch.setattr(discovery, "without_aliases", lambda paths: _Counted(dedupe(paths)))

    read_ahead = 0
    original = db.replace_document

    def watch(**kwargs: object) -> Any:
        nonlocal read_ahead
        if not read_ahead:
            read_ahead = taken
        return original(**kwargs)  # type: ignore[arg-type]

    db.replace_document = watch  # type: ignore[method-assign]
    try:
        report = Indexer(db, fake_embedder, workers=2).index_directory(root)
    finally:
        del db.replace_document  # type: ignore[attr-defined]

    assert report.files_indexed == 12
    assert 0 < read_ahead <= 4  # max(2 * workers, 2), and nowhere near the 12 on disk


def test_the_workers_really_do_embed_at_the_same_time(db: Database, tmp_path: Path) -> None:
    """Otherwise the whole feature could be a thread pool of one and every other test

    here would still pass: they compare what was written, and writing is deliberately
    serial. This one cannot finish unless two files are inside the embedder together.
    """

    class MeetsInTheMiddle(FakeEmbedder):
        def __init__(self) -> None:
            super().__init__()
            self.gate = threading.Barrier(2, timeout=30)

        def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
            self.gate.wait()  # BrokenBarrierError if nothing else arrives in time
            return super().embed_documents(texts)

    root = _corpus(tmp_path / "docs", 2)
    report = Indexer(db, MeetsInTheMiddle(), workers=2).index_directory(root)
    assert (report.files_indexed, report.errors) == (2, ())


def test_a_document_the_storage_layer_rejects_fails_alone(
    db: Database, tmp_path: Path, fake_embedder: FakeEmbedder
) -> None:
    """The write moved out of the worker and onto the driver, and for one revision it sat

    outside the per-file guard the read had: a single document the storage layer refused
    - a vector it will not take, a row that will not go in - then aborted the whole run
    instead of being recorded as that file's failure.
    """
    root = _corpus(tmp_path / "docs", 4)
    doomed = str(root / "doc02.md")
    original = db.replace_document

    def refuse(**kwargs: object) -> Any:
        if kwargs["file_path"] == doomed:
            raise DatabaseError("this document is not welcome here")
        return original(**kwargs)  # type: ignore[arg-type]

    db.replace_document = refuse  # type: ignore[method-assign]
    try:
        report = Indexer(db, fake_embedder, workers=2).index_directory(root)
    finally:
        del db.replace_document  # type: ignore[attr-defined]

    assert report.files_indexed == 3
    assert [failure.file_path for failure in report.errors] == [doomed]
    assert "not welcome" in report.errors[0].message


def test_a_file_whose_bytes_did_not_change_still_has_its_timestamp_brought_up_to_date(
    db: Database, tmp_path: Path, fake_embedder: FakeEmbedder
) -> None:
    """A skipped file used to keep whatever time it was stored with.

    A `touch`, a checkout, or a row migrated from a schema with no nanoseconds in it then
    left the freshness check hashing an unchanged file on every sweep, for ever, to
    conclude what the hash already in the row could have told it.
    """
    root = _corpus(tmp_path / "docs", 2)
    indexer = Indexer(db, fake_embedder, workers=2)
    indexer.index_directory(root)

    touched = root / "doc00.md"
    os.utime(touched, (1, 1))  # same bytes, a time from 1970
    report = indexer.index_directory(root)

    assert (report.files_indexed, report.files_unchanged) == (0, 2)  # nothing re-embedded
    content_hash, mtime_ns = db.document_fingerprints(str(root))[str(touched)]
    assert mtime_ns == touched.stat().st_mtime_ns
    assert content_hash == hashlib.sha256(touched.read_bytes()).hexdigest()


class _DiesAfter(_PinnedWeights):
    """Named weights whose process is killed after embedding ``files`` documents."""

    def __init__(self, revision: str, files: int) -> None:
        super().__init__(revision)
        self._left = files

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if self._left == 0:
            raise ModelLoadError("killed")
        self._left -= 1
        return super().embed_documents(texts)


def test_a_repair_killed_partway_resumes_where_it_stopped(db: Database, tmp_path: Path) -> None:
    """Each document's stamp is written with its vectors, so a run that dies halfway has

    already recorded which documents it repaired. The next run re-embeds only the rest,
    and in between keyword search answers while vectors are ranked by no model at all.
    """
    root = tmp_path / "docs"
    root.mkdir()
    for name in ("a", "b", "c"):
        (root / f"{name}.md").write_text(f"# {name}\n\nhello {name}\n")
    Indexer(db, _PinnedWeights("a" * 40), workers=1).index_directory(root)

    with pytest.raises(ModelLoadError):
        Indexer(db, _DiesAfter("b" * 40, files=2), workers=1).index_directory(root)

    assert db.count_rows("documents") == 3
    assert db.get_meta("embedding_weights_revision") == WEIGHTS_REVOKED
    assert not _ranks_by_vector(db, _PinnedWeights("a" * 40))
    assert not _ranks_by_vector(db, _PinnedWeights("b" * 40))

    report = Indexer(db, _PinnedWeights("b" * 40), workers=1).index_directory(root)
    assert (report.files_indexed, report.files_unchanged) == (1, 2)
    assert db.get_meta("embedding_weights_revision") == "b" * 40
    assert _ranks_by_vector(db, _PinnedWeights("b" * 40))


class _NothingDiscardedFirst(FakeEmbedder):
    """A renamed model with named weights that checks, as it embeds, what is still stored."""

    def __init__(self, db: Database, expected: int, lazy: bool = False) -> None:
        super().__init__(model_name="renamed", weights="b" * 40)
        self._db = db
        self._expected = expected
        self._loaded = not lazy

    @property
    def weights_revision(self) -> str | None:
        return self._weights if self._loaded else None

    def warm_up(self) -> None:
        self._loaded = True

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        assert self._db.count_rows("documents") == self._expected, "discarded first"
        return super().embed_documents(texts)


def test_a_renamed_model_with_named_weights_repairs_instead_of_discarding(
    db: Database, tmp_path: Path
) -> None:
    """An EmbeddingGemma upgrade changes the model's name as well as its weights, and the

    name change used to empty the whole index before a single file was re-embedded:
    keyword search went with it for the length of the rebuild. Weights that name
    themselves let the stamps do the work, so every document stays until it is replaced.
    """
    root = tmp_path / "docs"
    root.mkdir()
    for name in ("a", "b"):
        (root / f"{name}.md").write_text(f"# {name}\n\nhello {name}\n")
    Indexer(db, FakeEmbedder(model_name="original", weights="a" * 40)).index_directory(root)

    report = Indexer(db, _NothingDiscardedFirst(db, expected=2)).index_directory(root)

    assert report.files_indexed == 2
    assert report.notes == (), "nothing was discarded, so nothing to announce"
    assert db.get_meta("embedding_weights_revision") == "b" * 40


def test_a_renamed_model_that_names_its_weights_once_loaded_repairs_too(
    db: Database, tmp_path: Path
) -> None:
    """A lazily loaded model cannot name its weights until it loads, and the rename was

    decided before that - so it discarded the index anyway. It is loaded first: the run
    re-embeds everything regardless, and knowing the weights is what spares the index.
    """
    root = tmp_path / "docs"
    root.mkdir()
    for name in ("a", "b"):
        (root / f"{name}.md").write_text(f"# {name}\n\nhello {name}\n")
    Indexer(db, FakeEmbedder(model_name="original", weights="a" * 40)).index_directory(root)

    embedder = _NothingDiscardedFirst(db, expected=2, lazy=True)
    report = Indexer(db, embedder).index_directory(root)

    assert report.files_indexed == 2
    assert report.notes == ()
    assert db.get_meta("embedding_weights_revision") == "b" * 40


class _FirstLoadFails(_RevisionAfterLoading):
    """Named weights whose first load fails and whose second succeeds."""

    def __init__(self, revision: str) -> None:
        super().__init__(revision)
        self._failed = False

    def warm_up(self) -> None:
        if not self._failed:
            self._failed = True
            raise ModelLoadError("not yet")
        super().warm_up()


def test_vectors_nobody_vouched_for_are_not_ranked_beside_named_ones(
    db: Database, tmp_path: Path
) -> None:
    """An index built while the weights could not be named carries no revision at all.

    Named weights joining it used to find "no provenance to contradict" and write beside
    it, and search - which also saw no revision - ranked the mixture. Their first write
    revokes the index instead, and it stays revoked while an unvouched document remains.
    """
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.md").write_text("# a\n\nhello a\n")
    (root / "b.md").write_text("# b\n\nhello b\n")
    Indexer(db, _LazyWeights(None)).index_directory(root)
    assert db.get_meta("embedding_weights_revision") is None

    # Loaded at the start to repair them, the model would re-embed both; this one fails
    # that first load, so b.md is skipped on content and the two really are mixed.
    (root / "a.md").write_text("# a\n\nhello again\n")
    Indexer(db, _FirstLoadFails("b" * 40)).index_directory(root)

    assert db.get_meta("embedding_weights_revision") == WEIGHTS_REVOKED
    assert not _ranks_by_vector(db, _PinnedWeights("b" * 40))


def test_a_document_of_headings_alone_never_holds_up_the_certificate(
    db: Database, tmp_path: Path
) -> None:
    """A heading-only document embeds nothing, so whatever weights its row names, it has

    no vector for them to be wrong about. Counting it as stale would keep an index that is
    whole again ranked by keyword alone until somebody edited a table of contents.
    """
    root = tmp_path / "docs"
    root.mkdir()
    (root / "TOC.md").write_text("# One\n\n## Two\n")
    (root / "README.md").write_text("# Readme\n\nhello\n")
    Indexer(db, _PinnedWeights("a" * 40)).index_directory(root)

    (root / "README.md").write_text("# Readme\n\nhello again\n")
    Indexer(db, _RevisionAfterLoading("b" * 40)).index_directory(root)  # TOC.md skipped

    assert db.get_meta("embedding_weights_revision") == "b" * 40


def test_a_stale_document_no_walk_reaches_is_named_until_its_directory_is_indexed(
    db: Database, tmp_path: Path
) -> None:
    """A document indexed deliberately inside a pruned directory is never visited by a

    walk of its parent, so re-indexing the parent cannot repair it. The certificate stays
    withheld - vouching for vectors from other weights is the failure it exists to rule
    out - and the message names the directory to index instead.
    """
    root = tmp_path / "docs"
    hidden = root / ".venv" / "pkg"
    hidden.mkdir(parents=True)
    (root / "README.md").write_text("# Readme\n\nhello\n")
    (hidden / "NOTES.md").write_text("# Notes\n\nhello notes\n")
    Indexer(db, _PinnedWeights("a" * 40)).index_directory(root)
    Indexer(db, _PinnedWeights("a" * 40)).index_directory(hidden)

    Indexer(db, _PinnedWeights("b" * 40)).index_directory(root)

    assert db.get_meta("embedding_weights_revision") == WEIGHTS_REVOKED
    assert str(hidden) in (db.get_meta("embedding_weights_mismatch") or "")

    Indexer(db, _PinnedWeights("b" * 40)).index_directory(hidden)
    assert db.get_meta("embedding_weights_revision") == "b" * 40
    assert db.get_meta("embedding_weights_mismatch") is None
