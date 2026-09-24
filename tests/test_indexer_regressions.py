"""Regressions in the embedders: how the ONNX sessions they build are configured.

One test per defect found in code review; each fails when its fix is reverted.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeEmbedder

from markdown_memory.db import Database
from markdown_memory.embedders import EmbeddingGemmaEmbedder, FastEmbedEmbedder
from markdown_memory.exceptions import DatabaseError, IndexingError, ModelLoadError
from markdown_memory.indexer import Indexer
from markdown_memory.model_cache import BGE_SMALL_MODEL_NAME, GEMMA_FILES

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


def _session_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, derived: bool = True
) -> tuple[str, Any]:
    """Load a Gemma embedder against stub files; return what it asked onnxruntime for.

    The stubs are written where the code now looks - the revision-keyed folder - and the
    manifest is stubbed to match them, so verification passes offline. `snapshot_download`
    is replaced by a failure: a test about session options has no business fetching
    330 MB, and before this was pinned these tests only passed on a machine whose
    Hugging Face cache happened to be warm.
    """
    import huggingface_hub
    import onnxruntime
    import tokenizers

    from markdown_memory import embedders, model_cache

    manifest = {
        name: (len(data), hashlib.sha256(data).hexdigest()) for name, data in _FILES.items()
    }
    monkeypatch.setattr(model_cache, "GEMMA_MANIFEST", manifest)
    monkeypatch.setattr(model_cache, "GEMMA_FILES", tuple(manifest))
    monkeypatch.setattr(model_cache, "GEMMA_MODEL_FILE", "model.onnx")
    monkeypatch.setattr(model_cache, "DERIVED_GRAPH_FILE", "derived.onnx")
    monkeypatch.setattr(model_cache, "DERIVED_GRAPH_SHA256", hashlib.sha256(_DERIVED).hexdigest())
    monkeypatch.setattr(
        embedders, "gather_before_dequantize", lambda graph: _DERIVED if derived else None
    )
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
#: What the stubbed rewrite produces from _FILES["model.onnx"].
_DERIVED = b"the same graph, gathering first"


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Cache:
    """A Gemma embedder whose manifest is two tiny files and whose download is a stub."""
    import huggingface_hub

    from markdown_memory import embedders, model_cache

    manifest = {
        name: (len(data), hashlib.sha256(data).hexdigest()) for name, data in _FILES.items()
    }
    monkeypatch.setattr(model_cache, "GEMMA_MANIFEST", manifest)
    monkeypatch.setattr(model_cache, "GEMMA_FILES", tuple(manifest))
    monkeypatch.setattr(model_cache, "GEMMA_MODEL_FILE", "model.onnx")
    monkeypatch.setattr(model_cache, "DERIVED_GRAPH_FILE", "derived.onnx")
    monkeypatch.setattr(model_cache, "DERIVED_GRAPH_SHA256", hashlib.sha256(_DERIVED).hexdigest())
    harness = _Cache(tmp_path)
    monkeypatch.setattr(embedders, "gather_before_dequantize", harness.rewrite)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", harness.download)
    monkeypatch.setattr(EmbeddingGemmaEmbedder, "_open", lambda self: harness.open())
    return harness


class _Cache:
    """The scratch model cache, its download stub, and what each of them was asked to do."""

    def __init__(self, root: Path) -> None:
        from markdown_memory.model_cache import gemma_model_dir

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


def test_a_model_whose_weights_changed_may_not_write_into_the_index(
    db: Database, one_document: Path
) -> None:
    """fastembed pins no revision, so a re-download can bring different weights under the

    same model name. Noticing that after the run is noticing it too late: the files that
    changed have already been re-embedded, and the index holds two models' vectors with
    nothing saying so. The run stops before writing anything instead.
    """
    Indexer(db, _PinnedWeights("a" * 40)).index_directory(one_document)
    assert db.get_meta("embedding_weights_revision") == "a" * 40
    (one_document / "GUIDE.md").write_text("# Guide\n\nnew file\n")

    with pytest.raises(IndexingError, match="changed since this index was built") as refused:
        Indexer(db, _PinnedWeights("b" * 40)).index_directory(one_document)

    # What the refusal promises, exactly: no vector from the new model, not "no writes".
    # A document of headings alone embeds nothing and may already have been stored.
    assert "no vector from the new model has been stored" in str(refused.value)
    assert "headings alone" in str(refused.value)

    # Nothing discarded, nothing added, and the record still describes what is stored.
    assert db.count_rows("documents") == 1
    assert db.get_meta("embedding_weights_revision") == "a" * 40


def test_an_index_answering_from_another_models_vectors_says_so_in_its_status(
    db: Database, one_document: Path
) -> None:
    """Every search and listing carries `index_status`, and until this it could read

    `verified` while the stored vectors and the query's came from different models.
    """
    Indexer(db, _PinnedWeights("a" * 40)).index_directory(one_document)
    assert db.index_status(str(one_document)).verified

    (one_document / "GUIDE.md").write_text("# Guide\n\nnew file\n")
    with pytest.raises(IndexingError):
        Indexer(db, _PinnedWeights("b" * 40)).index_directory(one_document)

    status = db.index_status(str(one_document))
    assert not status.verified
    assert status.to_dict()["coverage"] == "unknown"
    assert "different models" in (status.message() or "")

    # Back on the weights it was built with, the warning goes away on its own.
    Indexer(db, _PinnedWeights("a" * 40)).index_directory(one_document)
    assert db.index_status(str(one_document)).verified


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
    """
    first = _RevisionAfterLoading("a" * 40)
    first.warm_up()
    Indexer(db, first).index_directory(one_document)
    (one_document / "GUIDE.md").write_text("# Guide\n\nnew file\n")

    with pytest.raises(IndexingError, match="changed since this index was built"):
        Indexer(db, _RevisionAfterLoading("b" * 40)).index_directory(one_document)


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
    model's vectors, and the answer came back looking semantic.
    """
    from markdown_memory.search import HybridSearcher

    Indexer(db, _PinnedWeights("a" * 40)).index_directory(one_document)
    Indexer(db, _PinnedWeights("b" * 40)).index_directory(one_document)  # nothing to embed

    searcher = HybridSearcher(db, _PinnedWeights("b" * 40))
    try:
        results = searcher.search("hello")
    finally:
        searcher.close()
    assert all(result.vec_rank is None for result in results)


def test_a_run_with_nothing_to_embed_does_not_refuse_and_does_not_load_the_model(
    db: Database, one_document: Path
) -> None:
    """The check runs where a vector is about to be produced, so a run over unchanged

    files never reaches it - and never loads a model it does not need. Search is what
    protects the reader in that case, by asking for itself rather than trusting a flag
    this run would have had to write.
    """
    Indexer(db, _PinnedWeights("a" * 40)).index_directory(one_document)

    embedder = _CountingWarmUp("b" * 40)
    report = Indexer(db, embedder).index_directory(one_document)

    assert report.files_unchanged == 1
    assert embedder.warm_ups == 0
    assert db.get_meta("embedding_weights_revision") == "a" * 40


class _CountingWarmUp(_PinnedWeights):
    """A pinned embedder that counts how often it was asked to load."""

    def __init__(self, revision: str) -> None:
        super().__init__(revision)
        self.warm_ups = 0

    def warm_up(self) -> None:
        self.warm_ups += 1


def test_emptying_the_index_clears_a_mismatch_recorded_against_what_was_in_it(
    db: Database, one_document: Path
) -> None:
    """A mismatch describes the vectors that were stored. Discard them - a new embedding

    size, a model change, a purge - and the flag describes nothing, while still marking
    the index unverified and holding semantic ranking off for good.
    """
    Indexer(db, _PinnedWeights("a" * 40)).index_directory(one_document)
    (one_document / "GUIDE.md").write_text("# Guide\n\nnew file\n")
    with pytest.raises(IndexingError):
        Indexer(db, _PinnedWeights("b" * 40)).index_directory(one_document)
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
    """The derived graph is made here, so it may never appear in what the Hub is asked for."""
    assert "derived.onnx" not in GEMMA_FILES


def test_the_embedder_runs_the_derived_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of deriving it: ~1 GB per query lives in this one path."""
    from markdown_memory.model_cache import gemma_model_dir

    path, _ = _session_call(tmp_path, monkeypatch)
    assert path == str(gemma_model_dir(tmp_path) / "derived.onnx")


def test_a_refused_rewrite_falls_back_to_the_published_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal costs the memory saving; it must not cost a server that starts."""
    from markdown_memory.model_cache import gemma_model_dir

    path, _ = _session_call(tmp_path, monkeypatch, derived=False)
    assert path == str(gemma_model_dir(tmp_path) / "model.onnx")


def test_moving_the_derived_pin_invalidates_a_stamp_that_still_matches_the_files(
    cache: _Cache, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An upgrade that changes the rewrite must take effect on an existing cache.

    Every file identity still matches after the package changes, so without the pin in
    the stamp the old derived graph would keep being loaded for ever.
    """
    from markdown_memory import embedders, model_cache

    cache.embedder().warm_up()
    assert cache.rewrites == 1

    newer = b"a differently rewritten graph"
    monkeypatch.setattr(model_cache, "DERIVED_GRAPH_SHA256", hashlib.sha256(newer).hexdigest())
    monkeypatch.setattr(embedders, "gather_before_dequantize", lambda graph: newer)
    cache.embedder().warm_up()
    assert (cache.model_dir / "derived.onnx").read_bytes() == newer
    assert cache.downloads == [cache.model_dir]  # regenerated, never re-fetched


def test_a_derived_graph_that_is_a_symlink_is_never_trusted(cache: _Cache) -> None:
    """lstat records no identity for a symlink, and `None == None` would match for ever,

    leaving a target that is swapped afterwards trusted without ever being verified.
    """
    cache.embedder().warm_up()
    derived = cache.model_dir / "derived.onnx"
    elsewhere = cache.root / "elsewhere.onnx"
    elsewhere.write_bytes(_DERIVED)
    derived.unlink()
    derived.symlink_to(elsewhere)

    cache.embedder().warm_up()
    assert not derived.is_symlink()
    assert derived.read_bytes() == _DERIVED


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


def test_the_weights_are_recorded_only_for_an_index_this_run_built_whole(
    db: Database, one_document: Path
) -> None:
    """The embedder loads lazily, so the first run of a fresh install starts with no

    revision to report. Recording one later - when the vectors were embedded blind, or by
    somebody else - would put a revision on them that may simply be wrong.
    """
    Indexer(db, _LazyWeights(None)).index_directory(one_document)
    assert db.get_meta("embedding_weights_revision") is None

    # The index is no longer empty, so this run cannot vouch for what is in it.
    (one_document / "README.md").write_text("# Readme\n\nedited\n")
    Indexer(db, _LazyWeights("b" * 40)).index_directory(one_document)
    assert db.get_meta("embedding_weights_revision") is None

    # Built from nothing: now every vector came from these weights.
    db.clear()
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
    it bounds is how far ahead the *walk* runs, not how much is embedded: embedding is
    serial per worker either way, so counting embeddings would score an unbounded
    submission green.
    """
    from markdown_memory import discovery

    root = _corpus(tmp_path / "docs", 12)
    walked = 0
    walk = discovery.iter_markdown_files

    def counting(*args: Any, **kwargs: Any) -> Any:
        nonlocal walked
        for path in walk(*args, **kwargs):
            walked += 1
            yield path

    monkeypatch.setattr(discovery, "iter_markdown_files", counting)

    read_ahead = 0
    original = db.replace_document

    def watch(**kwargs: object) -> Any:
        nonlocal read_ahead
        if not read_ahead:
            read_ahead = walked
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
