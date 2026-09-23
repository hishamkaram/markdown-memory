"""Incremental indexing: directory scan, SHA-256 change detection, embedding sync.

Two local ONNX embedders are available. ``EmbeddingGemmaEmbedder`` (the default) is what
retrieval quality was tuned on; ``FastEmbedEmbedder`` (bge-small) is the light option:
a tenth of the size and ~25x faster, at a clear cost in recall on paraphrased queries.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import math
import os
import shutil
import stat
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from fnmatch import fnmatchcase
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from markdown_memory.db import (
    DEFAULT_EMBEDDING_DIM,
    VECTOR_FORMAT,
    WEIGHTS_META_KEY,
    Database,
)
from markdown_memory.exceptions import (
    EmbeddingError,
    ForeignWeightsError,
    IndexBusyError,
    IndexingError,
    MarkdownMemoryError,
    ModelLoadError,
)
from markdown_memory.graph_patch import gather_before_dequantize
from markdown_memory.models import FileFailure, IndexReport, SectionVectors
from markdown_memory.parser import MarkdownParser

if TYPE_CHECKING:
    from fastembed import TextEmbedding
    from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDER = "embeddinggemma"
BGE_SMALL_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# bge v1.5 is asymmetric: queries (never passages) need this instruction, which
# fastembed's query_embed() does not add.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

GEMMA_REPOSITORY = "onnx-community/embeddinggemma-300m-ONNX"
# Pinned so that an upstream re-export can never silently change stored vectors.
GEMMA_REVISION = "5090578d9565bb06545b4552f76e6bc2c93e4a66"
GEMMA_MODEL_FILE = "onnx/model_quantized.onnx"
# Size and sha256 of every file at GEMMA_REVISION, from the Hub's paths-info API. This is
# what stands between a damaged cache and onnxruntime: huggingface_hub checks only the
# size of what it downloads, and hands back a file that is already on disk without reading
# it at all.
GEMMA_MANIFEST: Mapping[str, tuple[int, str]] = {
    GEMMA_MODEL_FILE: (
        567_874,
        "172efde319fe1542dc41f31be6154910b05b78f7a861c265c4600eec906bd6d8",
    ),
    GEMMA_MODEL_FILE + "_data": (
        308_890_624,
        "705626e28e4c23c82ade34566b4197d97f534c12275fa406dfb71e9937d388c0",
    ),
    "tokenizer.json": (
        20_323_312,
        "4dda02faaf32bc91031dc8c88457ac272b00c1016cc679757d1c441b248b9c47",
    ),
}
GEMMA_FILES: tuple[str, ...] = tuple(GEMMA_MANIFEST)
# The graph the embedder actually runs is derived from the downloaded one on this machine:
# the same weights, with the vocabulary gathered before it is dequantized, which is worth
# about 1 GB per query (see graph_patch). It is never fetched, and it is regenerated - not
# re-downloaded - whenever it is missing or does not match. Its sha256 is also the
# rewrite's version: changing the rewriter changes this, and the tests say so.
DERIVED_GRAPH_FILE = "onnx/model_quantized.gather_first.onnx"
DERIVED_GRAPH_SHA256 = "ce47d05e0aa9abd97a474a7a951c2814060ddc2f9822dbb0ad30a407aa6e95ea"
# Versions before this one kept the files in `_GEMMA_DIR_PREFIX` itself, with no revision
# anywhere in the path: moving GEMMA_REVISION would have kept serving the old weights
# under a model name that claims to be the new ones. That folder is migrated, not
# re-downloaded, the first time this runs.
_GEMMA_DIR_PREFIX = "embeddinggemma-300m-onnx"
# One lock for every revision, so two versions starting at once still exclude each other.
_GEMMA_LOCK_NAME = f"{_GEMMA_DIR_PREFIX}.lock"
_VERIFIED_STAMP = ".verified"
GEMMA_DIMENSION = 768
GEMMA_MAX_TOKENS = 512
# Prompts from the EmbeddingGemma model card; the model is trained to expect them.
GEMMA_QUERY_PROMPT = "task: search result | query: "
GEMMA_DOCUMENT_PROMPT = "title: none | text: "
# Measured, and kept at 4 deliberately. Embedding passages in isolation, a larger batch
# looks much faster; indexing a real directory it is not, because each file is embedded on
# its own and its sections and passages differ enough in length that the padding eats the
# gain. End to end over the same corpus: batch 4 gave 3.76 vectors/s at 1,639 MB peak RSS,
# batch 16 gave 4.41 vectors/s at 2,780 MB. +17% throughput does not buy +1.1 GB on a tool
# that runs beside an editor. Sorting by token count instead of characters was measured
# too: worth ~30% at batch 4 only, which a second tokenisation pass cancels out.
_GEMMA_BATCH_SIZE = 4
# Thread count is left to onnxruntime; what is *not* left to it is spinning (see
# _SPIN_CONFIG). Pinning the count was measured from 4 to 16 threads and every value sat
# inside the run-to-run noise on wall time. That measurement missed the cost that matters
# for a tool running beside an editor: with spinning off, 16 threads and 4 threads differ
# by ~30% of CPU and nothing in wall time, so the count stays onnxruntime's business and
# only a machine that disagrees with it needs MARKDOWN_MEMORY_THREADS.
_THREADS_ENV = "MARKDOWN_MEMORY_THREADS"
# onnxruntime's intra-op threads spin-wait between operators by default. That is a good
# trade for a server answering back-to-back requests and a bad one here: measured on a
# 16-core machine, one warm query cost 7.2 s of CPU across 16 spinning threads, and the
# pool kept burning ~0.5 core-seconds per second *after* the query returned. Turning
# spinning off made the same query 0.6 s of CPU and ~40% faster in wall time, because the
# spinners were competing with the thread doing the work. Queries here arrive seconds
# apart, so the wake-up cost spinning buys is never recovered.
_SPIN_CONFIG = ("session.intra_op.allow_spinning", "0")
#: What ``_printable`` leaves where it could not decode a byte of a file name.
_UNDECODABLE = "�"
MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})
MAX_FILE_BYTES = 10 * 1024 * 1024
# Below this the pooled direction is rounding noise rather than a direction. Unit vectors
# that genuinely cancel land near 1e-16; a real centroid of normalised passages is >= 1/n
# of one passage, which for the 64-passage ceiling is ~0.015.
_MIN_POOLED_NORM = 1e-6
_EMBED_BATCH_SIZE = 32
_MODEL_META_KEY = "embedding_model"
_SKIPPED_DIRECTORIES = frozenset(
    {
        ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__",
        ".mypy_cache", ".ruff_cache", ".pytest_cache", ".tox", "site-packages",
    }
)  # fmt: skip


class Embedder(Protocol):
    """Turns text into fixed-size vectors. Implementations must be thread-safe."""

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def weights_revision(self) -> str | None: ...

    def warm_up(self) -> None: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """Local ONNX embeddings via ``fastembed``; the model loads lazily on first use."""

    def __init__(
        self,
        model_name: str = BGE_SMALL_MODEL_NAME,
        *,
        cache_dir: Path | None = None,
        dimension: int = DEFAULT_EMBEDDING_DIM,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._dimension = dimension
        self._lock = threading.Lock()
        self._model: TextEmbedding | None = None
        self._weights_revision: str | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def weights_revision(self) -> str | None:
        """Which snapshot the loaded weights came from, or None before they are loaded.

        fastembed pins no revision, so a deleted cache can come back with different
        weights under an unchanged model name - and stored passage vectors would then be
        compared against query vectors from a different model, with nothing to notice it.
        huggingface_hub records the snapshot it fetched in `refs/main`.

        Read once, when the model loads, and not on every call: the file can change under
        a running process, and what matters is the weights that produced the vectors, not
        whatever is on disk by the time somebody asks.
        """
        return self._weights_revision

    def _read_weights_revision(self) -> str | None:
        directory = fastembed_model_dir(self._cache_dir)
        if directory is None:
            return None
        try:
            return (directory / "refs" / "main").read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    def warm_up(self) -> None:
        """Load (and if necessary download) the model now instead of on first query."""
        self._load()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        try:
            vectors = [
                _to_floats(vector.tolist())
                for vector in model.embed(list(texts), batch_size=_EMBED_BATCH_SIZE)
            ]
        except Exception as exc:  # onnxruntime raises a variety of unrelated types
            raise EmbeddingError(f"Embedding {len(texts)} passages failed: {exc}") from exc
        return self._validated(vectors, expected=len(texts))

    def embed_query(self, text: str) -> list[float]:
        model = self._load()
        try:
            prompt = _BGE_QUERY_INSTRUCTION if "bge-" in self._model_name else ""
            vectors = [_to_floats(vector.tolist()) for vector in model.query_embed(prompt + text)]
        except Exception as exc:
            raise EmbeddingError(f"Embedding the query failed: {exc}") from exc
        return self._validated(vectors, expected=1)[0]

    def _validated(self, vectors: list[list[float]], *, expected: int) -> list[list[float]]:
        if len(vectors) != expected:
            raise EmbeddingError(f"Model returned {len(vectors)} vectors for {expected} texts")
        for vector in vectors:
            if len(vector) != self._dimension:
                raise EmbeddingError(
                    f"Model {self._model_name} produced {len(vector)}-dimensional vectors, "
                    f"expected {self._dimension}"
                )
        return vectors

    def _load(self) -> TextEmbedding:
        with self._lock:
            if self._model is None:
                started = time.perf_counter()
                try:
                    from fastembed import TextEmbedding  # heavy import: defer until needed

                    if self._cache_dir is not None:
                        self._cache_dir.mkdir(parents=True, exist_ok=True)
                    self._model = TextEmbedding(
                        model_name=self._model_name,
                        cache_dir=None if self._cache_dir is None else str(self._cache_dir),
                        # fastembed builds its own session, so the override reaches it only
                        # through this argument; without it MARKDOWN_MEMORY_THREADS was
                        # documented but ignored for this preset. fastembed exposes no
                        # spinning switch (its add_extra_session_options knows only
                        # enable_cpu_mem_arena), so capping the threads is the whole lever
                        # here: at 4, a query cost 95 ms of CPU instead of 718 ms.
                        threads=_inference_threads() or None,
                    )
                    self._weights_revision = self._read_weights_revision()
                except Exception as exc:
                    raise ModelLoadError(
                        f"Cannot load embedding model {self._model_name}: {exc}"
                    ) from exc
                logger.info(
                    "Loaded embedding model %s in %.2fs",
                    self._model_name,
                    time.perf_counter() - started,
                )
            return self._model


def model_cache_root(cache_dir: Path | None = None) -> Path:
    """Where every model this package downloads is kept."""
    return cache_dir or Path.home() / ".cache" / "markdown-memory" / "models"


def gemma_model_dir(cache_dir: Path | None = None) -> Path:
    """The folder holding the pinned EmbeddingGemma revision.

    A *sibling* of the unversioned folder older versions used, never a child, so the
    migration can be one rename inside one directory - which no reader can catch
    half-done.
    """
    return model_cache_root(cache_dir) / f"{_GEMMA_DIR_PREFIX}-{GEMMA_REVISION[:12]}"


def fastembed_model_dir(cache_dir: Path | None) -> Path | None:
    """The folder fastembed keeps bge-small in, or None when it cannot be derived.

    Not guessable from the model name: fastembed downloads its own re-export of the
    model (`qdrant/bge-small-en-v1.5-onnx-q`), so the repository is read out of its
    registry rather than assumed.
    """
    if cache_dir is None:
        return None
    from fastembed import TextEmbedding

    for entry in TextEmbedding.list_supported_models():
        if not isinstance(entry, dict) or entry.get("model") != BGE_SMALL_MODEL_NAME:
            continue
        sources = entry.get("sources")
        repository = sources.get("hf") if isinstance(sources, dict) else None
        if isinstance(repository, str) and repository:
            return cache_dir / ("models--" + repository.replace("/", "--"))
    return None


def _cache_path(model_dir: Path, name: str) -> Path | None:
    """``model_dir/name``, or None when any directory on the way there is a symlink.

    `_file_identity` only ever looked at the last component, so an `onnx` that pointed
    somewhere else was trusted - and then repaired, which meant deleting and overwriting
    files outside the cache entirely.
    """
    current = model_dir
    for part in Path(name).parts:
        if current.is_symlink():
            return None
        current = current / part
    return current


def _file_identity(path: Path) -> dict[str, int] | None:
    """What a stamp remembers about one model file; None when it is not a plain file.

    `ctime_ns` earns its place: `cp -p`, `tar x` and `rsync --inplace` all rewrite a
    file's contents and then restore its old mtime, so size, mtime and inode can agree
    across different bytes. Nothing in user space can set ctime back.
    """
    try:
        info = path.lstat()
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None  # a symlink into a blob store is not a file this cache vouches for
    return {
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
        "inode": info.st_ino,
        "device": info.st_dev,
    }


def _stamped_files(model_dir: Path) -> list[str]:
    """Everything a stamp vouches for: the downloaded files, and the derived graph."""
    names = list(GEMMA_MANIFEST)
    if _identity_in(model_dir, DERIVED_GRAPH_FILE) is not None:
        names.append(DERIVED_GRAPH_FILE)
    return names


def _identity_in(model_dir: Path, name: str) -> dict[str, int] | None:
    """The identity of one cached file, refusing a path that leaves the cache."""
    path = _cache_path(model_dir, name)
    return None if path is None else _file_identity(path)


def _stamp_is_current(model_dir: Path) -> bool:
    """Whether every file still looks exactly as it did when it was last verified.

    Hashing 330 MB costs most of a second of one core, which is too much for every
    server start when an editor starts one per session. This is a handful of `stat`
    calls; anything that disagrees sends the files back to be hashed.
    """
    try:
        stamp = json.loads((model_dir / _VERIFIED_STAMP).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(stamp, dict) or stamp.get("revision") != GEMMA_REVISION:
        return False
    if stamp.get("derived") != DERIVED_GRAPH_SHA256:
        return False
    recorded = stamp.get("files")
    if not isinstance(recorded, dict) or set(recorded) != set(_stamped_files(model_dir)):
        return False  # a derived graph that has gone missing is regenerated, not ignored
    if model_dir.is_symlink():
        return False
    for name in recorded:
        identity = _identity_in(model_dir, name)
        # `None` means the path is not a regular file - a symlink, most likely. Nothing
        # should be able to stamp one (`_stamped_files` leaves it out, `_unverified`
        # calls it wrong and `_derive_graph` replaces it), and this is the line that
        # makes a stamp that somehow recorded `None` stop matching `None` for ever.
        if identity is None or recorded[name] != identity:
            return False
    return True


def _write_stamp(model_dir: Path) -> None:
    """Record what was just verified. Atomically: a half-written stamp is a false claim."""
    stamp = {
        "revision": GEMMA_REVISION,
        # The rewrite's version. Without it, a package update that moves the pin - or
        # fixes the rewriter - leaves the old derived graph in place, because the file
        # itself has not changed and every identity still matches.
        "derived": DERIVED_GRAPH_SHA256,
        "files": {name: _file_identity(model_dir / name) for name in _stamped_files(model_dir)},
    }
    temporary = model_dir / f"{_VERIFIED_STAMP}.{os.getpid()}"
    _remove(temporary)  # a stamp left behind by a crash under this same pid
    _write_new_file(temporary, json.dumps(stamp).encode("utf-8"))
    os.replace(temporary, model_dir / _VERIFIED_STAMP)


def _write_new_file(path: Path, payload: bytes) -> None:
    """Create ``path`` with its contents, refusing to follow a symlink or reuse a file.

    The temporary names these writes use are predictable (the pid), and a plain write
    follows a symlink planted at one of them: the caller would then overwrite whatever it
    points at, anywhere the user can write. `O_EXCL | O_NOFOLLOW` makes both refusals the
    kernel's.
    """
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
    except BaseException:
        with contextlib.suppress(OSError):
            path.unlink()
        raise


def _remove(path: Path) -> None:
    """Delete whatever sits at ``path``, file or directory.

    A directory where a model file belongs is not something `unlink` can clear, and a
    cache that cannot be repaired is a server that never starts again.
    """
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)
    if path.exists() or path.is_symlink():
        # Say so here, where the path is known, rather than failing three lines later on
        # a rename that cannot explain itself.
        raise ModelLoadError(f"Cannot clear {path} to repair the model cache")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _unverified(model_dir: Path) -> list[str]:
    """The model files that are missing, the wrong size, or the wrong bytes."""
    problems: list[str] = []
    for name, (size, checksum) in GEMMA_MANIFEST.items():
        path = _cache_path(model_dir, name)
        before = None if path is None else _file_identity(path)
        if path is None or before is None or before["size"] != size:
            problems.append(name)
        elif _hash_file(path) != checksum or _file_identity(path) != before:
            # Second identity: a file rewritten while it was being read was never hashed
            # as it now stands, so the answer that came back means nothing.
            problems.append(name)
    return problems


@contextlib.contextmanager
def _model_cache_lock(cache_dir: Path | None, *, exclusive: bool) -> Iterator[None]:
    """Serialise verification, repair and session construction across processes.

    Shared while a verified cache is being opened, exclusive while it is being changed,
    so nothing can repair files another process has verified but not yet handed to
    onnxruntime. The kernel drops a `flock` when the process dies, so a crash leaves
    nothing held. A model cache on NFS or SMB shared between machines is out of scope:
    `flock` can be local-only there - the boundary SQLite's WAL already has.
    """
    root = model_cache_root(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    with (root / _GEMMA_LOCK_NAME).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class EmbeddingGemmaEmbedder:
    """Google's EmbeddingGemma-300m (quantized ONNX, 768 dimensions) run with onnxruntime.

    The weights live in an external data file next to the graph. onnxruntime refuses
    such a file when it is a symlink out of the model directory - which is how the
    Hugging Face cache stores it - so the files are downloaded as real files into
    ``cache_dir``. Nothing is fetched when they are already there.
    """

    def __init__(self, *, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir
        self._model_dir = gemma_model_dir(cache_dir)
        self._lock = threading.Lock()
        self._session: _OrtSession | None = None
        self._tokenizer: Tokenizer | None = None

    @property
    def model_name(self) -> str:
        return f"{GEMMA_REPOSITORY}@{GEMMA_REVISION[:12]}/{GEMMA_MODEL_FILE}"

    @property
    def dimension(self) -> int:
        return GEMMA_DIMENSION

    @property
    def weights_revision(self) -> str | None:
        return GEMMA_REVISION

    def warm_up(self) -> None:
        """Download (first run only) and load the model now instead of on first use."""
        self._load()
        self._report_other_versions()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed([GEMMA_DOCUMENT_PROMPT + text for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([GEMMA_QUERY_PROMPT + text])[0]

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        session, tokenizer = self._load()
        import numpy as np

        vectors: list[list[float]] = [[] for _ in texts]
        try:
            # Similar lengths share a batch: padding is wasted compute.
            order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
            for start in range(0, len(order), _GEMMA_BATCH_SIZE):
                batch = order[start : start + _GEMMA_BATCH_SIZE]
                encodings = tokenizer.encode_batch([texts[index] for index in batch])
                outputs = session.run(
                    ["sentence_embedding"],
                    {
                        "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
                        "attention_mask": np.array(
                            [e.attention_mask for e in encodings], dtype=np.int64
                        ),
                    },
                )[0]
                norms = np.maximum(np.linalg.norm(outputs, axis=-1, keepdims=True), 1e-12)
                for index, vector in zip(batch, (outputs / norms).tolist(), strict=True):
                    vectors[index] = _to_floats(vector)
        except Exception as exc:  # onnxruntime raises a variety of unrelated types
            raise EmbeddingError(f"Embedding {len(texts)} texts failed: {exc}") from exc
        if any(len(vector) != GEMMA_DIMENSION for vector in vectors):
            raise EmbeddingError(f"Model did not return {GEMMA_DIMENSION}-dimensional vectors")
        return vectors

    def _load(self) -> tuple[_OrtSession, Tokenizer]:
        with self._lock:
            if self._session is not None and self._tokenizer is not None:
                return self._session, self._tokenizer
            started = time.perf_counter()
            # Two passes at most: the first can find nothing worth trusting, and the
            # second runs on a cache that was repaired under the exclusive lock in
            # between - by this process or by whichever one held the lock first.
            for attempt in range(2):
                with _model_cache_lock(self._cache_dir, exclusive=False):
                    if _stamp_is_current(self._model_dir):
                        self._session, self._tokenizer = self._open()
                        logger.info(
                            "Loaded embedding model %s in %.2fs",
                            GEMMA_REPOSITORY,
                            time.perf_counter() - started,
                        )
                        return self._session, self._tokenizer
                if attempt == 0:
                    with _model_cache_lock(self._cache_dir, exclusive=True):
                        try:
                            self._repair()
                        except MarkdownMemoryError:
                            raise
                        except Exception as exc:
                            # Downloading, hashing and writing the stamp all raise things
                            # the SDK would hide behind "Error executing tool": a full
                            # disk, a revoked token, a read-only cache.
                            raise ModelLoadError(
                                f"Cannot prepare the model cache at {self._model_dir}: {exc}"
                            ) from exc
            raise ModelLoadError(
                f"The files under {self._model_dir} still do not match {GEMMA_REPOSITORY} "
                f"at {GEMMA_REVISION[:12]} after being replaced"
            )

    def _open(self) -> tuple[_OrtSession, Tokenizer]:
        """Build the tokenizer and session from files verification has just trusted.

        A failure here is not a corruption signal, because these bytes were checked
        against the manifest moments ago: nothing is deleted and nothing is downloaded.
        What is left is an onnxruntime that cannot load this graph, a permission problem,
        or a machine out of memory, and the original exception says which. (The server
        still retries on the next request; what it will not do is fetch 330 MB again.)
        """
        try:
            import onnxruntime
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(self._model_dir / "tokenizer.json"))
            tokenizer.enable_truncation(max_length=GEMMA_MAX_TOKENS)
            tokenizer.enable_padding()
            options = onnxruntime.SessionOptions()
            options.add_session_config_entry(*_SPIN_CONFIG)
            if threads := _inference_threads():
                options.intra_op_num_threads = threads
            derived = _cache_path(self._model_dir, DERIVED_GRAPH_FILE)
            graph = derived if derived and _file_identity(derived) else None
            if graph is None:  # refused, or not a regular file inside the cache
                graph = self._model_dir / GEMMA_MODEL_FILE
            session: _OrtSession = onnxruntime.InferenceSession(
                str(graph),
                options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise ModelLoadError(f"Cannot load embedding model {GEMMA_REPOSITORY}: {exc}") from exc
        return session, tokenizer

    def _repair(self) -> None:
        """Bring the cache up to the manifest. Runs under the exclusive lock."""
        if _stamp_is_current(self._model_dir):
            return  # another process did the work while this one waited for the lock
        if not self._model_dir.exists():
            self._migrate_unversioned()
        self._model_dir.mkdir(parents=True, exist_ok=True)
        wrong = _unverified(self._model_dir)
        for name in wrong:
            path = _cache_path(self._model_dir, name)
            if path is None:  # a symlinked directory on the way: refuse to write through it
                raise ModelLoadError(
                    f"{self._model_dir / name} leaves the model cache through a symlinked "
                    "directory; move it aside by hand"
                )
            _remove(path)  # only what is proven wrong
        if wrong:
            self._fetch()
            still_wrong = _unverified(self._model_dir)
            if still_wrong:
                raise ModelLoadError(
                    f"Downloaded {GEMMA_REPOSITORY} at {GEMMA_REVISION[:12]}, but "
                    f"{', '.join(sorted(still_wrong))} does not match the expected size "
                    "and checksum"
                )
        self._derive_graph()
        _write_stamp(self._model_dir)

    def _derive_graph(self) -> None:
        """Write the gather-first graph beside the downloaded one, from verified bytes.

        It shares `model_quantized.onnx_data` untouched: the initializers name that file
        relatively, and onnxruntime resolves it against the graph's own folder.
        """
        derived = _cache_path(self._model_dir, DERIVED_GRAPH_FILE)
        if derived is None:
            return  # the path leaves the cache; the published graph is still correct
        if _file_identity(derived) is not None and _hash_file(derived) == DERIVED_GRAPH_SHA256:
            return
        _remove(derived)
        rewritten = gather_before_dequantize((self._model_dir / GEMMA_MODEL_FILE).read_bytes())
        if rewritten is None:
            return  # refused, and graph_patch has said why: run the published graph
        temporary = derived.with_name(f"{derived.name}.{os.getpid()}")
        _remove(temporary)
        _write_new_file(temporary, rewritten)
        os.replace(temporary, derived)
        if _hash_file(derived) != DERIVED_GRAPH_SHA256:
            # The source was verified, so this is the rewriter and the pin disagreeing.
            # Neither is worth refusing to start over: run the published graph instead.
            logger.warning(
                "The rewritten embedding graph does not match its pinned checksum; "
                "running the published graph, which costs about 1 GB more per query"
            )
            derived.unlink(missing_ok=True)

    def _migrate_unversioned(self) -> None:
        """Adopt the pre-versioning folder, if its bytes are the pinned revision's."""
        legacy = model_cache_root(self._cache_dir) / _GEMMA_DIR_PREFIX
        if not legacy.is_dir() or _unverified(legacy):
            return  # nothing there, or bytes that have to be fetched anyway
        # One rename, inside one directory: nobody sees half the files moved. Descriptors
        # and mappings another process already holds keep reading the same inodes.
        os.replace(legacy, self._model_dir)
        logger.info("Moved the model cache %s to %s", legacy, self._model_dir)

    def _fetch(self) -> None:
        from huggingface_hub import snapshot_download

        logger.info("Downloading %s (~330 MB, first run only)", GEMMA_REPOSITORY)
        snapshot_download(
            GEMMA_REPOSITORY,
            revision=GEMMA_REVISION,
            allow_patterns=list(GEMMA_FILES),
            local_dir=self._model_dir,  # real files, not symlinks into a blob store
        )

    def _report_other_versions(self) -> None:
        """Say what other revisions cost, once, and never delete any of them.

        Weights another checkout is using, or one pinned deliberately, are not this
        process's to remove; saying how much room they take is.
        """
        others: dict[Path, int] = {}
        for path in sorted(model_cache_root(self._cache_dir).glob(f"{_GEMMA_DIR_PREFIX}*")):
            if path == self._model_dir or not path.is_dir():
                continue
            with contextlib.suppress(OSError):
                others[path] = sum(
                    entry.stat().st_size for entry in path.rglob("*") if entry.is_file()
                )
        if others:
            logger.info(
                "The model cache also holds %d older copy/copies of %s (%.0f MB in total): "
                "%s. Nothing is deleted automatically; remove them to reclaim the space.",
                len(others),
                GEMMA_REPOSITORY,
                sum(others.values()) / 1e6,
                ", ".join(str(path) for path in others),
            )


class _OrtSession(Protocol):
    """The one onnxruntime call this module makes (the package ships no type stubs)."""

    def run(
        self, output_names: Sequence[str], input_feed: dict[str, object]
    ) -> Sequence[_Array]: ...


class _Array(Protocol):
    def tolist(self) -> list[list[float]]: ...


def create_embedder(preset: str = DEFAULT_EMBEDDER, *, cache_dir: Path | None = None) -> Embedder:
    """Build one of the supported local embedders by preset name."""
    if preset == "embeddinggemma":
        return EmbeddingGemmaEmbedder(cache_dir=cache_dir)
    if preset == "bge-small":
        return FastEmbedEmbedder(BGE_SMALL_MODEL_NAME, cache_dir=cache_dir)
    raise IndexingError(f"Unknown embedder {preset!r}; choose 'embeddinggemma' or 'bge-small'")


def _inference_threads() -> int:
    """Thread count for one embedding pass: onnxruntime's own choice unless overridden.

    Deriving it from the machine was tried and rejected: on a 16-core VM the topology
    says 16, which measured worse than the default, while every count from 4 to 12 sat
    inside the run-to-run noise. A wrong number is slower than no number.

    That comparison was wall time only, which is the smaller half of the story. Once
    spinning is off (``_SPIN_CONFIG``), the count barely moves wall time but does move
    CPU: indexing the same passages took ~52 s of CPU at onnxruntime's count and ~37 s
    capped at 4. The default stays onnxruntime's, because the right cap depends on what
    else the machine is doing; this is the knob for saying so.
    """
    override = os.environ.get(_THREADS_ENV, "").strip()
    return int(override) if override.isdigit() and int(override) > 0 else 0


def _section_vector(units: Sequence[Sequence[float]]) -> list[float] | None:
    """The vector stored for a section: pooled, or one of its passages if pooling fails.

    A section with passages must have a vector - the storage layer rejects the whole file
    otherwise - so passages that cancel each other out cannot be allowed to cost the file
    its place in the index. Falling back to the first passage keeps a direction that is
    at least the section's own text.
    """
    if not units:
        return None
    pooled = _mean_vector(units)
    return pooled if pooled is not None else list(units[0])


def _mean_vector(vectors: Sequence[Sequence[float]]) -> list[float] | None:
    """The centroid of ``vectors``, renormalised, or ``None`` for a section with no body.

    A section's own vector used to be a separate embedding of its whole text, which the
    model truncates at 512 tokens: 126 of 1,589 sections in the vendored corpus were
    longer than that, the largest half again over, and their tails were simply absent
    from the section-level signal. Averaging the passages covers the section entirely,
    and costs one embedding call fewer per section rather than one more.
    """
    if not vectors:
        return None
    totals = [math.fsum(values) for values in zip(*vectors, strict=True)]
    norm = math.sqrt(math.fsum(value * value for value in totals))
    # Not `== 0.0`: passages that point opposite ways cancel to float residue near 1e-16,
    # and dividing that by its own magnitude turns rounding noise into a full-length
    # vector aimed in an arbitrary direction, which then matches arbitrary queries.
    if norm < _MIN_POOLED_NORM:
        return None
    return [value / norm for value in totals]


def _to_floats(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values]


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def iter_markdown_files(
    directory: Path,
    on_error: Callable[[OSError], None] | None = None,
    exclude: Sequence[str] = (),
) -> Iterator[Path]:
    """Yield Markdown files beneath ``directory`` in a stable order, pruning vendored trees.

    ``on_error`` receives the ``OSError`` for every sub-directory that cannot be listed
    (``os.walk`` would otherwise skip it silently).

    ``exclude`` holds glob patterns matched against each path *relative to* ``directory``
    (``scripts/eval_data/*``, ``**/vendor/**``, ``CHANGELOG.md``). A repository that keeps
    fixtures, vendored documentation or a test corpus in-tree would otherwise index them
    as if they were its own documentation. A matching directory is pruned, so its subtree
    costs nothing to skip.
    """
    for root, dirnames, filenames in os.walk(directory, followlinks=False, onerror=on_error):
        here = Path(root)
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in _SKIPPED_DIRECTORIES
            and not _is_excluded(here / name, directory, exclude)
        )
        for filename in sorted(filenames):
            path = here / filename
            if path.suffix.lower() in MARKDOWN_SUFFIXES and not _is_excluded(
                path, directory, exclude
            ):
                yield path


def _behind_symlink(root: Path, path: str) -> bool:
    """True when reaching ``path`` from ``root`` passes through a symlinked directory.

    The walk sets ``followlinks=False``, so it never descends into one - it has no idea
    what is in there, which is the same position an unreadable directory leaves it in. A
    run that treats "I did not look" as "there is nothing there" purges documents that are
    still on disk and still readable at that path, and clears failures it never rechecked.
    """
    current = root
    for part in os.path.relpath(path, root).split(os.sep)[:-1]:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _certainly_gone(root: Path, path: str) -> bool:
    """True when ``path`` is observably absent, rather than merely out of the walk's sight.

    Everything else here refuses to draw conclusions from what the walk did not visit, and
    that refusal has one consequence nobody wanted: a row about a pruned, excluded or
    symlink-shadowed path can never be retired, because the walk that would retire it never
    goes there. The file is then deleted and the row outlives it - the root it sits under
    reports itself incomplete forever, naming a path that no longer exists, and no run can
    ever change that answer.

    Not looking is not evidence. Looking at that one path is: `lstat` separates "there is
    nothing here" (`ENOENT`) from "I am not allowed to know" (`EACCES`, a loop, a dead
    mount), and only the first retires anything. That is a direct observation of one name,
    not an inference from a walk's silence, which is why it is safe where the walk is not.

    Two things make that observation worthless, and both answer `ENOENT` about a path that
    was never the file's. A name that is not valid UTF-8 reached the row through
    `_printable`, which substitutes U+FFFD for the bytes it could not decode - the stored
    string is a rendering of the name, not the name, and nothing is at it. (Testing that
    the string survives `_printable` does not find this: the replacement character is
    itself valid UTF-8 and round-trips.) And a path reached through a symlink answers for
    the link's target: replace a real directory with a broken link and every document under
    it reports `ENOENT` while the files sit untouched wherever they were moved to. Neither
    is evidence of a deletion.
    """
    if _UNDECODABLE in path:
        return False
    if _behind_symlink(root, path) or os.path.islink(path):
        return False
    try:
        os.lstat(path)
    except FileNotFoundError:
        return True
    except OSError:  # no permission, symlink loop, unreachable mount: no evidence either way
        return False
    return False


def _is_shadowing_symlink(path: str) -> bool:
    """True when ``path`` is a symlink the walk sees the name of but never follows.

    `os.walk(followlinks=False)` lists a symlinked directory among its parent's names and
    stops there, so a failure recorded against that directory cannot be rechecked by any
    walk of the tree above it. `_behind_symlink` cannot answer this: it tests the
    components *before* the last one, which is right for a file inside a linked tree and
    blind to the linked directory itself. A symlink to a *file* is walked and indexed like
    any other file, so only one that does not resolve to a file is out of reach.
    """
    return os.path.islink(path) and not os.path.isfile(path)


def _is_excluded(path: Path, root: Path, patterns: Sequence[str]) -> bool:
    """True when ``path`` matches a pattern, tested against its path relative to ``root``.

    A pattern with no ``/`` matches a *name* anywhere in the tree, the way ``.gitignore``
    treats one: ``eval_data`` excludes ``scripts/eval_data/corpus/a.md``. Requiring the
    full relative path there was a trap - the pattern looked right, matched nothing, and
    the files were indexed silently. A pattern that does contain ``/`` is anchored at the
    root and matched with ``fnmatchcase`` (never the platform's case folding), plus an
    implied ``/*`` so naming a directory covers its subtree.
    """
    if not patterns:
        return False
    try:
        relative = path.relative_to(root)
    except ValueError:  # outside the root: nothing to match against
        return False
    text = relative.as_posix()
    parts = relative.parts
    for pattern in patterns:
        if "/" in pattern:
            anchored = pattern.rstrip("/")
            if fnmatchcase(text, anchored) or fnmatchcase(text, anchored + "/*"):
                return True
        elif any(fnmatchcase(part, pattern) for part in parts):
            return True
    return False


def parse_exclusions(value: str) -> tuple[str, ...]:
    """Split a configured exclusion list on commas; blanks and stray ``./`` dropped.

    Comma only: a colon separator would split a pattern that contains one, and silently
    excluding the wrong thing is worse than not accepting the separator.
    """
    patterns = []
    for part in value.split(","):
        # One leading "./" only: `lstrip("./")` would eat the dot of `.hidden` and
        # exclude a `hidden` directory instead of the one that was named.
        cleaned = part.strip().removeprefix("./").rstrip("/")
        if cleaned:
            patterns.append(cleaned)
    return tuple(patterns)


def _key(root: Path) -> str:
    """One spelling per documentation root.

    `index_directory` resolves its argument before anything else, and that resolved root
    is the only one this key is ever built from. Normalising a second time here would
    only hide it if that ever stopped being true.
    """
    return str(root)


def _printable(path: str) -> str:
    """``path`` safe to log and to send as JSON (undecodable bytes become U+FFFD)."""
    return os.fsencode(path).decode("utf-8", errors="replace")


def _is_walkable(relative_directories: Sequence[str]) -> bool:
    return not any(name in _SKIPPED_DIRECTORIES for name in relative_directories)


class Indexer:
    """Keeps the database in sync with the Markdown files of a directory tree."""

    def __init__(
        self,
        db: Database,
        embedder: Embedder,
        parser: MarkdownParser | None = None,
        exclude: Sequence[str] = (),
    ) -> None:
        if embedder.dimension != db.embedding_dim:
            raise IndexingError(
                f"Embedder produces {embedder.dimension}-dimensional vectors but the "
                f"database stores {db.embedding_dim}"
            )
        self._db = db
        self._embedder = embedder
        self._parser = parser or MarkdownParser()
        self._exclude = tuple(exclude)
        self._run_lock = threading.Lock()

    @contextlib.contextmanager
    def _scan_lock(self) -> Iterator[None]:
        """One scan at a time over this database, across threads and across processes.

        A thread lock cannot see another process, and every ordering rule this feature
        tried instead of a lock was wrong in one direction or the other. Both halves
        refuse rather than wait: a scan can run for 25 minutes, and a tool call that
        blocks that long is a client timeout, which reads to the agent as a broken
        server rather than a busy one.

        `flock` is released by the kernel when the process dies, so a killed run cannot
        strand it - the one guarantee a row in the database could not give.
        """
        if not self._run_lock.acquire(blocking=False):
            raise IndexBusyError(
                "Another index run is in progress in this process; try again shortly."
            )
        try:
            # Inside the try: opening the lock file can fail on its own (a read-only
            # directory, no file descriptors left), and a thread lock taken above and
            # never released would refuse every later run in this process for good.
            # Resolved: two spellings of one database - a symlink, a relative path -
            # would otherwise take two different locks and both scans would proceed.
            lock_path = str(Path(self._db.path).resolve()) + ".lock"
            handle = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        except OSError:
            self._run_lock.release()
            raise
        try:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise IndexBusyError(
                    f"Another process is indexing {self._db.path}; try again shortly."
                ) from error
            os.truncate(handle, 0)
            os.write(handle, f"{os.getpid()}\n".encode("ascii"))
            yield
        finally:
            os.close(handle)
            self._run_lock.release()

    def index_directory(self, directory: Path) -> IndexReport:
        """Index new/changed files, skip unchanged ones, purge files that disappeared.

        A failure in one file is recorded in the report and does not abort the run.
        """
        try:
            root = directory.expanduser().resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as exc:  # missing, ~unknown, NUL, loop
            raise IndexingError(f"Directory does not exist: {directory}") from exc
        if not root.is_dir():
            raise IndexingError(f"Not a directory: {root}")
        try:
            str(root).encode("utf-8")
        except UnicodeEncodeError:
            raise IndexingError(
                f"Directory name is not valid UTF-8 and cannot be indexed: {_printable(str(root))}"
            ) from None

        with self._scan_lock():
            started = time.perf_counter()
            # Set at the first write of the run, not here: a run that changes nothing has
            # no business retracting a certificate that is still true. Once it does write,
            # the certificate stays retracted until a full pass finishes - so a run killed
            # partway leaves the tree honestly described as unvouched-for, with no marker
            # to clean up and nothing to go stale.
            retracted = False

            def about_to_write() -> None:
                nonlocal retracted
                if not retracted:
                    self._db.mark_scan_started(str(root))
                    retracted = True

            previous_model = self._db.get_meta(_MODEL_META_KEY)
            if previous_model not in {None, self._embedder.model_name}:
                # Vectors from different models are not comparable, and they share one
                # vector table: every document has to go, not only those under `root`.
                # Nothing is announced when a size change already emptied (and announced) it.
                self._db.clear(
                    notice=lambda discarded: (
                        f"Embedding model changed ({previous_model} -> "
                        f"{self._embedder.model_name}): discarded all {discarded} previously "
                        "indexed documents from every directory. Re-run index_directory for "
                        "any other documentation root."
                    )
                )
            self._db.set_meta(_MODEL_META_KEY, self._embedder.model_name)
            # Whatever emptied the index (new format, new vector size, new model) left a
            # notice. They are dismissed only once the report carrying them exists: a run
            # that aborts - the model cannot be loaded - leaves them for the next one.
            notices = self._db.pending_notices()
            # Captured here, after this run has done its own discarding and just before it
            # reads the hashes it will trust: a discard *after* this point means the walk
            # measured a database that no longer exists. Captured any earlier and the run
            # counts its own model-change wipe as somebody else's, then refuses to certify
            # the index it just rebuilt from scratch.
            # Whether this run is the one that built everything in the index. Only then
            # can the weights it embedded with describe every vector stored.
            started_empty = self._db.count_rows("documents") == 0
            generation = self._db.generation()
            known_hashes = self._db.document_hashes(str(root))

            seen: set[str] = set()
            indexed = unchanged = sections_indexed = passages_indexed = 0
            failures: list[FileFailure] = []
            unreadable: list[str] = []

            def record_unreadable(error: OSError) -> None:
                location = str(error.filename or root)
                unreadable.append(location)
                failures.append(
                    FileFailure(
                        file_path=_printable(location),
                        message=f"Cannot list directory: {error.strerror or error}",
                    )
                )

            for path in iter_markdown_files(root, record_unreadable, self._exclude):
                file_path = str(path)
                seen.add(file_path)
                try:
                    counts = self._index_file(path, known_hashes.get(file_path), about_to_write)
                except (ModelLoadError, ForeignWeightsError):
                    raise  # not this file's fault: every other file would fail identically
                except (MarkdownMemoryError, OSError) as exc:
                    logger.warning("Failed to index %s: %s", _printable(file_path), exc)
                    failures.append(FileFailure(file_path=_printable(file_path), message=str(exc)))
                    continue
                if counts is None:
                    unchanged += 1
                else:
                    indexed += 1
                    sections_indexed += counts[0]
                    passages_indexed += counts[1]

            vanished = self._vanished(root, known_hashes, seen, unreadable)
            if vanished:
                about_to_write()  # deleting is changing it, even if no file was read
            purged = self._db.delete_documents(vanished)
            # This run's own failures are already in `errors`, and the search tools read
            # the recorded ones straight from the database, so nothing is added to
            # `notes`: a warning repeated in three places is how a warning becomes noise.
            # Only what this walk could have reached: a failure inside a pruned directory
            # or one that could not be listed is not this run's to forget, however far
            # under its root it sits.
            reachable = self._reachable(root, self._db.failure_paths(str(root)), unreadable)
            self._db.record_failures(
                reachable,
                {
                    failure.file_path: f"{failure.message} (indexing {_printable(str(root))})"
                    for failure in failures
                },
            )
            # The walk finished, which is all this records; what it could not read is
            # recorded separately, and `index_status` refuses to call a tree whole while
            # anything under it is still listed there. Two facts, two places, one answer.
            # Before the certificate, so a crash between the two leaves the tree
            # honestly unvouched-for rather than vouched-for with no provenance.
            self._record_weights_revision(started_empty, indexed)
            self._db.mark_scan_complete(str(root), generation)
            report = IndexReport(
                directory=_printable(str(root)),
                files_scanned=len(seen),
                files_indexed=indexed,
                files_unchanged=unchanged,
                files_purged=purged,
                sections_indexed=sections_indexed,
                passages_indexed=passages_indexed,
                elapsed_seconds=time.perf_counter() - started,
                errors=tuple(failures),
                notes=tuple(notices.values()),
            )
            self._db.dismiss_notices(notices)
        logger.info(report.summary())
        return report

    def _refuse_foreign_weights(self) -> None:
        """Stop before embedding if the model is not the one whose vectors are stored.

        Called at the point of use - the line before a vector is produced - because that
        is the only place a lazily-loaded embedder can be asked what it is without making
        a run that needs no model load one. Nothing is discarded; a rebuild costs a
        quarter of an hour and is the user's to ask for. But nothing new is written
        either, and every `index_status` says why until it is resolved.
        """
        recorded = self._db.get_meta(WEIGHTS_META_KEY)
        if recorded is None:
            return  # no provenance to contradict
        if self._db.count_rows("documents") == 0:
            # The revision describes vectors that are gone: `delete_documents` and the
            # rebuild for a new vector size empty the index without touching the meta
            # keys. Left standing, it would refuse every future run over a database with
            # nothing in it to protect - and keep a mismatch flag that suppresses ranking
            # on whatever is built next.
            self._db.forget_weights_revision()
            self._db.record_weights_mismatch(None)
            return
        # The load this run is about to do anyway. A model that will not load raises here
        # exactly as it would one line later, and a run with nothing to embed never
        # arrives.
        self._embedder.warm_up()
        weights = self._embedder.weights_revision
        if weights == recorded:
            self._db.record_weights_mismatch(None)
            return
        if weights is None:
            # The model loaded, so something answered - it just cannot say which weights
            # it is. That is not "nothing to compare": the vectors written now would be
            # unlabelled and indistinguishable from the ones already stored, which is the
            # state this guard exists to prevent.
            message = (
                f"Which weights {self._embedder.model_name} is running could not be read, so "
                "there is no way to tell whether they are the ones that built this index "
                f"({recorded[:12]}). Nothing has been discarded and nothing new is being "
                "indexed; repair the model cache and run index_directory again."
            )
        else:
            message = (
                f"The weights behind {self._embedder.model_name} changed since this index was "
                f"built ({recorded[:12]} -> {weights[:12]}), so its vectors and the ones a "
                "query would produce now come from different models. Nothing has been "
                "discarded and nothing new is being indexed; re-index this documentation "
                f"root from scratch (delete {self._db.path} and run index_directory) to make "
                "them comparable again."
            )
        self._db.record_weights_mismatch(message)
        self._db.revoke_coverage()
        raise ForeignWeightsError(message)

    def _record_weights_revision(self, started_empty: bool, embedded: int) -> None:
        """Note which weights produced the vectors this index now holds.

        A model *name* is not enough for bge-small: fastembed pins no revision, so a
        re-download can bring different weights under the same name and nothing in the
        index would notice. Only a run that built the index from nothing can say where
        all of it came from, so only such a run records it; anything else would put a
        provenance on vectors it never saw written. A run that finds a *different*
        revision never reaches here - `_refuse_foreign_weights` aborts it at the vector
        that would have been the first, which is the last point at which stopping helps.
        """
        if started_empty:
            # Nothing was here to lose. Whatever this run embedded - possibly nothing -
            # is the whole index, so any revision recorded before it describes vectors
            # that no longer exist, whether they were purged, rebuilt or discarded.
            self._db.forget_weights_revision()
        weights = self._embedder.weights_revision
        if weights is None:
            return  # nothing worth recording
        if started_empty and embedded and self._db.get_meta(WEIGHTS_META_KEY) is None:
            self._db.set_meta(WEIGHTS_META_KEY, weights)

    def _reachable(self, root: Path, paths: Sequence[str], unreadable: Sequence[str]) -> list[str]:
        """The subset of ``paths`` a walk of ``root`` would have visited.

        Pruned directories (`.venv`, `node_modules`), directories excluded by
        configuration, and directories that could not be listed are never entered, so this
        run saw nothing inside them and may not speak for what it did not see.

        Every component is tested, including the last. A recorded failure is usually a
        file, but an unreadable *directory* is recorded under its own path - and dropping
        the final component would ask whether `.venv`'s parent is walkable rather than
        whether `.venv` is, and then clear it.

        A path out of the walk's sight is still retired once it is observably gone
        (`_certainly_gone`), or its row would outlive the file and no run could ever
        retire it.
        """
        blocked = tuple(location.rstrip(os.sep) + os.sep for location in unreadable)
        visitable = []
        for path in paths:
            if self._walk_would_visit(root, path, blocked) or _certainly_gone(root, path):
                visitable.append(path)
        return visitable

    def _walk_would_visit(self, root: Path, path: str, blocked: tuple[str, ...]) -> bool:
        """Whether a walk of ``root`` reaches ``path``, given the directories it could not list."""
        if blocked and path.startswith(blocked):
            return False
        if not _is_walkable(os.path.relpath(path, root).split(os.sep)):
            return False
        if _behind_symlink(root, path) or _is_shadowing_symlink(path):
            return False
        return not (self._exclude and _is_excluded(Path(path), root, self._exclude))

    @staticmethod
    def _vanished(
        root: Path,
        known: dict[str, tuple[str, int]],
        seen: set[str],
        unreadable: Sequence[str],
    ) -> list[str]:
        """Known documents that this walk *would* have found had they still existed.

        A document is only purged when its absence is evidence of deletion. It is kept
        when the walk could not have reached it: it lives under a directory that could
        not be listed, or under a pruned tree (``node_modules`` ...) that was indexed
        explicitly by pointing ``index_directory`` inside it.

        Unless the file is observably gone (`_certainly_gone`). Not being visited is not
        evidence of deletion; `ENOENT` on that one name is exactly that evidence, and
        without it a deleted document under a pruned tree keeps answering searches with
        text that is not on disk any more, until someone re-indexes that tree by hand.
        """
        blocked = tuple(location.rstrip(os.sep) + os.sep for location in unreadable)
        vanished: list[str] = []
        for file_path in sorted(set(known) - seen):
            if _certainly_gone(root, file_path):
                vanished.append(file_path)
                continue
            if blocked and file_path.startswith(blocked):
                continue
            relative = os.path.relpath(file_path, root)
            if not _is_walkable(relative.split(os.sep)[:-1]):
                continue
            if _behind_symlink(root, file_path):
                continue
            vanished.append(file_path)
        return vanished

    def _index_file(
        self,
        path: Path,
        known: tuple[str, int] | None,
        about_to_write: Callable[[], None],
    ) -> tuple[int, int] | None:
        """Index one file. Returns ``(sections, passages)``, or ``None`` when unchanged."""
        file_path = str(path)
        try:
            file_path.encode("utf-8")
        except UnicodeEncodeError:
            raise IndexingError("File name is not valid UTF-8; skipped") from None
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            # A FIFO or device named *.md would block or stream forever when read.
            raise IndexingError("Not a regular file; skipped")
        with path.open("rb") as handle:
            data = handle.read(MAX_FILE_BYTES + 1)  # st_size cannot be trusted for the cap
        if len(data) > MAX_FILE_BYTES:
            raise IndexingError(f"File is larger than {MAX_FILE_BYTES} bytes; skipped")
        content_hash = hash_bytes(data)
        # The format counts as much as the content: a file whose bytes never changed still
        # has to be rebuilt if its vectors were pooled by an older scheme, or it would keep
        # them forever and the table would answer one query two different ways.
        if known is not None and known == (content_hash, VECTOR_FORMAT):
            return None
        parsed = self._parser.parse(
            data.decode("utf-8", errors="replace"), fallback_title=path.stem
        )
        # One embedding call per file, over the passages alone. The section vector is the
        # mean of its passages rather than a separate embedding of the whole section:
        # that text ran past the model's 512-token limit for 7.9% of the vendored corpus
        # and lost its tail, and embedding it cost one extra call per section.
        texts: list[str] = []
        for section in parsed.sections:
            texts.extend(section.unit_texts)
        self._refuse_foreign_weights()
        embeddings = self._embedder.embed_documents(texts)
        if len(embeddings) != len(texts):
            raise EmbeddingError(f"Got {len(embeddings)} vectors for {len(texts)} texts")
        embedded = iter(embeddings)
        vectors = []
        for section in parsed.sections:
            units = tuple(next(embedded) for _ in section.units)
            vectors.append(SectionVectors(section=_section_vector(units), units=units))
        # Here, and not a line earlier: parsing and embedding can fail without touching
        # the index, and a run that changed nothing must leave a standing certificate
        # alone. A model that will not load fails identically on every file.
        about_to_write()
        self._db.replace_document(
            file_path=file_path,
            title=parsed.title,
            content_hash=content_hash,
            last_modified=int(info.st_mtime),
            sections=parsed.sections,
            vectors=vectors,
        )
        return len(parsed.sections), sum(len(section.units) for section in parsed.sections)
