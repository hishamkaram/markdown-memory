"""The embedders: two local ONNX models behind one protocol.

``EmbeddingGemmaEmbedder`` (the default) is what retrieval quality was tuned on;
``FastEmbedEmbedder`` (bge-small) is the light option: a tenth of the size and ~25x
faster, at a clear cost in recall on paraphrased queries. ``numpy`` and ``onnxruntime``
live here and nowhere else in the package.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from markdown_memory import model_cache
from markdown_memory.db import DEFAULT_EMBEDDING_DIM
from markdown_memory.exceptions import (
    EmbeddingError,
    IndexingError,
    MarkdownMemoryError,
    ModelLoadError,
)

if TYPE_CHECKING:
    from fastembed import TextEmbedding
    from tokenizers import Tokenizer

logger = logging.getLogger(__name__)


DEFAULT_EMBEDDER = "embeddinggemma"


# bge v1.5 is asymmetric: queries (never passages) need this instruction, which
# fastembed's query_embed() does not add.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


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


_EMBED_BATCH_SIZE = 32


def short_weights(identity: str | None) -> str:
    """A weights identity short enough for a message, keeping what distinguishes it.

    Twelve characters of the revision used to be enough. It is not any more: one revision
    publishes several graphs, so both sides of "the weights changed from X to Y" would
    print the same string and the message would read as nonsense while being true.
    """
    if not identity:
        return "no readable revision"
    revision, separator, graph = identity.partition("/")
    return f"{revision[:12]}/{graph}" if separator else revision[:12]


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
        model_name: str = model_cache.BGE_SMALL_MODEL_NAME,
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
        if self._model_name != model_cache.BGE_SMALL_MODEL_NAME:
            # `model_cache.fastembed_model_dir` resolves one model's folder. Reading it for a
            # different model would report a revision belonging to weights that are not
            # the ones answering - worse than reporting none, which is merely unknown.
            return None
        directory = model_cache.fastembed_model_dir(self._cache_dir)
        if directory is None:
            return None
        try:
            return (directory / "refs" / "main").read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    def warm_up(self) -> None:
        """Load (and if necessary download) the model now instead of on first query.

        A model loaded while its revision could not be read is loaded again: the index
        asks this when it needs to know which weights it is running, and reading the
        revision alone, after the fact, could name files that were not the ones loaded.
        """
        with self._lock:
            if self._weights_revision is None:
                self._model = None
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


class EmbeddingGemmaEmbedder:
    """Google's EmbeddingGemma-300m (quantized ONNX, 768 dimensions) run with onnxruntime.

    The weights live in an external data file next to the graph. onnxruntime refuses
    such a file when it is a symlink out of the model directory - which is how the
    Hugging Face cache stores it - so the files are downloaded as real files into
    ``cache_dir``. Nothing is fetched when they are already there.
    """

    def __init__(self, *, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir
        self._model_dir = model_cache.gemma_model_dir(cache_dir)
        self._lock = threading.Lock()
        self._session: _OrtSession | None = None
        self._tokenizer: Tokenizer | None = None

    @property
    def model_name(self) -> str:
        return (
            f"{model_cache.GEMMA_REPOSITORY}@{model_cache.GEMMA_REVISION[:12]}"
            f"/{model_cache.GEMMA_MODEL_FILE}"
        )

    @property
    def dimension(self) -> int:
        return GEMMA_DIMENSION

    @property
    def weights_revision(self) -> str | None:
        """Which weights these vectors came from - the revision *and* the graph.

        The revision alone is not enough. This repository publishes several graphs at one
        revision, and they do not agree: swapping the int8 graph for the 4-bit one moves a
        query's vector by about 0.03 cosine, which is far more than the distance search
        ranks on. With the bare revision, an index built by one graph and searched by the
        other passes this check, so the query is embedded by one model and compared
        against another's vectors - no error, just quietly worse answers.
        """
        return f"{model_cache.GEMMA_REVISION}/{model_cache.GEMMA_MODEL_FILE}"

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
                with model_cache._model_cache_lock(self._cache_dir, exclusive=False):
                    if model_cache._stamp_is_current(self._model_dir):
                        self._session, self._tokenizer = self._open()
                        logger.info(
                            "Loaded embedding model %s in %.2fs",
                            model_cache.GEMMA_REPOSITORY,
                            time.perf_counter() - started,
                        )
                        return self._session, self._tokenizer
                if attempt == 0:
                    with model_cache._model_cache_lock(self._cache_dir, exclusive=True):
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
                f"The files under {self._model_dir} still do not match "
                f"{model_cache.GEMMA_REPOSITORY} "
                f"at {model_cache.GEMMA_REVISION[:12]} after being replaced"
            )

    def _open(self) -> tuple[_OrtSession, Tokenizer]:
        """Build the tokenizer and session from files verification has just trusted.

        A failure here is not a corruption signal, because these bytes were checked
        against the manifest moments ago: nothing is deleted and nothing is downloaded.
        What is left is an onnxruntime that cannot load this graph, a permission problem,
        or a machine out of memory, and the original exception says which. (The server
        still retries on the next request; what it will not do is fetch 218 MB again.)
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
            session: _OrtSession = onnxruntime.InferenceSession(
                str(self._model_dir / model_cache.GEMMA_MODEL_FILE),
                options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Cannot load embedding model {model_cache.GEMMA_REPOSITORY}: {exc}"
            ) from exc
        return session, tokenizer

    def _repair(self) -> None:
        """Bring the cache up to the manifest. Runs under the exclusive lock."""
        if model_cache._stamp_is_current(self._model_dir):
            return  # another process did the work while this one waited for the lock
        self._model_dir.mkdir(parents=True, exist_ok=True)
        wrong = model_cache._unverified(self._model_dir)
        for name in wrong:
            path = model_cache._cache_path(self._model_dir, name)
            if path is None:  # a symlinked directory on the way: refuse to write through it
                raise ModelLoadError(
                    f"{self._model_dir / name} leaves the model cache through a symlinked "
                    "directory; move it aside by hand"
                )
            model_cache._remove(path)  # only what is proven wrong
        if wrong:
            self._fetch()
            still_wrong = model_cache._unverified(self._model_dir)
            if still_wrong:
                raise ModelLoadError(
                    f"Downloaded {model_cache.GEMMA_REPOSITORY} at "
                    f"{model_cache.GEMMA_REVISION[:12]}, but "
                    f"{', '.join(sorted(still_wrong))} does not match the expected size "
                    "and checksum"
                )
        model_cache._write_stamp(self._model_dir)

    def _fetch(self) -> None:
        from huggingface_hub import snapshot_download

        logger.info("Downloading %s (~218 MB, first run only)", model_cache.GEMMA_REPOSITORY)
        snapshot_download(
            model_cache.GEMMA_REPOSITORY,
            revision=model_cache.GEMMA_REVISION,
            allow_patterns=list(model_cache.GEMMA_FILES),
            local_dir=self._model_dir,  # real files, not symlinks into a blob store
        )

    def _report_other_versions(self) -> None:
        """Say what weights this version does not use cost, once, and delete none of them.

        Weights another checkout is using, or one pinned deliberately, are not this
        process's to remove; saying how much room they take is. Two kinds qualify: a folder
        for another revision, and - because one revision publishes several graphs - a file
        sitting in *this* folder that the manifest does not name. The second is what an
        upgrade from the int8 graph leaves behind, and looking only at other folders would
        miss all 310 MB of it.
        """
        others: dict[Path, int] = {}
        for path in sorted(
            model_cache.model_cache_root(self._cache_dir).glob(f"{model_cache._GEMMA_DIR_PREFIX}*")
        ):
            if path == self._model_dir or not path.is_dir():
                continue
            with contextlib.suppress(OSError):
                others[path] = sum(
                    entry.stat().st_size for entry in path.rglob("*") if entry.is_file()
                )
        with contextlib.suppress(OSError):
            wanted = {self._model_dir / name for name in model_cache.GEMMA_FILES}
            for entry in sorted(self._model_dir.rglob("*")):
                stamp = entry.name.startswith(model_cache._VERIFIED_STAMP)
                if entry.is_file() and entry not in wanted and not stamp:
                    others[entry] = entry.stat().st_size
        if others:
            logger.info(
                "The model cache also holds %d copy/copies of %s this version does not use "
                "(%.0f MB in total): %s. Nothing is deleted automatically; remove them to "
                "reclaim the space.",
                len(others),
                model_cache.GEMMA_REPOSITORY,
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
        return FastEmbedEmbedder(model_cache.BGE_SMALL_MODEL_NAME, cache_dir=cache_dir)
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


def _to_floats(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values]
