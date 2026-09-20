"""Incremental indexing: directory scan, SHA-256 change detection, embedding sync.

Two local ONNX embedders are available. ``EmbeddingGemmaEmbedder`` (the default) is what
retrieval quality was tuned on; ``FastEmbedEmbedder`` (bge-small) is the light option:
a tenth of the size and ~25x faster, at a clear cost in recall on paraphrased queries.
"""

from __future__ import annotations

import hashlib
import logging
import os
import stat
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from markdown_memory.db import DEFAULT_EMBEDDING_DIM, Database
from markdown_memory.exceptions import (
    EmbeddingError,
    IndexingError,
    MarkdownMemoryError,
    ModelLoadError,
)
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
GEMMA_FILES = (GEMMA_MODEL_FILE, GEMMA_MODEL_FILE + "_data", "tokenizer.json")
GEMMA_DIMENSION = 768
GEMMA_MAX_TOKENS = 512
# Prompts from the EmbeddingGemma model card; the model is trained to expect them.
GEMMA_QUERY_PROMPT = "task: search result | query: "
GEMMA_DOCUMENT_PROMPT = "title: none | text: "
_GEMMA_BATCH_SIZE = 4  # measured: faster than 1, 2 and 8 on CPU; memory is the same
MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})
MAX_FILE_BYTES = 10 * 1024 * 1024
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

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

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
                    )
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
        base = cache_dir or Path.home() / ".cache" / "markdown-memory" / "models"
        self._model_dir = base / "embeddinggemma-300m-onnx"
        self._lock = threading.Lock()
        self._session: _OrtSession | None = None
        self._tokenizer: Tokenizer | None = None

    @property
    def model_name(self) -> str:
        return f"{GEMMA_REPOSITORY}@{GEMMA_REVISION[:12]}/{GEMMA_MODEL_FILE}"

    @property
    def dimension(self) -> int:
        return GEMMA_DIMENSION

    def warm_up(self) -> None:
        """Download (first run only) and load the model now instead of on first use."""
        self._load()

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
            if self._session is None or self._tokenizer is None:
                started = time.perf_counter()
                try:
                    import onnxruntime
                    from tokenizers import Tokenizer

                    self._download()
                    tokenizer = Tokenizer.from_file(str(self._model_dir / "tokenizer.json"))
                    tokenizer.enable_truncation(max_length=GEMMA_MAX_TOKENS)
                    tokenizer.enable_padding()
                    self._session = onnxruntime.InferenceSession(
                        str(self._model_dir / GEMMA_MODEL_FILE),
                        providers=["CPUExecutionProvider"],
                    )
                    self._tokenizer = tokenizer
                except Exception as exc:
                    raise ModelLoadError(
                        f"Cannot load embedding model {GEMMA_REPOSITORY}: {exc}"
                    ) from exc
                logger.info(
                    "Loaded embedding model %s in %.2fs",
                    GEMMA_REPOSITORY,
                    time.perf_counter() - started,
                )
            return self._session, self._tokenizer

    def _download(self) -> None:
        if all((self._model_dir / name).is_file() for name in GEMMA_FILES):
            return
        from huggingface_hub import snapshot_download

        logger.info("Downloading %s (~330 MB, first run only)", GEMMA_REPOSITORY)
        snapshot_download(
            GEMMA_REPOSITORY,
            revision=GEMMA_REVISION,
            allow_patterns=list(GEMMA_FILES),
            local_dir=self._model_dir,  # real files, not symlinks into a blob store
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


def _to_floats(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values]


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def iter_markdown_files(
    directory: Path, on_error: Callable[[OSError], None] | None = None
) -> Iterator[Path]:
    """Yield Markdown files beneath ``directory`` in a stable order, pruning vendored trees.

    ``on_error`` receives the ``OSError`` for every sub-directory that cannot be listed
    (``os.walk`` would otherwise skip it silently).
    """
    for root, dirnames, filenames in os.walk(directory, followlinks=False, onerror=on_error):
        dirnames[:] = sorted(name for name in dirnames if name not in _SKIPPED_DIRECTORIES)
        for filename in sorted(filenames):
            if Path(filename).suffix.lower() in MARKDOWN_SUFFIXES:
                yield Path(root, filename)


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
    ) -> None:
        if embedder.dimension != db.embedding_dim:
            raise IndexingError(
                f"Embedder produces {embedder.dimension}-dimensional vectors but the "
                f"database stores {db.embedding_dim}"
            )
        self._db = db
        self._embedder = embedder
        self._parser = parser or MarkdownParser()
        self._run_lock = threading.Lock()

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

        with self._run_lock:
            started = time.perf_counter()
            # Migrations that emptied the index (new format, new vector size) left a notice.
            notes: list[str] = self._db.take_notices()
            previous_model = self._db.get_meta(_MODEL_META_KEY)
            if previous_model not in {None, self._embedder.model_name}:
                # Vectors from different models are not comparable, and they share one
                # vector table: every document has to go, not only those under `root`.
                discarded = self._db.clear()
                if discarded:  # 0 when a size change already emptied (and announced) it
                    notes.append(
                        f"Embedding model changed ({previous_model} -> "
                        f"{self._embedder.model_name}): discarded all {discarded} previously "
                        "indexed documents from every directory. Re-run index_directory for "
                        "any other documentation root."
                    )
                    logger.warning(notes[-1])
            self._db.set_meta(_MODEL_META_KEY, self._embedder.model_name)
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

            for path in iter_markdown_files(root, record_unreadable):
                file_path = str(path)
                seen.add(file_path)
                try:
                    counts = self._index_file(path, known_hashes.get(file_path))
                except ModelLoadError:
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

            purged = self._db.delete_documents(self._vanished(root, known_hashes, seen, unreadable))
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
                notes=tuple(notes),
            )
        logger.info(report.summary())
        return report

    @staticmethod
    def _vanished(
        root: Path, known: dict[str, str], seen: set[str], unreadable: Sequence[str]
    ) -> list[str]:
        """Known documents that this walk *would* have found had they still existed.

        A document is only purged when its absence is evidence of deletion. It is kept
        when the walk could not have reached it: it lives under a directory that could
        not be listed, or under a pruned tree (``node_modules`` ...) that was indexed
        explicitly by pointing ``index_directory`` inside it.
        """
        blocked = tuple(location.rstrip(os.sep) + os.sep for location in unreadable)
        vanished: list[str] = []
        for file_path in sorted(set(known) - seen):
            if blocked and file_path.startswith(blocked):
                continue
            relative = os.path.relpath(file_path, root)
            if not _is_walkable(relative.split(os.sep)[:-1]):
                continue
            vanished.append(file_path)
        return vanished

    def _index_file(self, path: Path, known_hash: str | None) -> tuple[int, int] | None:
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
        if content_hash == known_hash:
            return None
        parsed = self._parser.parse(
            data.decode("utf-8", errors="replace"), fallback_title=path.stem
        )
        # One embedding call per file: each section with a body contributes its own text
        # followed by its passages; heading-only sections contribute nothing.
        texts: list[str] = []
        for section in parsed.sections:
            if section.units:
                texts.append(section.embedding_text)
                texts.extend(section.unit_texts)
        embeddings = self._embedder.embed_documents(texts)
        if len(embeddings) != len(texts):
            raise EmbeddingError(f"Got {len(embeddings)} vectors for {len(texts)} texts")
        embedded = iter(embeddings)
        vectors = [
            SectionVectors(
                section=next(embedded) if section.units else None,
                units=tuple(next(embedded) for _ in section.units),
            )
            for section in parsed.sections
        ]
        self._db.replace_document(
            file_path=file_path,
            title=parsed.title,
            content_hash=content_hash,
            last_modified=int(info.st_mtime),
            sections=parsed.sections,
            vectors=vectors,
        )
        return len(parsed.sections), sum(len(section.units) for section in parsed.sections)
