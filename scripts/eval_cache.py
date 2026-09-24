"""Reuse one evaluation index across runs, and keep two evaluations off the same machine.

Building the index costs about a minute on the 54-section corpus the gate scores, and
~25 minutes and 1.6 GB of resident memory on the v2 corpus; the queries it answers take
seconds. This module keeps the built database under
``$XDG_CACHE_HOME/markdown-memory/eval/`` keyed on everything that can change a stored
vector, so a repeat run of the gate is index-free.

A cache key alone is not trusted. A key is a claim about the inputs; the checks in
``validate`` are evidence about the artifact, and a stale index that scores well is worse
than no cache at all. Three checks run before a cached database is used:

* the **parse fingerprint** - ordered ``(heading path, unit ordinal, text hash)`` over the
  whole corpus, recomputed by parsing (no embedding, a second or two). This is what
  catches a chunking change that happens to preserve section counts, which is exactly the
  shape of the parser redesign this repository already went through once;
* ``Database.integrity_problems()`` - schema, orphans, vector/row agreement;
* a **vector probe**: a passage the index already holds is embedded again and the index
  must return that same passage as its nearest neighbour. A query cannot do this job -
  search fuses a keyword ranking with the vector ranking, so an identifier query returns
  the right section even when every vector in the database came from a different model.

``lock`` serialises runs. The gate reports median and p95 latency, and a second
evaluation (or an indexing job) sharing the CPU moves those numbers by more than the
effects being measured - this repository has retracted latency deltas for exactly that
reason. The lock is advisory, ``LOCK_EX | LOCK_NB``, and a blocked run fails loudly
rather than quietly producing numbers that cannot be compared with anything.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import stat
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from markdown_memory.db import SCHEMA_VERSION, Database
from markdown_memory.discovery import MAX_FILE_BYTES, iter_markdown_files
from markdown_memory.embedders import (
    DEFAULT_EMBEDDER,
    GEMMA_DIMENSION,
    GEMMA_DOCUMENT_PROMPT,
    GEMMA_MAX_TOKENS,
    GEMMA_QUERY_PROMPT,
    Embedder,
)
from markdown_memory.model_cache import (
    GEMMA_MODEL_FILE,
    GEMMA_REVISION,
    fastembed_model_dir,
    gemma_model_dir,
)
from markdown_memory.parser import (
    DEFAULT_MAX_SECTION_CHARS,
    MAX_UNIT_CHARS,
    MAX_UNITS_PER_SECTION,
    MarkdownParser,
)

SOURCE = Path(__file__).parent.parent / "src" / "markdown_memory"
# Modules whose source decides what goes into the index. `search.py` is deliberately
# absent: ranking changes are what the gate exists to measure, and re-indexing for one
# would make every comparison cost 25 minutes.
INDEX_SOURCES = (
    "parser.py",
    "indexer.py",
    "embedders.py",
    "model_cache.py",
    "discovery.py",
    "db.py",
    "models.py",
)
CACHE_VERSION = 1  # bump when the layout of the cache directory itself changes


def cache_root() -> Path:
    home = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(home).expanduser() if home else Path.home() / ".cache"
    return base / "markdown-memory" / "eval"


# Cosine distance between a passage and its own stored vector. Measured on the v1
# corpus with the quantized EmbeddingGemma build: the passage itself comes back at
# -6e-8 (the float32 round trip through sqlite-vec, and batched indexing versus a
# single re-embedding, cost nothing measurable), while the next-nearest passage sits at
# 0.384. The threshold is in that gap, far from both ends.
VECTOR_PROBE_TOLERANCE = 1e-3


@dataclass(slots=True, frozen=True)
class Probe:
    """One passage of the corpus, used to interrogate the stored vectors.

    ``text`` is the passage as the database stores it; ``embedding_text`` is what the
    indexer embedded, which carries the breadcrumb. Both come from ``SectionDraft`` so
    this module never restates how a passage is composed.
    """

    text: str
    embedding_text: str


@dataclass(slots=True, frozen=True)
class CacheKey:
    """Identity of a built index: corpus, chunking, embedder and storage format."""

    corpus: str
    chunking: str
    embedder: str
    storage: str

    @property
    def digest(self) -> str:
        blob = json.dumps(
            {
                "version": CACHE_VERSION,
                "corpus": self.corpus,
                "chunking": self.chunking,
                "embedder": self.embedder,
                "storage": self.storage,
            },
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def corpus_digest(corpus: Path, exclude: Sequence[str] = ()) -> str:
    """Hash the corpus by content, not by manifest.

    A manifest records what was *fetched*; this records what will be *indexed*, so an
    edited or half-written vendored file cannot silently reuse an index built from the
    original.
    """
    digest = hashlib.sha256()
    for path in iter_markdown_files(corpus, exclude=exclude):
        digest.update(path.relative_to(corpus).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_digest(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    """Hash a corpus file's bytes, or stand in for a file the indexer will refuse.

    ``iter_markdown_files`` yields whatever is named ``*.md``, and the indexer decides
    what it will accept - a named pipe is reported as "not a regular file", an oversized
    file is reported too. Reading either one here would hang the run or hash megabytes
    the index will never contain, so they are identified rather than read.
    """
    try:
        info = path.stat()
    except OSError as error:
        return f"unstattable:{error.errno}"
    if not stat.S_ISREG(info.st_mode):
        return f"not-a-regular-file:{stat.S_IFMT(info.st_mode)}"
    if info.st_size > MAX_FILE_BYTES:
        return f"too-large:{info.st_size}"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_indexable(path: Path) -> bool:
    """Whether the indexer would read this file at all (see ``_file_digest``)."""
    try:
        info = path.stat()
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and info.st_size <= MAX_FILE_BYTES


def _source_digest(names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(hashlib.sha256((SOURCE / name).read_bytes()).digest())
    return digest.hexdigest()


def _model_identity(model_cache_dir: Path | None, embedder: str) -> str:
    """Identify the weights on disk, not just the revision they were requested from.

    The constants say which revision was *asked for*; this says which bytes are there.
    A re-download, a hand-edited tokenizer or a half-written file all change the vectors
    while every constant stays put. Size and mtime, not content: the weights are ~330 MB
    and hashing them on every run would cost more than the check is worth, while the
    vector probes catch a change that somehow survives both.

    Only the selected model's own directory is walked. Scanning the whole cache would
    make installing an unrelated model discard this one's index.
    """
    if model_cache_dir is None:
        return embedder
    # Each preset's folder is asked of the code that creates it. Guessing bge-small's from
    # its model name gave `models--BAAI--bge-small-en-v1.5`, while fastembed downloads its
    # own re-export into `models--qdrant--bge-small-en-v1.5-onnx-q`, so this part of the
    # key was an empty string for that preset however the weights changed.
    if embedder == DEFAULT_EMBEDDER:
        directories = [gemma_model_dir(model_cache_dir)]
    else:
        fastembed_directory = fastembed_model_dir(model_cache_dir)
        directories = [fastembed_directory] if fastembed_directory else []
    entries: list[str] = []
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            try:
                info = path.stat()
            except OSError:
                continue
            if stat.S_ISREG(info.st_mode):
                entries.append(
                    f"{path.relative_to(directory).as_posix()}:{info.st_size}:{info.st_mtime_ns}"
                )
    return json.dumps(sorted(entries))


def build_key(
    corpus: Path,
    embedder: str,
    *,
    exclude: Sequence[str] = (),
    model_cache_dir: Path | None = None,
) -> CacheKey:
    """Everything that can change a stored vector, in four independent digests."""
    chunking = json.dumps(
        {
            "max_section_chars": DEFAULT_MAX_SECTION_CHARS,
            "max_units_per_section": MAX_UNITS_PER_SECTION,
            "max_unit_chars": MAX_UNIT_CHARS,
            "source": _source_digest(INDEX_SOURCES),
        },
        sort_keys=True,
    )
    model = {
        "name": embedder,
        "revision": GEMMA_REVISION,
        "file": GEMMA_MODEL_FILE,
        "dimension": GEMMA_DIMENSION,
        "query_prompt": GEMMA_QUERY_PROMPT,
        "document_prompt": GEMMA_DOCUMENT_PROMPT,
        "max_tokens": GEMMA_MAX_TOKENS,
        "artifacts": _model_identity(model_cache_dir, embedder),
        # Read at index time, and a thread count can change floating-point summation
        # order. Cheap to key on; impossible to notice if it is left out.
        "threads": os.environ.get("MARKDOWN_MEMORY_THREADS", ""),
        "exclude": list(exclude),
    }
    storage = {
        "schema": SCHEMA_VERSION,
        "sqlite": sqlite3.sqlite_version,
        "vec": _vec_version(),
    }
    return CacheKey(
        corpus=corpus_digest(corpus, exclude),
        chunking=hashlib.sha256(chunking.encode("utf-8")).hexdigest(),
        embedder=hashlib.sha256(json.dumps(model, sort_keys=True).encode("utf-8")).hexdigest(),
        storage=hashlib.sha256(json.dumps(storage, sort_keys=True).encode("utf-8")).hexdigest(),
    )


def _vec_version() -> str:
    """The sqlite-vec build in use; its vector format is part of the storage identity."""
    connection = sqlite3.connect(":memory:")
    try:
        import sqlite_vec

        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        row = connection.execute("SELECT vec_version()").fetchone()
        return str(row[0]) if row else "unknown"
    except Exception:  # pragma: no cover - a missing extension fails later, loudly
        return "unknown"
    finally:
        connection.close()


def parse_fingerprint(corpus: Path, exclude: Sequence[str] = ()) -> str:
    """Fingerprint the chunking itself: every section path, every unit, in order.

    Parsing the corpus costs a second or two and needs no model, which is what makes this
    affordable on every cache hit. Counts are not enough - a change that moves text
    between two units keeps both the section count and the unit count.

    Everything indexing consumes is hashed verbatim: the section content (which is what
    FTS5 indexes) and each passage exactly as it is embedded, breadcrumb included. An
    earlier version normalised whitespace first, which hid a change from ``a  b`` to
    ``a b`` - two different vectors behind one fingerprint.
    """
    parser = MarkdownParser()
    digest = hashlib.sha256()
    for path in iter_markdown_files(corpus, exclude=exclude):
        digest.update(path.relative_to(corpus).as_posix().encode("utf-8"))
        if not _is_indexable(path):
            digest.update(b"\x00unreadable")
            continue
        parsed = parser.parse(
            path.read_text(encoding="utf-8", errors="replace"), fallback_title=path.stem
        )
        for section in parsed.sections:
            digest.update(b"\x1e")
            digest.update(section.heading_path.encode("utf-8"))
            digest.update(f"{section.start_line}:{section.end_line}".encode("ascii"))
            digest.update(hashlib.sha256(section.content.encode("utf-8")).digest())
            for ordinal, unit in enumerate(section.unit_texts):
                digest.update(b"\x1f")
                digest.update(str(ordinal).encode("ascii"))
                digest.update(hashlib.sha256(unit.encode("utf-8")).digest())
    return digest.hexdigest()


def probe_passages(corpus: Path, count: int = 3, exclude: Sequence[str] = ()) -> tuple[Probe, ...]:
    """``count`` passages spread evenly through the corpus, in walk order.

    One passage only tells you about one passage. A build that went wrong partway - a
    file that failed to embed, a model swapped mid-run - leaves the first passage intact
    and everything after it wrong, so the probes are taken from the start, the middle and
    the end of the corpus.
    """
    parser = MarkdownParser()
    found: list[Probe] = []
    for path in iter_markdown_files(corpus, exclude=exclude):
        if not _is_indexable(path):
            continue
        parsed = parser.parse(
            path.read_text(encoding="utf-8", errors="replace"), fallback_title=path.stem
        )
        for section in parsed.sections:
            for unit, embedding_text in zip(section.units, section.unit_texts, strict=True):
                if unit.strip():
                    found.append(Probe(text=unit, embedding_text=embedding_text))
    if not found or count < 1:
        return ()
    step = max(1, len(found) // count)
    spread = found[::step][:count]
    return tuple(spread) if spread else (found[0],)


def check_vectors(database: Database, embedder: Embedder, probes: Sequence[Probe]) -> None:
    """Raise ``StaleCacheError`` unless the index recognises a passage it already holds.

    This is the only check that reads the vectors. Re-embedding the passage with the
    embedder about to be scored and asking for its nearest neighbour compares the stored
    vector against a freshly computed one; a database written by another model, or with
    different prompts, fails it immediately.
    """
    if not probes:
        raise StaleCacheError("no passage to probe the index with")
    for vector, probe in zip(
        embedder.embed_documents([probe.embedding_text for probe in probes]), probes, strict=True
    ):
        nearest = database.unit_search(vector, 1)
        if not nearest:
            raise StaleCacheError("the index holds no passage vectors")
        _, distance, passage = nearest[0]
        if distance > VECTOR_PROBE_TOLERANCE or passage != probe.text:
            raise StaleCacheError(
                f"stored vectors are not {embedder.model_name}'s: re-embedding a passage "
                f"put it at cosine distance {distance:.4f} from its nearest stored vector"
            )


class StaleCacheError(Exception):
    """A cached index exists under the key but does not answer for itself."""


def validate(db_path: Path, expected_fingerprint: str, key: CacheKey) -> None:
    """Raise ``StaleCacheError`` unless the stored index was built from these inputs.

    Metadata only, so it runs before the embedding model is loaded and a stale index
    costs nothing to detect. ``check_integrity`` and ``check_vectors`` run afterwards,
    against the open database.

    The recorded key is compared field by field rather than trusted through the directory
    name: the name is a 64-bit truncation, and a cache directory copied between machines
    or keys would otherwise be read as its neighbour's.
    """
    if not db_path.exists():
        raise StaleCacheError("no database")
    meta = _read_meta(db_path)
    stored = meta.get("parse_fingerprint")
    if not isinstance(stored, str):
        raise StaleCacheError("cache metadata has no parse fingerprint")
    if stored != expected_fingerprint:
        raise StaleCacheError(
            f"parse fingerprint {stored[:12]}... != {expected_fingerprint[:12]}..."
        )
    for field in ("corpus", "chunking", "embedder", "storage"):
        if meta.get(field) != getattr(key, field):
            raise StaleCacheError(f"cached index was built with a different {field}")


def check_integrity(database: Database) -> None:
    """Raise ``StaleCacheError`` if the cached database disagrees with itself."""
    problems = database.integrity_problems()
    if problems:
        raise StaleCacheError("; ".join(problems))


def _meta_path(db_path: Path) -> Path:
    return db_path.with_suffix(".meta.json")


def _read_meta(db_path: Path) -> dict[str, object]:
    try:
        loaded = json.loads(_meta_path(db_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise StaleCacheError(f"unreadable cache metadata: {error}") from error
    if not isinstance(loaded, dict):
        raise StaleCacheError("cache metadata is not an object")
    return loaded


def record(db_path: Path, key: CacheKey, fingerprint: str) -> None:
    """Mark a freshly built index as complete.

    This file is the commit marker: it is written last, and written atomically, so a run
    killed at any point leaves either no claim or a complete one - never a half-written
    metadata file that the next run would have to guess about.
    """
    payload = (
        json.dumps(
            {
                "version": CACHE_VERSION,
                "digest": key.digest,
                "parse_fingerprint": fingerprint,
                "corpus": key.corpus,
                "chunking": key.chunking,
                "embedder": key.embedder,
                "storage": key.storage,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    target = _meta_path(db_path)
    temporary = target.with_suffix(".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def discard(db_path: Path) -> None:
    """Remove a cached index and everything SQLite keeps beside it."""
    _meta_path(db_path).unlink(missing_ok=True)
    _meta_path(db_path).with_suffix(".tmp").unlink(missing_ok=True)
    for suffix in ("", "-wal", "-shm"):
        Path(str(db_path) + suffix).unlink(missing_ok=True)


def prune(root: Path, keep: str) -> None:
    """Drop cached indexes for other keys; each costs ~40 MB and only one is ever current."""
    if not root.is_dir():
        return
    for entry in root.iterdir():
        if entry.is_dir() and entry.name != keep:
            shutil.rmtree(entry, ignore_errors=True)


def confirm_stable(corpus: Path, fingerprint: str, exclude: Sequence[str] = ()) -> None:
    """Raise ``StaleCacheError`` if the corpus moved while it was being indexed.

    The corpus is read to build the key, again to fingerprint it and again to index it.
    A file edited between those reads produces a database and a key describing different
    corpora - and once the edit is reverted, that database validates cleanly forever.
    """
    if parse_fingerprint(corpus, exclude) != fingerprint:
        raise StaleCacheError(
            "the corpus changed while it was being indexed; this build describes no "
            "single state of it"
        )


class BusyError(Exception):
    """Another evaluation holds the lock; its latency numbers and ours would both be wrong."""


@contextmanager
def lock(root: Path) -> Iterator[None]:
    """Hold an exclusive advisory lock for the whole run, or refuse to start."""
    root.mkdir(parents=True, exist_ok=True)
    handle = os.open(root / "eval.lock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise BusyError(
                "another evaluation is running (lock held on "
                f"{root / 'eval.lock'}); latency measured against a busy CPU is not "
                "comparable with the baseline"
            ) from error
        os.truncate(handle, 0)
        os.write(handle, f"{os.getpid()}\n".encode("ascii"))
        yield
    finally:
        os.close(handle)
