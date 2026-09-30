"""SQLite storage: connection factory, migrations, and the section repository.

The store combines three indexes over the same ``sections`` rows:

* ``sections``      - canonical relational data (cascade-deleted with its document)
* ``sections_fts``  - FTS5 external-content index (BM25 keyword search)
* ``sections_vec``  - sqlite-vec ``vec0`` index, one vector per section with a body
* ``units`` / ``units_vec`` - the section's passages (paragraph, list item, table row,
  code block) and one vector for each; a section is ranked by its best passage

Triggers keep both virtual tables in lock-step with ``sections``, including rows
removed by ``ON DELETE CASCADE``, so callers only ever write to ``sections``.

Connections are per-thread (MCP tool handlers run in worker threads and hybrid
search queries both indexes concurrently); WAL mode lets readers proceed while
an indexing transaction is open.
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import struct
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Self

import sqlite_vec

from markdown_memory.exceptions import DatabaseError
from markdown_memory.models import (
    Document,
    DocumentSummary,
    FileFailure,
    IndexStatus,
    Section,
    SectionDraft,
    SectionVectors,
)

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_DIM = 384
SCHEMA_VERSION = 6

#: How a section's vector is built. 1 embedded the whole section text, truncated at the
#: model's token limit; 2 is the mean of the section's passage vectors. Stored per document
#: so that a document written under the old scheme re-indexes itself and one written under
#: the new one is left alone - a format change repairs a tree file by file, and resumes
#: where it stopped if it is interrupted.
VECTOR_FORMAT = 2
#: Meta key holding the weights revision the stored vectors were built from. It lives here
#: because `clear()` has to forget it in the same transaction that deletes them.
WEIGHTS_META_KEY = "embedding_weights_revision"
#: Set when the weights behind an unchanged model name changed under an existing index.
#: It holds the sentence an agent is shown, because the index is then answering from
#: vectors one model built while the next query would be embedded by another.
WEIGHTS_MISMATCH_KEY = "embedding_weights_mismatch"
#: What `WEIGHTS_META_KEY` holds while an index is being re-embedded by other weights. It
#: equals no revision, so every search - under the old weights or the new - finds it
#: differs from its own and ranks by keyword alone until a run re-certifies the index. A
#: revision is a hash, sometimes with a graph path after it; this can be neither.
WEIGHTS_REVOKED = "(revoked: being re-embedded)"
_LEGACY_VECTORS = 1

_SECTION_ID_META_KEY = "next_section_id"
_GENERATION_META_KEY = "index_generation"
_BUSY_TIMEOUT_MS = 10_000
_WAL_ATTEMPTS = 40
_WAL_RETRY_SECONDS = 0.05
_SQL_VARIABLE_BATCH = 500

_SECTION_COLUMNS = (
    "id, doc_id, heading_title, heading_level, heading_path, content, "
    "start_line, end_line, part_index"
)


def _schema_v1(embedding_dim: int) -> tuple[str, ...]:
    return (
        """
        CREATE TABLE meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE documents (
            id            INTEGER PRIMARY KEY,
            file_path     TEXT NOT NULL UNIQUE,
            title         TEXT NOT NULL,
            content_hash  TEXT NOT NULL,
            last_modified INTEGER NOT NULL
        )
        """,
        """
        CREATE TABLE sections (
            id            INTEGER PRIMARY KEY,
            doc_id        INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            heading_title TEXT NOT NULL,
            heading_level INTEGER NOT NULL,
            heading_path  TEXT NOT NULL,
            content       TEXT NOT NULL,
            start_line    INTEGER NOT NULL,
            end_line      INTEGER NOT NULL,
            part_index    INTEGER NOT NULL DEFAULT 0
        )
        """,
        "CREATE INDEX idx_sections_doc ON sections(doc_id, id)",
        """
        CREATE VIRTUAL TABLE sections_fts USING fts5(
            heading_title,
            heading_path,
            content,
            content='sections',
            content_rowid='id',
            tokenize='porter unicode61 remove_diacritics 2'
        )
        """,
        # Headings are short and highly descriptive: weight them above body text.
        "INSERT INTO sections_fts(sections_fts, rank) VALUES ('rank', 'bm25(5.0, 3.0, 1.0)')",
        _vector_tables(embedding_dim)[0],
        """
        CREATE TRIGGER sections_after_insert AFTER INSERT ON sections BEGIN
            INSERT INTO sections_fts(rowid, heading_title, heading_path, content)
            VALUES (new.id, new.heading_title, new.heading_path, new.content);
        END
        """,
        """
        CREATE TRIGGER sections_after_delete AFTER DELETE ON sections BEGIN
            INSERT INTO sections_fts(sections_fts, rowid, heading_title, heading_path, content)
            VALUES ('delete', old.id, old.heading_title, old.heading_path, old.content);
            DELETE FROM sections_vec WHERE section_id = old.id;
        END
        """,
        """
        CREATE TRIGGER sections_after_update AFTER UPDATE ON sections BEGIN
            INSERT INTO sections_fts(sections_fts, rowid, heading_title, heading_path, content)
            VALUES ('delete', old.id, old.heading_title, old.heading_path, old.content);
            INSERT INTO sections_fts(rowid, heading_title, heading_path, content)
            VALUES (new.id, new.heading_title, new.heading_path, new.content);
        END
        """,
    )


def _vector_tables(embedding_dim: int) -> tuple[str, ...]:
    return (
        f"""
        CREATE VIRTUAL TABLE sections_vec USING vec0(
            section_id INTEGER PRIMARY KEY,
            embedding FLOAT[{embedding_dim}] distance_metric=cosine
        )
        """,
        f"""
        CREATE VIRTUAL TABLE units_vec USING vec0(
            unit_id INTEGER PRIMARY KEY,
            embedding FLOAT[{embedding_dim}] distance_metric=cosine
        )
        """,
    )


def _schema_v2(embedding_dim: int) -> tuple[str, ...]:
    """Passage-level vectors: ``units`` rows cascade with their section."""
    return (
        """
        CREATE TABLE units (
            id         INTEGER PRIMARY KEY,
            section_id INTEGER NOT NULL REFERENCES sections(id) ON DELETE CASCADE,
            ordinal    INTEGER NOT NULL,
            content    TEXT NOT NULL
        )
        """,
        "CREATE INDEX idx_units_section ON units(section_id, ordinal)",
        _vector_tables(embedding_dim)[1],
        """
        CREATE TRIGGER units_after_delete AFTER DELETE ON units BEGIN
            DELETE FROM units_vec WHERE unit_id = old.id;
        END
        """,
    )


def _schema_v4() -> tuple[str, ...]:
    """What is known to be wrong with the index, and whether anything vouches for it.

    Two questions, kept apart because they have different answers. `index_failures` says
    which paths could not be read, one row per path, written by whichever scan last looked
    at that path. `index_coverage` says whether a full walk of a root ever finished without
    failures - the only thing that can distinguish "every file was seen" from "only these
    files were seen", which no amount of per-file state can tell you: a run killed on its
    tenth file leaves the other 990 with no rows at all.

    Earlier attempts hashed the root (so containment could not be asked), then kept a
    wall-clock guard (wrong in both orderings), then a crash marker that masked the file
    rows. Those are gone. What makes this sound instead is the scan lock: one scan at a
    time, so the run that just walked a tree is entitled to speak for it.
    """
    return (
        """
        CREATE TABLE index_failures (
            file_path TEXT PRIMARY KEY,
            message   TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE index_coverage (
            root     TEXT PRIMARY KEY,
            verified INTEGER NOT NULL
        )
        """,
    )


def serialize_embedding(embedding: Sequence[float]) -> bytes:
    """Pack a vector into the little-endian float32 blob format sqlite-vec expects."""
    return struct.pack(f"<{len(embedding)}f", *embedding)


def _is_usable_vector(embedding: Sequence[float]) -> bool:
    """Cosine distance is undefined (NaN) for non-finite or zero-length vectors."""
    return all(math.isfinite(value) for value in embedding) and any(embedding)


def _bump_generation(conn: sqlite3.Connection) -> None:
    """Mark that everything indexed before this moment is gone."""
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, '1') "
        "ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)",
        (_GENERATION_META_KEY,),
    )


def _forget_weights_without_vectors(conn: sqlite3.Connection) -> None:
    """Drop the weights metadata once the vectors it describes are gone.

    Which weights produced the vectors is a fact about the vectors, so it belongs to the
    same transaction that removes them - a purge of the last document, a rebuild for a new
    vector size, a discard for a changed model. Kept behind, it would have the next run
    compare a new model against the revision of a model whose output no longer exists, and
    have search rank on keywords alone over an index with nothing wrong with it.

    Heading-only documents hold no vectors, so the test is the vectors themselves rather
    than the document rows above them.
    """
    if conn.execute("SELECT 1 FROM units_vec LIMIT 1").fetchone() is not None:
        return
    conn.execute("DELETE FROM meta WHERE key = ?", (WEIGHTS_META_KEY,))
    conn.execute("DELETE FROM meta WHERE key = ?", (WEIGHTS_MISMATCH_KEY,))


def _revoke_weights(conn: sqlite3.Connection, message: str) -> None:
    """Mark the index as being re-embedded, and say why, in the caller's transaction.

    One transaction for both: a revocation with no message would leave `index_status`
    calling the index healthy while search refuses its vectors, and a message with no
    revocation would let search rank the mixture the message warns about.
    """
    conn.executemany(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        ((WEIGHTS_META_KEY, WEIGHTS_REVOKED), (WEIGHTS_MISMATCH_KEY, message)),
    )


def _directory_prefix(directory: str) -> str:
    return directory if directory.endswith(os.sep) else directory + os.sep


class Database:
    """Thread-safe SQLite client and repository for documents and sections.

    Use as a context manager to guarantee every per-thread connection is closed::

        with Database(path) as db:
            with db.transaction() as conn:
                ...
    """

    def __init__(self, path: str | Path, *, embedding_dim: int = DEFAULT_EMBEDDING_DIM) -> None:
        if str(path) == ":memory:" or str(path).startswith("file::memory:"):
            raise DatabaseError(
                "In-memory databases are not supported: connections are per-thread "
                "and would each see an empty database. Use a file path."
            )
        if embedding_dim <= 0:
            raise DatabaseError(f"embedding_dim must be positive, got {embedding_dim}")
        self._path = Path(path).expanduser()
        self._embedding_dim = embedding_dim
        self._local = threading.local()
        self._connections: list[tuple[threading.Thread, sqlite3.Connection]] = []
        self._lock = threading.Lock()
        self._closed = False
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise DatabaseError(f"Cannot create database directory {self._path.parent}") from exc
        self._migrate()

    # ------------------------------------------------------------------ lifecycle

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def close(self) -> None:
        """Close every connection opened by any thread. Idempotent."""
        with self._lock:
            self._closed = True
            connections, self._connections = self._connections, []
        for _, conn in connections:
            _close_quietly(conn)

    @property
    def open_connection_count(self) -> int:
        """Connections currently held open across all threads (diagnostics)."""
        with self._lock:
            return len(self._connections)

    def connection(self) -> sqlite3.Connection:
        """Return this thread's connection, opening and configuring it on first use.

        Tool handlers run on pooled worker threads that the runtime retires when idle,
        so opening a connection is also the moment the connections of threads that have
        since exited are closed; otherwise a long-lived server leaks file descriptors.
        """
        if self._closed:
            raise DatabaseError("Database is closed")
        existing: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if existing is not None:
            return existing
        conn = self._open()
        abandoned: list[sqlite3.Connection] = []
        with self._lock:
            if self._closed:
                conn.close()
                raise DatabaseError("Database is closed")
            alive: list[tuple[threading.Thread, sqlite3.Connection]] = []
            for thread, other in self._connections:
                if thread.is_alive():
                    alive.append((thread, other))
                else:
                    abandoned.append(other)
            alive.append((threading.current_thread(), conn))
            self._connections = alive
        self._local.conn = conn
        for other in abandoned:
            _close_quietly(other)
        return conn

    def _open(self) -> sqlite3.Connection:
        try:
            # isolation_level=None: autocommit; transactions are explicit (see transaction()).
            # check_same_thread=False only so close() may run from another thread;
            # each connection is otherwise used exclusively by the thread that opened it.
            conn = sqlite3.connect(
                self._path,
                isolation_level=None,
                check_same_thread=False,
                timeout=_BUSY_TIMEOUT_MS / 1000,
            )
        except sqlite3.Error as exc:
            raise DatabaseError(f"Cannot open database at {self._path}: {exc}") from exc
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except (sqlite3.Error, AttributeError) as exc:
            conn.close()
            raise DatabaseError(
                "Cannot load the sqlite-vec extension. This Python build's sqlite3 module "
                f"must support loadable extensions: {exc}"
            ) from exc
        try:
            conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            _enable_wal(conn)
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA foreign_keys = ON")
        except sqlite3.Error as exc:
            conn.close()
            raise DatabaseError(f"Cannot configure database connection: {exc}") from exc
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a write transaction: commit on success, roll back on any exception."""
        conn = self.connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.Error as exc:
            raise DatabaseError(f"Cannot begin transaction: {exc}") from exc
        try:
            yield conn
            conn.execute("COMMIT")
        except BaseException as exc:
            if conn.in_transaction:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:  # pragma: no cover - rollback is best effort
                    logger.exception("Rollback failed")
            if isinstance(exc, sqlite3.Error):
                raise DatabaseError(f"Transaction failed: {exc}") from exc
            raise

    @contextmanager
    def _reading(self) -> Iterator[sqlite3.Connection]:
        """Yield the connection for reads, translating driver errors."""
        conn = self.connection()
        try:
            yield conn
        except sqlite3.Error as exc:
            raise DatabaseError(f"Query failed: {exc}") from exc

    # ------------------------------------------------------------------ migrations

    def _migrate(self) -> None:
        conn = self.connection()
        try:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        except sqlite3.Error as exc:
            raise DatabaseError(f"Cannot read schema version: {exc}") from exc
        if version > SCHEMA_VERSION:
            raise DatabaseError(
                f"Database schema v{version} is newer than this build supports "
                f"(v{SCHEMA_VERSION}); upgrade markdown-memory."
            )
        if version < SCHEMA_VERSION:
            applied: list[int] = []
            with self.transaction() as tx:
                # Re-check under the write lock: another process starting at the same
                # moment may have migrated the schema while this one waited for it.
                current = int(tx.execute("PRAGMA user_version").fetchone()[0])
                if current < 1:
                    for statement in _schema_v1(self._embedding_dim):
                        tx.execute(statement)
                    tx.execute(
                        "INSERT INTO meta(key, value) VALUES ('embedding_dim', ?)",
                        (str(self._embedding_dim),),
                    )
                    applied.append(1)
                if current < 2:
                    stored = tx.execute(
                        "SELECT value FROM meta WHERE key = 'embedding_dim'"
                    ).fetchone()
                    for statement in _schema_v2(int(stored[0])):
                        tx.execute(statement)
                    # Sections indexed before v2 have no passages, and their unchanged
                    # SHA-256 would make every later run skip them. The index is a cache
                    # of the files: drop it so the next index_directory rebuilds it whole.
                    dropped = tx.execute("DELETE FROM documents").rowcount
                    if dropped:
                        _add_notice(
                            tx,
                            f"The index format changed (passage-level vectors): discarded all "
                            f"{dropped} previously indexed documents from every directory. "
                            "Re-run index_directory for each documentation root.",
                        )
                    applied.append(2)
                if current < 4:
                    for statement in _schema_v4():
                        tx.execute(statement)
                    tx.execute(
                        "ALTER TABLE documents "
                        f"ADD COLUMN vector_format INTEGER NOT NULL DEFAULT {_LEGACY_VECTORS}"
                    )
                    # v3 was never released; it exists only in working copies of the
                    # abandoned design. Its table is dropped rather than migrated.
                    tx.execute("DROP TABLE IF EXISTS index_problems")
                    tx.execute("DELETE FROM meta WHERE key LIKE 'incomplete:%'")
                    applied.append(4)
                if current < 5:
                    # Rows written before v5 recorded whole seconds, which cannot see an
                    # edit made in the same second as the scan that indexed it. They get
                    # NULL, which means "no modification time was recorded" - not a
                    # timestamp of any kind, so no real one can collide with it, the epoch
                    # included. The freshness check reads their bytes instead of trusting a
                    # time, which is the slow answer but never the wrong one, and the next
                    # index_directory writes a real value and the file stops paying it.
                    # Nothing is discarded for this: the vectors are still good, and a
                    # rebuild would cost half an hour of embedding to learn nothing.
                    tx.execute("ALTER TABLE documents ADD COLUMN mtime_ns INTEGER")
                    applied.append(5)
                if current < 6:
                    # Which weights embedded each document, so a change of model repairs the
                    # index file by file instead of discarding it. Copied from the index-wide
                    # revision where one is recorded, and that copy is exact: the revision is
                    # written only while no vector exists, every vector after it was checked
                    # against it, and it is forgotten only once no vector is left. Where none
                    # is recorded but vectors are, nobody can say what built them - they are
                    # quarantined now, in this transaction, rather than ranked against a query
                    # until some later run happens to notice.
                    tx.execute("ALTER TABLE documents ADD COLUMN weights_revision TEXT")
                    recorded = tx.execute(
                        "SELECT value FROM meta WHERE key = ?", (WEIGHTS_META_KEY,)
                    ).fetchone()
                    if recorded is not None:
                        tx.execute("UPDATE documents SET weights_revision = ?", (recorded[0],))
                    elif tx.execute("SELECT 1 FROM units_vec LIMIT 1").fetchone() is not None:
                        _revoke_weights(
                            tx,
                            "No record says which weights built this index's vectors, so "
                            "they are not compared with a query: only keyword ranking is used "
                            "until index_directory re-embeds them.",
                        )
                    applied.append(6)
                tx.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            if applied:
                logger.info("Applied schema migration(s) %s at %s", applied, self._path)
        self._seed_section_ids()
        stored_dim = self.get_meta("embedding_dim")
        if stored_dim is not None and int(stored_dim) != self._embedding_dim:
            self._rebuild_for_dimension(int(stored_dim))

    def _seed_section_ids(self) -> None:
        """Give a database from an earlier release its section-id high-water mark.

        Without it the mark would be derived from ``MAX(id)`` after rows had already been
        deleted, so a re-index - or a purge followed by one - would hand the freed ids
        straight back out, which is exactly what ``_next_section_id`` exists to prevent.
        """
        if self.get_meta(_SECTION_ID_META_KEY) is not None:
            return
        with self.transaction() as tx:
            if tx.execute("SELECT 1 FROM meta WHERE key = ?", (_SECTION_ID_META_KEY,)).fetchone():
                return  # another process seeded it while this one waited for the lock
            tx.execute(
                "INSERT INTO meta(key, value) "
                "SELECT ?, CAST(COALESCE(MAX(id), 0) + 1 AS TEXT) FROM sections",
                (_SECTION_ID_META_KEY,),
            )

    def _rebuild_for_dimension(self, stored_dim: int) -> None:
        """Re-create the vector tables for a model with a different output size.

        The index is a cache of the Markdown files: vectors of another dimensionality
        are useless, so everything is dropped and the next ``index_directory`` rebuilds it.
        """
        logger.warning(
            "Embedding size changed (%d -> %d): discarding the index at %s; re-run "
            "index_directory to rebuild it",
            stored_dim,
            self._embedding_dim,
            self._path,
        )
        with self.transaction() as tx:
            dropped = tx.execute("DELETE FROM documents").rowcount
            _forget_weights_without_vectors(tx)
            # Every root's documents are gone, including roots this process never looked
            # at; a certificate that survived would vouch for an empty tree.
            self.revoke_coverage(tx)
            if dropped:
                _add_notice(
                    tx,
                    f"The embedding size changed ({stored_dim} -> {self._embedding_dim} "
                    f"dimensions): discarded all {dropped} previously indexed documents from "
                    "every directory. Re-run index_directory for each documentation root.",
                )
            tx.execute("DROP TABLE sections_vec")
            tx.execute("DROP TABLE units_vec")
            for statement in _vector_tables(self._embedding_dim):
                tx.execute(statement)
            tx.execute(
                "UPDATE meta SET value = ? WHERE key = 'embedding_dim'",
                (str(self._embedding_dim),),
            )

    # ------------------------------------------------------------------ meta / pragmas

    def get_meta(self, key: str) -> str | None:
        with self._reading() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def set_meta(self, key: str, value: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def pending_notices(self) -> dict[str, str]:
        """Messages left by whatever discarded the index, oldest first, keyed for dismissal.

        They are persisted, not logged only, so that whoever next runs ``index_directory``
        - possibly another process, much later - is told why the index was empty. Reading
        does not clear them: a run that fails before it can report them must not eat them.
        """
        with self._reading() as conn:
            rows = conn.execute(
                "SELECT key, value FROM meta WHERE key LIKE 'notice:%' "
                "ORDER BY CAST(substr(key, 8) AS INTEGER)"
            ).fetchall()
        return {str(key): str(value) for key, value in rows}

    def failure_paths(self, root: str) -> list[str]:
        """Every path recorded as unreadable under ``root``."""
        prefix = _directory_prefix(root)
        with self._reading() as conn:
            rows = conn.execute(
                "SELECT file_path FROM index_failures "
                "WHERE file_path = ? OR substr(file_path, 1, length(?)) = ?",
                (root, prefix, prefix),
            ).fetchall()
        return [str(row[0]) for row in rows]

    def record_failures(self, clear: Sequence[str], failures: Mapping[str, str]) -> None:
        """Forget the failures in ``clear``, then record ``failures``.

        The caller names what to forget rather than passing a root, because a walk does
        not reach everything beneath its root: `.venv` and `node_modules` are pruned, and
        a directory that cannot be listed is skipped. Clearing by prefix would erase what
        a scan never looked at - a file recorded as broken inside a pruned directory would
        be quietly declared fine by a run of its parent.

        What is cleared is replaced wholesale, because a failure can outlive every chance
        to clear it one at a time: a file that fails on its *first* index never reaches
        `replace_document`, so it never enters `documents` and a later purge cannot find
        it either. Sound because one scan runs at a time - the run that just walked these
        paths is the freshest word on them.
        """
        with self.transaction() as conn:
            conn.executemany(
                "DELETE FROM index_failures WHERE file_path = ?", [(path,) for path in clear]
            )
            if failures:
                # An upsert, so a row this run could not clear - one out of its sight -
                # is restated rather than failing the whole run on a duplicate.
                conn.executemany(
                    "INSERT INTO index_failures(file_path, message) VALUES (?, ?) "
                    "ON CONFLICT(file_path) DO UPDATE SET message = excluded.message",
                    sorted(failures.items()),
                )

    def mark_scan_started(self, root: str) -> None:
        """This root, and every root containing it, is no longer vouched for.

        Called at a scan's first write, not at its start: a run that changes nothing -
        the model will not load, every file is unchanged - has no business retracting a
        certificate. A scan of `docs/api` retracts `docs` too, because a half-written
        subtree is a half-written tree.
        """
        prefix = _directory_prefix(root)
        with self.transaction() as conn:
            conn.execute(
                "UPDATE index_coverage SET verified = 0 "
                # itself, anything containing it, and anything inside it: this scan may
                # rewrite any of them, and none may go on vouching for itself while it does
                "WHERE root = ? "
                "OR substr(?, 1, length(root) + 1) = root || ? "
                "OR substr(root, 1, length(?)) = ?",
                (root, root, os.sep, prefix, prefix),
            )
            conn.execute(
                "INSERT INTO index_coverage(root, verified) VALUES (?, 0) "
                "ON CONFLICT(root) DO UPDATE SET verified = 0",
                (root,),
            )

    def mark_scan_complete(self, root: str, generation: int) -> None:
        """A full walk of ``root`` ran to the end.

        Not "and everything was readable" - that is what `index_failures` is for, and
        `index_status` will not call a tree whole while anything under it is listed there.
        Keeping the two apart means a run does not have to decide what a later question
        will mean.

        Only this root's row is written. An earlier draft also deleted the rows of roots
        inside it, on the grounds that this walk covered them - but it does not cover a
        pruned subtree, and a walk that hit failures covered even less. Deleting them
        turned a nested root that was perfectly fine into one that reported unknown,
        which is a worse answer than the one it replaced.
        """
        with self.transaction() as conn:
            current = int(
                (
                    conn.execute(
                        "SELECT value FROM meta WHERE key = ?", (_GENERATION_META_KEY,)
                    ).fetchone()
                    or ("0",)
                )[0]
            )
            if current != generation:
                # The index was discarded while this scan was walking; what it just
                # measured describes a database that no longer exists.
                return
            conn.execute(
                "INSERT INTO index_coverage(root, verified) VALUES (?, 1) "
                "ON CONFLICT(root) DO UPDATE SET verified = 1",
                (root,),
            )

    def revoke_coverage(self, conn: sqlite3.Connection | None = None) -> None:
        """Nothing is vouched for any more - the index itself was discarded.

        A model or dimension change empties every document in the database, including
        roots this process never looked at. A certificate that outlives its subject is
        worse than none: it says a tree is whole when nothing of it is left.

        The generation is bumped in the same breath. Revoking only settles the
        certificates that exist *now*; a scan already running has read its file hashes,
        will skip every file as unchanged, and would write a fresh certificate over an
        empty database. It compares the generation instead and stands down.
        """
        if conn is not None:
            _bump_generation(conn)
            conn.execute("DELETE FROM index_coverage")
            return
        with self.transaction() as owned:
            _bump_generation(owned)
            owned.execute("DELETE FROM index_coverage")

    def generation(self) -> int:
        """How many times this database has been emptied wholesale."""
        return int(self.get_meta(_GENERATION_META_KEY) or "0")

    def index_status(self, root: str, scope: str | None = None) -> IndexStatus:
        """What can honestly be said about answers drawn from ``root``.

        ``scope`` narrows *what is named* - the failures and stale documents worth
        mentioning - without changing whose coverage is being reported: a caller asking
        about one directory is still served from the whole root, and the certificate
        belongs to the root.

        Every read is one snapshot. Taken separately, a scan committing between them hands
        back a verdict that was never true at any instant: the failures read as empty, the
        certificate still reads valid, and the answer claims a whole tree while the row
        proving otherwise is already committed. Composing two snapshots in the caller has
        exactly the same hole, which is why the narrowing happens here.
        """
        named = scope if scope is not None else root
        prefix = _directory_prefix(named)
        with self._reading() as conn:
            conn.execute("BEGIN")
            try:
                rows = conn.execute(
                    "SELECT file_path, message FROM index_failures "
                    "WHERE file_path = ? OR substr(file_path, 1, length(?)) = ? "
                    "ORDER BY file_path",
                    (named, prefix, prefix),
                ).fetchall()
                certificate = conn.execute(
                    "SELECT verified FROM index_coverage WHERE root = ?", (root,)
                ).fetchone()
                # In the same snapshot as the rest: a verdict that mixes one moment's
                # certificate with another's provenance describes no moment at all.
                mismatch = conn.execute(
                    "SELECT value FROM meta WHERE key = ?", (WEIGHTS_MISMATCH_KEY,)
                ).fetchone()
                stale_vectors = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM documents "
                        "WHERE (file_path = ? OR substr(file_path, 1, length(?)) = ?) "
                        "AND vector_format != ?",
                        (named, prefix, prefix, VECTOR_FORMAT),
                    ).fetchone()[0]
                )
                # Whether the ROOT is whole, in the same snapshot. A caller asking about
                # one directory is still answered from the whole root, so a clean
                # subdirectory of a root that lost files is not itself a safe answer -
                # only what is *named* narrows.
                whole = rows == [] and stale_vectors == 0
                if named != root:
                    root_prefix = _directory_prefix(root)
                    whole = not conn.execute(
                        "SELECT 1 FROM index_failures "
                        "WHERE file_path = ? OR substr(file_path, 1, length(?)) = ? "
                        "UNION ALL SELECT 1 FROM documents "
                        "WHERE (file_path = ? OR substr(file_path, 1, length(?)) = ?) "
                        "AND vector_format != ? LIMIT 1",
                        (root, root_prefix, root_prefix,
                         root, root_prefix, root_prefix, VECTOR_FORMAT),
                    ).fetchone()  # fmt: skip
            finally:
                conn.execute("COMMIT")
        failures = tuple(
            FileFailure(file_path=str(path), message=str(message)) for path, message in rows
        )
        verified = certificate is not None and bool(certificate[0]) and whole
        weights_mismatch = str(mismatch[0]) if mismatch else None
        return IndexStatus(
            # A walk that read every file still cannot vouch for vectors built by a
            # model that is no longer the one answering.
            verified=verified and weights_mismatch is None,
            failures=failures,
            stale_vectors=stale_vectors,
            weights_mismatch=weights_mismatch,
        )

    def dismiss_notices(self, keys: Iterable[str]) -> None:
        """Forget the notices that have been delivered; any added since are kept."""
        with self.transaction() as conn:
            conn.executemany("DELETE FROM meta WHERE key = ?", [(key,) for key in keys])

    def pragma(self, name: str) -> str:
        """Return a PRAGMA's current value on this thread's connection."""
        if not name.isidentifier():
            raise DatabaseError(f"Invalid pragma name: {name!r}")
        with self._reading() as conn:
            row = conn.execute(f"PRAGMA {name}").fetchone()
        return "" if row is None else str(row[0])

    # ------------------------------------------------------------------ documents

    def get_document(self, file_path: str) -> Document | None:
        with self._reading() as conn:
            row = conn.execute(
                "SELECT id, file_path, title, content_hash, last_modified "
                "FROM documents WHERE file_path = ?",
                (file_path,),
            ).fetchone()
        return None if row is None else _document_from_row(row)

    def find_documents_by_suffix(self, relative_path: str) -> list[Document]:
        """Documents whose stored path ends with ``/<relative_path>``."""
        suffix = os.sep + relative_path.lstrip("/\\")
        with self._reading() as conn:
            rows = conn.execute(
                "SELECT id, file_path, title, content_hash, last_modified FROM documents "
                "WHERE substr(file_path, -length(?)) = ? ORDER BY file_path",
                (suffix, suffix),
            ).fetchall()
        return [_document_from_row(row) for row in rows]

    def document_fingerprints(self, directory: str) -> dict[str, tuple[str, int | None]]:
        """Map ``file_path -> (content_hash, mtime_ns)`` under ``directory``.

        What a caller needs to ask the filesystem whether the index is still current:
        the cheap question first (has the modification time moved?) and the expensive
        one - re-hashing the bytes - only for the files where it has.
        """
        prefix = _directory_prefix(directory)
        with self._reading() as conn:
            rows = conn.execute(
                "SELECT file_path, content_hash, mtime_ns FROM documents "
                "WHERE substr(file_path, 1, length(?)) = ?",
                (prefix, prefix),
            ).fetchall()
        return {
            str(path): (str(content_hash), None if mtime_ns is None else int(mtime_ns))
            for path, content_hash, mtime_ns in rows
        }

    def document_hashes(self, directory: str) -> dict[str, tuple[str, int, int | None, str | None]]:
        """Map ``file_path -> (hash, vector_format, mtime_ns, weights_revision)`` under it.

        The format and the weights travel with the hash because all three answer the same
        question - may this file be skipped? - and a file whose vectors predate the current
        pooling, or came from other weights, must be rebuilt however unchanged its bytes
        are. The recorded modification time travels with them because a file that may be
        skipped still has to have that time brought up to date, or the freshness check reads
        the bytes of an unchanged file for ever.
        """
        prefix = _directory_prefix(directory)
        with self._reading() as conn:
            rows = conn.execute(
                "SELECT file_path, content_hash, vector_format, mtime_ns, weights_revision "
                "FROM documents WHERE substr(file_path, 1, length(?)) = ?",
                (prefix, prefix),
            ).fetchall()
        return {
            str(path): (
                str(content_hash),
                int(vector_format),
                None if mtime_ns is None else int(mtime_ns),
                None if weights is None else str(weights),
            )
            for path, content_hash, vector_format, mtime_ns, weights in rows
        }

    def record_modification_time(
        self, file_path: str, content_hash: str, previous_ns: int | None, mtime_ns: int
    ) -> None:
        """Note when an unchanged file was last written, without touching its content.

        A file whose bytes are what was indexed is skipped, and used to keep whatever time
        it was stored with - a `touch`, a checkout, or a row migrated from a schema that
        had no nanoseconds at all. Every freshness sweep then found a time that did not
        match and hashed the file again, forever, to conclude what the hash it already
        held could have said. One narrow UPDATE ends that: no sections, no vectors, no
        reindex.

        A compare-and-swap on both the hash and the time the caller started from. Whoever
        checked those bytes did so outside this transaction: an indexing run may have
        replaced the document in between - writing a time against somebody else's content
        is how a stale row comes to look current - or may have recorded a *newer* time for
        the same content, which this must not roll back, or the file it just verified gets
        hashed all over again. `IS` rather than `=` so a row that had no time recorded at
        all is matched rather than skipped.
        """
        with self.transaction() as conn:
            conn.execute(
                "UPDATE documents SET last_modified = ?, mtime_ns = ? "
                "WHERE file_path = ? AND content_hash = ? AND mtime_ns IS ?",
                (mtime_ns // 1_000_000_000, mtime_ns, file_path, content_hash, previous_ns),
            )

    def list_documents(self, directory: str = "") -> list[DocumentSummary]:
        """All documents (optionally restricted to ``directory``) with section counts."""
        sql = (
            "SELECT d.file_path, d.title, COUNT(s.id), d.last_modified "
            "FROM documents d LEFT JOIN sections s ON s.doc_id = d.id "
        )
        params: tuple[str, ...] = ()
        if directory:
            prefix = _directory_prefix(directory)
            sql += "WHERE substr(d.file_path, 1, length(?)) = ? "
            params = (prefix, prefix)
        sql += "GROUP BY d.id ORDER BY d.file_path"
        with self._reading() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            DocumentSummary(
                file_path=str(path),
                title=str(title),
                section_count=int(count),
                last_modified=int(modified),
            )
            for path, title, count, modified in rows
        ]

    def replace_document(
        self,
        *,
        file_path: str,
        title: str,
        content_hash: str,
        last_modified: int,
        mtime_ns: int,
        sections: Sequence[SectionDraft],
        vectors: Sequence[SectionVectors],
        weights_revision: str | None = None,
    ) -> Document:
        """Atomically insert or fully replace one document, its sections and vectors.

        ``weights_revision`` names the weights that produced ``vectors``, and is written in
        the same transaction as them: it is the one moment the two are known to belong
        together. None means unknown, which no known revision will match.
        """
        if len(sections) != len(vectors):
            raise DatabaseError(
                f"Got {len(sections)} sections but {len(vectors)} vector sets for {file_path}"
            )
        for section, vector in zip(sections, vectors, strict=True):
            if len(vector.units) != len(section.units):
                raise DatabaseError(
                    f"Section '{section.heading_path}' of {file_path} has {len(section.units)} "
                    f"passages but {len(vector.units)} passage vectors"
                )
            if (vector.section is None) != (not section.units):
                raise DatabaseError(
                    f"Section '{section.heading_path}' of {file_path}: a section vector is "
                    "required exactly when the section has passages"
                )
            present = [] if vector.section is None else [vector.section]
            for embedding in (*present, *vector.units):
                self._check_vector(embedding, f"Embedding for {file_path}")
        with self.transaction() as conn:
            # Sampled before the delete below: on a database from an earlier release the
            # high-water mark is missing, and MAX(id) taken afterwards would hand the ids
            # of the rows just deleted straight back out.
            section_id = _next_section_id(conn) - 1
            row = conn.execute(
                "SELECT id FROM documents WHERE file_path = ?", (file_path,)
            ).fetchone()
            if row is None:
                cursor = conn.execute(
                    "INSERT INTO documents(file_path, title, content_hash, last_modified, "
                    "mtime_ns, vector_format, weights_revision) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        file_path,
                        title,
                        content_hash,
                        last_modified,
                        mtime_ns,
                        VECTOR_FORMAT,
                        weights_revision,
                    ),
                )
                if cursor.lastrowid is None:  # pragma: no cover - sqlite always sets it
                    raise DatabaseError("INSERT INTO documents returned no rowid")
                doc_id = cursor.lastrowid
            else:
                doc_id = int(row[0])
                conn.execute(
                    "UPDATE documents SET title = ?, content_hash = ?, last_modified = ?, "
                    "mtime_ns = ?, vector_format = ?, weights_revision = ? WHERE id = ?",
                    (
                        title,
                        content_hash,
                        last_modified,
                        mtime_ns,
                        VECTOR_FORMAT,
                        weights_revision,
                        doc_id,
                    ),
                )
                conn.execute("DELETE FROM sections WHERE doc_id = ?", (doc_id,))
            for section, vector in zip(sections, vectors, strict=True):
                section_id += 1
                conn.execute(
                    "INSERT INTO sections(id, doc_id, heading_title, heading_level, "
                    "heading_path, content, start_line, end_line, part_index) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        section_id,
                        doc_id,
                        section.heading_title,
                        section.heading_level,
                        section.heading_path,
                        section.content,
                        section.start_line,
                        section.end_line,
                        section.part_index,
                    ),
                )
                if vector.section is not None:
                    conn.execute(
                        "INSERT INTO sections_vec(section_id, embedding) VALUES (?, ?)",
                        (section_id, serialize_embedding(vector.section)),
                    )
                for ordinal, (text, embedding) in enumerate(
                    zip(section.units, vector.units, strict=True)
                ):
                    unit_id = conn.execute(
                        "INSERT INTO units(section_id, ordinal, content) VALUES (?, ?, ?)",
                        (section_id, ordinal, text),
                    ).lastrowid
                    conn.execute(
                        "INSERT INTO units_vec(unit_id, embedding) VALUES (?, ?)",
                        (unit_id, serialize_embedding(embedding)),
                    )
            conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (_SECTION_ID_META_KEY, str(section_id + 1)),
            )
            # This write can remove the last vector in the index as well as add one: a
            # document whose prose became headings alone embeds nothing, and its old
            # vectors went with its old sections.
            _forget_weights_without_vectors(conn)
        return Document(
            id=doc_id,
            file_path=file_path,
            title=title,
            content_hash=content_hash,
            last_modified=last_modified,
        )

    def _check_vector(self, embedding: Sequence[float], what: str) -> None:
        if len(embedding) != self._embedding_dim:
            raise DatabaseError(
                f"{what} has {len(embedding)} dimensions, expected {self._embedding_dim}"
            )
        if not _is_usable_vector(embedding):
            raise DatabaseError(f"{what} is all zeros or contains NaN/inf values")

    def delete_documents(self, file_paths: Iterable[str]) -> int:
        """Delete documents by path; sections, FTS rows and vectors cascade. Returns count."""
        paths = list(file_paths)
        if not paths:
            return 0
        deleted = 0
        with self.transaction() as conn:
            for path in paths:
                deleted += conn.execute(
                    "DELETE FROM documents WHERE file_path = ?", (path,)
                ).rowcount
            _forget_weights_without_vectors(conn)
        return deleted

    def clear(self, notice: Callable[[int], str] | None = None) -> int:
        """Delete every document (and, by cascade, every section and vector).

        ``notice`` words the message for the number of documents discarded; it is persisted
        in the same transaction, so the explanation cannot be lost while the data is.
        """
        with self.transaction() as conn:
            discarded = conn.execute("DELETE FROM documents").rowcount
            _forget_weights_without_vectors(conn)
            self.revoke_coverage(conn)
            if discarded and notice is not None:
                _add_notice(conn, notice(discarded))
            return discarded

    # ------------------------------------------------------------------ sections

    def record_weights_mismatch(self, message: str | None, *, replace: bool = True) -> None:
        """Remember (or clear) that the index and the loaded model disagree.

        Persisted rather than held in memory: every `search_docs` and `list_documents`
        answer carries an `index_status`, and a fact this serious may not depend on
        which process, or which run, happens to have noticed it. ``replace=False`` keeps a
        message already recorded, in the same statement that would have written this one.
        """
        with self.transaction() as conn:
            if message is None:
                conn.execute("DELETE FROM meta WHERE key = ?", (WEIGHTS_MISMATCH_KEY,))
            else:
                conn.execute(
                    "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) "
                    + ("DO UPDATE SET value = excluded.value" if replace else "DO NOTHING"),
                    (WEIGHTS_MISMATCH_KEY, message),
                )

    def revoke_weights(self, message: str) -> None:
        """Withdraw the index's revision before other weights write into it.

        From here until `settle_weights` re-certifies it, the index holds - or may hold -
        vectors from two models, and no search may rank them against one query.
        """
        with self.transaction() as conn:
            _revoke_weights(conn, message)

    def settle_weights(self, weights: str | None) -> None:
        """Close a run: vouch for the vectors again, or say what still stands in the way.

        The revision may be written back only once every vector in the database - every
        root's, and rows no walk reaches, such as a document indexed inside `.venv` - came
        from ``weights``. Checking only what this run visited would re-certify an index
        still holding another model's vectors, which is the failure a certificate exists
        to rule out. An index holding no vector is never certified: there is nothing to
        vouch for, and a revision over nothing would turn the next model away. None means
        this run could not tell which weights it ran, and so vouches for nothing.
        """
        recorded = self.get_meta(WEIGHTS_META_KEY)
        mismatch = self.get_meta(WEIGHTS_MISMATCH_KEY)
        if (recorded, mismatch) != (None, None) and self.count_rows("units_vec") == 0:
            # A revision claimed for vectors that never arrived - the run died between
            # the claim and the write - describes nothing, whoever is asking.
            with self.transaction() as conn:
                _forget_weights_without_vectors(conn)
            return
        # Unknown weights vouch for nothing. Known and already certified, with nothing
        # said against it: the invariant holds by construction, so the whole-database
        # check would find nothing, and a no-op run need not take the write lock to learn
        # that.
        if weights is None or (recorded == weights and mismatch is None):
            return
        with self.transaction() as conn:
            if conn.execute("SELECT 1 FROM units_vec LIMIT 1").fetchone() is None:
                _forget_weights_without_vectors(conn)
                return
            stale = [
                str(row[0])
                for row in conn.execute(
                    "SELECT d.file_path FROM documents AS d "
                    "WHERE (d.weights_revision IS NULL OR d.weights_revision != ?) "
                    "AND EXISTS (SELECT 1 FROM sections AS s JOIN units AS u "
                    "ON u.section_id = s.id WHERE s.doc_id = d.id) "
                    "ORDER BY d.file_path",
                    (weights,),
                )
            ]
            if not stale:
                conn.execute(
                    "INSERT INTO meta (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (WEIGHTS_META_KEY, weights),
                )
                conn.execute("DELETE FROM meta WHERE key = ?", (WEIGHTS_MISMATCH_KEY,))
                return
            # Named by directory, because a row this run could not reach - one indexed
            # deliberately inside a pruned directory - is repaired only by indexing that
            # directory itself, and a count alone would not say where to point it.
            folders = sorted({os.path.dirname(path) for path in stale})
            shown = ", ".join(folders[:3]) + (
                f" and {len(folders) - 3} more" if len(folders) > 3 else ""
            )
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (
                    WEIGHTS_MISMATCH_KEY,
                    f"{len(stale)} document(s) still hold vectors from other weights than "
                    f"the ones answering now, under {shown}. Only keyword ranking is used "
                    "until index_directory re-embeds them - run it on those directories.",
                ),
            )

    def forget_weights_revision(self) -> None:
        """Drop the recorded weights revision: no documents, so nothing it can describe.

        `clear()` does this in the same transaction as the delete. This exists for every
        other way the index empties - a purge of the last document, a rebuild for a new
        vector size, the v1 format discard - where the rows go without going through it.
        """
        with self.transaction() as conn:
            conn.execute("DELETE FROM meta WHERE key = ?", (WEIGHTS_META_KEY,))

    def get_sections(self, doc_id: int) -> list[Section]:
        """Every section of a document in source order."""
        with self._reading() as conn:
            rows = conn.execute(
                f"SELECT {_SECTION_COLUMNS} FROM sections WHERE doc_id = ? ORDER BY id",
                (doc_id,),
            ).fetchall()
        return [_section_from_row(row) for row in rows]

    def get_sections_with_documents(
        self, section_ids: Sequence[int]
    ) -> dict[int, tuple[Section, Document]]:
        """Hydrate section ids into ``(Section, Document)`` pairs."""
        hydrated: dict[int, tuple[Section, Document]] = {}
        columns = ", ".join(f"s.{name.strip()}" for name in _SECTION_COLUMNS.split(","))
        with self._reading() as conn:
            for start in range(0, len(section_ids), _SQL_VARIABLE_BATCH):
                batch = section_ids[start : start + _SQL_VARIABLE_BATCH]
                placeholders = ", ".join("?" for _ in batch)
                rows = conn.execute(
                    f"SELECT {columns}, d.id, d.file_path, d.title, d.content_hash, "
                    "d.last_modified FROM sections s JOIN documents d ON d.id = s.doc_id "
                    f"WHERE s.id IN ({placeholders})",
                    tuple(batch),
                ).fetchall()
                for row in rows:
                    section = _section_from_row(row[:9])
                    hydrated[section.id] = (section, _document_from_row(row[9:]))
        return hydrated

    # ------------------------------------------------------------------ search primitives

    def fts_search(self, match_query: str, limit: int, scope: str | None = None) -> list[int]:
        """Section ids matching an FTS5 query, best BM25 rank first.

        ``scope`` restricts the search to documents under one directory. The predicate
        joins inside the query so the limit applies to what survives it: filtering a
        page afterwards would return fewer rows than asked for whenever a neighbouring
        documentation root in the same database ranks higher.
        """
        with self._reading() as conn:
            if scope is None:
                rows = conn.execute(
                    "SELECT rowid FROM sections_fts WHERE sections_fts MATCH ? "
                    "ORDER BY rank LIMIT ?",
                    (match_query, limit),
                ).fetchall()
            else:
                prefix = _directory_prefix(scope)
                rows = conn.execute(
                    "SELECT f.rowid FROM sections_fts f "
                    "JOIN sections s ON s.id = f.rowid JOIN documents d ON d.id = s.doc_id "
                    "WHERE sections_fts MATCH ? AND substr(d.file_path, 1, length(?)) = ? "
                    "ORDER BY rank LIMIT ?",
                    (match_query, prefix, prefix, limit),
                ).fetchall()
        return [int(row[0]) for row in rows]

    def vec_search(self, embedding: Sequence[float], limit: int) -> list[tuple[int, float]]:
        """``(section_id, cosine_distance)`` for the nearest section vectors, closest first."""
        self._check_vector(embedding, "Query embedding")
        with self._reading() as conn:
            rows = conn.execute(
                "SELECT section_id, distance FROM sections_vec "
                "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (serialize_embedding(embedding), limit),
            ).fetchall()
        return [
            (int(section_id), float(distance))
            for section_id, distance in rows
            if distance is not None  # defensive: NaN distances surface as NULL
        ]

    def unit_search(self, embedding: Sequence[float], limit: int) -> list[tuple[int, float, str]]:
        """``(section_id, cosine_distance, passage)`` for the nearest passages, closest first.

        A section appears once per matching passage; callers keep its best one.
        """
        self._check_vector(embedding, "Query embedding")
        with self._reading() as conn:
            rows = conn.execute(
                "WITH nearest AS (SELECT unit_id, distance FROM units_vec "
                "WHERE embedding MATCH ? AND k = ?) "
                "SELECT u.section_id, nearest.distance, u.content FROM nearest "
                "JOIN units u ON u.id = nearest.unit_id ORDER BY nearest.distance",
                (serialize_embedding(embedding), limit),
            ).fetchall()
        return [
            (int(section_id), float(distance), str(content))
            for section_id, distance, content in rows
            if distance is not None
        ]

    def fts_matching(self, term: str, within: Sequence[int]) -> set[int]:
        """Which of the sections ``within`` match the single FTS5 ``term``."""
        if not within:
            return set()
        placeholders = ", ".join("?" for _ in within)
        with self._reading() as conn:
            rows = conn.execute(
                "SELECT rowid FROM sections_fts WHERE sections_fts MATCH ? "
                f"AND rowid IN ({placeholders})",
                (term, *within),
            ).fetchall()
        return {int(row[0]) for row in rows}

    def fts_document_frequency(self, term: str) -> int:
        """Number of sections matching the single FTS5 ``term``."""
        with self._reading() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM sections_fts WHERE sections_fts MATCH ?", (term,)
            ).fetchone()
        return int(row[0])

    def sections_with_passages(self, section_ids: Sequence[int]) -> set[int]:
        """The subset of ``section_ids`` that has a body (heading-only sections have none)."""
        if not section_ids:
            return set()
        placeholders = ", ".join("?" for _ in section_ids)
        with self._reading() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT section_id FROM units WHERE section_id IN ({placeholders})",
                tuple(section_ids),
            ).fetchall()
        return {int(row[0]) for row in rows}

    def sections_under(self, section_ids: Sequence[int], directory: str) -> set[int]:
        """The subset of ``section_ids`` whose document lives under ``directory``.

        Search is scoped with this rather than with a predicate inside the FTS5 and
        vec0 queries: both apply their own limit before any join would filter, so a
        scoped predicate there silently returns fewer results than asked for.
        """
        if not section_ids:
            return set()
        prefix = _directory_prefix(directory)
        placeholders = ", ".join("?" for _ in section_ids)
        with self._reading() as conn:
            rows = conn.execute(
                f"SELECT s.id FROM sections s JOIN documents d ON d.id = s.doc_id "
                f"WHERE s.id IN ({placeholders}) "
                "AND substr(d.file_path, 1, length(?)) = ?",
                (*section_ids, prefix, prefix),
            ).fetchall()
        return {int(row[0]) for row in rows}

    def integrity_problems(self) -> list[str]:
        """Everything that is wrong with the store; an empty list means it is sound.

        Covers SQLite's own page and foreign-key checks, the FTS5 index against the
        ``sections`` table, and the invariants the triggers exist to uphold: one FTS row
        per section, one vector per passage, a section vector exactly for the sections
        that have passages, and every stored vector of the configured dimension.
        """
        problems: list[str] = []
        blob_bytes = self._embedding_dim * 4
        conn = self.connection()
        try:
            # rank = 1 makes FTS5 compare the index with the external content table; the
            # plain form only checks the index's internal consistency. The command is an
            # INSERT, so it needs the write lock - failing to get it proves nothing.
            conn.execute(
                "INSERT INTO sections_fts(sections_fts, rank) VALUES ('integrity-check', 1)"
            )
        except sqlite3.Error as exc:
            if _is_lock_error(exc):
                problems.append(
                    "could not verify the FTS5 index: the database is locked by another "
                    "writer (not a sign of damage - retry when indexing has finished)"
                )
            else:
                problems.append(f"FTS5 index does not match the sections table: {exc}")
        try:
            conn.execute("BEGIN")  # one read snapshot: counts taken mid-write would disagree
            try:
                version = int(conn.execute("PRAGMA user_version").fetchone()[0])
                pages = [str(row[0]) for row in conn.execute("PRAGMA integrity_check")]
                orphans = conn.execute("PRAGMA foreign_key_check").fetchall()
                counts = {
                    table: int(conn.execute(f"SELECT COUNT(*) FROM {source}").fetchone()[0])
                    for table, source in (
                        ("sections", "sections"),
                        ("sections_fts", "sections_fts_docsize"),
                        ("sections_vec", "sections_vec"),
                        ("units", "units"),
                        ("units_vec", "units_vec"),
                    )
                }
                with_passages = int(
                    conn.execute("SELECT COUNT(DISTINCT section_id) FROM units").fetchone()[0]
                )
                wrong_size = sum(
                    int(
                        conn.execute(
                            f"SELECT COUNT(*) FROM {table} WHERE length(embedding) != ?",
                            (blob_bytes,),
                        ).fetchone()[0]
                    )
                    for table in ("sections_vec", "units_vec")
                )
                row = conn.execute("SELECT value FROM meta WHERE key = 'embedding_dim'").fetchone()
                stored_dim = None if row is None else str(row[0])
            finally:
                conn.execute("ROLLBACK")
        except sqlite3.Error as exc:
            raise DatabaseError(f"Integrity check could not read the database: {exc}") from exc

        if version != SCHEMA_VERSION:
            problems.append(f"schema version is {version}, expected {SCHEMA_VERSION}")
        if pages != ["ok"]:
            problems.append("PRAGMA integrity_check: " + "; ".join(pages[:5]))
        if orphans:
            problems.append(f"PRAGMA foreign_key_check: {len(orphans)} orphaned row(s)")
        if counts["sections"] != counts["sections_fts"]:
            problems.append(f"{counts['sections']} sections but {counts['sections_fts']} FTS rows")
        if counts["units"] != counts["units_vec"]:
            problems.append(f"{counts['units']} passages but {counts['units_vec']} passage vectors")
        if counts["sections_vec"] != with_passages:
            problems.append(
                f"{counts['sections_vec']} section vectors but "
                f"{with_passages} sections with passages"
            )
        if wrong_size:
            problems.append(
                f"{wrong_size} stored vector(s) are not {self._embedding_dim}-dimensional"
            )
        if stored_dim != str(self._embedding_dim):
            problems.append(f"meta embedding_dim is {stored_dim}, expected {self._embedding_dim}")
        return problems

    def count_rows(self, table: str) -> int:
        """Row count of one of the known tables (diagnostics and integrity tests)."""
        if table not in {
            "documents", "sections", "sections_fts", "sections_vec", "units", "units_vec",
            "index_failures", "index_coverage",
        }:  # fmt: skip
            raise DatabaseError(f"Unknown table: {table!r}")
        # COUNT(*) on an external-content FTS5 table is answered from `sections`; the
        # docsize shadow table has one row per entry actually present in the index.
        source = "sections_fts_docsize" if table == "sections_fts" else table
        with self._reading() as conn:
            row = conn.execute(f"SELECT COUNT(*) FROM {source}").fetchone()
        return int(row[0])


def _next_section_id(conn: sqlite3.Connection) -> int:
    """A section id that was never used before, not even by a row deleted since.

    SQLite hands the rowid of a deleted row out again. A search ranks ids on one connection
    and fetches them on another, so a re-index in between must make the old ids *vanish*
    (the search then ranks again) rather than point at whatever section was stored next.
    """
    stored = conn.execute("SELECT value FROM meta WHERE key = ?", (_SECTION_ID_META_KEY,))
    row = stored.fetchone()
    highest = conn.execute("SELECT COALESCE(MAX(id), 0) FROM sections").fetchone()[0]
    return max(int(row[0]) if row is not None else 1, int(highest) + 1)


def _add_notice(conn: sqlite3.Connection, message: str) -> None:
    logger.warning(message)
    # Numbered after the newest, not by count: notices are dismissed one by one. The key is
    # TEXT, so the successor is taken numerically - 'notice:10000' sorts below 'notice:9999'.
    newest = conn.execute(
        "SELECT MAX(CAST(substr(key, 8) AS INTEGER)) FROM meta WHERE key LIKE 'notice:%'"
    ).fetchone()[0]
    number = 0 if newest is None else int(newest) + 1
    conn.execute("INSERT INTO meta(key, value) VALUES (?, ?)", (f"notice:{number:04d}", message))


def _close_quietly(conn: sqlite3.Connection) -> None:
    try:
        conn.close()
    except sqlite3.Error:  # pragma: no cover - best effort
        logger.warning("Failed to close a SQLite connection", exc_info=True)


def _is_lock_error(exc: sqlite3.Error) -> bool:
    message = str(exc).lower()
    return "locked" in message or "busy" in message


def _enable_wal(conn: sqlite3.Connection) -> None:
    """Switch to WAL, retrying while another process holds the lock.

    Changing the journal mode needs an exclusive lock and SQLite reports contention
    on it immediately instead of honouring the busy timeout.
    """
    for attempt in range(_WAL_ATTEMPTS):
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            return
        except sqlite3.OperationalError as exc:
            if attempt == _WAL_ATTEMPTS - 1 or not _is_lock_error(exc):
                raise
            time.sleep(_WAL_RETRY_SECONDS * (attempt + 1))


def _document_from_row(row: Sequence[object]) -> Document:
    doc_id, file_path, title, content_hash, last_modified = row
    return Document(
        id=_as_int(doc_id),
        file_path=str(file_path),
        title=str(title),
        content_hash=str(content_hash),
        last_modified=_as_int(last_modified),
    )


def _section_from_row(row: Sequence[object]) -> Section:
    (
        section_id,
        doc_id,
        heading_title,
        heading_level,
        heading_path,
        content,
        start_line,
        end_line,
        part_index,
    ) = row
    return Section(
        id=_as_int(section_id),
        doc_id=_as_int(doc_id),
        heading_title=str(heading_title),
        heading_level=_as_int(heading_level),
        heading_path=str(heading_path),
        content=str(content),
        start_line=_as_int(start_line),
        end_line=_as_int(end_line),
        part_index=_as_int(part_index),
    )


def _as_int(value: object) -> int:
    if isinstance(value, int):
        return value
    raise DatabaseError(f"Expected an integer column value, got {type(value).__name__}")
