"""Storage tests: schema integrity, WAL mode, sqlite-vec, cascades, transactions."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import replace
from pathlib import Path

import pytest
from fakes import FakeEmbedder, vectors_for

from markdown_memory.db import SCHEMA_VERSION, Database, serialize_embedding
from markdown_memory.exceptions import DatabaseError
from markdown_memory.models import Document, SectionDraft, SectionVectors


def draft(title: str, content: str, *, path: str | None = None, line: int = 1) -> SectionDraft:
    return SectionDraft(
        heading_title=title,
        heading_level=2,
        heading_path=path or f"Doc > {title}",
        base_path=path or f"Doc > {title}",
        content=content,
        start_line=line,
        end_line=line + content.count("\n"),
        units=(content.split("\n\n", 1)[-1],),
    )


def store(
    db: Database, embedder: FakeEmbedder, file_path: str, sections: list[SectionDraft]
) -> Document:
    return db.replace_document(
        file_path=file_path,
        title="Doc",
        content_hash="hash-" + file_path,
        last_modified=1_700_000_000,
        mtime_ns=1_700_000_000,
        sections=sections,
        vectors=vectors_for(embedder, sections),
    )


SECTIONS = [
    draft("Install", "## Install\n\nRun the installer script.", line=1),
    draft("Retries", "## Retries\n\nSet ORBIT_MAX_RETRIES to control retry attempts.", line=5),
    draft("Shutdown", "## Shutdown\n\nThe proxy drains connections on SIGTERM.", line=9),
]


class TestConnection:
    def test_wal_mode_and_pragmas(self, db: Database) -> None:
        assert db.pragma("journal_mode") == "wal"
        assert db.pragma("foreign_keys") == "1"
        assert db.pragma("synchronous") == "1"  # NORMAL

    def test_wal_file_is_created_on_write(self, db: Database, fake_embedder: FakeEmbedder) -> None:
        store(db, fake_embedder, "/docs/a.md", SECTIONS)
        assert db.path.with_name(db.path.name + "-wal").exists()

    def test_sqlite_vec_extension_is_loaded(self, db: Database) -> None:
        version = db.connection().execute("SELECT vec_version()").fetchone()[0]
        assert str(version).startswith("v")

    def test_pragmas_apply_to_connections_opened_by_other_threads(self, db: Database) -> None:
        seen: dict[str, str] = {}

        def probe() -> None:
            seen["journal_mode"] = db.pragma("journal_mode")
            seen["foreign_keys"] = db.pragma("foreign_keys")
            seen["vec"] = str(db.connection().execute("SELECT vec_version()").fetchone()[0])

        thread = threading.Thread(target=probe)
        thread.start()
        thread.join()
        assert seen["journal_mode"] == "wal"
        assert seen["foreign_keys"] == "1"
        assert seen["vec"].startswith("v")

    def test_each_thread_gets_its_own_connection(self, db: Database) -> None:
        connections: list[sqlite3.Connection] = []
        thread = threading.Thread(target=lambda: connections.append(db.connection()))
        thread.start()
        thread.join()
        assert connections[0] is not db.connection()
        assert db.connection() is db.connection()

    def test_in_memory_database_is_rejected(self) -> None:
        with pytest.raises(DatabaseError, match="In-memory"):
            Database(":memory:")

    def test_closed_database_refuses_work(self, tmp_path: Path) -> None:
        database = Database(tmp_path / "x.db")
        database.close()
        database.close()  # idempotent
        with pytest.raises(DatabaseError, match="closed"):
            database.list_documents()

    def test_unopenable_path_raises_database_error(self, tmp_path: Path) -> None:
        blocker = tmp_path / "file"
        blocker.write_text("not a directory")
        with pytest.raises(DatabaseError):
            Database(blocker / "nested" / "index.db")

    def test_invalid_pragma_name_is_rejected(self, db: Database) -> None:
        with pytest.raises(DatabaseError, match="Invalid pragma"):
            db.pragma("journal_mode; DROP TABLE documents")


class TestSchema:
    def test_tables_triggers_and_version(self, db: Database) -> None:
        conn = db.connection()
        names = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','trigger')"
            )
        }
        assert {"documents", "sections", "sections_fts", "sections_vec", "meta"} <= names
        assert {"sections_after_insert", "sections_after_delete", "sections_after_update"} <= names
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == SCHEMA_VERSION

    def test_column_layout_matches_the_specification(self, db: Database) -> None:
        conn = db.connection()
        documents = [str(row[1]) for row in conn.execute("PRAGMA table_info(documents)")]
        sections = [str(row[1]) for row in conn.execute("PRAGMA table_info(sections)")]
        assert documents == [
            "id",
            "file_path",
            "title",
            "content_hash",
            "last_modified",
            "vector_format",
            "mtime_ns",
            "weights_revision",
        ]
        assert sections == [
            "id", "doc_id", "heading_title", "heading_level", "heading_path",
            "content", "start_line", "end_line", "part_index",
        ]  # fmt: skip

    def test_file_path_is_unique(self, db: Database) -> None:
        with pytest.raises(DatabaseError), db.transaction() as conn:
            for _ in range(2):
                conn.execute(
                    "INSERT INTO documents(file_path, title, content_hash, last_modified) "
                    "VALUES ('/same.md', 't', 'h', 0)"
                )

    def test_foreign_key_is_enforced(self, db: Database) -> None:
        with pytest.raises(DatabaseError), db.transaction() as conn:
            conn.execute(
                "INSERT INTO sections(doc_id, heading_title, heading_level, heading_path, "
                "content, start_line, end_line) VALUES (999, 't', 1, 'p', 'c', 1, 1)"
            )

    def test_reopening_is_a_no_op_migration(self, tmp_path: Path) -> None:
        path = tmp_path / "again.db"
        with Database(path):
            pass
        with Database(path) as reopened:
            assert reopened.get_meta("embedding_dim") == "384"

    @staticmethod
    def contents(path: Path) -> tuple[object, ...]:
        """Everything a refused open must leave as it was, read without opening a Database."""
        plain = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            counts = tuple(
                plain.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("documents", "sections", "units", "index_coverage")
            )
            meta = plain.execute("SELECT key, value FROM meta ORDER BY key").fetchall()
            version = plain.execute("PRAGMA user_version").fetchone()[0]
            columns = plain.execute("SELECT name FROM pragma_table_info('index_coverage')")
            return (*counts, tuple(meta), version, tuple(columns.fetchall()))
        finally:
            plain.close()

    def test_a_database_of_another_dimension_is_refused_and_left_unchanged(
        self, tmp_path: Path, fake_embedder: FakeEmbedder
    ) -> None:
        """Opening is all a search does: it must never cost the index its vectors (#82)."""
        path = tmp_path / "dim.db"
        with Database(path, embedding_dim=384) as small:
            store(small, fake_embedder, "/docs/a.md", SECTIONS)
            vectors = small.count_rows("units_vec"), small.count_rows("sections_vec")
        before = self.contents(path)
        with pytest.raises(DatabaseError, match="Nothing was changed"):
            Database(path, embedding_dim=768)
        assert self.contents(path) == before
        with Database(path, embedding_dim=384) as again:
            assert (again.count_rows("units_vec"), again.count_rows("sections_vec")) == vectors
            assert len(again.unit_search(fake_embedder.embed_query("installer"), 5)) == 3

    def test_the_dimension_is_checked_before_any_migration_runs(
        self, tmp_path: Path, fake_embedder: FakeEmbedder
    ) -> None:
        # One migration deletes documents; a refusal that came after them would not be one.
        path = tmp_path / "old.db"
        with Database(path, embedding_dim=384) as small:
            store(small, fake_embedder, "/docs/a.md", SECTIONS)
            conn = small.connection()
            conn.execute("ALTER TABLE index_coverage DROP COLUMN gitignore")  # back to v6
            conn.execute("PRAGMA user_version = 6")
        before = self.contents(path)
        with pytest.raises(DatabaseError, match="384-dimensional"):
            Database(path, embedding_dim=768)
        assert self.contents(path) == before

    def test_a_refused_open_lets_go_of_the_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "held.db"
        with Database(path, embedding_dim=384):
            pass
        closed: list[Path] = []
        real_close = Database.close
        monkeypatch.setattr(Database, "close", lambda db: (closed.append(db.path), real_close(db)))
        with pytest.raises(DatabaseError):
            Database(path, embedding_dim=768)
        assert closed == [path]  # nobody else holds the object, so nobody else could close it

    def test_the_refusal_names_the_path_both_sizes_and_the_model(self, tmp_path: Path) -> None:
        recorded, unrecorded = tmp_path / "recorded.db", tmp_path / "unrecorded.db"
        with Database(recorded, embedding_dim=384) as database:
            database.set_meta("embedding_model", "BAAI/bge-small-en-v1.5")
        with Database(unrecorded, embedding_dim=384):
            pass
        with pytest.raises(DatabaseError) as named:
            Database(recorded, embedding_dim=768)
        message = str(named.value)
        assert str(recorded) in message and "384-dimensional" in message
        assert "produces 768" in message and "built by BAAI/bge-small-en-v1.5" in message
        assert "MARKDOWN_MEMORY_DB" in message and "-wal and -shm" in message
        with pytest.raises(DatabaseError, match="built by an unrecorded model"):
            Database(unrecorded, embedding_dim=768)

    def test_newer_schema_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "future.db"
        with Database(path) as database:
            database.connection().execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
        with pytest.raises(DatabaseError, match="newer"):
            Database(path)


class TestRepository:
    def test_units_of_reads_a_sections_passages_in_ordinal_order(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        section = replace(draft("A", "## A\n\nfirst\n\nsecond"), units=("first", "second"))
        document = store(db, fake_embedder, "/d/a.md", [section])
        (stored,) = db.get_sections(document.id)
        # Row ids follow insertion; the ordinal is what says where a passage sits.
        with db.transaction() as conn:
            conn.execute(
                "UPDATE units SET ordinal = 1 - ordinal WHERE section_id = ?", (stored.id,)
            )
        assert db.units_of(stored.id) == ["second", "first"]
        assert db.units_of(stored.id + 1) == []

    def test_replace_document_populates_all_three_indexes(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        document = store(db, fake_embedder, "/docs/a.md", SECTIONS)
        assert db.get_document("/docs/a.md") == document
        assert db.count_rows("documents") == 1
        assert db.count_rows("sections") == 3
        assert db.count_rows("sections_fts") == 3
        assert db.count_rows("sections_vec") == 3
        stored = db.get_sections(document.id)
        assert [s.heading_path for s in stored] == [s.heading_path for s in SECTIONS]
        assert [s.content for s in stored] == [s.content for s in SECTIONS]
        assert [(s.start_line, s.end_line) for s in stored] == [(1, 3), (5, 7), (9, 11)]

    def test_fts_and_vector_primitives(self, db: Database, fake_embedder: FakeEmbedder) -> None:
        document = store(db, fake_embedder, "/docs/a.md", SECTIONS)
        ids = {s.heading_title: s.id for s in db.get_sections(document.id)}
        assert db.fts_search('"ORBIT_MAX_RETRIES"', 10) == [ids["Retries"]]
        assert db.fts_search('"sigterm"', 10) == [ids["Shutdown"]]
        nearest = db.vec_search(fake_embedder.embed_query("drains connections on SIGTERM"), 3)
        assert nearest[0][0] == ids["Shutdown"]
        distances = [distance for _, distance in nearest]
        assert distances == sorted(distances)
        assert 0.0 <= distances[0] < distances[-1] <= 2.0  # cosine distance

    def test_porter_stemming_is_active(self, db: Database, fake_embedder: FakeEmbedder) -> None:
        store(db, fake_embedder, "/docs/a.md", SECTIONS)
        assert len(db.fts_search('"draining"', 10)) == 1
        assert len(db.fts_search('"connection"', 10)) == 1

    def test_deleting_a_document_cascades_to_sections_fts_and_vectors(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        store(db, fake_embedder, "/docs/a.md", SECTIONS)
        keep = store(db, fake_embedder, "/docs/b.md", [draft("Other", "## Other\n\nunrelated")])
        assert db.delete_documents(["/docs/a.md", "/docs/missing.md"]) == 1
        assert db.count_rows("documents") == 1
        assert db.count_rows("sections") == 1
        assert db.count_rows("sections_fts") == 1
        assert db.count_rows("sections_vec") == 1
        assert db.fts_search('"ORBIT_MAX_RETRIES"', 10) == []
        survivors = [sid for sid, _ in db.vec_search(fake_embedder.embed_query("retry"), 10)]
        assert survivors == [s.id for s in db.get_sections(keep.id)]
        db.connection().execute(
            "INSERT INTO sections_fts(sections_fts, rank) VALUES ('integrity-check', 1)"
        )

    def test_raw_sql_delete_also_cascades(self, db: Database, fake_embedder: FakeEmbedder) -> None:
        """The cascade is enforced by the schema itself, not by repository code."""
        store(db, fake_embedder, "/docs/a.md", SECTIONS)
        with db.transaction() as conn:
            conn.execute("DELETE FROM documents WHERE file_path = '/docs/a.md'")
        for table in ("sections", "sections_fts", "sections_vec"):
            assert db.count_rows(table) == 0

    def test_replacing_a_document_keeps_its_id_and_swaps_sections(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        first = store(db, fake_embedder, "/docs/a.md", SECTIONS)
        second = store(db, fake_embedder, "/docs/a.md", [draft("New", "## New\n\nfresh words")])
        assert second.id == first.id
        assert [s.heading_title for s in db.get_sections(first.id)] == ["New"]
        assert db.count_rows("sections_vec") == 1
        assert db.fts_search('"installer"', 10) == []
        assert len(db.fts_search('"fresh"', 10)) == 1
        db.connection().execute(
            "INSERT INTO sections_fts(sections_fts, rank) VALUES ('integrity-check', 1)"
        )

    def test_document_without_sections(self, db: Database, fake_embedder: FakeEmbedder) -> None:
        document = store(db, fake_embedder, "/docs/empty.md", [])
        assert db.get_sections(document.id) == []
        assert db.list_documents()[0].section_count == 0

    def test_failed_transaction_rolls_back_everything(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        store(db, fake_embedder, "/docs/a.md", SECTIONS)
        bad_vectors = vectors_for(fake_embedder, SECTIONS[:1])
        with pytest.raises(DatabaseError, match="vector sets"):
            db.replace_document(
                file_path="/docs/a.md", title="T", content_hash="new", last_modified=1, mtime_ns=1,
                sections=SECTIONS, vectors=bad_vectors,
            )  # fmt: skip

        class BoomError(RuntimeError):
            pass

        with pytest.raises(BoomError), db.transaction() as conn:
            conn.execute("DELETE FROM documents")
            raise BoomError
        assert db.count_rows("documents") == 1
        assert db.count_rows("sections_vec") == 3
        current = db.get_document("/docs/a.md")
        assert current is not None
        assert current.content_hash == "hash-/docs/a.md"
        assert not db.connection().in_transaction

    def test_wrong_embedding_width_is_rejected_before_writing(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        with pytest.raises(DatabaseError, match="dimensions"):
            db.replace_document(
                file_path="/docs/a.md", title="T", content_hash="h", last_modified=1, mtime_ns=1,
                sections=SECTIONS[:1],
                vectors=[SectionVectors(section=[0.1, 0.2], units=([0.1, 0.2],))],
            )  # fmt: skip
        assert db.count_rows("documents") == 0
        with pytest.raises(DatabaseError, match="dimensions"):
            db.vec_search([1.0, 0.0], 5)

    def test_data_survives_close_and_reopen(
        self, tmp_path: Path, fake_embedder: FakeEmbedder
    ) -> None:
        path = tmp_path / "persist.db"
        with Database(path) as first:
            store(first, fake_embedder, "/docs/a.md", SECTIONS)
        with Database(path) as second:
            assert second.pragma("journal_mode") == "wal"
            document = second.get_document("/docs/a.md")
            assert document is not None
            assert len(second.get_sections(document.id)) == 3
            assert len(second.fts_search('"installer"', 5)) == 1
            assert len(second.vec_search(fake_embedder.embed_query("installer"), 5)) == 3

    def test_reader_in_another_thread_sees_committed_writes(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        store(db, fake_embedder, "/docs/a.md", SECTIONS)
        counts: list[int] = []
        thread = threading.Thread(target=lambda: counts.append(db.count_rows("sections")))
        thread.start()
        thread.join()
        assert counts == [3]

    def test_directory_filter_does_not_match_sibling_prefixes(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        for path in ("/work/docs/a.md", "/work/docs/sub/b.md", "/work/docs2/c.md"):
            store(db, fake_embedder, path, SECTIONS[:1])
        listed = [d.file_path for d in db.list_documents("/work/docs")]
        assert listed == ["/work/docs/a.md", "/work/docs/sub/b.md"]
        assert set(db.document_hashes("/work/docs")) == set(listed)
        assert len(db.list_documents()) == 3
        assert [d.section_count for d in db.list_documents()] == [1, 1, 1]

    def test_like_wildcards_in_paths_are_literal(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        store(db, fake_embedder, "/work/100%_done/a.md", SECTIONS[:1])
        store(db, fake_embedder, "/work/100x_done/b.md", SECTIONS[:1])
        assert [d.file_path for d in db.list_documents("/work/100%_done")] == [
            "/work/100%_done/a.md"
        ]

    def test_suffix_lookup(self, db: Database, fake_embedder: FakeEmbedder) -> None:
        for path in ("/w/docs/guide.md", "/w/other/guide.md", "/w/docs/myguide.md"):
            store(db, fake_embedder, path, SECTIONS[:1])
        assert [d.file_path for d in db.find_documents_by_suffix("docs/guide.md")] == [
            "/w/docs/guide.md"
        ]
        assert len(db.find_documents_by_suffix("guide.md")) == 2  # not myguide.md

    def test_hydration_joins_sections_with_their_documents(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        document = store(db, fake_embedder, "/docs/a.md", SECTIONS)
        ids = [s.id for s in db.get_sections(document.id)]
        hydrated = db.get_sections_with_documents([*ids, 12345])
        assert set(hydrated) == set(ids)
        section, owner = hydrated[ids[1]]
        assert section.heading_title == "Retries"
        assert owner == document

    def test_clear_empties_every_index(self, db: Database, fake_embedder: FakeEmbedder) -> None:
        store(db, fake_embedder, "/docs/a.md", SECTIONS)
        assert db.clear() == 1
        for table in ("documents", "sections", "sections_fts", "sections_vec"):
            assert db.count_rows(table) == 0


def test_embedding_serialization_is_little_endian_float32() -> None:
    blob = serialize_embedding([1.0, -2.5])
    assert blob == b"\x00\x00\x80\x3f\x00\x00\x20\xc0"
