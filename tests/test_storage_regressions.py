"""Storage regressions: connections, migrations, vectors, indexing, integrity, notices.

One test per defect found in code review; each fails when its fix is reverted.
"""

from __future__ import annotations

import fcntl
import gzip
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
from collections.abc import Sequence
from pathlib import Path

import pytest
from fakes import FakeEmbedder, vectors_for
from helpers import draft, store
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

import markdown_memory.indexer as indexer_module
from markdown_memory.db import SCHEMA_VERSION, Database
from markdown_memory.exceptions import (
    DatabaseError,
    IndexBusyError,
    IndexingError,
    ModelLoadError,
)
from markdown_memory.indexer import Indexer, iter_markdown_files
from markdown_memory.models import (
    SectionVectors,
)
from markdown_memory.search import HybridSearcher
from markdown_memory.server import MarkdownMemoryService, ServerConfig, create_server


class TestConnectionLifecycle:
    def test_connections_of_finished_threads_are_closed(self, db: Database) -> None:
        opened: list[sqlite3.Connection] = []

        def work() -> None:
            opened.append(db.connection())
            db.count_rows("documents")

        for _ in range(6):
            thread = threading.Thread(target=work)
            thread.start()
            thread.join()
        # main thread + the most recent worker (reaped when the next connection opens)
        assert db.open_connection_count == 2
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            opened[0].execute("SELECT 1")
        opened[-1].execute("SELECT 1")  # not reaped yet, still usable until close()

    @pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="needs /proc")
    def test_file_descriptors_do_not_grow_across_thread_generations(self, db: Database) -> None:
        def descriptors() -> int:
            count = 0
            for entry in Path("/proc/self/fd").iterdir():
                try:
                    if os.readlink(entry).startswith(str(db.path)):
                        count += 1
                except OSError:
                    continue
            return count

        def generation() -> None:
            threads = [
                threading.Thread(target=lambda: db.count_rows("documents")) for _ in range(5)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        for _ in range(10):
            generation()
        # Connections of one finished generation linger until the next connection opens,
        # so the count is bounded by two generations (+ this thread), not by their total.
        # Without reaping this reaches 51 connections and 100+ descriptors.
        assert db.open_connection_count <= 1 + 2 * 5
        assert descriptors() <= 3 * (1 + 2 * 5)  # <= 3 fds per connection: db, -wal, -shm


class TestMigrationRace:
    def test_processes_racing_to_create_the_schema_all_succeed(self, tmp_path: Path) -> None:
        program = (
            "import sys; from markdown_memory.db import Database; "
            "db = Database(sys.argv[1]); print(db.pragma('journal_mode')); db.close()"
        )
        for attempt in range(4):
            target = tmp_path / f"race-{attempt}.db"
            processes = [
                subprocess.Popen(
                    [sys.executable, "-c", program, str(target)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )  # fmt: skip
                for _ in range(4)
            ]
            for process in processes:
                out, err = process.communicate(timeout=120)
                assert process.returncode == 0, err
                assert out.strip() == "wal"
            with Database(target) as database:
                assert database.get_meta("embedding_dim") == "384"


class TestVectorValidation:
    @pytest.mark.parametrize("bad", [0.0, math.nan, math.inf])
    def test_degenerate_embeddings_are_rejected_on_write(
        self, db: Database, fake_embedder: FakeEmbedder, bad: float
    ) -> None:
        vector = [bad] * db.embedding_dim if bad == 0.0 else [1.0, bad] + [0.0] * 382
        with pytest.raises(DatabaseError, match="zeros or contains NaN/inf"):
            db.replace_document(
                file_path="/d/a.md", title="T", content_hash="h", last_modified=1,
                sections=[draft("A", "## A\n\nbody")],
                vectors=[SectionVectors(section=vector, units=(vector,))],
            )  # fmt: skip
        assert db.count_rows("documents") == 0

    def test_degenerate_query_vector_is_a_domain_error(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        store(db, fake_embedder, "/d/a.md")
        with pytest.raises(DatabaseError, match="zeros or contains NaN/inf"):
            db.vec_search([0.0] * db.embedding_dim, 5)

    def test_search_falls_back_to_keywords_when_the_query_vector_is_degenerate(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        class ZeroQueries(FakeEmbedder):
            def embed_query(self, text: str) -> list[float]:
                return [0.0] * self.dimension

        store(db, fake_embedder, "/d/a.md")
        searcher = HybridSearcher(db, ZeroQueries())
        try:
            results = searcher.search("body number")
        finally:
            searcher.close()
        assert results and all(r.vec_rank is None for r in results)


def test_fts_row_count_reads_the_index_not_the_content_table(
    db: Database, fake_embedder: FakeEmbedder
) -> None:
    store(db, fake_embedder, "/d/a.md")
    assert db.count_rows("sections_fts") == 3
    db.connection().execute("INSERT INTO sections_fts(sections_fts) VALUES ('delete-all')")
    assert db.count_rows("sections") == 3
    assert db.count_rows("sections_fts") == 0  # COUNT(*) on the FTS table would still say 3


class TestIndexerSafety:
    def test_fifo_named_like_markdown_does_not_hang_the_run(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        if not hasattr(os, "mkfifo"):
            pytest.skip("no FIFOs on this platform")
        root = tmp_path / "docs"
        root.mkdir()
        (root / "real.md").write_text("# Real\n")
        os.mkfifo(root / "pipe.md")
        reports = []
        worker = threading.Thread(
            target=lambda: reports.append(Indexer(db, fake_embedder).index_directory(root)),
            daemon=True,
        )
        worker.start()
        worker.join(timeout=20)
        assert not worker.is_alive(), "index_directory blocked on a FIFO"
        report = reports[0]
        assert report.files_indexed == 1
        assert [(Path(e.file_path).name, e.message) for e in report.errors] == [
            ("pipe.md", "Not a regular file; skipped")
        ]

    def test_size_cap_does_not_rely_on_st_size(
        self,
        db: Database,
        fake_embedder: FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(indexer_module, "MAX_FILE_BYTES", 32)
        root = tmp_path / "docs"
        root.mkdir()
        (root / "big.md").write_text("# Big\n\n" + "x" * 100)
        (root / "small.md").write_text("# Small\n")
        real_stat = Path.stat

        def lying_stat(self: Path, **kwargs: bool) -> os.stat_result:
            result = real_stat(self, **kwargs)
            if self.name != "big.md":
                return result
            fields = list(result)
            fields[6] = 0  # st_size, as reported for /proc files, devices and growing files
            return os.stat_result(fields)

        monkeypatch.setattr(Path, "stat", lying_stat)
        report = Indexer(db, fake_embedder).index_directory(root)
        assert report.files_indexed == 1
        assert [Path(e.file_path).name for e in report.errors] == ["big.md"]
        assert "larger than 32 bytes" in report.errors[0].message

    @pytest.mark.skipif(
        hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores directory permissions"
    )
    def test_unlistable_directory_is_reported_and_its_documents_are_kept(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        root = tmp_path / "docs"
        (root / "locked").mkdir(parents=True)
        (root / "locked" / "secret.md").write_text("# Secret\n")
        (root / "open.md").write_text("# Open\n")
        indexer = Indexer(db, fake_embedder)
        assert indexer.index_directory(root).files_indexed == 2
        (root / "locked").chmod(0o000)
        try:
            report = indexer.index_directory(root)
        finally:
            (root / "locked").chmod(0o755)
        assert report.files_purged == 0
        assert db.get_document(str(root / "locked" / "secret.md")) is not None
        assert [e.message.split(":")[0] for e in report.errors] == ["Cannot list directory"]
        assert report.errors[0].file_path == str(root / "locked")

    def test_documents_indexed_inside_a_pruned_tree_survive_indexing_an_ancestor(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        project = tmp_path / "proj"
        dependency_docs = project / "node_modules" / "pkg" / "docs"
        dependency_docs.mkdir(parents=True)
        (dependency_docs / "api.md").write_text("# Dependency API\n")
        (project / "README.md").write_text("# Project\n")
        indexer = Indexer(db, fake_embedder)
        assert indexer.index_directory(dependency_docs).files_indexed == 1
        report = indexer.index_directory(project)
        assert (report.files_indexed, report.files_purged) == (1, 0)
        assert db.get_document(str(dependency_docs / "api.md")) is not None
        (dependency_docs / "api.md").unlink()  # its own root still notices real deletions
        assert indexer.index_directory(dependency_docs).files_purged == 1

    def test_model_change_reports_everything_it_discarded(
        self, db: Database, tmp_path: Path
    ) -> None:
        first, second = tmp_path / "one", tmp_path / "two"
        for root in (first, second):
            root.mkdir()
            (root / "doc.md").write_text(f"# {root.name}\n")
        old = Indexer(db, FakeEmbedder(model_name="model-a"))
        old.index_directory(first)
        old.index_directory(second)
        report = Indexer(db, FakeEmbedder(model_name="model-b")).index_directory(first)
        assert len(report.notes) == 1
        assert "model-a -> model-b" in report.notes[0]
        assert "discarded all 2 previously indexed documents" in report.notes[0]
        assert "NOTE Embedding model changed" in report.summary()
        assert [Path(d.file_path).parent.name for d in db.list_documents()] == ["one"]
        assert Indexer(db, FakeEmbedder(model_name="model-b")).index_directory(first).notes == ()

    def test_undecodable_file_name_fails_alone_and_stays_json_safe(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        root = tmp_path / "docs"
        root.mkdir()
        (root / "good.md").write_text("# Good\n")
        try:
            with open(os.path.join(os.fsencode(root), b"caf\xe9.md"), "wb") as handle:
                handle.write(b"# Bad name\n")
        except OSError:
            pytest.skip("file system rejects non-UTF-8 names")
        report = Indexer(db, fake_embedder).index_directory(root)
        assert report.files_indexed == 1
        assert [e.message for e in report.errors] == ["File name is not valid UTF-8; skipped"]
        report.summary().encode("utf-8")  # must be sendable as JSON: no lone surrogates
        assert "caf�.md" in report.errors[0].file_path
        assert Indexer(db, fake_embedder).index_directory(root).files_unchanged == 1


class TestWholeRunFailures:
    def test_unloadable_model_aborts_the_run_instead_of_failing_every_file(
        self, db: Database, tmp_path: Path
    ) -> None:
        class NoModel(FakeEmbedder):
            def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
                self.document_calls.append(list(texts))
                raise ModelLoadError("Cannot load embedding model: offline")

        root = tmp_path / "docs"
        root.mkdir()
        for n in range(5):
            (root / f"doc{n}.md").write_text(f"# Doc {n}\n")
        embedder = NoModel()
        with pytest.raises(ModelLoadError, match="offline"):
            Indexer(db, embedder).index_directory(root)
        assert len(embedder.document_calls) == 1  # not retried once per file

    async def test_directory_with_undecodable_name_is_a_domain_error(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        raw = os.path.join(os.fsencode(tmp_path), b"caf\xe9-docs")
        try:
            os.mkdir(raw)
        except OSError:
            pytest.skip("file system rejects non-UTF-8 names")
        with open(os.path.join(raw, b"a.md"), "wb") as handle:
            handle.write(b"# A\n")
        bad_root = Path(os.fsdecode(raw))
        with pytest.raises(IndexingError, match="not valid UTF-8") as raised:
            Indexer(db, fake_embedder).index_directory(bad_root)
        str(raised.value).encode("utf-8")  # the message itself must be sendable
        service = MarkdownMemoryService(
            ServerConfig(db_path=tmp_path / "s.db", docs_dir=tmp_path), embedder=FakeEmbedder()
        )
        try:
            server = create_server(service=service)
            for tool, arguments in (
                ("index_directory", {"directory": str(bad_root)}),
                ("list_documents", {"directory": str(bad_root)}),
                ("get_document_outline", {"file_path": str(bad_root / "a.md")}),
            ):
                with pytest.raises(ToolError) as failure:
                    await server.call_tool(tool, arguments)
                assert not isinstance(failure.value, UnexpectedToolError), tool
        finally:
            service.close()


class TestMigrationInvalidatesTheIndex:
    @staticmethod
    def rewind_to_v1(path: Path) -> None:
        """Turn a populated current database into what the v1 release left behind."""
        with Database(path) as database:
            conn = database.connection()
            conn.execute("DROP TRIGGER units_after_delete")
            conn.execute("DROP TABLE units_vec")
            conn.execute("DROP TABLE units")
            conn.execute("DROP TABLE index_failures")
            conn.execute("DROP TABLE index_coverage")
            conn.execute("ALTER TABLE documents DROP COLUMN vector_format")
            conn.execute("PRAGMA user_version = 1")

    def test_v1_documents_are_dropped_so_reindexing_rebuilds_them_with_passages(
        self, tmp_path: Path, fake_embedder: FakeEmbedder
    ) -> None:
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.md").write_text("# A\n\nalpha body\n")
        path = tmp_path / "old.db"
        with Database(path) as database:
            Indexer(database, fake_embedder).index_directory(docs)
        self.rewind_to_v1(path)
        with Database(path) as migrated:
            assert migrated.count_rows("documents") == 0
            assert migrated.integrity_problems() == []
            report = Indexer(migrated, fake_embedder).index_directory(docs)
            assert (report.files_indexed, report.files_unchanged) == (1, 0)  # not skipped by hash
            assert report.passages_indexed == 1
            assert len(report.notes) == 1
            assert "discarded all 1 previously indexed documents" in report.notes[0]
            assert "passage-level vectors" in report.notes[0]
            assert migrated.integrity_problems() == []
            assert Indexer(migrated, fake_embedder).index_directory(docs).notes == ()  # told once

    @staticmethod
    def rewind_to_v2(path: Path) -> None:
        """Turn a current database into what the previous release left behind."""
        with Database(path) as database:
            conn = database.connection()
            conn.execute("DROP TABLE index_failures")
            conn.execute("DROP TABLE index_coverage")
            conn.execute("ALTER TABLE documents DROP COLUMN vector_format")
            conn.execute("PRAGMA user_version = 2")

    def test_a_v2_database_keeps_everything_and_rebuilds_its_vectors_in_place(
        self, tmp_path: Path, fake_embedder: FakeEmbedder
    ) -> None:
        """A section vector used to mean one thing and now means another.

        It was one embedding of the whole section, cut off at the model's token limit; it
        is now the mean of the section's passage vectors. Files are skipped by content
        hash, so an untouched file would keep whatever it was given and the table would
        answer one query two different ways.

        Nothing is destroyed to fix that. Opening the database changes no data - the
        documents, their sections and the FTS index are all still there and still
        answering - and every document simply carries the older format until the next run
        of its root rebuilds it.
        """
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.md").write_text("# A\n\nalpha body\n")
        path = tmp_path / "old.db"
        with Database(path) as database:
            Indexer(database, fake_embedder).index_directory(docs)
            before = database.count_rows("documents"), database.count_rows("units")
        self.rewind_to_v2(path)

        with Database(path) as migrated:
            assert (migrated.count_rows("documents"), migrated.count_rows("units")) == before
            assert migrated.integrity_problems() == []
            assert migrated.pending_notices() == {}, "an upgrade that changes nothing says nothing"
            # Nothing vouches for it, because nothing has walked it under this format.
            assert not migrated.index_status(str(docs)).verified

            report = Indexer(migrated, fake_embedder).index_directory(docs)
            assert (report.files_indexed, report.files_unchanged) == (1, 0)  # not skipped
            assert migrated.index_status(str(docs)).verified
            assert migrated.integrity_problems() == []

            again = Indexer(migrated, fake_embedder).index_directory(docs)
            assert (again.files_indexed, again.files_unchanged) == (0, 1), "rebuilt twice"

    def test_dimension_change_reports_how_many_documents_it_really_discarded(
        self, tmp_path: Path
    ) -> None:
        first, second = tmp_path / "one", tmp_path / "two"
        for root in (first, second):
            root.mkdir()
            (root / "doc.md").write_text(f"# {root.name}\n\nbody\n")
        path = tmp_path / "switch.db"
        small = FakeEmbedder(dimension=384, model_name="small")
        with Database(path, embedding_dim=384) as database:
            indexer = Indexer(database, small)
            indexer.index_directory(first)
            indexer.index_directory(second)
        wide = FakeEmbedder(dimension=768, model_name="wide")
        with Database(path, embedding_dim=768) as database:
            report = Indexer(database, wide).index_directory(first)
        assert len(report.notes) == 1
        assert "384 -> 768 dimensions" in report.notes[0]
        assert "discarded all 2 previously indexed documents" in report.notes[0]
        assert "discarded all 0" not in report.summary()


class TestIntegrityCheckUnderContention:
    def test_a_locked_database_is_unverified_not_corrupt(
        self, db: Database, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store(db, fake_embedder, "/d/a.md")
        monkeypatch.setattr("markdown_memory.db._BUSY_TIMEOUT_MS", 200)
        other = sqlite3.connect(db.path, isolation_level=None, timeout=0.2)
        db.connection().execute("PRAGMA busy_timeout = 200")
        other.execute("BEGIN IMMEDIATE")  # another process is in the middle of indexing
        try:
            problems = db.integrity_problems()
        finally:
            other.execute("ROLLBACK")
            other.close()
        assert len(problems) == 1
        assert "could not verify the FTS5 index" in problems[0]
        assert "not a sign of damage" in problems[0]
        assert db.integrity_problems() == []  # and once the writer is done, all is well


class TestNoticesSurviveAnAbortedRun:
    class NoModel(FakeEmbedder):
        def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
            raise ModelLoadError("Cannot load embedding model: offline")

    @staticmethod
    def two_roots(tmp_path: Path) -> tuple[Path, Path]:
        first, second = tmp_path / "one", tmp_path / "two"
        for root in (first, second):
            root.mkdir()
            (root / "doc.md").write_text(f"# {root.name}\n\nbody\n")
        return first, second

    def test_dimension_change_notice_is_kept_until_a_run_reports_it(self, tmp_path: Path) -> None:
        first, second = self.two_roots(tmp_path)
        path = tmp_path / "switch.db"
        with Database(path, embedding_dim=384) as database:
            indexer = Indexer(database, FakeEmbedder(dimension=384, model_name="small"))
            indexer.index_directory(first)
            indexer.index_directory(second)
        with Database(path, embedding_dim=768) as database, pytest.raises(ModelLoadError):
            Indexer(database, self.NoModel(768, "wide")).index_directory(first)
        with Database(path, embedding_dim=768) as database:
            indexer = Indexer(database, FakeEmbedder(dimension=768, model_name="wide"))
            report = indexer.index_directory(first)
            assert len(report.notes) == 1
            assert "discarded all 2 previously indexed documents" in report.notes[0]
            assert indexer.index_directory(first).notes == ()  # told once

    def test_model_change_notice_is_kept_until_a_run_reports_it(self, tmp_path: Path) -> None:
        first, second = self.two_roots(tmp_path)
        with Database(tmp_path / "model.db") as database:
            indexer = Indexer(database, FakeEmbedder(model_name="old"))
            indexer.index_directory(first)
            indexer.index_directory(second)
            with pytest.raises(ModelLoadError):
                Indexer(database, self.NoModel(model_name="new")).index_directory(first)
            assert database.count_rows("documents") == 0
            report = Indexer(database, FakeEmbedder(model_name="new")).index_directory(first)
            assert len(report.notes) == 1
            assert "Embedding model changed (old -> new)" in report.notes[0]
            assert "discarded all 2 previously indexed documents" in report.notes[0]

    def test_notice_added_after_a_partial_dismissal_gets_a_fresh_key(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        for number in range(3):
            store(db, fake_embedder, f"/d/{number}.md", count=1)
            assert db.clear(notice=lambda count, n=number: f"run {n}: dropped {count}") == 1
            if number == 1:
                db.dismiss_notices(["notice:0000"])
        assert list(db.pending_notices().values()) == ["run 1: dropped 1", "run 2: dropped 1"]
        assert db.clear(notice=lambda count: "nothing was dropped") == 0
        assert len(db.pending_notices()) == 2  # an empty index is not announced


class TestSymlinkedDirectories:
    def test_a_symlink_back_to_an_ancestor_neither_hangs_nor_indexes_twice(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        root = tmp_path / "docs"
        (root / "sub").mkdir(parents=True)
        (root / "sub" / "a.md").write_text("# A\n\nalpha body\n")
        try:
            (root / "sub" / "back").symlink_to(root, target_is_directory=True)
        except OSError:  # pragma: no cover - Windows without developer mode
            pytest.skip("this file system does not support symlinks")
        report = Indexer(db, fake_embedder).index_directory(root)  # must terminate
        assert (report.files_scanned, report.files_indexed) == (1, 1)

    def test_a_file_reachable_only_through_a_symlinked_directory_is_not_indexed(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        root, outside = tmp_path / "docs", tmp_path / "outside"
        root.mkdir()
        outside.mkdir()
        (root / "here.md").write_text("# Here\n\nbody\n")
        (outside / "there.md").write_text("# There\n\nbody\n")
        try:
            (root / "link").symlink_to(outside, target_is_directory=True)
        except OSError:  # pragma: no cover - Windows without developer mode
            pytest.skip("this file system does not support symlinks")
        report = Indexer(db, fake_embedder).index_directory(root)
        assert report.files_scanned == 1
        indexed = [document.file_path for document in db.list_documents()]
        assert indexed == [str(root / "here.md")]


class TestSectionIdsAreNeverReused:
    def test_a_deleted_section_does_not_lend_its_id_to_the_next_one(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        def ids() -> list[int]:
            return [int(row[0]) for row in db.connection().execute("SELECT id FROM sections")]

        store(db, fake_embedder, "/d/a.md", count=3)
        first = ids()
        db.delete_documents(["/d/a.md"])
        store(db, fake_embedder, "/d/b.md", count=3)
        # SQLite hands a deleted rowid out again; a search that ranked the old ids would
        # then fetch whatever took their place instead of noticing they are gone.
        assert not set(first) & set(ids())

    def test_ids_keep_rising_across_reopens(
        self, tmp_path: Path, fake_embedder: FakeEmbedder
    ) -> None:
        path = tmp_path / "ids.db"
        seen: set[int] = set()
        for round_number in range(3):
            with Database(path) as database:
                store(database, fake_embedder, f"/d/{round_number}.md", count=2)
                ids = {int(r[0]) for r in database.connection().execute("SELECT id FROM sections")}
                assert not seen & ids
                seen |= ids
                database.delete_documents([f"/d/{round_number}.md"])


class TestSectionIdsOnAnUpgradedDatabase:
    """A database written before the high-water mark existed must not reuse ids either."""

    @staticmethod
    def section_ids(db: Database) -> list[int]:
        return [int(row[0]) for row in db.connection().execute("SELECT id FROM sections")]

    @staticmethod
    def downgrade(db: Database) -> None:
        """What the previous release left behind: rows, but no high-water mark in meta."""
        with db.transaction() as conn:
            conn.execute("DELETE FROM meta WHERE key = 'next_section_id'")

    def test_reindexing_does_not_hand_back_the_ids_it_just_deleted(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        first = [draft(f"S{n}", f"## S{n}\n\nbody {n}") for n in range(3)]
        db.replace_document(
            file_path="/d/a.md", title="D", content_hash="h1", last_modified=1,
            sections=first, vectors=vectors_for(fake_embedder, first),
        )  # fmt: skip
        before = self.section_ids(db)
        self.downgrade(db)
        second = [draft(f"T{n}", f"## T{n}\n\nother {n}") for n in range(3)]
        db.replace_document(
            file_path="/d/a.md", title="D", content_hash="h2", last_modified=2,
            sections=second, vectors=vectors_for(fake_embedder, second),
        )  # fmt: skip
        # The mark is sampled before the document's old sections are deleted.
        assert not set(before) & set(self.section_ids(db))

    def test_opening_an_older_database_records_the_mark_before_any_write(
        self, tmp_path: Path, fake_embedder: FakeEmbedder
    ) -> None:
        path = tmp_path / "upgrade.db"
        with Database(path) as database:
            store(database, fake_embedder, "/d/a.md", count=3)
            highest = max(self.section_ids(database))
            self.downgrade(database)
        with Database(path) as upgraded:
            assert upgraded.get_meta("next_section_id") == str(highest + 1)
            upgraded.delete_documents(["/d/a.md"])  # purge, then index something else
            store(upgraded, fake_embedder, "/d/b.md", count=3)
            assert min(self.section_ids(upgraded)) > highest


class TestNoticeNumberingBeyondFourDigits:
    def test_the_ten_thousandth_notice_does_not_collide(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        with db.transaction() as conn:  # as if 10_000 notices had been raised over time
            conn.execute("INSERT INTO meta(key, value) VALUES ('notice:9999', 'older')")
            conn.execute("INSERT INTO meta(key, value) VALUES ('notice:10000', 'newer')")
        store(db, fake_embedder, "/d/a.md", count=1)
        assert db.clear(notice=lambda count: f"newest: dropped {count}") == 1
        # 'notice:10000' sorts below 'notice:9999' as text: both the successor key and the
        # delivery order have to be computed numerically.
        assert list(db.pending_notices().values()) == ["older", "newer", "newest: dropped 1"]


class TestMigratingARealOldDatabase:
    """Databases written by earlier releases, kept as fixtures and migrated forward.

    The other migration tests simulate an old database by dropping tables from a current
    one, which tests the simulation. These files were written by the schema statements of
    their own version, with the rows that version wrote and no key it never wrote - the
    missing ``next_section_id`` is how the section-id reuse bug reached users.
    """

    FIXTURES = Path(__file__).parent / "fixtures" / "databases"

    @staticmethod
    def unpack(archive: Path, destination: Path) -> Path:
        """Opening a database migrates it, so every test gets its own copy."""
        path = destination / archive.with_suffix("").name
        with gzip.open(archive, "rb") as packed, path.open("wb") as raw:
            shutil.copyfileobj(packed, raw)
        return path

    @pytest.fixture(params=[1, 2])
    def old_database(self, request: pytest.FixtureRequest, tmp_path: Path) -> Path:
        return self.unpack(self.FIXTURES / f"v{request.param}_seed.db.gz", tmp_path)

    def test_the_fixture_is_the_version_it_claims(self, old_database: Path) -> None:
        conn = sqlite3.connect(old_database)
        try:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            tables = {
                str(row[0])
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
            keys = {str(row[0]) for row in conn.execute("SELECT key FROM meta")}
        finally:
            conn.close()
        assert version == int(old_database.stem[1])
        assert ("units" in tables) == (version >= 2)
        assert "next_section_id" not in keys  # no release wrote one before this fix

    def test_opening_it_migrates_to_the_current_schema(self, old_database: Path) -> None:
        with Database(old_database) as database:
            assert database.get_meta("embedding_dim") == "384"
            conn = database.connection()
            assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == SCHEMA_VERSION
            assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
            assert database.integrity_problems() == []

    def test_the_migrated_schema_matches_a_fresh_database(
        self, old_database: Path, tmp_path: Path
    ) -> None:
        def schema(path: Path) -> list[tuple[str, str]]:
            with Database(path) as database:
                rows = database.connection().execute(
                    "SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY name"
                )
                return [(str(name), " ".join(str(sql).split())) for name, sql in rows]

        assert schema(old_database) == schema(tmp_path / "fresh.db")

    def test_reindexing_after_the_upgrade_does_not_reuse_section_ids(
        self, old_database: Path, fake_embedder: FakeEmbedder
    ) -> None:
        def section_ids(database: Database) -> set[int]:
            rows = database.connection().execute("SELECT id FROM sections")
            return {int(row[0]) for row in rows}

        with Database(old_database) as database:
            before = section_ids(database)
            database.delete_documents(
                [document.file_path for document in database.list_documents()]
            )
            store(database, fake_embedder, "/docs/gamma.md", count=3)
            assert not before & section_ids(database)

    def test_a_v1_database_is_told_why_its_documents_are_gone(self, tmp_path: Path) -> None:
        # v1 had no passage vectors, so the v2 migration discards the index and says so.
        path = self.unpack(self.FIXTURES / "v1_seed.db.gz", tmp_path)
        with Database(path) as database:
            assert database.count_rows("documents") == 0
            notices = list(database.pending_notices().values())
        assert len(notices) == 1
        assert "discarded all 2 previously indexed documents" in notices[0]

    def test_a_v2_database_keeps_its_documents(self, tmp_path: Path) -> None:
        path = self.unpack(self.FIXTURES / "v2_seed.db.gz", tmp_path)
        with Database(path) as database:
            assert database.count_rows("documents") == 2
            assert database.count_rows("sections") == 4
            assert database.pending_notices() == {}


class TestExcludedPaths:
    """A repository's fixtures and vendored docs are not its documentation.

    Found by pointing the server at this repository: it would have indexed the 1,682
    sections of vendored evaluation corpus under scripts/eval_data as if they were the
    project's own docs.
    """

    @staticmethod
    def tree(root: Path) -> None:
        (root / "guide.md").write_text("# Guide\n\nreal documentation\n")
        for relative in ("fixtures/sample.md", "vendor/upstream/readme.md", "notes/keep.md"):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {path.stem}\n\nbody of {relative}\n")

    def indexed(self, db: Database, embedder: FakeEmbedder, root: Path, *exclude: str) -> set[str]:
        Indexer(db, embedder, exclude=exclude).index_directory(root)
        return {str(Path(document.file_path).relative_to(root)) for document in db.list_documents()}

    def test_adding_an_exclusion_purges_what_was_already_indexed(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """Excluding a tree has to clean up, not just stop adding.

        A repository that indexed its fixtures before the exclusion was configured would
        otherwise keep serving them from the database forever, and no amount of
        re-indexing would remove them.
        """
        self.tree(tmp_path)
        assert "fixtures/sample.md" in self.indexed(db, fake_embedder, tmp_path)
        report = Indexer(db, fake_embedder, exclude=("fixtures",)).index_directory(tmp_path)
        assert report.files_purged == 1
        remaining = {
            str(Path(document.file_path).relative_to(tmp_path)) for document in db.list_documents()
        }
        assert "fixtures/sample.md" not in remaining
        assert "guide.md" in remaining

    def test_nothing_is_excluded_by_default(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        self.tree(tmp_path)
        assert self.indexed(db, fake_embedder, tmp_path) == {
            "guide.md",
            "fixtures/sample.md",
            "vendor/upstream/readme.md",
            "notes/keep.md",
        }

    @pytest.mark.parametrize(
        ("patterns", "expected"),
        [
            (("fixtures",), {"guide.md", "vendor/upstream/readme.md", "notes/keep.md"}),
            (("fixtures/*",), {"guide.md", "vendor/upstream/readme.md", "notes/keep.md"}),
            (("vendor",), {"guide.md", "fixtures/sample.md", "notes/keep.md"}),
            (("fixtures", "vendor"), {"guide.md", "notes/keep.md"}),
            (("guide.md",), {"fixtures/sample.md", "vendor/upstream/readme.md", "notes/keep.md"}),
            (("*/sample.md",), {"guide.md", "vendor/upstream/readme.md", "notes/keep.md"}),
        ],
    )
    def test_patterns_match_paths_relative_to_the_docs_root(
        self,
        db: Database,
        fake_embedder: FakeEmbedder,
        tmp_path: Path,
        patterns: tuple[str, ...],
        expected: set[str],
    ) -> None:
        self.tree(tmp_path)
        assert self.indexed(db, fake_embedder, tmp_path, *patterns) == expected

    def test_an_excluded_directory_is_never_walked(self, tmp_path: Path) -> None:
        self.tree(tmp_path)
        walked = list(iter_markdown_files(tmp_path, None, ("vendor",)))
        assert all("vendor" not in path.parts for path in walked)

    def test_the_eval_corpus_can_be_kept_out_of_this_repository(self, db: Database) -> None:
        # The case that prompted the feature, run against the real tree.
        root = Path(__file__).parent.parent
        found = list(iter_markdown_files(root, None, ("scripts/eval_data", ".venv")))
        assert not any("eval_data" in path.parts for path in found)
        assert any(path.name == "CLAUDE.md" for path in found)

    def test_configuration_is_read_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("MARKDOWN_MEMORY_EXCLUDE", " fixtures , ./vendor/** , notes/ ")
        monkeypatch.setenv("MARKDOWN_MEMORY_DOCS_DIR", str(tmp_path))
        assert ServerConfig.from_env().exclude == ("fixtures", "vendor/**", "notes")

    def test_a_pattern_containing_a_colon_is_one_pattern(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A colon separator would silently exclude two things nobody asked for."""
        monkeypatch.setenv("MARKDOWN_MEMORY_EXCLUDE", "reports:old")
        monkeypatch.setenv("MARKDOWN_MEMORY_DOCS_DIR", str(tmp_path))
        assert ServerConfig.from_env().exclude == ("reports:old",)

    def test_a_dot_prefixed_name_is_not_mistaken_for_a_relative_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Stripping every leading "." and "/" turned `.hidden` into `hidden`.

        That left the directory the user named indexed, and excluded a different one.
        """
        monkeypatch.setenv("MARKDOWN_MEMORY_EXCLUDE", ".hidden, ./docs/build")
        monkeypatch.setenv("MARKDOWN_MEMORY_DOCS_DIR", str(tmp_path))
        assert ServerConfig.from_env().exclude == (".hidden", "docs/build")

    def test_a_bare_name_excludes_that_directory_at_any_depth(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """`eval_data` has to mean what it looks like it means.

        Anchoring every pattern at the root made a bare name match nothing below the top
        level: the pattern looked correct, excluded nothing, and the files were indexed.
        """
        self.tree(tmp_path)
        assert self.indexed(db, fake_embedder, tmp_path, "upstream") == {
            "guide.md",
            "fixtures/sample.md",
            "notes/keep.md",
        }

    def test_an_anchored_pattern_stays_anchored(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """A pattern with a separator is still relative to the docs root only."""
        (tmp_path / "deep" / "fixtures").mkdir(parents=True)
        (tmp_path / "deep" / "fixtures" / "nested.md").write_text("# Nested\n\nbody\n")
        self.tree(tmp_path)
        indexed = self.indexed(db, fake_embedder, tmp_path, "fixtures/*")
        assert "fixtures/sample.md" not in indexed
        assert "deep/fixtures/nested.md" in indexed

    def test_matching_does_not_depend_on_the_platform_case_rules(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """`fnmatch` folds case on some platforms; the docs tree must not."""
        self.tree(tmp_path)
        assert "fixtures/sample.md" in self.indexed(db, fake_embedder, tmp_path, "FIXTURES")


class TestAnIndexAnsweringFromAnotherModelsVectors:
    """A walk that read every file still cannot vouch for vectors another model built."""

    def test_a_recorded_mismatch_is_enough_to_withdraw_the_certificate(
        self, db: Database, tmp_path: Path, fake_embedder: FakeEmbedder
    ) -> None:
        """Tested on its own, without revoking coverage, because the two protect

        different things: coverage says the walk was whole, this says the vectors are
        the ones a query would be comparable with. A future caller that records the
        mismatch without revoking must not be able to report `verified` either.
        """
        root = tmp_path / "docs"
        root.mkdir()
        (root / "README.md").write_text("# Readme\n\nhello\n")
        Indexer(db, fake_embedder).index_directory(root)
        assert db.index_status(str(root)).verified

        db.record_weights_mismatch("another model's vectors are in here")
        status = db.index_status(str(root))
        assert not status.verified
        assert status.weights_mismatch == "another model's vectors are in here"
        assert status.message() == "another model's vectors are in here"

        db.record_weights_mismatch(None)
        assert db.index_status(str(root)).verified


class StopsAfterOneFile:
    """Embeds the first file, then behaves like a Ctrl-C partway through the second."""

    def __init__(self, inner: FakeEmbedder) -> None:
        self._inner = inner
        self._files = 0

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    @property
    def weights_revision(self) -> str | None:
        return self._inner.weights_revision

    def warm_up(self) -> None:
        self._inner.warm_up()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self._files += 1
        if self._files > 1:
            raise KeyboardInterrupt
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)


class RefusesToLoad:
    """The model is not on disk and there is no network: every file fails identically."""

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return "fake"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def weights_revision(self) -> str | None:
        return None

    def warm_up(self) -> None:
        raise ModelLoadError("the model is not on disk and there is no network")

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise ModelLoadError("Cannot load the embedding model")

    def embed_query(self, text: str) -> list[float]:
        raise ModelLoadError("Cannot load the embedding model")


class TestAPartialIndexSaysSo:
    """An index missing part of its tree answers as though it were whole.

    Indexing collected per-file failures and printed ERROR lines, then returned a
    success-shaped report: the errors were listed, but nothing durable said so, and the
    next session searched a tree with holes in it and was told nothing.
    """

    @staticmethod
    def tree(root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / "good.md").write_text("# Good\n\nreadable documentation\n")
        broken = root / "broken.md"
        broken.write_text("# Broken\n\nbody\n")
        broken.chmod(0o000)
        return broken

    def test_the_run_that_hits_it_calls_the_index_incomplete(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        broken = self.tree(tmp_path)
        try:
            report = Indexer(db, fake_embedder).index_directory(tmp_path)
            assert report.errors
            assert "INCOMPLETE" in report.summary()
            assert report.files_indexed == 1  # the readable file still landed
        finally:
            broken.chmod(0o644)

    def test_every_later_session_is_told_while_it_is_still_broken(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """Not once, and not only to whoever ran the indexer: until it is fixed."""
        broken = self.tree(tmp_path)
        indexer = Indexer(db, fake_embedder)
        try:
            for _ in range(3):
                indexer.index_directory(tmp_path)
                status = db.index_status(str(tmp_path))
                assert not status.verified
                assert [f.file_path for f in status.failures] == [str(broken)]
        finally:
            broken.chmod(0o644)

    def test_one_root_never_wears_another_root_s_failure(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """One database holds several roots, and the default one holds every project."""
        alpha = tmp_path / "alpha"
        broken = self.tree(alpha)
        beta = tmp_path / "beta"
        beta.mkdir()
        (beta / "fine.md").write_text("# Fine\n\nnothing wrong here\n")
        indexer = Indexer(db, fake_embedder)
        try:
            indexer.index_directory(alpha)
            indexer.index_directory(beta)
            assert db.index_status(str(beta)).verified, "beta wore alpha's failure"
            assert not db.index_status(str(alpha)).verified, "beta's run silenced alpha"
        finally:
            broken.chmod(0o644)

    def test_a_failure_under_a_subdirectory_is_visible_from_the_root(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """Indexing is often pointed at a subdirectory; searching is not."""
        broken = self.tree(tmp_path / "api")
        try:
            Indexer(db, fake_embedder).index_directory(tmp_path / "api")
        finally:
            broken.chmod(0o644)
        status = db.index_status(str(tmp_path))
        assert not status.verified
        assert [f.file_path for f in status.failures] == [str(broken)]

    def test_a_clean_run_clears_it(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """A warning that outlives its problem is noise, and noise gets ignored."""
        broken = self.tree(tmp_path)
        indexer = Indexer(db, fake_embedder)
        try:
            indexer.index_directory(tmp_path)
        finally:
            broken.chmod(0o644)
        indexer.index_directory(tmp_path)
        assert db.index_status(str(tmp_path)).verified

    def test_a_file_that_failed_then_vanished_stops_being_reported(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """The row can outlive every chance to clear it one file at a time.

        A file that fails on its first index never reaches `replace_document`, so it never
        enters `documents` - and a purge works from `documents`. Delete the file and
        nothing would ever visit its row again.
        """
        broken = self.tree(tmp_path)
        indexer = Indexer(db, fake_embedder)
        try:
            indexer.index_directory(tmp_path)
            assert not db.index_status(str(tmp_path)).verified
        finally:
            broken.chmod(0o644)
        broken.unlink()
        indexer.index_directory(tmp_path)
        assert db.index_status(str(tmp_path)).verified, "the row outlived the file"

    def test_a_file_that_failed_then_came_back_unchanged_stops_being_reported(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """The mirror case: the hash check skips it, so `replace_document` never runs."""
        target = tmp_path / "a.md"
        target.write_text("# A\n\nfirst body\n")
        indexer = Indexer(db, fake_embedder)
        indexer.index_directory(tmp_path)
        original = target.read_text()

        target.write_text("# A\n\nedited body\n")
        target.chmod(0o000)
        try:
            indexer.index_directory(tmp_path)
            assert not db.index_status(str(tmp_path)).verified
        finally:
            target.chmod(0o644)
        target.write_text(original)  # back to exactly what was indexed
        indexer.index_directory(tmp_path)
        assert db.index_status(str(tmp_path)).verified

    def test_an_unreadable_directory_is_reported_against_itself(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """The walker reports the directory, not the files it could not see inside it."""
        (tmp_path / "good.md").write_text("# Good\n\nreadable\n")
        closed = tmp_path / "closed"
        closed.mkdir()
        (closed / "hidden.md").write_text("# Hidden\n\nbody\n")
        closed.chmod(0o000)
        try:
            Indexer(db, fake_embedder).index_directory(tmp_path)
            reported = [f.file_path for f in db.index_status(str(closed)).failures]
            assert reported == [str(closed)]
        finally:
            closed.chmod(0o755)


class TestOnlyAWholeWalkVouchesForATree:
    """`verified` claims a full pass read everything it found - nothing less."""

    def test_a_tree_nobody_has_indexed_is_not_vouched_for(
        self, db: Database, tmp_path: Path
    ) -> None:
        assert not db.index_status(str(tmp_path)).verified

    def test_a_run_that_dies_partway_leaves_the_tree_unvouched_for(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """Whatever it committed stays and answers; nothing says the rest never arrived.

        The tree is certified first, so the certificate has something to lose: a fresh
        database reads unverified whatever the run does, and a test starting there would
        pass with the retraction deleted outright.
        """
        (tmp_path / "a.md").write_text("# A\n\nalpha body\n")
        (tmp_path / "b.md").write_text("# B\n\nbeta body\n")
        Indexer(db, fake_embedder).index_directory(tmp_path)
        assert db.index_status(str(tmp_path)).verified

        (tmp_path / "a.md").write_text("# A\n\nedited\n")
        (tmp_path / "b.md").write_text("# B\n\nedited too\n")
        with pytest.raises(KeyboardInterrupt):
            Indexer(db, StopsAfterOneFile(fake_embedder)).index_directory(tmp_path)
        assert not db.index_status(str(tmp_path)).verified

    def test_a_run_that_committed_nothing_leaves_the_certificate_alone(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """The model would not load. That says nothing about the index already there."""
        (tmp_path / "a.md").write_text("# A\n\nalpha body\n")
        Indexer(db, fake_embedder).index_directory(tmp_path)
        assert db.index_status(str(tmp_path)).verified
        # Edited, so the run really does reach the model rather than skipping the file.
        (tmp_path / "a.md").write_text("# A\n\nedited body\n")
        with pytest.raises(ModelLoadError):
            Indexer(db, RefusesToLoad(fake_embedder.dimension)).index_directory(tmp_path)
        assert db.index_status(str(tmp_path)).verified, "a failed process blamed the index"

    def test_a_scan_of_a_subdirectory_retracts_the_root_it_sits_in(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """A half-written subtree is a half-written tree.

        Certify the root, start rewriting one directory inside it, and stop. Nothing
        failed, so no failure row exists - only the certificate can say the tree is no
        longer the one that was vouched for.
        """
        api = tmp_path / "api"
        api.mkdir()
        (tmp_path / "top.md").write_text("# Top\n\nbody\n")
        (api / "a.md").write_text("# A\n\nalpha body\n")
        (api / "b.md").write_text("# B\n\nbeta body\n")
        indexer = Indexer(db, fake_embedder)
        indexer.index_directory(tmp_path)
        assert db.index_status(str(tmp_path)).verified

        (api / "a.md").write_text("# A\n\nrewritten\n")
        (api / "b.md").write_text("# B\n\nrewritten too\n")
        with pytest.raises(KeyboardInterrupt):
            Indexer(db, StopsAfterOneFile(fake_embedder)).index_directory(api)
        message = "the root still vouched for itself"
        assert not db.index_status(str(tmp_path)).verified, message

    def test_the_same_root_spelled_two_ways_is_one_root(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """A detour through a subdirectory leads to the same tree.

        `Path` folds away `.` and doubled separators on its own, so a test using those
        proves nothing; `..` survives until something resolves it. Two spellings would be
        two states, and a clean run under one would not clear what the other recorded.
        """
        (tmp_path / "nested").mkdir()
        (tmp_path / "a.md").write_text("# A\n\nbody\n")
        detour = tmp_path / "nested" / ".."
        assert str(detour) != str(tmp_path), "pathlib normalised the detour away"

        indexer = Indexer(db, fake_embedder)
        indexer.index_directory(detour)
        assert db.index_status(str(tmp_path)).verified, "the odd spelling was a second root"

    def test_a_scan_retracts_the_roots_nested_inside_it_too(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """Read the other way round: this scan rewrites those files as well.

        One database can hold several servers' roots at different depths. A scan of
        `docs` rewrites everything under `docs/api`, so a server configured at `docs/api`
        must stop vouching for it - otherwise it answers confidently from a subtree that
        something else is halfway through replacing.
        """
        api = tmp_path / "api"
        api.mkdir()
        (api / "a.md").write_text("# A\n\nalpha body\n")
        indexer = Indexer(db, fake_embedder)
        indexer.index_directory(api)
        assert db.index_status(str(api)).verified

        db.mark_scan_started(str(tmp_path))  # a scan of the tree containing it begins
        assert not db.index_status(str(api)).verified, "the nested root still vouched for itself"

    def test_a_scan_leaves_a_similarly_named_neighbour_alone(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """`/docs` and `/docs-api` share a prefix and share nothing else."""
        docs, neighbour = tmp_path / "docs", tmp_path / "docs-api"
        for root in (docs, neighbour):
            root.mkdir()
            (root / "a.md").write_text(f"# {root.name}\n\nbody\n")
        indexer = Indexer(db, fake_embedder)
        indexer.index_directory(docs)
        indexer.index_directory(neighbour)
        db.mark_scan_started(str(docs))
        assert db.index_status(str(neighbour)).verified, "a bare prefix match caught the neighbour"

    def test_a_whole_run_of_the_root_vouches_for_it_again(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """And it leaves the nested root's own record alone.

        An earlier version deleted the rows of roots inside the one it walked, reasoning
        that this walk covered them. It does not cover a pruned subtree, and a walk that
        hit failures covered less still - so deleting turned a nested root that was
        perfectly fine into one reporting unknown. The row stays; the scan retracted it on
        the way in, and only a walk of that root itself can vouch for it again.
        """
        api = tmp_path / "api"
        api.mkdir()
        (api / "a.md").write_text("# A\n\nalpha body\n")
        indexer = Indexer(db, fake_embedder)
        indexer.index_directory(api)
        assert db.index_status(str(api)).verified

        indexer.index_directory(tmp_path)
        assert db.index_status(str(tmp_path)).verified
        assert db.count_rows("index_coverage") == 2, "the nested root's record was destroyed"
        # This walk rewrote nothing - every file was already indexed - so it had no reason
        # to retract anything, and `api` is as true as it was a moment ago.
        assert db.index_status(str(api)).verified

    def test_a_second_process_is_refused_rather_than_queued(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """A 25-minute wait inside a tool call is a client timeout, not patience."""
        (tmp_path / "a.md").write_text("# A\n\nbody\n")
        handle = os.open(str(db.path) + ".lock", os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(IndexBusyError):
                Indexer(db, fake_embedder).index_directory(tmp_path)
        finally:
            os.close(handle)

    def test_a_lock_file_that_cannot_be_opened_does_not_wedge_the_process(
        self,
        db: Database,
        fake_embedder: FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The thread lock is taken first, so whatever fails next has to give it back.

        Otherwise one transient failure - no file descriptors, a read-only directory -
        refuses every later run in this process for as long as it lives, and the message
        blames another run that is not there.
        """
        (tmp_path / "a.md").write_text("# A\n\nbody\n")
        indexer = Indexer(db, fake_embedder)

        def refuse(*arguments: object, **keywords: object) -> int:
            raise OSError(24, "Too many open files")

        monkeypatch.setattr(indexer_module.os, "open", refuse)
        with pytest.raises(OSError):
            indexer.index_directory(tmp_path)
        monkeypatch.undo()
        indexer.index_directory(tmp_path)  # the process is not wedged
        assert db.index_status(str(tmp_path)).verified

    def test_two_spellings_of_one_database_take_one_lock(
        self, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """A symlinked database is the same database, and must be the same lock.

        The lock file is named after the database. Named after whatever spelling reached
        it, two processes pointed at one file through different paths take two different
        locks, both believe they are alone, and the one-scan-at-a-time guarantee that the
        failure bookkeeping rests on is simply false.
        """
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.md").write_text("# A\n\nbody\n")
        real = tmp_path / "index.db"
        with Database(real) as seed:
            seed.get_meta("anything")  # create the file
        alias = tmp_path / "alias.db"
        alias.symlink_to(real)

        with Database(real) as first, Database(alias) as second:
            held = Indexer(first, fake_embedder)
            with held._scan_lock(), pytest.raises(IndexBusyError):
                Indexer(second, fake_embedder).index_directory(docs)

    def test_a_failure_inside_a_pruned_directory_outlives_a_parent_scan(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """A walk that never went there has nothing to say about it.

        `.venv` and `node_modules` are pruned, so a scan of the tree above one of them
        never looks inside. Clearing failures by prefix would let that scan declare a file
        fine that it did not read, and the file stays broken and unindexed.
        """
        venv = tmp_path / ".venv"
        venv.mkdir()
        (tmp_path / "good.md").write_text("# Good\n\nreadable\n")
        broken = venv / "bad.md"
        broken.write_text("# Bad\n\nbody\n")
        broken.chmod(0o000)
        indexer = Indexer(db, fake_embedder)
        try:
            indexer.index_directory(venv)  # indexed deliberately, by pointing inside it
            assert len(db.index_status(str(venv)).failures) == 1

            indexer.index_directory(tmp_path)  # prunes .venv: it sees nothing in there
            assert len(db.index_status(str(venv)).failures) == 1, (
                "a scan spoke for a tree it pruned"
            )
        finally:
            broken.chmod(0o644)

    def test_a_document_left_on_an_older_vector_format_is_not_called_whole(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """Pruning means some documents are never reached again by a parent scan.

        A file indexed deliberately inside `.venv` keeps whatever vectors it was given:
        every later scan of the tree above prunes that directory, so the old-format vector
        stays, answering alongside pooled ones. Saying so is the difference between a
        bounded window and a permanent one nobody is told about.
        """
        venv = tmp_path / ".venv"
        venv.mkdir()
        (tmp_path / "good.md").write_text("# Good\n\nreadable\n")
        (venv / "old.md").write_text("# Old\n\nbody\n")
        indexer = Indexer(db, fake_embedder)
        indexer.index_directory(venv)
        indexer.index_directory(tmp_path)

        with db.transaction() as conn:  # what a pre-pooling release left behind
            conn.execute("UPDATE documents SET vector_format = 1 WHERE file_path LIKE '%old.md'")
        status = db.index_status(str(tmp_path))
        assert status.stale_vectors == 1
        assert not status.verified, "an older vector format answered without saying so"
        assert "older vector format" in (status.message() or "")

    def test_a_clean_run_after_a_model_change_vouches_for_the_tree(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """A run's own housekeeping is not an abort.

        Changing the embedding model makes every stored vector incomparable, so the run
        discards the index itself and rebuilds it. Counting that discard as somebody
        else's leaves the tree it just rebuilt from scratch reported as unvouched-for -
        for good, since every later run finds nothing to do.
        """
        (tmp_path / "a.md").write_text("# A\n\nbody\n")
        Indexer(db, FakeEmbedder(model_name="alpha")).index_directory(tmp_path)
        assert db.index_status(str(tmp_path)).verified

        report = Indexer(db, FakeEmbedder(model_name="beta")).index_directory(tmp_path)
        assert (report.files_indexed, report.errors) == (1, ())
        assert db.index_status(str(tmp_path)).verified, "the run blamed itself for its own wipe"

    def test_a_pruned_directory_that_could_not_be_listed_keeps_its_own_failure(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """The recorded path is the directory itself, not a file inside it.

        Most failures name a file, so asking whether the *parent* is walkable works by
        accident. An unreadable directory is recorded under its own path, and that question
        then asks whether `.venv`'s parent is walkable - which it is - and clears a row for
        a tree the walk never entered.
        """
        venv = tmp_path / ".venv"
        venv.mkdir()
        (tmp_path / "good.md").write_text("# Good\n\nreadable\n")
        (venv / "inner.md").write_text("# Inner\n\nbody\n")
        venv.chmod(0o000)
        try:
            Indexer(db, fake_embedder).index_directory(venv)
            assert [f.file_path for f in db.index_status(str(venv)).failures] == [str(venv)]

            Indexer(db, fake_embedder).index_directory(tmp_path)  # prunes .venv entirely
            assert [f.file_path for f in db.index_status(str(venv)).failures] == [str(venv)], (
                "a scan cleared a directory it never entered"
            )
        finally:
            venv.chmod(0o755)

    def test_a_failure_behind_a_symlinked_directory_outlives_a_parent_scan(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """Same question for the failure rows: the walk never went in there.

        The failure is recorded while the directory is real, so the row sits under a path
        that later turns into a symlink. The walk stops at the link, learns nothing about
        what is behind it, and must not declare it fine.
        """
        root = tmp_path / "root"
        (root / "vendor").mkdir(parents=True)
        (root / "top.md").write_text("# Top\n\nbody\n")
        broken = root / "vendor" / "bad.md"
        broken.write_text("# Bad\n\nbody\n")
        broken.chmod(0o000)
        indexer = Indexer(db, fake_embedder)
        try:
            indexer.index_directory(root)
            assert [f.file_path for f in db.index_status(str(root)).failures] == [str(broken)]
        finally:
            broken.chmod(0o644)

        elsewhere = tmp_path / "elsewhere"  # the tree moves and is linked back
        (root / "vendor").rename(elsewhere)
        (root / "vendor").symlink_to(elsewhere, target_is_directory=True)

        indexer.index_directory(root)
        assert len(db.index_status(str(root)).failures) == 1, (
            "a scan cleared what it could not see behind a symlink"
        )

    def test_a_directory_that_became_a_symlink_costs_nothing(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """The walk does not follow links, so it cannot report on what is behind one.

        A real directory becomes a symlink - a vendored tree moved and linked back, a
        checkout rearranged. Its files are still there and still readable at that path,
        but the walk no longer descends. Treating "I did not look" as "there is nothing
        there" purges the documents and clears the failures.
        """
        root, elsewhere = tmp_path / "root", tmp_path / "elsewhere"
        (root / "link").mkdir(parents=True)
        elsewhere.mkdir()
        (root / "top.md").write_text("# Top\n\nbody\n")
        (root / "link" / "inner.md").write_text("# Inner\n\nindexed while it was real\n")
        indexer = Indexer(db, fake_embedder)
        indexer.index_directory(root)
        assert len(db.list_documents(str(root))) == 2

        (root / "link" / "inner.md").rename(elsewhere / "inner.md")
        (root / "link").rmdir()
        (root / "link").symlink_to(elsewhere, target_is_directory=True)

        indexer.index_directory(root)
        assert len(db.list_documents(str(root))) == 2, "a symlinked directory cost its documents"

    def test_a_failure_inside_an_excluded_directory_outlives_a_parent_scan(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """Exclusions prune the walk exactly as `.venv` does, and mean the same thing."""
        drafts = tmp_path / "drafts"
        drafts.mkdir()
        (tmp_path / "good.md").write_text("# Good\n\nreadable\n")
        broken = drafts / "bad.md"
        broken.write_text("# Bad\n\nbody\n")
        broken.chmod(0o000)
        try:
            Indexer(db, fake_embedder).index_directory(drafts)
            assert len(db.index_status(str(drafts)).failures) == 1

            Indexer(db, fake_embedder, exclude=("drafts",)).index_directory(tmp_path)
            assert len(db.index_status(str(drafts)).failures) == 1, (
                "a scan spoke for a tree its exclusions kept it out of"
            )
        finally:
            broken.chmod(0o644)

    def test_a_failure_out_of_the_walk_s_reach_goes_when_the_file_does(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """Refusing to speak for what was not seen must not make a row immortal.

        The row sits inside a directory this scan excludes, so no walk of the root will
        ever look at it again. Delete the file and the root reported itself incomplete
        forever - naming a path that no longer exists, advising a re-index that could not
        change the answer. A walk's silence is not evidence; `ENOENT` at that one name is.
        """
        drafts = tmp_path / "drafts"
        drafts.mkdir()
        (tmp_path / "good.md").write_text("# Good\n\nreadable\n")
        broken = drafts / "bad.md"
        broken.write_text("# Bad\n\nbody\n")
        broken.chmod(0o000)
        try:
            Indexer(db, fake_embedder).index_directory(tmp_path)
            assert not db.index_status(str(tmp_path)).verified
        finally:
            broken.chmod(0o644)
        broken.unlink()

        Indexer(db, fake_embedder, exclude=("drafts",)).index_directory(tmp_path)
        status = db.index_status(str(tmp_path))
        assert status.failures == (), "a failure outlived the file it names"
        assert status.verified, "the root could never be called whole again"

    def test_a_deleted_document_inside_a_pruned_tree_stops_answering(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """The same evidence on the other side: a purge needs a deletion, and here it has one.

        A document indexed by pointing `index_directory` inside `.venv` survives every
        parent scan, because pruning is not evidence of anything. Once the file is gone,
        keeping the document means searches answer from text that is not on disk.
        """
        venv = tmp_path / ".venv"
        venv.mkdir()
        (tmp_path / "good.md").write_text("# Good\n\nreadable\n")
        inner = venv / "inner.md"
        inner.write_text("# Inner\n\nbody\n")
        indexer = Indexer(db, fake_embedder)
        indexer.index_directory(venv)
        indexer.index_directory(tmp_path)
        assert str(inner) in db.document_hashes(str(tmp_path)), "a pruned tree is not evidence"

        inner.unlink()
        indexer.index_directory(tmp_path)
        assert str(inner) not in db.document_hashes(str(tmp_path)), (
            "a deleted document kept answering"
        )

    def test_a_file_under_an_unreadable_directory_is_not_taken_for_deleted(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """`ENOENT` is evidence of deletion; `EACCES` is not, and one call returns both.

        Nothing under a directory that cannot be listed can be stat'ed either: every file
        in it answers "not allowed", never "not there". Reading the second from the first
        would purge a whole locked tree and clear every failure recorded beneath it.
        """
        locked = tmp_path / "locked"
        locked.mkdir()
        (tmp_path / "good.md").write_text("# Good\n\nreadable\n")
        inner = locked / "inner.md"
        inner.write_text("# Inner\n\nbody\n")
        indexer = Indexer(db, fake_embedder)
        indexer.index_directory(locked)
        locked.chmod(0o000)
        try:
            indexer.index_directory(tmp_path)
            assert str(inner) in db.document_hashes(str(tmp_path)), (
                "a locked directory was read as an empty one"
            )
        finally:
            locked.chmod(0o755)

    def test_a_directory_replaced_by_a_broken_symlink_keeps_its_documents(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """`ENOENT` through a symlink is the link being broken, not the file being deleted.

        The real directory is moved aside and a dangling link left in its place. Every
        document under it now stats as absent, though the files are intact where they were
        moved to. Purging them here would redo, through the new evidence, exactly the data
        loss this branch fixed: a directory becoming a symlink deleting what was beneath it.
        """
        root = tmp_path / "root"
        (root / "link").mkdir(parents=True)
        (root / "good.md").write_text("# G\n\nbody\n")
        (root / "link" / "inner.md").write_text("# I\n\nbody\n")
        indexer = Indexer(db, fake_embedder)
        indexer.index_directory(root)
        assert len(db.document_hashes(str(root))) == 2

        shutil.move(str(root / "link"), str(tmp_path / "moved"))
        (root / "link").symlink_to(tmp_path / "nowhere")
        report = indexer.index_directory(root)
        assert report.files_purged == 0, "a broken link was read as a deletion"
        assert len(db.document_hashes(str(root))) == 2

    def test_a_failure_on_a_directory_that_became_a_symlink_outlives_the_swap(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """`_behind_symlink` tests the components before the last one, and this row IS the last.

        The failure names the directory itself. Once that directory is a symlink the walk
        lists its name and never enters, so nothing rechecks it - but asking whether the
        path sits *behind* a symlink says no, and the row was cleared and the root
        certified with a whole subtree unindexed.
        """
        root = tmp_path / "root"
        vendor = root / "vendor"
        vendor.mkdir(parents=True)
        (root / "good.md").write_text("# G\n\nbody\n")
        (vendor / "inner.md").write_text("# I\n\nbody\n")
        vendor.chmod(0o000)
        try:
            Indexer(db, fake_embedder).index_directory(root)
            assert db.failure_paths(str(root)) == [str(vendor)]
        finally:
            vendor.chmod(0o755)
        shutil.move(str(vendor), str(tmp_path / "real_vendor"))
        (root / "vendor").symlink_to(tmp_path / "real_vendor")

        Indexer(db, fake_embedder).index_directory(root)
        status = db.index_status(str(root))
        assert len(status.failures) == 1, "a walk that stopped at the link spoke for it anyway"
        assert not status.verified

    def test_a_failure_whose_name_could_not_be_decoded_is_not_taken_for_deleted(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """The row spells the name with U+FFFD, so nothing is ever at the path it spells.

        `_printable` substitutes for bytes it cannot decode, and the result is itself valid
        UTF-8 - it survives a round trip through `_printable` unchanged, so only the
        replacement character gives it away. Stat that rendering and it is always absent,
        about a file that is still there and still broken.
        """
        root = tmp_path / "root"
        venv = root / ".venv"
        venv.mkdir(parents=True)
        (root / "good.md").write_text("# G\n\nbody\n")
        raw = os.path.join(os.fsdecode(bytes(venv)), os.fsdecode(b"caf\xe9.md"))
        with open(os.fsencode(raw), "wb") as handle:
            handle.write(b"# C\n\nbody\n")
        os.chmod(os.fsencode(raw), 0o000)
        try:
            Indexer(db, fake_embedder).index_directory(venv)
            assert len(db.failure_paths(str(venv))) == 1

            Indexer(db, fake_embedder).index_directory(root)  # prunes .venv
            assert os.path.exists(os.fsencode(raw))
            assert len(db.index_status(str(root)).failures) == 1, (
                "a name it could not read was taken for a file that is gone"
            )
        finally:
            os.chmod(os.fsencode(raw), 0o644)

    def test_a_newly_excluded_document_leaves_the_index(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """Exclusion is scope, not blindness: what it covers must stop being searchable.

        Pruned and unreadable trees are kept because the walk learned nothing about them.
        An exclusion is the opposite - the operator saying this is not documentation - so
        the documents go, and only the failure rows stay for whoever recorded them.
        """
        drafts = tmp_path / "drafts"
        drafts.mkdir()
        (tmp_path / "good.md").write_text("# Good\n\nreadable\n")
        (drafts / "a.md").write_text("# A\n\nbody\n")
        Indexer(db, fake_embedder).index_directory(tmp_path)
        assert len(db.document_hashes(str(tmp_path))) == 2

        report = Indexer(db, fake_embedder, exclude=("drafts",)).index_directory(tmp_path)
        assert report.files_purged == 1
        assert list(db.document_hashes(str(tmp_path))) == [str(tmp_path / "good.md")]

    def test_a_scan_does_not_certify_an_index_discarded_under_it(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """Its measurements describe a database that no longer exists.

        A scan reads the file hashes, then something empties the index wholesale - a model
        or embedding-size change. Every file now looks unchanged to that scan, so it walks
        to the end seeing nothing wrong and would certify an empty database as whole.
        """
        (tmp_path / "a.md").write_text("# A\n\nbody\n")
        Indexer(db, fake_embedder).index_directory(tmp_path)
        generation = db.generation()
        db.clear()  # the wholesale discard, mid-scan
        db.mark_scan_complete(str(tmp_path), generation)
        assert not db.index_status(str(tmp_path)).verified, "it vouched for an emptied index"

    def test_the_scan_lock_is_exclusive_not_merely_held(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """Asking for a shared lock would let every run in at once.

        Holding an exclusive lock in this test cannot show the difference - exclusive
        refuses a shared request too. A shared holder can: an exclusive request still has
        to be refused, while a shared one would sail past it and two scans would rewrite
        one database together.
        """
        (tmp_path / "a.md").write_text("# A\n\nbody\n")
        handle = os.open(str(db.path) + ".lock", os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
            with pytest.raises(IndexBusyError):
                Indexer(db, fake_embedder).index_directory(tmp_path)
        finally:
            os.close(handle)

    def test_a_second_thread_is_refused_the_same_way(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """`flock` is held per process, so the thread lock has to refuse too.

        Run in a worker with a deadline rather than asserting the exception directly: a
        thread lock that waits would block this test for as long as a real scan takes,
        and a test that hangs reports nothing - it just stops, and something upstream
        eventually kills it with no idea which behaviour was missing.
        """
        (tmp_path / "a.md").write_text("# A\n\nbody\n")
        indexer = Indexer(db, fake_embedder)
        outcome: list[BaseException | None] = []

        def second_call() -> None:
            try:
                indexer.index_directory(tmp_path)
                outcome.append(None)
            except BaseException as error:  # noqa: BLE001 - recorded, then asserted on
                outcome.append(error)

        indexer._run_lock.acquire()
        worker = threading.Thread(target=second_call, daemon=True)
        try:
            worker.start()
            worker.join(timeout=10)
            assert not worker.is_alive(), "the second call waited instead of refusing"
            assert isinstance(outcome[0], IndexBusyError)
        finally:
            indexer._run_lock.release()
            worker.join(timeout=10)
