"""Storage regressions: connections, migrations, vectors, indexing, integrity, notices.

One test per defect found in code review; each fails when its fix is reverted.
"""

from __future__ import annotations

import math
import os
import sqlite3
import subprocess
import sys
import threading
from collections.abc import Sequence
from pathlib import Path

import pytest
from fakes import FakeEmbedder
from helpers import draft, store
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

import markdown_memory.indexer as indexer_module
from markdown_memory.db import Database
from markdown_memory.exceptions import DatabaseError, IndexingError, ModelLoadError
from markdown_memory.indexer import Indexer
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
        """Turn a populated v2 database into what the previous release left behind."""
        with Database(path) as database:
            conn = database.connection()
            conn.execute("DROP TRIGGER units_after_delete")
            conn.execute("DROP TABLE units_vec")
            conn.execute("DROP TABLE units")
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
