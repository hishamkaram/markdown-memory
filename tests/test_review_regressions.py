"""Regression tests for defects found in code review. One test (at least) per finding."""

from __future__ import annotations

import math
import os
import sqlite3
import subprocess
import sys
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from fakes import FakeEmbedder, vectors_for
from mcp import Client
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

import markdown_memory.indexer as indexer_module
import markdown_memory.server as server_module
from markdown_memory.db import Database
from markdown_memory.exceptions import DatabaseError, IndexingError, ModelLoadError, SearchError
from markdown_memory.indexer import Indexer
from markdown_memory.models import (
    PREAMBLE_TITLE,
    OutlineNode,
    ParsedDocument,
    SectionDraft,
    SectionVectors,
)
from markdown_memory.parser import DEFAULT_MAX_SECTION_CHARS, MarkdownParser, join_parts
from markdown_memory.search import HybridSearcher, build_fts_query
from markdown_memory.server import MarkdownMemoryService, ServerConfig, create_server


def parse(text: str) -> ParsedDocument:
    return MarkdownParser().parse(text, fallback_title="fallback")


def paths(document: ParsedDocument) -> list[str]:
    return [section.heading_path for section in document.sections]


def draft(title: str, content: str) -> SectionDraft:
    return SectionDraft(
        heading_title=title, heading_level=2, heading_path=f"Doc > {title}",
        base_path=f"Doc > {title}", content=content, start_line=1, end_line=1,
        units=(content.split("\n\n", 1)[-1],),
    )  # fmt: skip


def store(db: Database, embedder: FakeEmbedder, file_path: str, count: int = 3) -> None:
    sections = [draft(f"S{n}", f"## S{n}\n\nbody number {n}") for n in range(count)]
    db.replace_document(
        file_path=file_path, title="Doc", content_hash="h", last_modified=1, sections=sections,
        vectors=vectors_for(embedder, sections),
    )  # fmt: skip


# ====================================================================== storage


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


# ====================================================================== indexer


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


# ====================================================================== search


class TestQueryBuilding:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("git log --all", '"git" OR "log" OR "--all"'),
            ("rsync --from --to", '"rsync" OR "--from" OR "--to"'),
            ("grep -i flag", '"grep" OR "-i" OR "flag"'),
            ("NOT NULL constraint", '"NOT" OR "NULL" OR "constraint"'),
            ("@Before annotation", '"@Before" OR "annotation"'),
            ("WHERE clause", '"WHERE" OR "clause"'),
            ("what is (the) Where, exactly?", '"exactly?"'),
        ],
    )
    def test_identifiers_that_spell_stopwords_are_kept(self, query: str, expected: str) -> None:
        assert build_fts_query(query) == expected

    def test_heading_keyword_is_found_together_with_other_terms(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        root = tmp_path / "docs"
        root.mkdir()
        filler = "\n\n".join(f"## Note {n}\n\nA clause about clause number {n}." for n in range(30))
        (root / "sql.md").write_text(f"# SQL\n\n## WHERE\n\nFilters rows.\n\n{filler}\n")
        Indexer(db, fake_embedder).index_directory(root)
        searcher = HybridSearcher(db, fake_embedder)
        try:
            results = searcher.search("WHERE clause")
        finally:
            searcher.close()
        assert "SQL > WHERE" in [r.heading_path for r in results]

    def test_nul_bytes_cannot_truncate_the_match_expression(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        store(db, fake_embedder, "/d/a.md")
        assert build_fts_query("body\x00number") == '"body" OR "number"'
        expression = build_fts_query("x\x00 body")
        assert expression is not None and "\x00" not in expression
        assert db.fts_search(expression, 5)  # the raw primitive, no fallback to hide errors


# ====================================================================== parser


class TestUniquePaths:
    def test_generated_suffix_never_collides_with_a_literal_heading(self) -> None:
        document = parse("# Log\n\n## Fixed\n\na\n\n## Fixed [2]\n\nb\n\n## Fixed\n\nc\n")
        assert paths(document) == ["Log", "Log > Fixed", "Log > Fixed [2]", "Log > Fixed [3]"]
        assert len(set(paths(document))) == len(document.sections)

    def test_many_duplicates_with_literal_lookalikes(self) -> None:
        document = parse("# X\n\n# X [3]\n\n# X\n\n# X\n\n# X [2]\n")
        assert len(set(paths(document))) == 5

    def test_real_heading_named_like_a_generated_part(self) -> None:
        body = "\n\n".join(f"Sentence number {n} is here. " * 12 for n in range(12))
        text = f"# Tutorial\n\n{body}\n\n# Tutorial (Part 2)\n\nThe real second chapter.\n"
        document = parse(text)
        found = paths(document)
        assert len(set(found)) == len(found)
        assert "Tutorial (Part 2)" in found  # the generated part keeps the plain name
        real = next(s for s in document.sections if "real second chapter" in s.content)
        assert real.heading_path == "Tutorial (Part 2) [2]"
        assert real.part_index == 0

    def test_literal_part_heading_before_the_oversized_section(self) -> None:
        body = "\n\n".join(f"Sentence number {n} is here. " * 12 for n in range(12))
        document = parse(f"# Guide (Part 1)\n\nliteral\n\n# Guide\n\n{body}\n")
        found = paths(document)
        assert len(set(found)) == len(found)
        assert found[0] == "Guide (Part 1)"
        assert found[1].startswith("Guide [2] (Part ")

    def test_heading_named_like_the_preamble(self) -> None:
        document = parse(f"intro\n\n# {PREAMBLE_TITLE}\n\nbody\n")
        assert paths(document) == [PREAMBLE_TITLE, f"{PREAMBLE_TITLE} [2]"]


class TestHeadingTitles:
    def test_type_parameters_survive_in_titles(self) -> None:
        text = (
            "# Types\n\n## Option<T>\n\na\n\n## Option<U>\n\nb\n\n## The <details> element\n\nc\n"
        )
        assert paths(parse(text)) == [
            "Types",
            "Types > Option<T>",
            "Types > Option<U>",
            "Types > The <details> element",
        ]

    def test_formatting_html_is_still_stripped(self) -> None:
        document = parse('# The <b>bold</b> H<sub>2</sub>O <a href="x">link</a><!-- note -->\n')
        assert paths(document) == ["The bold H2O link"]


class TestFrontMatterDetection:
    def test_leading_thematic_break_is_not_front_matter(self) -> None:
        document = parse("---\n\n# Project\n\nIntro\n\n---\n\n## Usage\n\nrun it\n")
        assert paths(document)[-2:] == ["Project", "Project > Usage"]
        assert document.title == "Project"

    def test_heading_then_ellipsis_is_not_front_matter(self) -> None:
        document = parse("---\n# Project\n\nWait for it\n...\n\n## Usage\n\nrun\n")
        assert "Project > Usage" in paths(document)

    def test_prose_with_a_colon_is_not_front_matter(self) -> None:
        document = parse("---\nNote: read this first\nSome prose follows here\n---\n\n# T\n")
        assert document.title == "T"
        assert not any("Note" in s.content for s in document.sections if s.heading_level == 0)

    def test_real_front_matter_with_lists_comments_and_blocks(self) -> None:
        text = (
            "---\n# generated by a tool\ntitle: 'Real'\ntags:\n  - a\n  - b\n"
            "summary: |\n  multi\n  line\n\nnested:\n  key: value\n---\n\n## Body\n\ntext\n"
        )
        document = parse(text)
        assert document.title == "Real"
        assert paths(document) == [PREAMBLE_TITLE, "Body"]


class TestUnclosedFenceBeforeLaterFences:
    def test_later_info_fence_does_not_hide_the_swallowed_heading(self) -> None:
        text = "# A\n\n```bash\nls\n\n## B\n\nB text\n\n```python\nx = 1\n```\n\n## C\n\nmore\n"
        document = parse(text)
        assert paths(document) == ["A", "A > B", "A > C"]
        assert "```python\nx = 1\n```" in document.sections[1].content

    def test_bare_later_fences_flip_parity_and_everything_is_recovered(self) -> None:
        text = (
            "# A\n\n```bash\nls\n\n## B\n\nB text\n\n```\nx = 1\n```\n\n## C\n\nmore\n\n"
            "```\ny\n```\n\n## D\n\nend\n"
        )
        assert paths(parse(text)) == ["A", "A > B", "A > C", "A > D"]

    def test_longer_outer_fence_with_an_inner_example_is_left_alone(self) -> None:
        text = "# Doc\n\n````md\n\n## Example\n\n```python\nx = 1\n```\n\n````\n\n## Real\n"
        assert paths(parse(text)) == ["Doc", "Doc > Real"]

    def test_tilde_fence_holding_a_backtick_example_is_left_alone(self) -> None:
        text = "# Doc\n\n~~~md\n\n## Example\n\n```python\nx = 1\n```\n\n~~~\n\n## Real\n"
        assert paths(parse(text)) == ["Doc", "Doc > Real"]

    def test_well_formed_fences_with_hash_comments_are_untouched(self) -> None:
        text = (
            "# Doc\n\n```bash\nmake\n\n## build everything\nmake all\n```\n\n## Next\n\n"
            "```yaml\na: 1\n\n## section comment\nb: 2\n```\n\n## Last\n"
        )
        assert paths(parse(text)) == ["Doc", "Doc > Next", "Doc > Last"]

    def test_candidate_that_would_not_repair_the_document_is_rejected(self) -> None:
        # The final bare fence really is unclosed; cutting the earlier, well-formed bash
        # fence at its "## comment" would flip every later marker instead of fixing it.
        text = (
            "# Doc\n\n```bash\nmake\n\n## comment\nmake all\n```\n\n## Middle\n\ntext\n\n"
            "```\nnever closed\n\n## After\n\nend\n"
        )
        assert paths(parse(text)) == ["Doc", "Doc > Middle", "Doc > After"]


class TestFenceSafeChunking:
    def test_fence_that_fits_alone_is_not_cut_because_of_the_heading(self) -> None:
        code = "```python\n" + "\n".join(f"line_{n} = {n}" for n in range(225)) + "\n```"
        assert len(code) <= DEFAULT_MAX_SECTION_CHARS
        text = f"# A heading that is a little longer\n\n{code}\n\nAfter paragraph.\n"
        assert len(text) > DEFAULT_MAX_SECTION_CHARS
        document = parse(text)
        holders = [s for s in document.sections if "line_0 = 0" in s.content]
        assert len(holders) == 1
        assert code in holders[0].content
        assert join_parts(document.sections) == text.rstrip("\n")

    def test_fence_indented_inside_a_list_item_is_kept_whole(self) -> None:
        filler = "\n\n".join("filler paragraph text " * 20 for _ in range(6))
        code = (
            "    ```python\n"
            + "\n\n".join(f"    def function_{n}():\n        return {n}" for n in range(30))
            + "\n    ```"
        )
        text = f"# Guide\n\n{filler}\n\n1. Install:\n\n{code}\n\n{filler}\n"
        # the fence starts before the 3200-character boundary and ends after it
        assert text.index("```python") < DEFAULT_MAX_SECTION_CHARS < text.index("\n    ```\n")
        assert len(code) <= DEFAULT_MAX_SECTION_CHARS
        document = parse(text)
        holders = [s for s in document.sections if "def function_" in s.content]
        assert len(holders) == 1
        assert code in holders[0].content
        assert join_parts(document.sections) == text.rstrip("\n")


# ====================================================================== server


ARROWS = """# API

## Step 1 -> Step 2

arrow body

### Detail

detail body

## x >= 5

compare body

## `Result<T, E>` handling

generic body

## Setup

lower body

## SETUP

upper body
"""


ArrowService = tuple[MarkdownMemoryService, "MCPServer[None]"]


@pytest.fixture
def arrow_service(tmp_path: Path) -> Iterator[ArrowService]:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "api.md").write_text(ARROWS, encoding="utf-8")
    body = "\n\n".join(f"Paragraph {n}. " + "words and more words " * 25 for n in range(10))
    (docs / "big.md").write_text(f"# Big\n\n## Huge\n\n{body}\n\n### Child\n\nchild body\n")
    service = MarkdownMemoryService(
        ServerConfig(db_path=tmp_path / "s.db", docs_dir=docs), embedder=FakeEmbedder()
    )
    service.index_directory()
    yield service, create_server(service=service)
    service.close()


class TestHeadingPathResolution:
    def test_paths_shown_by_the_outline_can_always_be_read(
        self, arrow_service: ArrowService
    ) -> None:
        service, _ = arrow_service

        def walk(nodes: Sequence[OutlineNode]) -> Iterator[str]:
            for node in nodes:
                yield node.heading_path
                yield from walk(node.children)

        listed = list(walk(service.get_document_outline("api.md")))
        assert "API > Step 1 -> Step 2" in listed
        assert "API > Result<T, E> handling" in listed
        for heading_path in listed:
            assert service.read_section("api.md", heading_path)

    def test_titles_containing_a_greater_than_sign(self, arrow_service: ArrowService) -> None:
        service, _ = arrow_service
        assert "arrow body" in service.read_section("api.md", "API > Step 1 -> Step 2")
        assert "arrow body" in service.read_section("api.md", "Step 1 -> Step 2")
        assert "arrow body" in service.read_section("api.md", "api>step 1 -> step 2")
        assert "compare body" in service.read_section("api.md", "API > x >= 5")
        assert "generic body" in service.read_section("api.md", "Result<T, E> handling")
        assert "detail body" in service.read_section("api.md", "API > Step 1 -> Step 2 > Detail")
        subtree = service.read_section("api.md", "Step 1 -> Step 2", include_subsections=True)
        assert "arrow body" in subtree and "detail body" in subtree
        assert "compare body" not in subtree

    def test_siblings_differing_only_in_case(self, arrow_service: ArrowService) -> None:
        service, _ = arrow_service
        assert "lower body" in service.read_section("api.md", "API > Setup")
        assert "upper body" in service.read_section("api.md", "API > SETUP")
        assert "upper body" in service.read_section("api.md", "SETUP")
        with pytest.raises(Exception, match="ambiguous") as raised:
            service.read_section("api.md", "api > setup")
        assert "API > Setup | API > SETUP" in str(raised.value)

    def test_part_path_with_include_subsections(self, arrow_service: ArrowService) -> None:
        service, _ = arrow_service
        plain = service.read_section("big.md", "Big > Huge (Part 2)")
        assert plain == service.read_section(
            "big.md", "Big > Huge (Part 2)", include_subsections=True
        )
        whole = service.read_section("big.md", "Big > Huge", include_subsections=True)
        assert plain in whole and whole.endswith("child body")

    async def test_unusable_paths_are_anticipated_errors(self, arrow_service: ArrowService) -> None:
        _, server = arrow_service
        for arguments in (
            {"file_path": "~no_such_user_zz/x.md"},
            {"file_path": "bad\x00name.md"},
        ):
            with pytest.raises(ToolError) as raised:
                await server.call_tool("get_document_outline", arguments)
            assert not isinstance(raised.value, UnexpectedToolError), arguments
        for directory in ("~no_such_user_zz/docs", "bad\x00dir"):
            with pytest.raises(ToolError) as raised:
                await server.call_tool("index_directory", {"directory": directory})
            assert not isinstance(raised.value, UnexpectedToolError), directory


class TestServiceOwnership:
    async def test_lifespan_leaves_a_callers_service_open(self, tmp_path: Path) -> None:
        service = MarkdownMemoryService(
            ServerConfig(db_path=tmp_path / "own.db", docs_dir=tmp_path), embedder=FakeEmbedder()
        )
        server: MCPServer[None] = create_server(service=service)
        try:
            for _ in range(2):  # a second session on the same server must still work
                async with Client(server) as client:
                    outcome = await client.call_tool("list_documents", {})
                    assert not outcome.is_error
            assert service.list_documents() == []
        finally:
            service.close()

    async def test_server_closes_a_service_it_created_itself(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created: list[MarkdownMemoryService] = []

        class Tracked(MarkdownMemoryService):
            def __init__(self, config: ServerConfig) -> None:
                super().__init__(config, embedder=FakeEmbedder())
                created.append(self)

        monkeypatch.setattr(server_module, "MarkdownMemoryService", Tracked)
        server = create_server(ServerConfig(db_path=tmp_path / "owned.db", docs_dir=tmp_path))
        async with Client(server) as client:
            assert not (await client.call_tool("list_documents", {})).is_error
        with pytest.raises(DatabaseError, match="closed"):
            created[0].list_documents()


class TestMainEntrypoint:
    @pytest.fixture
    def harness(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, object]:
        seen: dict[str, object] = {}

        class FakeServer:
            def run(self, transport: str) -> None:
                seen["transport"] = transport

        def fake_create_server(
            config: ServerConfig, *, service: MarkdownMemoryService
        ) -> FakeServer:
            seen["config"] = config
            seen["service"] = service
            return FakeServer()

        monkeypatch.setattr(server_module, "create_server", fake_create_server)
        monkeypatch.setattr(server_module, "_warm_up", lambda embedder: None)
        monkeypatch.setattr(
            server_module, "configure_logging", lambda level=None: seen.update(level=level)
        )
        monkeypatch.setenv("MARKDOWN_MEMORY_DB", str(tmp_path / "env.db"))
        monkeypatch.setenv("MARKDOWN_MEMORY_DOCS_DIR", str(tmp_path / "env-docs"))
        return seen

    def test_environment_configures_the_server(
        self, harness: dict[str, object], tmp_path: Path
    ) -> None:
        server_module.main([])
        config = harness["config"]
        assert isinstance(config, ServerConfig)
        assert config.db_path == tmp_path / "env.db"
        assert config.docs_dir == tmp_path / "env-docs"
        assert harness["transport"] == "stdio"
        assert harness["level"] is None

    def test_flags_override_the_environment(
        self, harness: dict[str, object], tmp_path: Path
    ) -> None:
        server_module.main(
            ["--db", str(tmp_path / "flag.db"), "--docs-dir", str(tmp_path), "--log-level", "debug"]
        )
        config = harness["config"]
        assert isinstance(config, ServerConfig)
        assert (config.db_path, config.docs_dir) == (tmp_path / "flag.db", tmp_path)
        assert harness["level"] == "debug"
        assert (tmp_path / "flag.db").exists()

    def test_service_is_closed_when_the_server_stops(self, harness: dict[str, object]) -> None:
        server_module.main([])
        service = harness["service"]
        assert isinstance(service, MarkdownMemoryService)
        with pytest.raises(DatabaseError, match="closed"):
            service.list_documents()

    def test_unusable_database_exits_with_status_one(
        self, harness: dict[str, object], tmp_path: Path
    ) -> None:
        blocker = tmp_path / "file"
        blocker.write_text("x")
        with pytest.raises(SystemExit) as raised:
            server_module.main(["--db", str(blocker / "nested" / "index.db")])
        assert raised.value.code == 1
        assert "service" not in harness


# ====================================================================== review round 2


SKILL_DOC = """# Skill Authoring

## Writing a Skill

Put this in `SKILL.md`:

```markdown
# My Skill

## Usage

Run the command:

```bash
my-skill run
```

That is all you need.

## Publishing

Push the folder.
"""

STRAY_FENCE_DOC = """# Data Pipeline

## Loading

```python
import pandas as pd

## Load the raw frame

frame = pd.read_csv("in.csv")
```

## Cleaning

Drop the nulls.

## Output

Write it back.

```
"""


class TestFenceRepairNeverDamagesWellFormedDocuments:
    def test_markdown_sample_holding_an_inner_fence_is_left_alone(self) -> None:
        document = parse(SKILL_DOC)
        assert paths(document) == [
            "Skill Authoring",
            "Skill Authoring > Writing a Skill",
            "Skill Authoring > Publishing",
        ]
        assert "```bash\nmy-skill run\n```" in document.sections[1].content

    @pytest.mark.parametrize("language", ["markdown", "md", "text", ""])
    def test_sample_languages_are_exempt_from_the_nested_opener_rule(self, language: str) -> None:
        text = (
            f"# Doc\n\nIntro.\n\n```{language}\n\n## Example\n\nbody\n\n```js\nx()\n```\n\nTail.\n"
        )
        assert paths(parse(text)) == ["Doc"]

    def test_stray_closing_fence_does_not_swallow_real_headings(self) -> None:
        document = parse(STRAY_FENCE_DOC)
        assert paths(document) == [
            "Data Pipeline",
            "Data Pipeline > Loading",
            "Data Pipeline > Cleaning",
            "Data Pipeline > Output",
        ]
        assert "## Load the raw frame" in document.sections[1].content  # still code

    def test_stray_fence_after_a_markdown_sample(self) -> None:
        text = (
            "# Doc\n\n```md\nintro\n\n## Sample\n\nbody\n```\n\n## Real Section\n\nprose\n\n```\n"
        )
        assert paths(parse(text)) == ["Doc", "Doc > Real Section"]

    def test_code_fence_holding_another_languages_opener_is_still_repaired(self) -> None:
        text = "# A\n\n```bash\nls\n\n## B\n\nB text\n\n```python\nx = 1\n```\n\n## C\n\nmore\n"
        assert paths(parse(text)) == ["A", "A > B", "A > C"]


class TestParserScaling:
    def test_each_section_is_built_once_however_many_duplicates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        body = "\n\n".join("paragraph of filler text " * 12 for _ in range(14))  # oversized
        text = "# Log\n\n" + "".join(f"## Entry\n\n{body}\n\n" for _ in range(60))
        builds = 0
        real = MarkdownParser._build_section

        def counting(self: MarkdownParser, *args: object, **kwargs: object) -> list[SectionDraft]:
            nonlocal builds
            builds += 1
            return real(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(MarkdownParser, "_build_section", counting)
        document = parse(text)
        assert builds == 62  # preamble + "Log" + 60 entries; it was 1 + 1 + (1 + 2 + ... + 60)
        found = paths(document)
        assert len(set(found)) == len(found)
        assert "Log > Entry [60] (Part 2)" in found
        entry_60 = [s for s in document.sections if s.base_path == "Log > Entry [60]"]
        assert all(s.heading_title == "Entry [60]" for s in entry_60)
        assert join_parts(entry_60) == f"## Entry\n\n{body}"


class TestFrontMatterKeys:
    @pytest.mark.parametrize(
        "front_matter",
        [
            '"title": Quoted',
            "'title': Quoted",
            "page title: Spaced\ntitle: Quoted",
            "title: Quoted",
        ],
    )
    def test_quoted_and_spaced_keys_are_yaml(self, front_matter: str) -> None:
        document = parse(f"---\n{front_matter}\n---\n\n## Body\n\ntext\n")
        assert paths(document) == [PREAMBLE_TITLE, "Body"]
        assert document.title in {"Quoted", "Body"}  # never the raw YAML as a setext heading
        assert not any(
            section.heading_level == 2 and "title" in section.heading_title.lower()
            for section in document.sections
        )


def test_heading_shares_part_one_with_an_over_long_line() -> None:
    text = "## Release Notes\n\n" + "word " * 1500 + "\n"
    document = parse(text)
    first = document.sections[0]
    assert first.heading_path == "Release Notes (Part 1)"
    assert first.content.startswith("## Release Notes\n\nword word")
    assert len(first.content) > 1000
    assert all(len(s.content) <= DEFAULT_MAX_SECTION_CHARS for s in document.sections)
    assert join_parts(document.sections) == text.rstrip("\n")  # verbatim, trailing space kept


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


class TestSearchRobustness:
    def test_upper_case_keyword_survives_its_lower_case_stopword_twin(self) -> None:
        assert build_fts_query("where is the WHERE clause") == '"WHERE" OR "clause"'
        assert build_fts_query("NOT not Not") == '"NOT"'
        assert build_fts_query("the The THE") == '"THE"'

    def test_lone_surrogate_cannot_crash_a_search(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        store(db, fake_embedder, "/d/a.md")
        expression = build_fts_query("body \ud800 number")
        assert expression is not None
        expression.encode("utf-8")
        assert db.fts_search(expression, 5)
        searcher = HybridSearcher(db, fake_embedder)
        try:
            assert searcher.search("body \ud800 number")
        finally:
            searcher.close()

    def test_unexpected_exception_in_one_index_degrades_to_the_other(
        self, db: Database, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store(db, fake_embedder, "/d/a.md")

        def broken(match_query: str, limit: int) -> list[int]:
            raise RuntimeError("not a domain error")

        monkeypatch.setattr(db, "fts_search", broken)
        searcher = HybridSearcher(db, fake_embedder)
        try:
            results = searcher.search("body number")
            assert results and all(r.fts_rank is None for r in results)
            monkeypatch.setattr(db, "vec_search", broken)
            with pytest.raises(SearchError, match="RuntimeError: not a domain error"):
                searcher.search("body number")
        finally:
            searcher.close()

    def test_search_after_close_is_a_domain_error(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        searcher = HybridSearcher(db, fake_embedder)
        searcher.close()
        with pytest.raises(SearchError, match="shut down"):
            searcher.search("anything")


class TestServerLifecycleRound2:
    def test_relative_lookup_survives_a_deleted_working_directory(
        self, arrow_service: ArrowService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service, _ = arrow_service
        doomed = tmp_path / "doomed"
        doomed.mkdir()
        monkeypatch.chdir(doomed)
        doomed.rmdir()
        try:
            with pytest.raises(OSError):
                Path.cwd()
            assert "lower body" in service.read_section("api.md", "API > Setup")
        finally:
            os.chdir(tmp_path)  # give monkeypatch a directory it can restore from

    async def test_server_that_owns_its_service_serves_many_sessions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created: list[MarkdownMemoryService] = []

        class Tracked(MarkdownMemoryService):
            def __init__(self, config: ServerConfig) -> None:
                super().__init__(config, embedder=FakeEmbedder())
                created.append(self)

        monkeypatch.setattr(server_module, "MarkdownMemoryService", Tracked)
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.md").write_text("# A\n\nsearchable body text\n")
        server = create_server(ServerConfig(db_path=tmp_path / "multi.db", docs_dir=docs))
        for session in range(3):
            async with Client(server) as client:
                assert not (await client.call_tool("index_directory", {})).is_error
                found = await client.call_tool("search_docs", {"query": "searchable body"})
                assert not found.is_error, session
        assert len(created) == 3  # one per session, each closed when its session ended
        for service in created:
            with pytest.raises(DatabaseError, match="closed"):
                service.list_documents()

    async def test_overlapping_sessions_share_one_owned_service(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created: list[MarkdownMemoryService] = []

        class Tracked(MarkdownMemoryService):
            def __init__(self, config: ServerConfig) -> None:
                super().__init__(config, embedder=FakeEmbedder())
                created.append(self)

        monkeypatch.setattr(server_module, "MarkdownMemoryService", Tracked)
        server = create_server(ServerConfig(db_path=tmp_path / "overlap.db", docs_dir=tmp_path))
        async with Client(server) as first:
            assert not (await first.call_tool("list_documents", {})).is_error
            async with Client(server) as second:
                assert not (await second.call_tool("list_documents", {})).is_error
            # the inner session ended, the outer one must still work
            assert not (await first.call_tool("list_documents", {})).is_error
        assert len(created) == 1
