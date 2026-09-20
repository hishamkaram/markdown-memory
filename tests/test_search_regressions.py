"""Search regressions: FTS query building, the IDF keyword gate, ranking and fusion.

One test per defect found in code review; each fails when its fix is reverted.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fakes import FakeEmbedder
from helpers import store

from markdown_memory.db import Database
from markdown_memory.exceptions import SearchError
from markdown_memory.indexer import Indexer
from markdown_memory.search import HybridSearcher, build_fts_query


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
