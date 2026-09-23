"""Search regressions: FTS query building, the IDF keyword gate, ranking and fusion.

One test per defect found in code review; each fails when its fix is reverted.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from fakes import FakeEmbedder, vectors_for
from helpers import draft, store

from markdown_memory.db import Database
from markdown_memory.exceptions import SearchError
from markdown_memory.indexer import Indexer
from markdown_memory.models import (
    Document,
    SearchResult,
    Section,
    SectionDraft,
)
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

        def broken(*_arguments: object) -> list[int]:
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

    def test_vectors_from_another_model_are_not_ranked_against_this_ones_query(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        """A recorded weights mismatch means the stored vectors and the vector this query

        would produce come from different models, so the distance between them measures
        nothing. Indexing already refuses; searching used to go on ranking on them and
        return the result as if it were semantic. Keyword ranking reads no vector, so it
        still answers - and no query is embedded at all.
        """
        store(db, fake_embedder, "/d/a.md")
        db.record_weights_mismatch("the weights changed")
        searcher = HybridSearcher(db, fake_embedder)
        try:
            embedded_before = len(fake_embedder.query_calls)
            results = searcher.search("body number")
            assert results and all(r.vec_rank is None for r in results)
            assert len(fake_embedder.query_calls) == embedded_before
        finally:
            searcher.close()

    def test_a_keyword_failure_during_a_mismatch_is_the_whole_search_failing(
        self, db: Database, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With vector ranking suppressed there is no second index to degrade to."""
        store(db, fake_embedder, "/d/a.md")
        db.record_weights_mismatch("the weights changed")

        def broken(*_arguments: object) -> list[int]:
            raise RuntimeError("not a domain error")

        monkeypatch.setattr(db, "fts_search", broken)
        searcher = HybridSearcher(db, fake_embedder)
        try:
            with pytest.raises(SearchError, match="RuntimeError: not a domain error"):
                searcher.search("body number")
        finally:
            searcher.close()

    def test_a_ranking_already_in_flight_when_the_weights_change_is_dropped(
        self, db: Database, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The flag is read before the vector work is submitted, so an indexer can record

        it while that work runs - against exactly the vectors it is about. Reading it
        again once the ranking is in hand costs one query and drops a ranking that means
        nothing.
        """
        store(db, fake_embedder, "/d/a.md")
        searcher = HybridSearcher(db, fake_embedder)
        original = searcher._vector_ranking

        def rank_then_change(query: str, limit: int) -> tuple[list[int], dict[int, str]]:
            result = original(query, limit)
            db.record_weights_mismatch("the weights changed mid-search")
            return result

        monkeypatch.setattr(searcher, "_vector_ranking", rank_then_change)
        try:
            results = searcher.search("body number")
            assert results and all(r.vec_rank is None for r in results)
        finally:
            searcher.close()

    def test_search_after_close_is_a_domain_error(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        searcher = HybridSearcher(db, fake_embedder)
        searcher.close()
        with pytest.raises(SearchError, match="shut down"):
            searcher.search("anything")


def keyword_sections(results: Sequence[SearchResult]) -> set[str]:
    return {result.heading_title for result in results if result.fts_rank is not None}


class TestIdentifierGateBypassNeedsRarity:
    ACRONYM_SECTIONS = (
        ("Upstream Deadlines", "ORBIT_UPSTREAM_TIMEOUT_MS: the deadline for one request."),
        ("Status Codes", "HTTP status codes from the backend are passed through unchanged."),
        ("Headers", "Hop-by-hop HTTP headers are removed before forwarding."),
        ("Compression", "HTTP bodies are compressed with gzip when the client accepts it."),
        ("Access Logs", "Every HTTP exchange is written to the access log."),
        ("Keep Alive", "Idle HTTP connections are closed after a minute."),
    )  # fmt: skip

    @staticmethod
    def fill(db: Database, embedder: FakeEmbedder, sections: Sequence[tuple[str, str]]) -> None:
        drafts = [draft(title, f"## {title}\n\n{body}") for title, body in sections]
        db.replace_document(
            file_path="/d/proxy.md", title="Doc", content_hash="h", last_modified=1,
            sections=drafts, vectors=vectors_for(embedder, drafts),
        )  # fmt: skip

    def test_common_acronym_does_not_switch_the_gate_off(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        self.fill(db, fake_embedder, self.ACRONYM_SECTIONS)
        searcher = HybridSearcher(db, fake_embedder)
        try:
            upper = searcher.search("HTTP request deadline", limit=6)
            lower = searcher.search("http request deadline", limit=6)
        finally:
            searcher.close()
        # "HTTP" is in five of six sections: a match on it alone says nothing.
        assert keyword_sections(upper) == {"Upstream Deadlines"}
        assert upper[0].heading_title == "Upstream Deadlines"
        assert [r.heading_title for r in upper] == [r.heading_title for r in lower]

    def test_rare_identifier_still_passes_the_gate_by_itself(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        sections = [*self.ACRONYM_SECTIONS, ("Disk Full", "Writes fail with ENOSPC.")]
        self.fill(db, fake_embedder, sections)
        searcher = HybridSearcher(db, fake_embedder)
        try:
            # Three words found nowhere dominate the query's IDF: only the bypass keeps it.
            results = searcher.search("ENOSPC zebra quokka wombat", limit=7)
        finally:
            searcher.close()
        assert keyword_sections(results) == {"Disk Full"}
        assert results[0].heading_title == "Disk Full"


class TestSearchDuringReindex:
    @staticmethod
    def fill(db: Database, embedder: FakeEmbedder) -> list[SectionDraft]:
        retries = [
            draft("Backoff", "## Backoff\n\nretries use exponential backoff"),
            draft("Deadlines", "## Deadlines\n\nretries stop at the deadline"),
        ]
        colours = [draft("Colours", "## Colours\n\nthe deadline banner is red")]
        for file_path, sections in (("/d/retries.md", retries), ("/d/colours.md", colours)):
            db.replace_document(
                file_path=file_path, title="Doc", content_hash="h", last_modified=1,
                sections=sections, vectors=vectors_for(embedder, sections),
            )  # fmt: skip
        return retries

    def test_sections_replaced_between_ranking_and_fetch_are_ranked_again(
        self, db: Database, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        retries = self.fill(db, fake_embedder)
        fetch = db.get_sections_with_documents
        reindexed: list[bool] = []

        def fetch_after_reindex(ids: Sequence[int]) -> dict[int, tuple[Section, Document]]:
            if not reindexed:  # a concurrent index_directory lands exactly here, once
                reindexed.append(True)
                db.replace_document(
                    file_path="/d/retries.md", title="Doc", content_hash="h2", last_modified=2,
                    sections=retries, vectors=vectors_for(fake_embedder, retries),
                )  # fmt: skip
            return fetch(ids)

        monkeypatch.setattr(db, "get_sections_with_documents", fetch_after_reindex)
        searcher = HybridSearcher(db, fake_embedder)
        try:
            results = searcher.search("retries backoff deadline", limit=5)
        finally:
            searcher.close()
        assert sorted(r.heading_title for r in results) == ["Backoff", "Colours", "Deadlines"]
        assert results[0].heading_title == "Backoff"
        current = {int(row[0]) for row in db.connection().execute("SELECT id FROM sections")}
        assert {r.section_id for r in results} == current  # no stale ids handed out

    def test_page_is_filled_from_the_next_best_when_every_pass_is_raced(
        self, db: Database, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        retries = self.fill(db, fake_embedder)
        fetch = db.get_sections_with_documents

        def fetch_after_reindex(ids: Sequence[int]) -> dict[int, tuple[Section, Document]]:
            db.replace_document(  # the re-index never stops: every fetch comes too late
                file_path="/d/retries.md", title="Doc", content_hash="h", last_modified=1,
                sections=retries, vectors=vectors_for(fake_embedder, retries),
            )  # fmt: skip
            return fetch(ids)

        monkeypatch.setattr(db, "get_sections_with_documents", fetch_after_reindex)
        searcher = HybridSearcher(db, fake_embedder)
        try:
            results = searcher.search("retries backoff deadline", limit=1)
        finally:
            searcher.close()
        assert [r.heading_title for r in results] == ["Colours"]  # not an empty page


class TestRankingIsActuallyTested:
    """Coverage for two ranking rules that every earlier test passed without."""

    def test_a_section_is_ranked_by_its_closest_passage_not_its_average(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        filler = "\n\n".join(f"unrelated billing invoice paragraph {n}" for n in range(12))
        target = SectionDraft(
            heading_title="Rotation", heading_level=2, heading_path="Doc > Rotation",
            base_path="Doc > Rotation", content=f"## Rotation\n\n{filler}", start_line=1,
            end_line=1, units=(*filler.split("\n\n"), "certificates expire after ninety days"),
        )  # fmt: skip
        # The decoy mentions two of the three query words in a short section: its own
        # vector beats the target's, which is twelve parts filler to one part answer.
        decoy = draft("Almanac", "## Almanac\n\ncertificates expire\n")
        noise = [draft(f"N{n}", f"## N{n}\n\nbilling invoice note {n}") for n in range(25)]
        sections = [target, decoy, *noise]
        db.replace_document(
            file_path="/d/a.md", title="Doc", content_hash="h", last_modified=1,
            sections=sections, vectors=vectors_for(fake_embedder, sections),
        )  # fmt: skip
        searcher = HybridSearcher(db, fake_embedder)
        try:
            ranking, passages = searcher._vector_ranking("certificates expire ninety", 20)
        finally:
            searcher.close()
        ids = {
            str(row[1]): int(row[0])
            for row in db.connection().execute("SELECT id, heading_title FROM sections")
        }
        target_id = ids["Rotation"]
        # Ranked by section vector alone the decoy wins; by closest passage the target does.
        section_only = dict(
            db.vec_search(fake_embedder.embed_query("certificates expire ninety"), 20)
        )  # noqa: E501
        assert section_only[ids["Almanac"]] < section_only[target_id]
        assert ranking[0] == target_id
        assert passages[target_id] == "certificates expire after ninety days"

    def test_keyword_hits_come_back_in_bm25_order_not_row_order(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        padding = " ".join(f"filler word {n}" for n in range(60))
        sections = [
            draft("Alpha Note", f"## Alpha Note\n\n{padding} compaction {padding}"),
            draft("Storage", f"## Storage\n\ncompaction {padding}"),
            draft("Compaction", "## Compaction\n\ncompaction compaction compaction"),
        ]
        db.replace_document(
            file_path="/d/a.md", title="Doc", content_hash="h", last_modified=1,
            sections=sections, vectors=vectors_for(fake_embedder, sections),
        )  # fmt: skip
        ids = {
            str(row[1]): int(row[0])
            for row in db.connection().execute("SELECT id, heading_title FROM sections")
        }
        expected = [ids["Compaction"], ids["Storage"], ids["Alpha Note"]]
        hits = db.fts_search('"compaction"', 10)
        assert hits[0] != min(hits)  # not simply the first row that matched
        assert hits == expected
        # A scoped search is a second query with its own ORDER BY: it ranks the same way
        # or a project-scoped server silently gets row order.
        assert db.fts_search('"compaction"', 10, "/d") == expected

    def test_a_match_in_the_heading_outranks_one_buried_in_a_body(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        padding = " ".join(f"filler word {n}" for n in range(60))
        # Six body hits against one heading hit, in bodies of the same length: with equal
        # column weights the body wins, so only bm25(5.0, 3.0, 1.0) puts the heading first.
        sections = [
            draft("Long Body", f"## Long Body\n\n{padding} " + "vacuuming " * 6),
            draft("Vacuuming", f"## Vacuuming\n\n{padding} filler word 60"),
        ]
        db.replace_document(
            file_path="/d/a.md", title="Doc", content_hash="h", last_modified=1,
            sections=sections, vectors=vectors_for(fake_embedder, sections),
        )  # fmt: skip
        first = db.fts_search('"vacuuming"', 10)[0]
        heading_match = db.connection().execute(
            "SELECT heading_title FROM sections WHERE id = ?", (first,)
        ).fetchone()  # fmt: skip
        assert heading_match[0] == "Vacuuming"  # bm25(5.0, 3.0, 1.0) weights the heading
