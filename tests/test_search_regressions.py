"""Search regressions: FTS query building, the IDF keyword gate, ranking and fusion.

One test per defect found in code review; each fails when its fix is reverted.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from fakes import FakeEmbedder, vectors_for
from helpers import draft, store

from markdown_memory.db import MODEL_META_KEY, WEIGHTS_META_KEY, WEIGHTS_MISMATCH_KEY, Database
from markdown_memory.discovery import walk_order
from markdown_memory.exceptions import DatabaseError, SearchError
from markdown_memory.indexer import Indexer
from markdown_memory.models import (
    Document,
    SearchResult,
    Section,
    SectionDraft,
)
from markdown_memory.search import (
    HybridSearcher,
    _is_identifier,
    _is_identifier_lookup,
    _Keyword,
    _Literal,
    build_fts_query,
    fts_terms,
    select_anchor,
    unnamed_rename,
)


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
        self, db: Database, tmp_path: Path
    ) -> None:
        """Search asks which weights it is running for itself, rather than trusting a flag

        an indexing run would have had to write. A cache whose weights changed while no
        document did leaves indexing a clean no-op, so no run would ever set that flag -
        and every query would go on being ranked against vectors from another model.
        """
        embedder = _RevisedEmbedder("b" * 40)
        store(db, embedder, "/d/a.md")
        db.set_meta(WEIGHTS_META_KEY, "a" * 40)
        searcher = HybridSearcher(db, embedder)
        try:
            results = searcher.search("body number")
            assert results and all(r.vec_rank is None for r in results)
        finally:
            searcher.close()

    def test_a_keyword_failure_during_a_mismatch_is_the_whole_search_failing(
        self, db: Database, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Losing one index degrades to the other; losing both is an error, and a

        suppressed vector ranking is one of them lost.
        """
        embedder = _RevisedEmbedder("b" * 40)
        store(db, embedder, "/d/a.md")
        db.set_meta(WEIGHTS_META_KEY, "a" * 40)

        def broken(*_arguments: object) -> list[int]:
            raise RuntimeError("not a domain error")

        monkeypatch.setattr(db, "fts_search", broken)
        searcher = HybridSearcher(db, embedder)
        try:
            with pytest.raises(SearchError, match="RuntimeError: not a domain error"):
                searcher.search("body number")
        finally:
            searcher.close()

    def test_the_weights_are_read_after_the_query_is_embedded_not_before(
        self, db: Database
    ) -> None:
        """The embedder loads lazily and cannot say which weights it is until it has

        loaded, so asking first would suppress vector ranking on every first query of a
        process - a healthy index answering as if it were broken.
        """
        embedder = _RevisedEmbedder("a" * 40, lazy=True)
        store(db, embedder, "/d/a.md")
        db.set_meta(WEIGHTS_META_KEY, "a" * 40)
        searcher = HybridSearcher(db, embedder)
        try:
            results = searcher.search("body number")
            assert any(r.vec_rank is not None for r in results)
        finally:
            searcher.close()

    def test_a_search_that_finds_the_weights_changed_says_so_in_the_index_status(
        self, db: Database
    ) -> None:
        """Weights can change while no document does, so no indexing run will ever write

        that down. Without this the agent got keyword-only results and an `index_status`
        still calling the index verified - told it was healthy by the one field that
        exists to say otherwise.
        """
        embedder = _RevisedEmbedder("b" * 40)
        store(db, embedder, "/d/a.md")
        db.set_meta(WEIGHTS_META_KEY, "a" * 40)
        assert "keyword ranking" not in (db.index_status("/d").message() or "")

        searcher = HybridSearcher(db, embedder)
        try:
            searcher.search("body number")
        finally:
            searcher.close()

        status = db.index_status("/d")
        assert not status.verified
        assert "only keyword ranking is used" in (status.message() or "")

    def test_weights_that_come_back_rank_again_and_leave_the_mismatch_to_a_run(
        self, db: Database, tmp_path: Path
    ) -> None:
        """Weights that agree with the index rank by vector at once, but do not withdraw a

        mismatch another process recorded: it may be the only thing telling the next index
        run that a repair is pending, and a search on the old weights used to cancel it.
        The run withdraws it once it has checked the whole index.
        """
        root = tmp_path / "docs"
        root.mkdir()
        (root / "a.md").write_text("# A\n\nbody number one\n")
        Indexer(db, _RevisedEmbedder("a" * 40)).index_directory(root)
        searcher = HybridSearcher(db, _RevisedEmbedder("b" * 40))
        try:
            searcher.search("body number")
            assert db.get_meta(WEIGHTS_MISMATCH_KEY) is not None
        finally:
            searcher.close()

        healthy = HybridSearcher(db, _RevisedEmbedder("a" * 40))
        try:
            results = healthy.search("body number")
        finally:
            healthy.close()
        assert any(r.vec_rank is not None for r in results)
        assert db.get_meta(WEIGHTS_MISMATCH_KEY) is not None, "a repair cancelled by search"

        Indexer(db, _RevisedEmbedder("a" * 40)).index_directory(root)
        assert db.get_meta(WEIGHTS_MISMATCH_KEY) is None
        assert db.index_status(str(root)).verified

    def test_vectors_written_unvouched_during_the_lookup_are_not_ranked(
        self, db: Database, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty index records no revision, so the check before the lookup passes. If a

        model that cannot name its weights stores vectors before the lookup reads them,
        the revision is still absent afterwards and the check after it used to pass too.
        """
        searcher = HybridSearcher(db, _RevisedEmbedder("b" * 40))
        lookup = searcher._nearest

        def racing(
            embedding: list[float], limit: int
        ) -> tuple[dict[int, float], dict[int, tuple[int, str]]]:
            store(db, FakeEmbedder(), "/d/a.md")  # another process, weights unnamed
            return lookup(embedding, limit)

        monkeypatch.setattr(searcher, "_nearest", racing)
        try:
            results = searcher.search("body number")
        finally:
            searcher.close()
        assert all(r.vec_rank is None for r in results)

    def test_a_search_does_not_replace_the_indexers_account_of_a_mismatch(
        self, db: Database
    ) -> None:
        """A run that finds stale vectors names the directories to re-index. A search by

        other weights used to overwrite that with a generic sentence of its own.
        """
        store(db, _RevisedEmbedder("a" * 40), "/d/a.md")
        db.set_meta(WEIGHTS_META_KEY, "a" * 40)
        db.record_weights_mismatch("stale vectors under /d/.venv")
        searcher = HybridSearcher(db, _RevisedEmbedder("b" * 40))
        try:
            results = searcher.search("body number")
        finally:
            searcher.close()
        assert results and all(r.vec_rank is None for r in results)
        assert db.get_meta(WEIGHTS_MISMATCH_KEY) == "stale vectors under /d/.venv"

    def test_a_revision_over_no_vectors_is_not_reported_as_a_mismatch(
        self, db: Database, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run that dies between claiming the revision and writing its first vector leaves

        a revision over nothing. Other weights searching it used to record a mismatch and
        mark the index unverified over vectors that do not exist - and if the claiming run
        comes back meanwhile, what its lookup finds is still not ranked by these weights.
        """
        db.set_meta(WEIGHTS_META_KEY, "a" * 40)
        searcher = HybridSearcher(db, _RevisedEmbedder("b" * 40))
        try:
            searcher.search("body number")
            assert db.get_meta(WEIGHTS_MISMATCH_KEY) is None

            lookup = searcher._nearest

            def racing(
                embedding: list[float], limit: int
            ) -> tuple[dict[int, float], dict[int, tuple[int, str]]]:
                store(db, _RevisedEmbedder("a" * 40), "/d/a.md")  # the claiming run returns
                return lookup(embedding, limit)

            monkeypatch.setattr(searcher, "_nearest", racing)
            results = searcher.search("body number")
        finally:
            searcher.close()
        # The keyword half ran before the write, so it may find nothing; the vector half
        # found the new rows, and must not have ranked them.
        assert all(r.vec_rank is None for r in results)

    def test_vectors_no_revision_vouches_for_are_not_ranked_by_named_weights(
        self, db: Database, tmp_path: Path
    ) -> None:
        """An index built while the weights could not be named records no revision. Weights

        that can name themselves have nothing saying they are the model that built it, so
        they rank by keyword alone, say why, and the next index run re-embeds the index.
        """
        root = tmp_path / "docs"
        root.mkdir()
        (root / "a.md").write_text("# A\n\nbody number one\n")
        Indexer(db, FakeEmbedder()).index_directory(root)
        assert db.get_meta(WEIGHTS_META_KEY) is None

        searcher = HybridSearcher(db, _RevisedEmbedder("b" * 40))
        try:
            results = searcher.search("body number")
        finally:
            searcher.close()
        assert results and all(r.vec_rank is None for r in results)
        assert not db.index_status(str(root)).verified

        Indexer(db, _RevisedEmbedder("b" * 40)).index_directory(root)
        assert db.get_meta(WEIGHTS_META_KEY) == "b" * 40

    def test_an_index_rebuilt_by_another_model_mid_search_is_not_ranked_on(
        self, db: Database, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checking the revision does not freeze it. Another process may re-embed the

        index while this one ranks, and rows read after the check are not the rows it
        vouched for - the query was embedded by one model and the vectors
        it is measured against were written by another.
        """
        embedder = _RevisedEmbedder("a" * 40)
        store(db, embedder, "/d/a.md")
        db.set_meta(WEIGHTS_META_KEY, "a" * 40)
        original = db.vec_search

        def rebuild_then_search(*arguments: object) -> list[tuple[int, float]]:
            db.set_meta(WEIGHTS_META_KEY, "b" * 40)  # another process got there first
            return original(*arguments)  # type: ignore[arg-type]

        monkeypatch.setattr(db, "vec_search", rebuild_then_search)
        searcher = HybridSearcher(db, embedder)
        try:
            results = searcher.search("body number")
            assert results and all(r.vec_rank is None for r in results)
        finally:
            searcher.close()

    def test_vectors_another_model_built_are_not_ranked_when_neither_names_its_weights(
        self, db: Database
    ) -> None:
        """With no revision on either side, only the stored model name can tell (#91).

        Indexing refuses to write over such an index, so it lasts until the person acts,
        and every query in between would have been measured against another model's
        vectors. Nothing is recorded: what is derived from the stored name goes away the
        moment the old model is back, where a recorded mismatch would outlive it.
        """
        store(db, FakeEmbedder(), "/d/a.md")
        db.set_meta(MODEL_META_KEY, "model-a")
        renamed = FakeEmbedder(model_name="model-b")
        assert "built by model-a" in (unnamed_rename(db, renamed) or "")
        for embedder, ranked in ((renamed, False), (FakeEmbedder(model_name="model-a"), True)):
            searcher = HybridSearcher(db, embedder)
            try:
                results = searcher.search("body number")
            finally:
                searcher.close()
            assert results and any(r.vec_rank is not None for r in results) is ranked
        assert db.get_meta(WEIGHTS_MISMATCH_KEY) is None

    def test_a_model_renamed_mid_search_is_not_ranked_on(
        self, db: Database, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When neither model names its weights, the revision stays None throughout, so

        the name is all that changes when an index is taken over - an empty one, filled by
        another model while this search was ranking.
        """
        embedder = FakeEmbedder(model_name="model-a")
        store(db, embedder, "/d/a.md")
        db.set_meta(MODEL_META_KEY, "model-a")
        original = db.vec_search

        def rename_then_search(*arguments: object) -> list[tuple[int, float]]:
            db.set_meta(MODEL_META_KEY, "model-b")  # another process took the index over
            return original(*arguments)  # type: ignore[arg-type]

        monkeypatch.setattr(db, "vec_search", rename_then_search)
        searcher = HybridSearcher(db, embedder)
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


class _RevisedEmbedder(FakeEmbedder):
    """A FakeEmbedder that reports a weights revision, optionally only once loaded."""

    def __init__(self, revision: str, *, lazy: bool = False) -> None:
        super().__init__()
        self._revision = revision
        self._loaded = not lazy

    @property
    def weights_revision(self) -> str | None:
        return self._revision if self._loaded else None

    def embed_query(self, text: str) -> list[float]:
        self._loaded = True  # the query is what loads the model
        return super().embed_query(text)


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
            file_path="/d/proxy.md", title="Doc", content_hash="h", last_modified=1, mtime_ns=1,
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
                file_path=file_path, title="Doc", content_hash="h", last_modified=1, mtime_ns=1,
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
                    mtime_ns=2,
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
                mtime_ns=1,
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
            file_path="/d/a.md", title="Doc", content_hash="h", last_modified=1, mtime_ns=1,
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
        )
        assert section_only[ids["Almanac"]] < section_only[target_id]
        assert ranking[0] == target_id
        ordinal, text = passages[target_id]
        assert text == "certificates expire after ninety days"
        assert db.units_of(target_id)[ordinal] == text  # the ordinal says where it sits

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
            file_path="/d/a.md", title="Doc", content_hash="h", last_modified=1, mtime_ns=1,
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
            file_path="/d/a.md", title="Doc", content_hash="h", last_modified=1, mtime_ns=1,
            sections=sections, vectors=vectors_for(fake_embedder, sections),
        )  # fmt: skip
        first = db.fts_search('"vacuuming"', 10)[0]
        heading_match = db.connection().execute(
            "SELECT heading_title FROM sections WHERE id = ?", (first,)
        ).fetchone()  # fmt: skip
        assert heading_match[0] == "Vacuuming"  # bm25(5.0, 3.0, 1.0) weights the heading


def test_a_revoked_index_keeps_the_reason_the_indexer_gave(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """While a run re-embeds the index, the revision matches no model, and search used to

    overwrite the indexer's account - which names what is left to do - with its own guess
    that the index needed rebuilding from scratch, naming the sentinel as a revision.
    """
    with Database(tmp_path / "index.db") as db:
        store(db, FakeEmbedder(), "/d/a.md")
        db.revoke_weights("re-embedding under /d")
        searcher = HybridSearcher(db, FakeEmbedder(weights="b" * 40))
        try:
            results = searcher.search("body number")
        finally:
            searcher.close()
        assert results and all(result.vec_rank is None for result in results)
        assert db.get_meta("embedding_weights_mismatch") == "re-embedding under /d"
    assert "being re-embedded" in caplog.text


class TestIdentifierLookups:
    """#75: an identifier lookup is answered by sections that name the identifier itself.

    The tokenizer reads `GH_REPO` as the words `gh repo` and `--pre` as `pre`, so prose
    holding those words used to rank like the identifier - `GH_REPO` returned a page that
    never named it.
    """

    @staticmethod
    def searcher(
        db: Database, fake_embedder: FakeEmbedder, root: Path, files: dict[str, str]
    ) -> HybridSearcher:
        root.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():  # in order: the first file gets the lowest ids
            (root / name).write_text(text)
            Indexer(db, fake_embedder).index_directory(root)
        return HybridSearcher(db, fake_embedder)

    @staticmethod
    def vectors_rank(
        searcher: HybridSearcher, monkeypatch: pytest.MonkeyPatch, *paths: str
    ) -> None:
        """Make the vector side rank these heading paths first, whatever the fake embedder says."""
        rows = searcher._db.connection().execute("SELECT id, heading_path FROM sections")
        ids = {path: int(sid) for sid, path in rows}
        ranking = [ids[path] for path in paths]
        monkeypatch.setattr(searcher, "_vector_ranking", lambda query, limit: (ranking, {}))

    @pytest.mark.parametrize(
        ("query", "text", "found"),
        [
            ("--pre", "run with --pre-glob '*.gz'", False),
            ("--pre", "run with --prefix", False),
            ("--pre", "run with --pre=cat", True),
            ("--pre", "flags (--pre) and more", True),
            ("GH_REPO", "set GH_REPOSITORY instead", False),
            ("GH_REPO", "export gh_repo=cli/cli", True),
            ("restart_policy", "see deploy.restart_policy below", True),
            ("v1.2", "since v1.2.3 only", False),
            ("v1.2", "since v1.2.", True),
            ("os.path", "call os.path.join", False),
            ("foo_bar", "then foo_bar..baz", True),
            ("foo_bar", "then foo_bar.-baz", True),
            ("histogram_quantile()", "histogram_quantile(0.9, rate(x[5m]))", True),
            ("histogram_quantile()", "histogram_quantile (0.9, x)", True),
            ("histogram_quantile()", "histogram_quantiles(x)", False),
            ("histogram_quantile()", "a bare histogram_quantile here", False),
            ("ENOSPC.", "fails with ENOSPC when full", True),
            ("`--pre`", "run with --pre cat", True),
            ("`GH_REPO`.", "export GH_REPO=cli/cli", True),
            ('"GH_REPO",', "export GH_REPO=cli/cli", True),
            ("'--pre'?", "run with --pre cat", True),
            ("`histogram_quantile`()", "histogram_quantile(0.9, x)", True),
        ],
    )
    def test_an_identifier_is_found_as_itself_and_not_inside_another(
        self, query: str, text: str, found: bool
    ) -> None:
        literal = _Literal(fts_terms(query)[0])
        assert literal.found(text) is found

    @pytest.mark.parametrize(
        "title",
        [
            "histogram_quantile()",
            "`histogram_quantile()`",
            "`histogram_quantile`()",
            "`histogram_quantile` ()",
        ],
    )
    def test_a_heading_naming_the_call_heads_it_however_it_is_quoted(self, title: str) -> None:
        assert _Literal(fts_terms("histogram_quantile()")[0]).heads(title)

    def test_exact_case_is_told_apart_from_a_case_folded_match(self) -> None:
        literal = _Literal(fts_terms("GH_REPO")[0])
        assert literal.found("export gh_repo=x") and not literal.exact("export gh_repo=x")
        assert literal.exact("export GH_REPO=x")
        assert not literal.exact("export GH_REPOSITORY=x"), "case is checked at the same edges"

    def test_the_section_naming_the_identifier_beats_prose_holding_its_words(
        self,
        db: Database,
        fake_embedder: FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        searcher = self.searcher(db, fake_embedder, tmp_path / "docs", {
            "a.md": "# Primer\n\n## Language\n\nRun gh repo view, then gh repo clone: gh repo.\n",
            "b.md": "# Hosts\n\n## Choose the host\n\nSet GH_REPO to pick the repository.\n",
        })  # fmt: skip
        try:
            # Vectors prefer the prose; on its own the keyword hit would tie at 1/61 and the
            # prose, which the walk reaches first, would win.
            self.vectors_rank(searcher, monkeypatch, "Primer > Language")
            page = searcher.search_page("GH_REPO", limit=1)
        finally:
            searcher.close()
        assert [r.heading_path for r in page.results] == ["Hosts > Choose the host"]
        assert page.keyword_match == "matched"

    def test_a_literal_beyond_the_first_twenty_keyword_hits_is_found(
        self,
        db: Database,
        fake_embedder: FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        prose = "".join(
            f"## pre_start {n}\n\nThe pre_start hook {n} runs pre tasks, pre first.\n\n"
            for n in range(25)
        )
        searcher = self.searcher(db, fake_embedder, tmp_path / "docs", {
            "compose.md": f"# Compose\n\n{prose}",
            "guide.md": "# Guide\n\n## Preprocessor\n\nRun a command on each file with --pre.\n",
        })  # fmt: skip
        try:
            monkeypatch.setattr(searcher, "_vector_ranking", lambda query, limit: ([], {}))
            assert "Guide > Preprocessor" not in [
                r.heading_path for r in searcher.search("pre", limit=20)
            ], "the fixture must bury the literal past the first twenty keyword hits"
            results = searcher.search("--pre", limit=5)
        finally:
            searcher.close()
        assert results[0].heading_path == "Guide > Preprocessor"

    def test_the_section_headed_by_the_identifier_comes_first_and_its_first_part_first(
        self,
        db: Database,
        fake_embedder: FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Later paragraphs name the function more often, so BM25 alone would put Part 2 first.
        definition = "\n\n".join(
            f"histogram_quantile(φ, b) paragraph {n}: "
            + "histogram_quantile() " * (n // 4)
            + "quantile words " * 30
            for n in range(12)
        )
        searcher = self.searcher(db, fake_embedder, tmp_path / "docs", {
            "functions.md": "# Functions\n\n## `histogram_quantiles()`\n\n"
            "Like histogram_quantile(), for several quantiles at once.\n\n"
            f"## `histogram_quantile()`\n\n{definition}\n",
        })  # fmt: skip
        try:
            parts = [
                path
                for (path,) in db.connection().execute("SELECT heading_path FROM sections")
                if path.startswith("Functions > histogram_quantile() (Part")
            ]
            assert len(parts) >= 2, "the fixture must split the definition into parts"
            # Vectors prefer the neighbour and the last part.
            self.vectors_rank(
                searcher, monkeypatch, "Functions > histogram_quantiles()", sorted(parts)[-1]
            )
            results = searcher.search("histogram_quantile()", limit=3)
        finally:
            searcher.close()
        assert results[0].heading_path == "Functions > histogram_quantile() (Part 1)"
        assert "Functions > histogram_quantiles()" not in [r.heading_path for r in results[:2]]

    def test_the_spelling_asked_for_ranks_first_among_sections_naming_it(
        self,
        db: Database,
        fake_embedder: FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        searcher = self.searcher(db, fake_embedder, tmp_path / "docs", {
            "a.md": "# Env\n\n## Lower\n\nexport gh_repo=cli/cli for scripts, gh_repo again.\n",
            "b.md": "# Env\n\n## Upper\n\nSet GH_REPO once.\n",
        })  # fmt: skip
        try:
            monkeypatch.setattr(searcher, "_vector_ranking", lambda query, limit: ([], {}))
            results = searcher.search("GH_REPO", limit=2)
        finally:
            searcher.close()
        assert [r.heading_path for r in results] == ["Env > Upper", "Env > Lower"]

    def test_an_identifier_named_nowhere_is_ranked_exactly_as_before(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        searcher = self.searcher(db, fake_embedder, tmp_path / "docs", {
            "a.md": "# Notes\n\n## Words\n\nA non existent value is non existent.\n",
        })  # fmt: skip
        try:
            assert searcher._keyword_pass("NON_EXISTENT", 20) == _Keyword(
                *searcher._keyword_ranking("NON_EXISTENT", 20)
            )
        finally:
            searcher.close()

    def test_an_identifier_named_everywhere_is_vocabulary(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        sections = "".join(f"## Part {n}\n\nServe HTTP on port {n}.\n\n" for n in range(6))
        searcher = self.searcher(db, fake_embedder, tmp_path / "docs", {
            "a.md": f"# Server\n\n{sections}",
        })  # fmt: skip
        try:
            plain = _Keyword(*searcher._keyword_ranking("HTTP", 20))
            assert searcher._keyword_pass("HTTP", 20) == plain
        finally:
            searcher.close()

    def test_a_term_common_beyond_the_candidates_checked_is_vocabulary(
        self,
        db: Database,
        fake_embedder: FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Three candidates checked, all naming HTTP, of eight that do: the sample scales up."""
        import markdown_memory.search as search

        monkeypatch.setattr(search, "LITERAL_CANDIDATES", 3)
        sections = "".join(f"## Part {n}\n\nServe HTTP on port {n}.\n\n" for n in range(8))
        searcher = self.searcher(db, fake_embedder, tmp_path / "docs", {
            "a.md": f"# Server\n\n{sections}",
        })  # fmt: skip
        try:
            assert searcher._keyword_pass("HTTP", 2) == _Keyword(
                *searcher._keyword_ranking("HTTP", 2)
            )
        finally:
            searcher.close()

    def test_an_identifier_whose_words_are_common_is_still_rare(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """`--files` matches every "files" in the keyword index; one section names the flag."""
        sections = "".join(f"## Part {n}\n\nList the files of part {n}.\n\n" for n in range(6))
        searcher = self.searcher(db, fake_embedder, tmp_path / "docs", {
            "a.md": f"# Guide\n\n{sections}## Listing\n\nPass --files to list them.\n",
        })  # fmt: skip
        try:
            ranked = searcher._keyword_pass("--files", 20)
        finally:
            searcher.close()
        titles = {
            int(sid): path
            for sid, path in db.connection().execute("SELECT id, heading_path FROM sections")
        }
        assert [titles[sid] for sid in ranked.ranking] == ["Guide > Listing"]

    def test_rarity_is_judged_per_term(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        # Every section survives the gate holding both terms' words; only one names GH_REPO.
        sections = "".join(
            f"## Part {n}\n\nServe HTTP from the gh repo on port {n}.\n\n" for n in range(6)
        )
        searcher = self.searcher(db, fake_embedder, tmp_path / "docs", {
            "a.md": f"# Server\n\n{sections}## Repo\n\nSet GH_REPO to serve HTTP.\n",
        })  # fmt: skip
        try:
            ranked = searcher._keyword_pass("HTTP GH_REPO", 20)
        finally:
            searcher.close()
        titles = {
            int(sid): path
            for sid, path in db.connection().execute("SELECT id, heading_path FROM sections")
        }
        assert [titles[sid] for sid in ranked.ranking] == ["Server > Repo"]

    def test_a_section_vanishing_while_checked_asks_for_a_second_pass(
        self,
        db: Database,
        fake_embedder: FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        searcher = self.searcher(db, fake_embedder, tmp_path / "docs", {
            "a.md": "# Hosts\n\n## One\n\nSet GH_REPO.\n\n## Two\n\nGH_REPO again.\n",
        })  # fmt: skip
        hydrate = searcher._db.get_sections_with_documents

        def lose_one(ids: Sequence[int]) -> dict[int, tuple[Section, Document]]:
            found = hydrate(ids)
            found.pop(next(iter(found)))  # replaced by a concurrent re-index
            return found

        try:
            monkeypatch.setattr(searcher._db, "get_sections_with_documents", lose_one)
            assert searcher._keyword_pass("GH_REPO", 20).stale
        finally:
            searcher.close()

    def test_losing_the_only_section_naming_it_still_asks_for_a_second_pass(
        self,
        db: Database,
        fake_embedder: FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        searcher = self.searcher(db, fake_embedder, tmp_path / "docs", {
            "a.md": "# Hosts\n\n## One\n\nSet GH_REPO.\n\n## Two\n\nRun gh repo view.\n",
        })  # fmt: skip
        hydrate = searcher._db.get_sections_with_documents

        def lose_the_literal(ids: Sequence[int]) -> dict[int, tuple[Section, Document]]:
            found = hydrate(ids)
            for sid, (section, _) in list(found.items()):
                if "GH_REPO" in section.content:
                    del found[sid]  # replaced by a concurrent re-index
            return found

        try:
            monkeypatch.setattr(searcher._db, "get_sections_with_documents", lose_the_literal)
            ranked = searcher._keyword_pass("GH_REPO", 20)
        finally:
            searcher.close()
        assert ranked.stale and not ranked.literal

    def test_an_empty_page_from_a_pass_that_lost_sections_is_ranked_again(
        self,
        db: Database,
        fake_embedder: FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The section naming it vanished, then nothing matched: the replacement is coming."""
        searcher = self.searcher(db, fake_embedder, tmp_path / "docs", {
            "a.md": "# Hosts\n\n## One\n\nSet GH_REPO.\n",
        })  # fmt: skip
        real = searcher._keyword_pass
        passes: list[str] = []

        def first_pass_raced(query: str, limit: int) -> _Keyword:
            passes.append(query)
            return _Keyword([], "no_match", stale=True) if len(passes) == 1 else real(query, limit)

        monkeypatch.setattr(searcher, "_keyword_pass", first_pass_raced)
        try:
            page = searcher.search_page("GH_REPO", 5)
        finally:
            searcher.close()
        assert len(passes) == 2 and [r.heading_path for r in page.results] == ["Hosts > One"]

    def test_a_term_is_sampled_among_the_candidates_matching_it_not_the_whole_pool(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """HTTP is named in 8 of 30 sections (rare is 3); the pool checked holds only 2."""
        serve = "".join(f"## Serve {n}\n\nServe HTTP on port {n}.\n\n" for n in range(8))
        words = "".join(f"## Words {n}\n\nThe gh repo command, take {n}.\n\n" for n in range(20))
        searcher = self.searcher(db, fake_embedder, tmp_path / "docs", {
            "a.md": f"# Guide\n\n{serve}{words}## Repo\n\nSet GH_REPO.\n",
        })  # fmt: skip
        rows = db.connection().execute("SELECT id, heading_path FROM sections").fetchall()
        ids = {path: int(sid) for sid, path in rows}
        pool = [ids["Guide > Repo"], ids["Guide > Serve 0"], ids["Guide > Serve 1"]]
        pool += [ids[f"Guide > Words {n}"] for n in range(17)]
        try:
            ranked = searcher._literal_ranking(fts_terms("GH_REPO HTTP"), pool, 20)
        finally:
            searcher.close()
        assert ranked.ranking == [ids["Guide > Repo"]]

    def test_rarity_in_one_root_is_not_judged_by_another_roots_words(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        words = "".join(f"## Words {n}\n\nThe gh repo command, take {n}.\n\n" for n in range(10))
        self.searcher(db, fake_embedder, tmp_path / "other", {
            "b.md": f"# Other\n\n{words}",
        }).close()  # fmt: skip
        mine = tmp_path / "mine"
        self.searcher(db, fake_embedder, mine, {
            "a.md": "# Mine\n\n## Host\n\nSet GH_REPO here.\n",
        }).close()  # fmt: skip
        scoped = HybridSearcher(db, fake_embedder, scope=str(mine))
        try:
            ranked = scoped._keyword_pass("GH_REPO", 20)
        finally:
            scoped.close()
        assert ranked.literal and len(ranked.ranking) == 1

    def test_a_scoped_search_never_answers_from_another_root(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        self.searcher(db, fake_embedder, tmp_path / "other", {
            "b.md": "# Other\n\n## Host\n\nSet GH_REPO here.\n",
        }).close()  # fmt: skip
        mine = tmp_path / "mine"
        searcher = self.searcher(db, fake_embedder, mine, {
            "a.md": "# Mine\n\n## Prose\n\nRun gh repo view.\n",
        })  # fmt: skip
        searcher.close()
        scoped = HybridSearcher(db, fake_embedder, scope=str(mine))
        try:
            paths = [r.file_path for r in scoped.search("GH_REPO", limit=5)]
        finally:
            scoped.close()
        assert paths and all(path.startswith(str(mine)) for path in paths)

    def test_a_query_mixing_words_and_an_identifier_is_untouched(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        searcher = self.searcher(db, fake_embedder, tmp_path / "docs", {
            "a.md": "# Primer\n\n## Language\n\nRun gh repo view.\n",
            "b.md": "# Hosts\n\n## Choose\n\nSet GH_REPO to pick the repository.\n",
        })  # fmt: skip
        try:
            query = "how do I set GH_REPO for scripts"
            assert searcher._keyword_pass(query, 20) == _Keyword(
                *searcher._keyword_ranking(query, 20)
            )
        finally:
            searcher.close()


class TestRarityIsCountedInTheSearchedRoot:
    """#83: another root in the same database leaves this root's gate and lookups as they were.

    Rarity was sized from every section in the database, and IDF from every root's words: a
    large neighbour made this root's common words rare, and its own words common.
    """

    @staticmethod
    def fill(db: Database, embedder: FakeEmbedder, path: str, bodies: Sequence[str]) -> None:
        drafts = [draft(f"S{n}", f"## S{n}\n\n{body}") for n, body in enumerate(bodies)]
        db.replace_document(
            file_path=path, title="Doc", content_hash="h", last_modified=1, mtime_ns=1,
            sections=drafts, vectors=vectors_for(embedder, drafts),
        )  # fmt: skip

    @staticmethod
    def titles(db: Database, ranked: tuple[list[int], object]) -> list[str]:
        found = db.get_sections_with_documents(ranked[0])
        return [found[sid][0].heading_title for sid in ranked[0]]

    def test_a_large_neighbour_does_not_make_a_common_term_rare(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        """HTTP is in 4 of 10 sections here (rare is 3); 310 sections in all would make it 15."""
        filler = [f"Plain words, take {n}." for n in range(5)]
        http = [f"Serve HTTP on port {n}." for n in range(4)]
        self.fill(db, fake_embedder, "/mine/a.md", ["Set the request deadline.", *http, *filler])
        searcher = HybridSearcher(db, fake_embedder, scope="/mine")
        try:
            alone = searcher._keyword_ranking("HTTP deadline", 20)
            self.fill(db, fake_embedder, "/other/b.md", ["Unrelated prose."] * 300)
            beside = searcher._keyword_ranking("HTTP deadline", 20)
        finally:
            searcher.close()
        assert alone == beside
        assert self.titles(db, alone) == ["S0"]

    def test_another_roots_words_do_not_reweigh_this_roots_query(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        """alpha (2 sections) outweighs beta (3) here; 300 more alphas would invert that."""
        bodies = ["alpha one.", "alpha two.", "beta one.", "beta two.", "beta three."]
        self.fill(db, fake_embedder, "/mine/a.md", [*bodies, *(f"Filler {n}." for n in range(5))])
        searcher = HybridSearcher(db, fake_embedder, scope="/mine")
        try:
            alone = searcher._keyword_ranking("alpha beta", 20)
            self.fill(db, fake_embedder, "/other/b.md", ["alpha elsewhere."] * 300)
            beside = searcher._keyword_ranking("alpha beta", 20)
        finally:
            searcher.close()
        assert alone == beside
        assert set(self.titles(db, alone)) == {"S0", "S1"}

    def test_a_large_neighbour_does_not_make_a_common_identifier_one(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        """The issue's case: API is in 5 of 10 sections here, vocabulary, not an identifier."""
        api = [f"Call the API, take {n}." for n in range(5)]
        self.fill(db, fake_embedder, "/mine/a.md", [*api, *(f"Filler {n}." for n in range(5))])
        searcher = HybridSearcher(db, fake_embedder, scope="/mine")
        try:
            alone = searcher._keyword_pass("API", 20)
            self.fill(db, fake_embedder, "/other/b.md", ["Unrelated prose."] * 300)
            beside = searcher._keyword_pass("API", 20)
        finally:
            searcher.close()
        assert not alone.literal and alone == beside


class TestPlainCalls:
    """#79: a plain call (`rate()`) is an identifier lookup, answered by sections calling it.

    The tokenizer reads `rate()` as the word `rate`, and `clamp()` matches `clamp_max()`'s
    words too: without being read as a call, a function was ranked like the word it spells.
    """

    searcher = staticmethod(TestIdentifierLookups.searcher)
    vectors_rank = staticmethod(TestIdentifierLookups.vectors_rank)

    @pytest.mark.parametrize(
        ("query", "definition", "distractor"),
        [
            ("clamp()", "Functions > clamp()", "Helpers > clamp_max()"),
            ("`clamp()`", "Functions > clamp()", "Helpers > clamp_max()"),
            ('"clamp()".', "Functions > clamp()", "Helpers > clamp_max()"),
            ("clamp()?", "Functions > clamp()", "Helpers > clamp_max()"),
            ("scalar()", "Functions > scalar()", "Helpers > Scalars"),
            ("`scalar()`", "Functions > scalar()", "Helpers > Scalars"),
        ],
    )
    def test_plain_call_definition_wins(
        self,
        db: Database,
        fake_embedder: FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        query: str,
        definition: str,
        distractor: str,
    ) -> None:
        searcher = self.searcher(db, fake_embedder, tmp_path / "docs", {
            "helpers.md": (
                "# Helpers\n\n## clamp_max()\n\nclamp_max(v, max) is clamp() with one bound: "
                "clamp clamp clamp clamp.\n\n## Scalars\n\nA scalar is a number: scalar "
                "scalar scalar values.\n"
            ),
            "functions.md": (
                "# Functions\n\n## clamp()\n\nclamp(v, min, max) bounds v.\n\n"
                "## scalar()\n\nscalar(v) returns its sample value.\n"
            ),
        })  # fmt: skip
        try:
            # Vectors and BM25 both prefer the distractor: only reading the call puts the
            # section it heads first.
            self.vectors_rank(searcher, monkeypatch, distractor)
            assert (
                searcher.search(query.strip('`"?.').removesuffix("()"), limit=1)[0].heading_path
                == distractor
            ), "the fixture must favour the distractor on words"
            page = searcher.search_page(query, limit=3)
        finally:
            searcher.close()
        top = page.results[0]
        assert (Path(top.file_path).name, top.heading_path) == ("functions.md", definition)
        assert page.keyword_match == "matched"

    @pytest.mark.parametrize("call", ["rate(http_requests_total[5m])", "rate (x)"])
    def test_plain_call_matches_arguments_at_identifier_boundaries(
        self,
        db: Database,
        fake_embedder: FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        call: str,
    ) -> None:
        searcher = self.searcher(db, fake_embedder, tmp_path / "docs", {
            "a.md": (
                "# Notes\n\n## Instant\n\nirate(x) is the instant rate: rate rate rate.\n\n"
                "## Limits\n\nrate_limit(x) caps the rate of rate rate requests.\n\n"
                "## Prose\n\nThe rate rate rate rate of requests.\n"
            ),
            "b.md": f"# Queries\n\n## Counters\n\nPer-second change: {call}.\n",
        })  # fmt: skip
        try:
            self.vectors_rank(searcher, monkeypatch, "Notes > Prose", "Notes > Limits")
            assert searcher.search("rate", limit=1)[0].heading_path != "Queries > Counters"
            page = searcher.search_page("rate()", limit=4)
        finally:
            searcher.close()
        assert page.results[0].heading_path == "Queries > Counters"
        assert page.keyword_match == "matched"
        literal = _Literal(fts_terms("rate()")[0])
        assert literal.found(call)
        assert not any(map(literal.found, ["irate(x)", "rate_limit(x)", "the rate of"]))

    @pytest.mark.parametrize("call", ["all()", "any()", "is()"])
    def test_stopword_call_survives_mixed_query(
        self,
        db: Database,
        fake_embedder: FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        call: str,
    ) -> None:
        name = call.removesuffix("()")
        searcher = self.searcher(db, fake_embedder, tmp_path / "docs", {
            "a.md": (
                f"# Functions\n\n## Predicates\n\nUse {name}(xs) on a list.\n\n"
                "## Sorting\n\nOrder values by key.\n"
            ),
        })  # fmt: skip
        try:
            monkeypatch.setattr(searcher, "_vector_ranking", lambda query, limit: ([], {}))
            page = searcher.search_page(f"{call} zebra quokka", limit=2)
        finally:
            searcher.close()
        assert [r.heading_path for r in page.results] == ["Functions > Predicates"]
        assert page.results[0].fts_rank is not None and page.keyword_match == "matched"

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("is latency", '"latency"'),
            ("is() latency", '"is()" OR "latency"'),
            ("is() the", '"is()"'),  # the call is meaningful: `the` no longer rides along
            ("is() IS() latency", '"is()" OR "latency"'),  # the first spelling is kept
            ("IS() is() latency", '"IS()" OR "latency"'),
            ("`is()` latency", '"`is()`" OR "latency"'),
        ],
    )
    def test_stopword_call_is_kept_and_deduplicated(
        self,
        db: Database,
        fake_embedder: FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        query: str,
        expected: str,
    ) -> None:
        searcher = self.searcher(db, fake_embedder, tmp_path / "docs", {
            "a.md": "# Checks\n\n## Predicates\n\nCall is(x) to test latency budgets.\n",
        })  # fmt: skip
        searched: list[str] = []
        fts_search = db.fts_search

        def recording(match_query: str, limit: int, scope: str | None = None) -> list[int]:
            searched.append(match_query)
            return fts_search(match_query, limit, scope)

        monkeypatch.setattr(db, "fts_search", recording)
        try:
            page = searcher.search_page(query, limit=1)
        finally:
            searcher.close()
        assert searched[0] == expected  # the expression the keyword index was asked
        assert page.keyword_match == "matched"
        assert page.results[0].heading_path == "Checks > Predicates"

    @staticmethod
    def corpus(db: Database, fake_embedder: FakeEmbedder, root: Path) -> HybridSearcher:
        notes = "".join(f"## Orbit {n}\n\nThe orbit of body {n}.\n\n" for n in range(6))
        return TestIdentifierLookups.searcher(db, fake_embedder, root, {
            "a.md": f"# Notes\n\n{notes}## Rates\n\nThe rate of requests per second.\n",
        })  # fmt: skip

    @pytest.mark.parametrize("query", ["zqxabsent()", "`zqxabsent()`", "zqxone() zqxtwo()"])
    def test_plain_call_nothing_contains_abstains(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path, query: str
    ) -> None:
        searcher = self.corpus(db, fake_embedder, tmp_path / "docs")
        try:
            page = searcher.search_page(query)
        finally:
            searcher.close()
        assert (page.results, page.keyword_match) == ((), "no_match")

    @pytest.mark.parametrize("count", [31, 32, 33])
    def test_plain_call_abstention_stops_where_the_query_was_cut(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path, count: int
    ) -> None:
        searcher = self.corpus(db, fake_embedder, tmp_path / "docs")
        try:
            # Letters only: a digit would already make each term an identifier.
            names = (f"absent{chr(97 + n // 26)}{chr(97 + n % 26)}()" for n in range(count))
            page = searcher.search_page(" ".join(names))
        finally:
            searcher.close()
        assert page.keyword_match == "no_match"
        assert bool(page.results) is (count >= 32)

    @pytest.mark.parametrize(
        "query",
        [
            "zebra giraffe",
            "zqxabsent(x)",
            "zqxabsent(",
            "zqxabsent( )",
            "cargo zqxabsent",
            "zqxabsent(s)",  # prose (#96): `flag(s)` is a plural, not a call
            "zqxabsent(v)",
        ],
    )
    def test_plain_call_syntax_does_not_expand_other_queries(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path, query: str
    ) -> None:
        searcher = self.corpus(db, fake_embedder, tmp_path / "docs")
        try:
            page = searcher.search_page(query)
        finally:
            searcher.close()
        assert page.keyword_match == "no_match" and page.results, "not a lookup: neighbours"

    @pytest.mark.parametrize(
        ("term", "expected"),
        [
            ("rate()", True),
            ("`rate()`", True),
            ("rate().", True),
            ("rate()..", True),  # already: a `.` before the last character marks a path
            ("rate", False),
            ("rate(x)", False),
            ("rate(", False),
            ("`rate`()", False),
        ],
    )
    def test_only_an_empty_call_is_read_as_one(self, term: str, expected: bool) -> None:
        assert _is_identifier(fts_terms(term)[0]) is expected
        assert not _is_identifier_lookup(fts_terms("cargo metadata"))

    @pytest.mark.parametrize(
        ("query", "terms"),
        [
            ("rate(x[5m])", ["rate()"]),
            ("`rate(x[5m])`", ["rate()"]),
            ("rate(x[5m])?", ["rate()"]),
            ("rate(my_metric[5m])", ["rate()"]),
            # The inner call closes the outer one: an argument, not a second call to rank.
            ("histogram_quantile(0.9, rate(x[5m]))", ["histogram_quantile()", "rate(x[5m]))"]),
            ("sum(rate(x[1m]))", ["sum()"]),  # the outer call only: arguments are not parsed
            ("step(1)", ["step()"]),
            ("rate(x[5m]) rate(y[1m])", ["rate()"]),
            ("how does rate(x[5m]) work", ["rate()", "work"]),
            ("abs(v)", ["abs(v)"]),  # nothing in the arguments reads as code
            ("rate(errors)", ["rate(errors)"]),
            ("flag(s)", ["flag(s)"]),
            ("option(s)", ["option(s)"]),
            ("e.g.(x)", ["e.g.(x)"]),
            ("obj.method(1)", ["obj.method(1)"]),
            ("x[5m]", ["x[5m]"]),
            ("rate(", ["rate("]),
        ],
    )
    def test_a_call_written_with_arguments_is_searched_as_the_call(
        self, query: str, terms: list[str]
    ) -> None:
        """#96: the arguments are the asker's own metric and range; the function is documented."""
        assert fts_terms(query) == [f'"{term}"' for term in terms]

    @pytest.mark.parametrize(
        ("query", "definition"),
        [
            ("rate(x[5m])", "rate()"),
            ("`rate(x[5m])`", "rate()"),
            ("rate(x[5m])?", "rate()"),
            ("rate(my_metric[5m])", "rate()"),
            ("histogram_quantile(0.9, rate(x[5m]))", "histogram_quantile()"),
        ],
    )
    def test_a_call_written_with_arguments_is_answered_by_its_definition(
        self,
        db: Database,
        fake_embedder: FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        query: str,
        definition: str,
    ) -> None:
        """No document holds the asker's metric, so the whole call matched nothing: empty page."""
        searcher = self.searcher(db, fake_embedder, tmp_path / "docs", {
            "functions.md": (
                "# Functions\n\n## irate()\n\nirate(v range-vector) is the instant rate.\n\n"
                "## rate()\n\nrate(v range-vector) is the per-second average rate.\n\n"
                "## histogram_quantile()\n\nhistogram_quantile(φ scalar, b instant-vector) "
                "takes the φ-quantile of the buckets.\n"
            ),
            "notes.md": "# Notes\n\n## Rates\n\nThe rate rate rate of requests.\n",
        })  # fmt: skip
        try:
            self.vectors_rank(searcher, monkeypatch, "Notes > Rates", "Functions > rate()")
            page = searcher.search_page(query, limit=3)
        finally:
            searcher.close()
        assert page.keyword_match == "matched"
        assert [r.heading_path for r in page.results][:1] == [f"Functions > {definition}"]

    def test_a_call_named_only_in_prose_keeps_its_keyword_hits(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        searcher = self.corpus(db, fake_embedder, tmp_path / "docs")
        try:
            page = searcher.search_page("rate()")
        finally:
            searcher.close()
        # No section calls rate(), so nothing is ranked as naming it; the word still matched.
        assert page.keyword_match == "matched"
        assert [r.heading_path for r in page.results if r.fts_rank is not None] == ["Notes > Rates"]

    def test_a_failed_keyword_index_never_abstains_on_a_call(
        self,
        db: Database,
        fake_embedder: FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        searcher = self.corpus(db, fake_embedder, tmp_path / "docs")

        def broken(match_query: str, limit: int, scope: str | None = None) -> list[int]:
            raise DatabaseError("fts index unavailable")

        monkeypatch.setattr(db, "fts_search", broken)
        try:
            page = searcher.search_page("zqxabsent()")
        finally:
            searcher.close()
        assert page.keyword_match == "unavailable" and page.results

    def test_a_common_call_the_gate_refuses_never_abstains(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        searcher = self.corpus(db, fake_embedder, tmp_path / "docs")
        try:
            # `orbit` is in six sections: found, too common to bypass the gate, refused.
            page = searcher.search_page("orbit() zqxone() zqxtwo()")
        finally:
            searcher.close()
        assert page.keyword_match == "filtered" and page.results

    def test_plain_call_gate_and_excerpt(self, db: Database, fake_embedder: FakeEmbedder) -> None:
        sections = [*TestIdentifierGateBypassNeedsRarity.ACRONYM_SECTIONS]
        sections.append(("Disk Full", "Writes fail when fallocate() returns an error."))
        TestIdentifierGateBypassNeedsRarity.fill(db, fake_embedder, sections)
        searcher = HybridSearcher(db, fake_embedder)
        try:
            rare = searcher.search("fallocate() zebra quokka wombat", limit=7)
            common = searcher.search("http() request deadline", limit=7)
        finally:
            searcher.close()
        assert keyword_sections(rare) == {"Disk Full"} and rare[0].heading_title == "Disk Full"
        # `http` is in five of seven sections: written as a call, it is still vocabulary.
        assert keyword_sections(common) == {"Upstream Deadlines"}

        passages = [
            "This is where counters live.",
            "Call rate(x) for a per-second rate, and is(x) to test one.",
            "Counters reset on restart.",
            "Nothing else is kept.",
        ]
        assert select_anchor(fts_terms("rate()"), passages, 3) == 1
        assert select_anchor(fts_terms("is()"), passages, 3) == 1
        assert select_anchor(fts_terms("rate()"), passages, 3, "`rate()`") is None

    PASSAGES = (
        "Counters only ever go up.",
        "This is where they live.",
        "They reset when the process restarts.",
        "Scrapes read them every minute.",
        "Gaps appear when a scrape fails.",
        "Call rate(x) for a per-second figure, and is(x) to test one.",
        "Graphs show the result.",
        "Nothing else is kept.",
    )

    @pytest.mark.parametrize(
        ("query", "heading", "excerpted"),
        [("rate()", "Counters", True), ("is()", "Counters", True), ("rate()", "`rate()`", False)],
    )
    def test_a_call_lookup_is_excerpted_where_the_call_is(
        self,
        db: Database,
        fake_embedder: FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        query: str,
        heading: str,
        excerpted: bool,
    ) -> None:
        body = "\n\n".join(self.PASSAGES)
        searcher = self.searcher(db, fake_embedder, tmp_path / "docs", {
            "a.md": f"# Metrics\n\n## {heading}\n\n{body}\n",
        })  # fmt: skip
        query_sql = "SELECT id FROM sections WHERE heading_level = 2"
        (sid,) = (s for (s,) in db.connection().execute(query_sql))
        # The vectors liked the first passage: only reading the call anchors on its own.
        ranking = ([int(sid)], {int(sid): (0, self.PASSAGES[0])})
        monkeypatch.setattr(searcher, "_vector_ranking", lambda query, limit: ranking)
        try:
            top = searcher.search_page(query, limit=1).results[0]
        finally:
            searcher.close()
        if not excerpted:  # headed by the call: the section is the answer, whole
            assert top.excerpt is None
            return
        assert top.excerpt is not None
        assert "rate(x)" in top.excerpt.text and self.PASSAGES[0] not in top.excerpt.text


class TestTheExcerptAnchor:
    """`select_anchor` (#76): which passage of the top hit its excerpt is centred on."""

    PASSAGES = [
        "The installer writes to the prefix.",
        "Set CARGO_INSTALL_ROOT to choose another root.",
        "Installs are cached between runs.",
        "Nothing else is written.",
    ]

    @pytest.mark.parametrize("query", ["the", "what is it", "`the`", "`what` `is`"])
    def test_a_query_of_stopwords_has_no_anchor_even_with_a_vector_passage(
        self, query: str
    ) -> None:
        from markdown_memory.search import select_anchor

        assert select_anchor(fts_terms(query), self.PASSAGES, 2) is None

    def test_an_identifier_lookup_is_anchored_where_the_identifier_is(self) -> None:
        from markdown_memory.search import select_anchor

        terms = fts_terms("CARGO_INSTALL_ROOT")
        assert select_anchor(terms, self.PASSAGES, 1) == 1
        assert select_anchor(terms, self.PASSAGES, 3) == 1  # the vector passage does not name it
        assert select_anchor(terms, self.PASSAGES, None) == 1

    def test_an_identifier_lookup_is_anchored_where_it_is_first_named(self) -> None:
        """Where a document introduces an identifier, not a later caveat the vector liked."""
        from markdown_memory.search import select_anchor

        passages = [*self.PASSAGES, "CARGO_INSTALL_ROOT only applies to this command."]
        assert select_anchor(fts_terms("CARGO_INSTALL_ROOT"), passages, 4) == 1

    def test_an_identifier_only_the_heading_names_gets_the_whole_section(self) -> None:
        from markdown_memory.search import select_anchor

        assert select_anchor(fts_terms("CARGO_HOME"), self.PASSAGES, 2) is None

    def test_a_section_whose_heading_names_the_whole_query_is_sent_whole(self) -> None:
        """A section headed by what was asked for is the answer: an excerpt cut its definition."""
        from markdown_memory.search import select_anchor

        terms = fts_terms("CARGO_INSTALL_ROOT")
        assert select_anchor(terms, self.PASSAGES, 1, "CARGO_INSTALL_ROOT") is None
        assert select_anchor(terms, self.PASSAGES, 1, "Install roots") == 1
        assert (
            select_anchor(fts_terms("installs cached"), self.PASSAGES, 0, "Installs cached") is None
        )
        # A query asking for more than the heading names is still cut.
        query = fts_terms("where are installs cached written")
        assert select_anchor(query, self.PASSAGES, 0, "Installs cached") == 0

    def test_any_other_query_is_anchored_on_the_vector_passage(self) -> None:
        from markdown_memory.search import select_anchor

        assert select_anchor(fts_terms("where are installs cached"), self.PASSAGES, 0) == 0

    def test_without_one_the_passage_holding_the_most_terms_lowest_first(self) -> None:
        from markdown_memory.search import select_anchor

        assert select_anchor(fts_terms("prefix installer"), self.PASSAGES, None) == 0
        assert select_anchor(fts_terms("written runs"), self.PASSAGES, None) == 2
        assert select_anchor(fts_terms("unrelated words"), self.PASSAGES, None) is None

    def test_terms_match_whole_words_only(self) -> None:
        from markdown_memory.search import select_anchor

        # `install` is not in `installs`, `installer` or `CARGO_INSTALL_ROOT`.
        assert select_anchor(fts_terms("install"), self.PASSAGES, None) is None

    def test_a_vector_ordinal_past_the_passages_is_ignored(self) -> None:
        from markdown_memory.search import select_anchor

        assert select_anchor(fts_terms("cached runs"), self.PASSAGES, 9) == 2


def _section(*blocks: str) -> str:
    return "\n\n".join(["## S", *blocks])


class TestTheExcerptWindow:
    """`excerpt_lines` (#76): the lines an excerpt shows, or None for the whole section."""

    @staticmethod
    def shown(content: str, anchor: int) -> list[str] | None:
        from markdown_memory.parser import MarkdownParser
        from markdown_memory.search import excerpt_lines

        passages = MarkdownParser().passages(content, skip_heading=True)
        window = excerpt_lines(content, passages, anchor)
        return None if window is None else content.split("\n")[window[0] : window[1]]

    PARAGRAPHS = _section(*(f"Paragraph {n} says something." for n in range(6)))

    def test_the_anchor_one_block_before_and_three_after(self) -> None:
        """The answer follows the passage that states the problem (#76 sealed run)."""

        def paragraphs(*numbers: int) -> list[str]:
            lines = [line for n in numbers for line in (f"Paragraph {n} says something.", "")]
            return lines[:-1]

        assert self.shown(self.PARAGRAPHS, 2) == paragraphs(1, 2, 3, 4, 5)
        assert self.shown(self.PARAGRAPHS, 0) == paragraphs(0, 1, 2, 3)
        assert self.shown(self.PARAGRAPHS, 4) == paragraphs(3, 4, 5)

    def test_three_passages_or_fewer_are_sent_whole(self) -> None:
        # One block before the anchor leaves the first of three out when the third is anchored.
        for anchor in range(3):
            assert self.shown(_section("One.", "Two.", "Three."), anchor) is None

    def test_a_table_row_brings_its_header_and_delimiter(self) -> None:
        rows = "\n".join(f"| r{n} | v{n} |" for n in range(8))
        content = _section("Intro.", f"| K | V |\n| - | - |\n{rows}", "Outro.")
        shown = self.shown(content, 3)  # the row r2: rows r1 to r5 are shown
        assert shown is not None
        assert shown[:3] == ["| K | V |", "| - | - |", "| r0 | v0 |"]
        assert shown[-1] == "| r5 | v5 |"

    def test_a_window_that_grows_over_every_passage_is_sent_whole(self) -> None:
        rows = "\n".join(f"| r{n} | v{n} |" for n in range(4))
        assert self.shown(_section(f"| K | V |\n| - | - |\n{rows}"), 3) is None

    def test_a_window_no_smaller_than_the_section_is_sent_whole(self) -> None:
        long = "word " * 40
        for padding in range(8):
            content = "\n\n".join([long + "a" * padding, long, long, long, long, "x"])  # no heading
            remainder = len("\n\nx")
            if -(-len(content) // 4) == -(-(len(content) - remainder) // 4):
                break
        else:
            raise AssertionError("no padding makes the two estimates equal")
        assert self.shown(content, 1) is None

    def test_a_window_that_leaves_a_fence_open_is_sent_whole(self) -> None:
        content = _section("One.", "Two.", "Three.", "Four.", "```sh\nopen fence")
        assert self.shown(content, 3) is None

    def test_a_list_item_does_not_bring_the_blank_line_after_it(self) -> None:
        content = _section("P0.", "P1.", "P2.", "P3.", "- one\n- two\n\n- three", "After.", "End.")
        shown = self.shown(content, 1)  # P0 to the list
        assert shown is not None and shown[-2:] == ["", "- three"]

    def test_a_list_is_one_block_with_the_sentence_that_introduces_it(self) -> None:
        """Three items cut from a longer list lost what the list was of (#76 dev check)."""
        items = "\n".join(f"- item {n}" for n in range(5))
        content = _section("Intro.", "The root is chosen from:", items, "After.", "Outro.")
        assert self.shown(content, 4) == [
            "The root is chosen from:",
            "",
            *(f"- item {n}" for n in range(5)),
            "",
            "After.",
            "",
            "Outro.",
        ]

    def test_a_passage_without_lines_sends_the_section_whole(self) -> None:
        from markdown_memory.parser import MarkdownParser, Passage
        from markdown_memory.search import excerpt_lines

        passages = list(MarkdownParser().passages(self.PARAGRAPHS, skip_heading=True))
        passages[2] = Passage(passages[2].text, None)
        assert excerpt_lines(self.PARAGRAPHS, passages, 3) is None


class TestTheTopHitExcerpt:
    """The excerpt end to end (#76): what `search_page` hands the server for its top hit."""

    BODY = "\n\n".join(
        [
            "# Guide",
            "## Rotation",
            "Certificates are issued by the internal authority.",
            "They expire after ninety days unless renewed.",
            "Renewal starts thirty days before expiry.",
            "A failed renewal pages the on-call engineer.",
            "Revoked certificates are published hourly.",
        ]
    )

    def top(self, db: Database, embedder: FakeEmbedder, root: Path, query: str) -> SearchResult:
        Indexer(db, embedder).index_directory(root)
        searcher = HybridSearcher(db, embedder)
        try:
            return searcher.search(query)[0]
        finally:
            searcher.close()

    def docs(self, tmp_path: Path, text: str) -> Path:
        root = tmp_path / "docs"
        root.mkdir()
        (root / "guide.md").write_text(text + "\n")
        return root

    def test_the_excerpt_is_verbatim_lines_of_the_file(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        root = self.docs(tmp_path, self.BODY)
        top = self.top(db, fake_embedder, root, "renewal starts thirty days")
        assert top.excerpt is not None
        lines = (root / "guide.md").read_text().split("\n")
        assert top.excerpt.text == "\n".join(
            lines[top.excerpt.start_line - 1 : top.excerpt.end_line]
        )
        assert "Renewal starts thirty days" in top.excerpt.text
        shown = top.to_dict()
        assert shown["excerpt"] is True and shown["content"] == top.excerpt.text
        assert shown["lines"] == f"{top.excerpt.start_line}-{top.excerpt.end_line}"
        assert shown["tokens"] == -(-len(top.content) // 4)  # what read_section costs
        assert list(shown) == ["file_path", "heading_path", "lines", "tokens", "excerpt", "content"]

    def test_passages_that_differ_from_the_stored_ones_send_the_section(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        root = self.docs(tmp_path, self.BODY)
        Indexer(db, fake_embedder).index_directory(root)
        with db.transaction() as conn:  # an index another parser version cut
            conn.execute("UPDATE units SET content = content || ' (old cut)' WHERE ordinal = 0")
        top = self.top(db, fake_embedder, root, "renewal starts thirty days")
        assert top.excerpt is None and "excerpt" not in top.to_dict()

    def test_a_section_at_the_passage_cap_is_sent_whole(
        self,
        db: Database,
        fake_embedder: FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import markdown_memory.search as search_module

        monkeypatch.setattr(search_module, "MAX_UNITS_PER_SECTION", 5)
        root = self.docs(tmp_path, self.BODY)
        assert self.top(db, fake_embedder, root, "renewal starts thirty days").excerpt is None

    def test_a_failure_while_cutting_sends_the_section(
        self,
        db: Database,
        fake_embedder: FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from markdown_memory.parser import MarkdownParser

        root = self.docs(tmp_path, self.BODY)
        Indexer(db, fake_embedder).index_directory(root)

        def broken(self: MarkdownParser, content: str, *, skip_heading: bool = False) -> None:
            raise RuntimeError("tokeniser crashed")

        monkeypatch.setattr(MarkdownParser, "passages", broken)
        assert self.top(db, fake_embedder, root, "renewal starts thirty days").excerpt is None

    def test_a_query_without_an_anchor_sends_the_section(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        root = self.docs(tmp_path, self.BODY)
        assert self.top(db, fake_embedder, root, "what is the").excerpt is None

    @pytest.mark.parametrize(("heading", "excerpted"), [("RENEW_DAYS", False), ("Rotation", True)])
    def test_a_section_headed_by_the_identifier_looked_up_is_sent_whole(
        self,
        db: Database,
        fake_embedder: FakeEmbedder,
        tmp_path: Path,
        heading: str,
        excerpted: bool,
    ) -> None:
        text = self.BODY.replace("## Rotation", f"## {heading}").replace(
            "Renewal starts thirty days", "RENEW_DAYS sets when renewal starts, thirty days"
        )
        root = self.docs(tmp_path, text)
        top = self.top(db, fake_embedder, root, "RENEW_DAYS")
        assert (top.excerpt is not None) is excerpted

    def test_a_preamble_without_a_heading_line_is_excerpted_too(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        text = self.BODY.replace("# Guide\n\n## Rotation\n\n", "")
        root = self.docs(tmp_path, text)
        top = self.top(db, fake_embedder, root, "renewal starts thirty days")
        assert top.excerpt is not None and "Renewal starts" in top.excerpt.text

    def test_only_the_top_hit_carries_an_excerpt(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        root = self.docs(tmp_path, self.BODY + "\n\n" + self.BODY.replace("# Guide", "# Copy"))
        Indexer(db, fake_embedder).index_directory(root)
        searcher = HybridSearcher(db, fake_embedder)
        try:
            results = searcher.search("renewal starts thirty days")
        finally:
            searcher.close()
        assert len(results) > 1 and all(result.excerpt is None for result in results[1:])


class TestPartsAreExcerptedOnlyWhenCleanlyCut:
    """A `(Part n)` is excerpted only when no block of its section crosses its boundaries."""

    @staticmethod
    def parts(db: Database, embedder: FakeEmbedder, tmp_path: Path, body: str) -> list[Section]:
        root = tmp_path / "docs"
        root.mkdir()
        (root / "big.md").write_text(f"# Big\n\n## Huge\n\n{body}\n")
        Indexer(db, embedder).index_directory(root)
        document = db.get_document(str((root / "big.md").resolve()))
        assert document is not None
        return [s for s in db.get_sections(document.id) if s.part_index]

    @staticmethod
    def excerpt(db: Database, embedder: FakeEmbedder, part: Section) -> object:
        searcher = HybridSearcher(db, embedder)
        try:
            stored = db.units_of(part.id)
            return searcher._excerpt(part, fts_terms(stored[len(stored) // 2]), len(stored) // 2)
        finally:
            searcher.close()

    def test_a_part_cut_between_paragraphs_is_excerpted(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        body = "\n\n".join(f"Paragraph {n} " + "filler words " * 20 for n in range(40))
        parts = self.parts(db, fake_embedder, tmp_path, body)
        assert len(parts) > 2
        assert self.excerpt(db, fake_embedder, parts[1]) is not None

    def test_a_part_cut_through_a_fence_is_sent_whole(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        code = "\n".join(f"line_{n} = compute({n})  # step" for n in range(200))
        prose = "\n\n".join(f"Paragraph {n} " + "filler words " * 20 for n in range(6))
        body = f"{prose}\n\n```python\n{code}\n```\n\n{prose}"
        parts = self.parts(db, fake_embedder, tmp_path, body)
        inside = [p for p in parts if "```" not in p.content and "line_" in p.content]
        assert inside, "the fence was not cut across parts"
        assert all(self.excerpt(db, fake_embedder, p) is None for p in inside)

    def test_a_part_cut_inside_one_long_line_is_sent_whole(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        prose = "\n\n".join(f"Paragraph {n} " + "filler words " * 20 for n in range(5))
        line = "<div>" + " ".join(f"cell{n}" for n in range(1500)) + "</div>"
        parts = self.parts(db, fake_embedder, tmp_path, f"{prose}\n\n{line}\n\n{prose}")
        sharing = [p for p in parts if "cell" in p.content]
        assert len(sharing) > 1, "the long line was not cut across parts"
        assert all(self.excerpt(db, fake_embedder, p) is None for p in sharing)


class TestEveryExcerptOfBothCorporaIsVerbatim:
    """Criterion 3 of #76, on real documentation: every passage of every section as anchor."""

    EVAL_DATA = Path(__file__).parent.parent / "scripts" / "eval_data"

    @pytest.mark.parametrize("corpus", ["corpus", "corpus_v2"])
    def test_excerpts_are_file_lines_with_headers_and_closed_fences(self, corpus: str) -> None:
        from markdown_memory.discovery import iter_markdown_files
        from markdown_memory.parser import (
            MarkdownParser,
            _split_lines,
            _table_headers,
            ends_inside_fence,
            join_parts,
        )
        from markdown_memory.search import excerpt_lines

        parser = MarkdownParser()
        root = self.EVAL_DATA / corpus
        checked = 0
        for path in iter_markdown_files(root):
            text = path.read_text(encoding="utf-8", errors="replace")
            file_lines = _split_lines(text)
            sections = parser.parse(text, fallback_title=path.stem).sections
            for section in sections:
                passages = parser.passages(section.content, skip_heading=True)
                if [p.text for p in passages] != list(section.units):
                    passages = parser.passages(section.content, skip_heading=False)
                if [p.text for p in passages] != list(section.units):
                    continue  # a part cut with a table header it does not hold: sent whole
                if section.part_index:
                    siblings = [s for s in sections if s.base_path == section.base_path]
                    offset = sum(
                        len(s.content) + max(0, n.start_line - s.end_line)
                        for s, n in zip(siblings, siblings[1:], strict=False)
                        if n.part_index <= section.part_index
                    )
                    cuts = [offset, offset + len(section.content)]
                    if parser.cuts_a_block(join_parts(siblings), cuts):
                        continue
                lines = section.content.split("\n")
                body_rows = _table_headers(lines)
                for anchor in range(len(passages)):
                    window = excerpt_lines(section.content, passages, anchor)
                    if window is None:
                        continue
                    first, end = window
                    shown = "\n".join(lines[first:end])
                    start = section.start_line + first
                    assert shown == "\n".join(file_lines[start - 1 : start - 1 + end - first])
                    assert first not in body_rows, f"{path}: a table row without its header"
                    assert not ends_inside_fence("\n".join(lines[:first])), (
                        f"{path}: opens mid-fence"
                    )
                    assert not ends_inside_fence(shown), f"{path}: leaves a fence open"
                    checked += 1
        assert checked > {"corpus": 12, "corpus_v2": 3000}[corpus], checked


class TestExactTiesFollowTheWalk:
    """#103: an exact tie goes to the section the walk reaches first, not to the lower id.

    Ids follow the walk only on a fresh build: an edited document is stored again under new,
    higher ids, so a tie decided by id would go to whichever document was edited longest ago.
    """

    @staticmethod
    def ids(searcher: HybridSearcher) -> dict[str, int]:
        rows = searcher._db.connection().execute("SELECT id, heading_path FROM sections")
        return {path: int(sid) for sid, path in rows}

    @classmethod
    def lanes(
        cls,
        searcher: HybridSearcher,
        monkeypatch: pytest.MonkeyPatch,
        keyword: Sequence[str],
        vector: Sequence[str],
    ) -> None:
        """Make each side rank these heading paths, in this order."""
        ids = cls.ids(searcher)
        by_keyword = _Keyword([ids[path] for path in keyword], "matched")
        by_vector = [ids[path] for path in vector]
        monkeypatch.setattr(searcher, "_keyword_pass", lambda query, limit: by_keyword)
        monkeypatch.setattr(searcher, "_vector_ranking", lambda query, limit: (by_vector, {}))

    def test_a_tie_between_documents_survives_an_edit(
        self,
        db: Database,
        fake_embedder: FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        root = tmp_path / "docs"
        root.mkdir()
        (root / "a.md").write_text("# Alpha\n\nalpha text\n")
        (root / "b.md").write_text("# Beta\n\nbeta text\n")
        indexer = Indexer(db, fake_embedder)
        indexer.index_directory(root)
        searcher = HybridSearcher(db, fake_embedder)
        pages: list[list[str]] = []
        try:
            for edit in ("alpha text, edited", "alpha text, edited again"):
                # Alpha: keyword 1, vector 2. Beta: keyword 2, vector 1. Both 1/61 + 1/62.
                self.lanes(searcher, monkeypatch, ["Alpha", "Beta"], ["Beta", "Alpha"])
                page = searcher.search_page("text", limit=2)
                pages.append([result.heading_path for result in page.results])
                (root / "a.md").write_text(f"# Alpha\n\n{edit}\n")
                indexer.index_directory(root)
                ids = self.ids(searcher)
                assert ids["Alpha"] > ids["Beta"], "the edit stored Alpha again, after Beta"
        finally:
            searcher.close()
        assert pages == [["Alpha", "Beta"], ["Alpha", "Beta"]]

    def test_a_vector_distance_tie_goes_to_the_walk_up_to_the_cut(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        # The same passage in two documents embeds to the same vector: an exact distance tie.
        # Beta is indexed first, so it holds the lower ids; the walk reaches Alpha first.
        root = tmp_path / "docs"
        root.mkdir()
        (root / "b.md").write_text("# Beta\n\nthe same passage in both\n")
        Indexer(db, fake_embedder).index_directory(root)
        (root / "a.md").write_text("# Alpha\n\nthe same passage in both\n")
        Indexer(db, fake_embedder).index_directory(root)
        searcher = HybridSearcher(db, fake_embedder)
        try:
            ids = self.ids(searcher)
            full, _ = searcher._vector_ranking("the same passage in both", 5)
            cut, _ = searcher._vector_ranking("the same passage in both", 1)
        finally:
            searcher.close()
        assert ids["Beta"] < ids["Alpha"]
        assert full == [ids["Alpha"], ids["Beta"]]
        assert cut == [ids["Alpha"]], "the cut keeps the one the walk reaches first"

    def test_a_tie_inside_one_document_goes_to_the_earlier_section(
        self,
        db: Database,
        fake_embedder: FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        root = tmp_path / "docs"
        root.mkdir()
        (root / "a.md").write_text(
            "# Guide\n\n## First\n\nfirst text\n\n## Second\n\nsecond text\n"
        )
        Indexer(db, fake_embedder).index_directory(root)
        searcher = HybridSearcher(db, fake_embedder)
        try:
            # Keyword lists Second first, so the order the scores were met in is no answer.
            self.lanes(
                searcher, monkeypatch, ["Guide > Second", "Guide > First"],
                ["Guide > First", "Guide > Second"],
            )  # fmt: skip
            page = searcher.search_page("text", limit=2)
        finally:
            searcher.close()
        assert [result.heading_path for result in page.results] == [
            "Guide > First",
            "Guide > Second",
        ]

    def test_a_tie_between_parts_of_one_line_goes_to_the_earlier_part(
        self,
        db: Database,
        fake_embedder: FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # One line too long for a section is cut into parts that all start on that line:
        # only the part number tells them apart.
        root = tmp_path / "docs"
        root.mkdir()
        (root / "a.md").write_text("# Long\n\n" + "word " * 6000 + "\n")
        Indexer(db, fake_embedder).index_directory(root)
        searcher = HybridSearcher(db, fake_embedder)
        try:
            lines = dict(
                searcher._db.connection().execute("SELECT heading_path, start_line FROM sections")
            )
            assert lines["Long (Part 2)"] == lines["Long (Part 3)"]
            self.lanes(
                searcher, monkeypatch, ["Long (Part 3)", "Long (Part 2)"],
                ["Long (Part 2)", "Long (Part 3)"],
            )  # fmt: skip
            page = searcher.search_page("word", limit=2)
        finally:
            searcher.close()
        assert [result.heading_path for result in page.results] == [
            "Long (Part 2)",
            "Long (Part 3)",
        ]

    def test_a_section_gone_since_the_ranking_goes_after_its_tie(
        self,
        db: Database,
        fake_embedder: FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        root = tmp_path / "docs"
        root.mkdir()
        (root / "a.md").write_text("# Alpha\n\nalpha text\n")
        Indexer(db, fake_embedder).index_directory(root)
        searcher = HybridSearcher(db, fake_embedder)
        rankings: list[str] = []

        def vector(query: str, limit: int) -> tuple[list[int], dict[int, tuple[int, str]]]:
            rankings.append(query)
            return [0], {}  # an id no section holds - lower than any that does

        try:
            alpha = self.ids(searcher)["Alpha"]
            by_keyword = _Keyword([alpha], "matched")
            monkeypatch.setattr(searcher, "_keyword_pass", lambda query, limit: by_keyword)
            monkeypatch.setattr(searcher, "_vector_ranking", vector)
            page = searcher.search_page("text", limit=1)
        finally:
            searcher.close()
        assert [result.heading_path for result in page.results] == ["Alpha"]
        assert len(rankings) == 1, "the page was full before the gone section: no second pass"

    def test_positions_are_read_only_when_something_ties(
        self,
        db: Database,
        fake_embedder: FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        root = tmp_path / "docs"
        root.mkdir()
        (root / "a.md").write_text("# Alpha\n\nalpha text\n")
        (root / "b.md").write_text("# Beta\n\nbeta text\n")
        Indexer(db, fake_embedder).index_directory(root)
        searcher = HybridSearcher(db, fake_embedder)
        looked_up: list[list[int]] = []

        def positions(section_ids: Sequence[int]) -> dict[int, tuple[str, int, int]]:
            looked_up.append(list(section_ids))
            return {}

        try:
            self.lanes(searcher, monkeypatch, ["Alpha", "Beta"], ["Alpha", "Beta"])  # no tie
            monkeypatch.setattr(searcher._db, "section_positions", positions)
            page = searcher.search_page("text", limit=2)
        finally:
            searcher.close()
        assert [result.heading_path for result in page.results] == ["Alpha", "Beta"]
        assert looked_up == []

    def test_a_fresh_build_numbers_sections_in_walk_order(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        """Why the change leaves every ranking of a fresh index as it was: ids already
        followed the walk there, so breaking a tie by the walk picks what the id picked.

        The v1 corpus only: corpus_v2 takes half a minute to build, and its eval indexes are
        checked the same way, read-only, whenever a change to ranking is measured."""
        root = Path(__file__).parent.parent / "scripts" / "eval_data" / "corpus"
        Indexer(db, fake_embedder).index_directory(root)
        rows = db.connection().execute(
            "SELECT d.file_path, s.start_line, s.part_index FROM sections s "
            "JOIN documents d ON d.id = s.doc_id ORDER BY s.id"
        )
        places = [(walk_order(path), line, part) for path, line, part in rows]
        assert len(places) > 50
        assert places == sorted(places)
        assert len(set(places)) == len(places), "two sections at one place: a tie left open"
