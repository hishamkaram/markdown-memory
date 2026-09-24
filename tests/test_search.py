"""Search tests: FTS5 query building, vector similarity, RRF fusion, hybrid ranking."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from fakes import FakeEmbedder

from markdown_memory.db import Database
from markdown_memory.embedders import Embedder
from markdown_memory.exceptions import DatabaseError, EmbeddingError
from markdown_memory.indexer import Indexer
from markdown_memory.search import (
    MAX_RESULT_LIMIT,
    RRF_K,
    HybridSearcher,
    build_fts_query,
    reciprocal_rank_fusion,
)

CORPUS = {
    "gateway.md": """# Orbit Gateway

A reverse proxy for internal services.

## Environment Variables

Set `ORBIT_UPSTREAM_TIMEOUT_MS` to change the upstream deadline. The default is 3000.

## Command Line Flags

Pass `--drain-seconds 30` to wait for in-flight requests before exiting.

## Throttling

Clients exceeding their quota receive HTTP 429 responses. Back off exponentially
and respect the Retry-After header before sending more traffic.

## Credentials

Rotate API keys every ninety days. Secrets are stored encrypted at rest and are
never written to log files.
""",
    "storage.md": """# Storage Engine

## Compaction

Background compaction merges small segment files into larger ones to reclaim disk space.

## Backups

Snapshots are uploaded to object storage nightly and retained for thirty days.
""",
}


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    root.mkdir()
    for name, text in CORPUS.items():
        (root / name).write_text(text, encoding="utf-8")
    return root


@pytest.fixture
def fake_searcher(
    db: Database, fake_embedder: FakeEmbedder, corpus_dir: Path
) -> Iterator[HybridSearcher]:
    Indexer(db, fake_embedder).index_directory(corpus_dir)
    searcher = HybridSearcher(db, fake_embedder)
    yield searcher
    searcher.close()


@pytest.fixture
def real_searcher(
    tmp_path: Path, real_embedder: Embedder, corpus_dir: Path
) -> Iterator[HybridSearcher]:
    with Database(tmp_path / "real.db", embedding_dim=real_embedder.dimension) as database:
        Indexer(database, real_embedder).index_directory(corpus_dir)
        searcher = HybridSearcher(database, real_embedder)
        yield searcher
        searcher.close()


# ---------------------------------------------------------------------- query building


class TestBuildFtsQuery:
    def test_terms_are_quoted_and_or_ed(self) -> None:
        assert build_fts_query("upstream timeout") == '"upstream" OR "timeout"'

    def test_flags_and_env_vars_are_kept_as_literal_phrases(self) -> None:
        assert build_fts_query("--drain-seconds") == '"--drain-seconds"'
        assert build_fts_query("ORBIT_UPSTREAM_TIMEOUT_MS") == '"ORBIT_UPSTREAM_TIMEOUT_MS"'

    def test_operators_are_neutralised(self) -> None:
        assert build_fts_query("foo NEAR bar") == '"foo" OR "NEAR" OR "bar"'
        assert build_fts_query("title:secret*") == '"title:secret*"'
        assert build_fts_query("AND OR NOT") == '"AND" OR "OR" OR "NOT"'  # literal, never syntax

    def test_stopwords_are_dropped_when_meaningful_terms_remain(self) -> None:
        query = "What happens when the system is overloaded?"
        assert build_fts_query(query) == '"happens" OR "system" OR "overloaded?"'
        assert build_fts_query("foo and not bar") == '"foo" OR "bar"'
        # upper-case words are keywords/identifiers (SQL "NOT NULL"), never stopwords
        assert build_fts_query("foo AND NOT bar") == '"foo" OR "AND" OR "NOT" OR "bar"'

    def test_stopword_only_query_is_kept_verbatim(self) -> None:
        assert build_fts_query("to be or not") == '"to" OR "be" OR "or" OR "not"'

    def test_identifiers_containing_stopwords_are_untouched(self) -> None:
        assert build_fts_query("--no-cache IS_ON") == '"--no-cache" OR "IS_ON"'

    def test_term_count_is_capped(self) -> None:
        query = build_fts_query(" ".join(f"term{n}" for n in range(100)))
        assert query is not None
        assert query.count(" OR ") == 31

    def test_embedded_quotes_are_escaped(self) -> None:
        assert build_fts_query('say "hello"') == '"say" OR """hello"""'

    def test_punctuation_only_terms_are_dropped(self) -> None:
        assert build_fts_query("-- && ** ???") is None
        assert build_fts_query("   ") is None
        assert build_fts_query("retry -- now") == '"retry" OR "now"'

    def test_duplicate_terms_are_collapsed(self) -> None:
        assert build_fts_query("Retry retry RETRY") == '"Retry"'

    @pytest.mark.parametrize(
        "query",
        [
            '"', '""', "'", "NEAR(", "NEAR(a b, 2)", "a AND", "OR", "NOT", "*", "^start",
            "col:val", "{a b}", "(unbalanced", "a + b", "semi;colon", "back\\slash",
            "--flag=value", "émoji 🚀 ünïcode", "-", "a" * 500,
        ],
    )  # fmt: skip
    def test_no_input_can_raise_an_fts5_syntax_error(
        self, db: Database, fake_searcher: HybridSearcher, query: str
    ) -> None:
        # Straight at the FTS5 primitive: HybridSearcher.search() would hide a syntax
        # error by degrading to vector-only results, which would make this test vacuous.
        expression = build_fts_query(query)
        if expression is not None:
            db.fts_search(expression, 20)
        fake_searcher.search(query)


# ---------------------------------------------------------------------- RRF


class TestReciprocalRankFusion:
    def test_formula(self) -> None:
        scores = reciprocal_rank_fusion([[10, 20, 30], [30, 10]])
        assert scores[10] == pytest.approx(1 / (RRF_K + 1) + 1 / (RRF_K + 2))
        assert scores[20] == pytest.approx(1 / (RRF_K + 2))
        assert scores[30] == pytest.approx(1 / (RRF_K + 3) + 1 / (RRF_K + 1))

    def test_k_defaults_to_sixty(self) -> None:
        assert RRF_K == 60
        assert reciprocal_rank_fusion([[1]])[1] == pytest.approx(1 / 61)

    def test_agreement_between_indexes_beats_a_single_top_rank(self) -> None:
        scores = reciprocal_rank_fusion([[1, 2, 3], [4, 2, 5]])
        ranked = sorted(scores, key=lambda item: -scores[item])
        assert ranked[0] == 2  # second in both lists beats first in only one

    def test_empty_rankings(self) -> None:
        assert reciprocal_rank_fusion([[], []]) == {}
        assert reciprocal_rank_fusion([[7], []]) == {7: pytest.approx(1 / 61)}


# ---------------------------------------------------------------------- hybrid (fake vectors)


class TestHybridSearch:
    def test_results_are_sorted_by_descending_score(self, fake_searcher: HybridSearcher) -> None:
        results = fake_searcher.search("compaction segment files disk", limit=10)
        scores = [result.score for result in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0].heading_path == "Storage Engine > Compaction"
        assert results[0].fts_rank == 1
        assert results[0].vec_rank == 1
        assert results[0].score == pytest.approx(2 / (RRF_K + 1))

    def test_result_carries_breadcrumb_location_and_content(
        self, fake_searcher: HybridSearcher, corpus_dir: Path
    ) -> None:
        top = fake_searcher.search("nightly snapshots object storage", limit=1)[0]
        assert top.file_path == str(corpus_dir / "storage.md")
        assert top.document_title == "Storage Engine"
        assert top.heading_title == "Backups"
        assert top.heading_path == "Storage Engine > Backups"
        assert top.content.startswith("## Backups")
        assert (top.start_line, top.end_line) == (7, 9)
        payload = top.to_dict()
        assert payload["heading_path"] == "Storage Engine > Backups"
        assert payload["lines"] == "7-9"
        assert payload["content"] == top.content

    def test_limit_is_respected_and_clamped(
        self, db: Database, fake_searcher: HybridSearcher, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert len(fake_searcher.search("the", limit=2)) == 2
        assert len(fake_searcher.search("the", limit=0)) == 1
        requested: list[int] = []
        original = db.vec_search

        def spy(embedding: Sequence[float], limit: int) -> list[tuple[int, float]]:
            requested.append(limit)
            return original(embedding, limit)

        monkeypatch.setattr(db, "vec_search", spy)
        fake_searcher.search("the", limit=10_000)
        fake_searcher.search("the", limit=3)
        assert requested == [MAX_RESULT_LIMIT, 20]  # clamped to 50; never below 20 candidates

    def test_keyword_failure_degrades_to_vector_only(
        self, db: Database, fake_searcher: HybridSearcher, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(match_query: str, limit: int) -> list[int]:
            raise DatabaseError("fts index unavailable")

        monkeypatch.setattr(db, "fts_search", broken)
        results = fake_searcher.search("compaction segment files")
        assert results[0].heading_path == "Storage Engine > Compaction"
        assert all(r.fts_rank is None for r in results)

    def test_blank_query_returns_nothing(self, fake_searcher: HybridSearcher) -> None:
        assert fake_searcher.search("") == []
        assert fake_searcher.search("   \n") == []

    def test_both_indexes_contribute_candidates(self, fake_searcher: HybridSearcher) -> None:
        results = fake_searcher.search("compaction", limit=20)
        assert any(r.fts_rank is not None for r in results)
        assert any(r.fts_rank is None and r.vec_rank is not None for r in results)

    def test_vector_failure_degrades_to_keyword_only(
        self, db: Database, fake_embedder: FakeEmbedder, corpus_dir: Path
    ) -> None:
        class BrokenQueries(FakeEmbedder):
            def embed_query(self, text: str) -> list[float]:
                raise EmbeddingError("model offline")

        Indexer(db, fake_embedder).index_directory(corpus_dir)
        searcher = HybridSearcher(db, BrokenQueries())
        try:
            results = searcher.search("ORBIT_UPSTREAM_TIMEOUT_MS")
        finally:
            searcher.close()
        assert [r.heading_title for r in results] == ["Environment Variables"]
        assert results[0].vec_rank is None

    def test_failure_of_both_indexes_is_raised(
        self, tmp_path: Path, fake_embedder: FakeEmbedder
    ) -> None:
        database = Database(tmp_path / "gone.db")
        searcher = HybridSearcher(database, fake_embedder)
        database.close()
        try:
            with pytest.raises(DatabaseError):
                searcher.search("anything")
        finally:
            searcher.close()

    def test_searches_run_on_dedicated_worker_threads(
        self, db: Database, fake_embedder: FakeEmbedder, corpus_dir: Path
    ) -> None:
        import threading

        names: list[str] = []

        class Recording(FakeEmbedder):
            def embed_query(self, text: str) -> list[float]:
                names.append(threading.current_thread().name)
                return super().embed_query(text)

        Indexer(db, fake_embedder).index_directory(corpus_dir)
        searcher = HybridSearcher(db, Recording())
        try:
            searcher.search("quota")
        finally:
            searcher.close()
        assert names and names[0].startswith("mdmem-search")


# ---------------------------------------------------------------------- real embeddings


def headings(results: Sequence[object]) -> list[str]:
    return [getattr(result, "heading_title", "") for result in results]


@pytest.mark.embedding
class TestKeywordVersusSemantic:
    """Keyword-only identifiers must come from FTS5; paraphrases must come from vectors."""

    def test_exact_env_var_is_retrieved_by_fts5(self, real_searcher: HybridSearcher) -> None:
        results = real_searcher.search("ORBIT_UPSTREAM_TIMEOUT_MS")
        assert results[0].heading_path == "Orbit Gateway > Environment Variables"
        assert results[0].fts_rank == 1
        # the identifier occurs in exactly one section, so FTS matched nothing else
        assert [r.fts_rank for r in results[1:]] == [None] * (len(results) - 1)

    def test_exact_cli_flag_is_retrieved_by_fts5(self, real_searcher: HybridSearcher) -> None:
        results = real_searcher.search("--drain-seconds")
        assert results[0].heading_path == "Orbit Gateway > Command Line Flags"
        assert results[0].fts_rank == 1

    def test_semantic_query_without_shared_words_is_retrieved_by_vectors(
        self, real_searcher: HybridSearcher
    ) -> None:
        query = "rate limiting"
        assert real_searcher._keyword_ranking(query, 20) == []  # no lexical match anywhere
        results = real_searcher.search(query)
        assert results[0].heading_path == "Orbit Gateway > Throttling"
        assert results[0].fts_rank is None
        assert results[0].vec_rank == 1

    def test_semantic_synonyms_for_credentials(self, real_searcher: HybridSearcher) -> None:
        query = "password security policy"
        assert real_searcher._keyword_ranking(query, 20) == []
        results = real_searcher.search(query)
        assert results[0].heading_path == "Orbit Gateway > Credentials"
        assert results[0].fts_rank is None

    def test_semantic_query_crosses_documents(self, real_searcher: HybridSearcher) -> None:
        results = real_searcher.search("disaster recovery copies")
        assert "Backups" in headings(results[:2])

    def test_real_embeddings_are_unit_vectors_of_the_declared_size(
        self, real_embedder: Embedder
    ) -> None:
        vector = real_embedder.embed_query("hello world")
        assert len(vector) == real_embedder.dimension == 768
        assert sum(value * value for value in vector) == pytest.approx(1.0, abs=1e-3)
        documents = real_embedder.embed_documents(["a", "b", "c"])
        assert [len(v) for v in documents] == [768, 768, 768]
        assert real_embedder.embed_documents([]) == []
