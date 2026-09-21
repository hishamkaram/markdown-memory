"""Passage-level retrieval: unit extraction, unit storage, keyword gate, embedder presets."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from fakes import FakeEmbedder, vectors_for

import markdown_memory.indexer as indexer_module
from markdown_memory.db import SCHEMA_VERSION, Database
from markdown_memory.exceptions import DatabaseError, IndexingError
from markdown_memory.indexer import (
    GEMMA_DOCUMENT_PROMPT,
    GEMMA_FILES,
    GEMMA_QUERY_PROMPT,
    Embedder,
    EmbeddingGemmaEmbedder,
    FastEmbedEmbedder,
    Indexer,
    create_embedder,
)
from markdown_memory.models import SectionDraft, SectionVectors
from markdown_memory.parser import MAX_UNIT_CHARS, MAX_UNITS_PER_SECTION, MarkdownParser
from markdown_memory.search import KEYWORD_GATE, HybridSearcher, _is_identifier, fts_terms

CONFIG_DOC = """# Helios

Helios ingests change events.

## Settings

### Ingest

| Variable | Default | Description |
| --- | --- | --- |
| `HELIOS_BATCH` | `500` | Records pulled from the broker per poll |
| `HELIOS_WORKERS` | `4` | Parallel decoder goroutines |

### Resources

```yaml
resources:
  requests:
    memory: 4Gi
```

## Operations

### Shutdown

- Send `SIGTERM` to drain in-flight requests.
- Send `SIGKILL` only as a last resort.
"""


def units_of(text: str, path: str) -> tuple[str, ...]:
    sections = MarkdownParser().parse(text).sections
    return next(s.units for s in sections if s.heading_path == path)


# ---------------------------------------------------------------------- unit extraction


class TestUnitExtraction:
    def test_table_rows_are_labelled_with_their_column_headers(self) -> None:
        assert units_of(CONFIG_DOC, "Helios > Settings > Ingest") == (
            "Variable: HELIOS_BATCH; Default: 500; "
            "Description: Records pulled from the broker per poll",
            "Variable: HELIOS_WORKERS; Default: 4; Description: Parallel decoder goroutines",
        )

    def test_code_block_is_one_unit_with_short_lines_kept(self) -> None:
        assert units_of(CONFIG_DOC, "Helios > Settings > Resources") == (
            "resources: requests: memory: 4Gi",
        )

    def test_each_list_item_is_a_unit(self) -> None:
        assert units_of(CONFIG_DOC, "Helios > Operations > Shutdown") == (
            "Send SIGTERM to drain in-flight requests.",
            "Send SIGKILL only as a last resort.",
        )

    def test_the_sections_own_heading_is_not_a_unit(self) -> None:
        assert units_of(CONFIG_DOC, "Helios") == ("Helios ingests change events.",)

    def test_heading_only_section_has_no_units(self) -> None:
        assert units_of(CONFIG_DOC, "Helios > Settings") == ()
        assert units_of(CONFIG_DOC, "Helios > Operations") == ()

    def test_markup_is_stripped_but_text_code_and_alt_text_survive(self) -> None:
        text = (
            "# T\n\nSee **bold** `code` [the link](http://x) ![diagram](d.png).\n\n"
            "> quoted advice\n"
        )
        assert units_of(text, "T") == ("See bold code the link diagram.", "quoted advice")

    def test_nested_list_content_stays_with_its_item(self) -> None:
        text = (
            "# T\n\n1. Install:\n\n   ```bash\n   make install\n   ```\n\n"
            "2. Run it\n   - with flags\n"
        )
        assert units_of(text, "T") == ("Install: make install", "Run it with flags")

    def test_preamble_and_setext_headings(self) -> None:
        sections = MarkdownParser().parse("badge text\n\nTitle\n=====\n\nbody\n").sections
        assert [s.units for s in sections] == [("badge text",), ("body",)]

    def test_every_part_of_an_oversized_section_gets_its_own_units(self) -> None:
        body = "\n\n".join(f"Paragraph {n} " + "filler words " * 25 for n in range(14))
        parts = MarkdownParser().parse(f"## Big\n\n{body}\n").sections
        assert len(parts) >= 2
        assert all(part.units for part in parts)
        assert parts[0].units[0].startswith("Paragraph 0 ")
        assert not any(unit.startswith("Big") for part in parts for unit in part.units)
        recovered = [unit for part in parts for unit in part.units]
        assert len(recovered) == 14

    def test_units_are_capped_in_number_and_length(self) -> None:
        many = "\n".join(f"- item {n}" for n in range(200))
        assert len(units_of(f"# T\n\n{many}\n", "T")) == MAX_UNITS_PER_SECTION
        # A long run is now windowed, not truncated: every character keeps a passage
        # vector, and no window exceeds the limit.
        long_unit = MarkdownParser().extract_units("x" * 5000)
        assert "".join(long_unit) == "x" * 5000
        assert all(0 < len(unit) <= MAX_UNIT_CHARS for unit in long_unit)

    def test_embedding_texts_carry_the_breadcrumb(self) -> None:
        section = next(
            s for s in MarkdownParser().parse(CONFIG_DOC).sections if s.heading_title == "Shutdown"
        )
        assert section.embedding_text == (
            "Helios > Operations > Shutdown\n\n"
            "Send SIGTERM to drain in-flight requests. Send SIGKILL only as a last resort."
        )
        assert section.unit_texts[0] == (
            "Helios > Operations > Shutdown: Send SIGTERM to drain in-flight requests."
        )


# ---------------------------------------------------------------------- storage


def draft(title: str, units: Sequence[str], line: int = 1) -> SectionDraft:
    return SectionDraft(
        heading_title=title, heading_level=2, heading_path=f"Doc > {title}",
        base_path=f"Doc > {title}", content=f"## {title}\n\n" + "\n\n".join(units),
        start_line=line, end_line=line, units=tuple(units),
    )  # fmt: skip


def store(db: Database, embedder: FakeEmbedder, path: str, sections: list[SectionDraft]) -> int:
    document = db.replace_document(
        file_path=path, title="Doc", content_hash="h", last_modified=1,
        sections=sections, vectors=vectors_for(embedder, sections),
    )  # fmt: skip
    return document.id


class TestUnitStorage:
    SECTIONS = [
        draft("Parent", []),
        draft("Retries", ["Set MAX_RETRIES to tune retry attempts.", "Backoff doubles each time."]),
        draft("Shutdown", ["The proxy drains connections on SIGTERM."]),
    ]

    def test_passages_and_vectors_are_stored_per_section(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        store(db, fake_embedder, "/d/a.md", self.SECTIONS)
        assert db.count_rows("sections") == 3
        assert db.count_rows("sections_vec") == 2  # the heading-only section has no vector
        assert db.count_rows("units") == db.count_rows("units_vec") == 3
        rows = (
            db.connection()
            .execute(
                "SELECT s.heading_title, u.ordinal, u.content FROM units u "
                "JOIN sections s ON s.id = u.section_id ORDER BY u.id"
            )
            .fetchall()
        )
        assert rows == [
            ("Retries", 0, "Set MAX_RETRIES to tune retry attempts."),
            ("Retries", 1, "Backoff doubles each time."),
            ("Shutdown", 0, "The proxy drains connections on SIGTERM."),
        ]

    def test_unit_search_returns_the_matching_passage(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        doc_id = store(db, fake_embedder, "/d/a.md", self.SECTIONS)
        ids = {s.heading_title: s.id for s in db.get_sections(doc_id)}
        hits = db.unit_search(fake_embedder.embed_query("Doc Retries backoff doubles"), 3)
        assert hits[0][0] == ids["Retries"]
        assert hits[0][2] == "Backoff doubles each time."
        assert [distance for _, distance, _ in hits] == sorted(d for _, d, _ in hits)
        assert db.sections_with_passages(list(ids.values())) == {ids["Retries"], ids["Shutdown"]}
        assert db.sections_with_passages([]) == set()

    def test_deleting_a_document_cascades_through_sections_to_passage_vectors(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        store(db, fake_embedder, "/d/a.md", self.SECTIONS)
        store(db, fake_embedder, "/d/b.md", [draft("Keep", ["kept passage"])])
        assert db.delete_documents(["/d/a.md"]) == 1
        assert db.count_rows("units") == db.count_rows("units_vec") == 1
        assert db.count_rows("sections_vec") == 1
        remaining = db.unit_search(fake_embedder.embed_query("retry attempts"), 10)
        assert [passage for _, _, passage in remaining] == ["kept passage"]

    def test_replacing_a_document_replaces_its_passages(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        store(db, fake_embedder, "/d/a.md", self.SECTIONS)
        store(db, fake_embedder, "/d/a.md", [draft("New", ["fresh passage"])])
        assert db.count_rows("units") == db.count_rows("units_vec") == 1
        assert db.count_rows("sections_vec") == 1

    def test_vector_sets_must_mirror_the_sections(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        section = draft("A", ["one", "two"])
        vector = fake_embedder.embed_query("x")
        for bad, message in (
            (SectionVectors(section=vector, units=(vector,)), "2 passages but 1 passage vectors"),
            (SectionVectors(section=None, units=(vector, vector)), "section vector is required"),
        ):
            with pytest.raises(DatabaseError, match=message):
                db.replace_document(
                    file_path="/d/a.md", title="T", content_hash="h", last_modified=1,
                    sections=[section], vectors=[bad],
                )  # fmt: skip
        stub = draft("Stub", [])
        with pytest.raises(DatabaseError, match="section vector is required"):
            db.replace_document(
                file_path="/d/a.md", title="T", content_hash="h", last_modified=1,
                sections=[stub], vectors=[SectionVectors(section=vector)],
            )  # fmt: skip
        assert db.count_rows("documents") == 0

    def test_version_one_database_is_migrated_in_place(
        self, tmp_path: Path, fake_embedder: FakeEmbedder
    ) -> None:
        path = tmp_path / "v1.db"
        with Database(path) as database:  # rewind a fresh database to the v1 layout
            conn = database.connection()
            conn.execute("DROP TRIGGER units_after_delete")
            conn.execute("DROP TABLE units_vec")
            conn.execute("DROP TABLE units")
            conn.execute("DROP TABLE index_failures")
            conn.execute("DROP TABLE index_coverage")
            conn.execute("ALTER TABLE documents DROP COLUMN vector_format")
            conn.execute("PRAGMA user_version = 1")
        with Database(path) as migrated:
            version = migrated.connection().execute("PRAGMA user_version").fetchone()[0]
            assert int(version) == SCHEMA_VERSION == 4
            store(migrated, fake_embedder, "/d/a.md", self.SECTIONS)
            assert migrated.count_rows("units_vec") == 3


# ---------------------------------------------------------------------- search


NOISE_CORPUS = {
    "deploy.md": """# Deployment Guide

## Kubernetes

Apply the manifests to deploy the service. Deploy with care and deploy often.

## Rolling Back

Run the migrate down command, then start the previous image tag again.
""",
    "reference.md": """# Reference

## Flags

Pass `--replay-from-offset` to start consuming again at an earlier offset.

## Operations

### Health

`GET /healthz` answers while the process accepts traffic.
""",
}


@pytest.fixture
def searcher(db: Database, fake_embedder: FakeEmbedder, tmp_path: Path) -> Iterator[HybridSearcher]:
    root = tmp_path / "corpus"
    root.mkdir()
    for name, text in NOISE_CORPUS.items():
        (root / name).write_text(text, encoding="utf-8")
    Indexer(db, fake_embedder).index_directory(root)
    engine = HybridSearcher(db, fake_embedder)
    yield engine
    engine.close()


def titles(db: Database, ids: Sequence[int]) -> list[str]:
    rows = db.get_sections_with_documents(ids)
    return [rows[i][0].heading_title for i in ids]


class TestKeywordGate:
    def test_gate_threshold_is_half_the_querys_information(self) -> None:
        assert KEYWORD_GATE == 0.5

    def test_stray_common_word_match_is_rejected(
        self, db: Database, searcher: HybridSearcher
    ) -> None:
        # only "deploy" occurs in the corpus; "undo" and "bad" carry most of the IDF mass
        assert fts_terms("how do I undo a bad deploy") == ['"undo"', '"bad"', '"deploy"']
        assert titles(db, db.fts_search('"undo" OR "bad" OR "deploy"', 20)) == ["Kubernetes"]
        assert searcher._keyword_ranking("how do I undo a bad deploy", 20) == []

    def test_hit_covering_most_of_the_query_passes(
        self, db: Database, searcher: HybridSearcher
    ) -> None:
        ranking = searcher._keyword_ranking("previous image tag", 20)
        assert titles(db, ranking) == ["Rolling Back"]

    def test_single_term_and_identifier_queries_always_pass(
        self, db: Database, searcher: HybridSearcher
    ) -> None:
        assert titles(db, searcher._keyword_ranking("deploy", 20)) == ["Kubernetes"]
        assert titles(db, searcher._keyword_ranking("--replay-from-offset", 20)) == ["Flags"]
        assert titles(db, searcher._keyword_ranking("/healthz", 20)) == ["Health"]

    @pytest.mark.parametrize(
        ("term", "expected"),
        [
            ('"--dry-run"', True), ('"-i"', True), ('"HELIOS_BATCH"', True), ('"/healthz"', True),
            ('"WHERE"', True), ('"v2.1"', True), ('"a.b"', True), ('"x=1"', True),
            ('"deploy"', False), ('"can\'t"', False), ('"in-flight"', False),
            ('"overloaded?"', False), ('"Deploy"', False), ('"I"', False), ('"file."', False),
        ],
    )  # fmt: skip
    def test_identifier_detection(self, term: str, expected: bool) -> None:
        assert _is_identifier(term) is expected

    def test_rare_identifier_outweighs_common_words_around_it(
        self, db: Database, searcher: HybridSearcher
    ) -> None:
        ranking = searcher._keyword_ranking("what does --replay-from-offset mean exactly", 20)
        assert titles(db, ranking) == ["Flags"]


class TestHeadingOnlySections:
    def test_they_have_no_vectors_and_never_win_the_vector_ranking(
        self, db: Database, searcher: HybridSearcher
    ) -> None:
        ranking, _ = searcher._vector_ranking("Reference Operations", 20)
        assert "Operations" not in titles(db, ranking)
        assert "Health" in titles(db, ranking)

    def test_keyword_hits_prefer_sections_with_a_body(
        self, db: Database, searcher: HybridSearcher
    ) -> None:
        # "operations" matches the heading-only parent and (via its breadcrumb) the child
        assert titles(db, searcher._keyword_ranking("operations", 20)) == ["Health"]

    def test_a_heading_only_section_is_still_returned_when_nothing_else_matches(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        root = tmp_path / "solo"
        root.mkdir()
        (root / "a.md").write_text("# Alpha\n\ntext\n\n## Zanzibar\n\n## Other\n\nbody\n")
        Indexer(db, fake_embedder).index_directory(root)
        engine = HybridSearcher(db, fake_embedder)
        try:
            assert titles(db, engine._keyword_ranking("zanzibar", 20)) == ["Zanzibar"]
            found = {r.heading_title: r for r in engine.search("zanzibar")}
            assert found["Zanzibar"].fts_rank == 1
            assert found["Zanzibar"].vec_rank is None  # it has no vector to be found by
        finally:
            engine.close()


class TestPassageRanking:
    def test_best_passage_decides_and_is_reported(self, searcher: HybridSearcher) -> None:
        top = searcher.search("Reference Operations Health healthz answers traffic")[0]
        assert top.heading_path == "Reference > Operations > Health"
        assert top.matched_passage == "GET /healthz answers while the process accepts traffic."
        assert top.to_dict()["matched_passage"] == top.matched_passage

    def test_result_without_a_winning_passage_omits_the_field(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        class NoVectors(FakeEmbedder):
            def embed_query(self, text: str) -> list[float]:
                raise indexer_module.EmbeddingError("offline")

        root = tmp_path / "kw"
        root.mkdir()
        (root / "a.md").write_text("# A\n\nquokka habitat\n")
        Indexer(db, fake_embedder).index_directory(root)
        engine = HybridSearcher(db, NoVectors())
        try:
            result = engine.search("quokka")[0]
        finally:
            engine.close()
        assert result.matched_passage is None
        assert "matched_passage" not in result.to_dict()

    def test_indexer_reports_sections_and_passages(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        root = tmp_path / "cfg"
        root.mkdir()
        (root / "helios.md").write_text(CONFIG_DOC)
        report = Indexer(db, fake_embedder).index_directory(root)
        assert (report.sections_indexed, report.passages_indexed) == (6, 6)
        assert "6 sections embedded (6 passages)" in report.summary()
        assert db.count_rows("sections_vec") == 4  # "Settings" and "Operations" are heading-only
        assert db.count_rows("units_vec") == 6


# ---------------------------------------------------------------------- embedders


class TestEmbedderPresets:
    def test_presets(self, tmp_path: Path) -> None:
        gemma = create_embedder(cache_dir=tmp_path)
        assert isinstance(gemma, EmbeddingGemmaEmbedder)
        assert gemma.dimension == 768
        assert "embeddinggemma-300m-ONNX@" in gemma.model_name  # pinned revision is part of it
        small = create_embedder("bge-small", cache_dir=tmp_path)
        assert isinstance(small, FastEmbedEmbedder)
        assert (small.dimension, small.model_name) == (384, "BAAI/bge-small-en-v1.5")
        with pytest.raises(IndexingError, match="Unknown embedder"):
            create_embedder("gpt-embed")

    def test_gemma_prompts_are_applied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[list[str]] = []
        embedder = EmbeddingGemmaEmbedder()

        def capture(texts: Sequence[str]) -> list[list[float]]:
            seen.append(list(texts))
            return [[1.0] + [0.0] * 767 for _ in texts]

        monkeypatch.setattr(embedder, "_embed", capture)
        embedder.embed_documents(["Doc > A\n\nbody"])
        embedder.embed_query("how do I")
        assert seen == [
            ["title: none | text: Doc > A\n\nbody"],
            ["task: search result | query: how do I"],
        ]
        assert (GEMMA_DOCUMENT_PROMPT, GEMMA_QUERY_PROMPT) == (
            "title: none | text: ",
            "task: search result | query: ",
        )

    def test_nothing_is_downloaded_when_the_files_are_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import huggingface_hub

        def forbidden(*args: object, **kwargs: object) -> str:
            raise AssertionError("network access attempted")

        monkeypatch.setattr(huggingface_hub, "snapshot_download", forbidden)
        embedder = EmbeddingGemmaEmbedder(cache_dir=tmp_path)
        for name in GEMMA_FILES:
            target = tmp_path / "embeddinggemma-300m-onnx" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x")
        embedder._download()  # must not raise
        (tmp_path / "embeddinggemma-300m-onnx" / GEMMA_FILES[0]).unlink()
        with pytest.raises(AssertionError, match="network access attempted"):
            embedder._download()

    def test_bge_queries_get_the_retrieval_instruction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        queries: list[str] = []

        class Vector:
            def tolist(self) -> list[float]:
                return [1.0] + [0.0] * 383

        class FakeModel:
            def query_embed(self, text: str) -> list[Vector]:
                queries.append(text)
                return [Vector()]

        embedder = FastEmbedEmbedder()
        monkeypatch.setattr(embedder, "_load", lambda: FakeModel())
        embedder.embed_query("rotate credentials")
        assert queries == [
            "Represent this sentence for searching relevant passages: rotate credentials"
        ]


@pytest.fixture(scope="module")
def engine(
    real_embedder: Embedder, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[HybridSearcher]:
    root = tmp_path_factory.mktemp("quality")
    (root / "docs").mkdir()
    (root / "docs" / "helios.md").write_text(CONFIG_DOC, encoding="utf-8")
    for name, text in NOISE_CORPUS.items():
        (root / "docs" / name).write_text(text, encoding="utf-8")
    with Database(root / "q.db", embedding_dim=real_embedder.dimension) as database:
        Indexer(database, real_embedder).index_directory(root / "docs")
        searcher = HybridSearcher(database, real_embedder)
        yield searcher
        searcher.close()


@pytest.mark.embedding
class TestRetrievalQuality:
    """Paraphrases of table rows, code and lists: the cases section-level vectors missed."""

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("tune how many items are fetched at once", "Helios > Settings > Ingest"),
            ("how much RAM does the container ask for", "Helios > Settings > Resources"),
            ("stop the service without dropping requests", "Helios > Operations > Shutdown"),
            ("how do I undo a bad deploy", "Deployment Guide > Rolling Back"),
            ("is the process alive endpoint", "Reference > Operations > Health"),
            ("HELIOS_WORKERS", "Helios > Settings > Ingest"),
            ("--replay-from-offset", "Reference > Flags"),
        ],
    )
    def test_top_result(self, engine: HybridSearcher, query: str, expected: str) -> None:
        results = engine.search(query)
        assert results[0].heading_path == expected, [r.heading_path for r in results]

    def test_matched_passage_points_at_the_table_row(self, engine: HybridSearcher) -> None:
        top = engine.search("tune how many items are fetched at once")[0]
        assert top.matched_passage is not None
        assert "HELIOS_BATCH" in top.matched_passage
