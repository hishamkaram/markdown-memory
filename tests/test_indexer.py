"""Indexer tests: scanning, SHA-256 change detection, purging, failure isolation."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import pytest
from fakes import FakeEmbedder

from markdown_memory.db import Database
from markdown_memory.discovery import hash_bytes, iter_markdown_files
from markdown_memory.exceptions import EmbeddingError, IndexingError
from markdown_memory.indexer import Indexer
from markdown_memory.parser import MarkdownParser


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    (root / "guides" / "deep").mkdir(parents=True)
    (root / "node_modules" / "pkg").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "README.md").write_text("# Readme\n\nhello\n\n## Usage\n\nrun it\n")
    (root / "guides" / "setup.md").write_text("# Setup\n\nsteps\n")
    (root / "guides" / "deep" / "NOTES.MD").write_text("# Notes\n\nupper-case suffix\n")
    (root / "guides" / "legacy.markdown").write_text("# Legacy\n\nold suffix\n")
    (root / "guides" / "image.png").write_bytes(b"\x89PNG")
    (root / "notes.txt").write_text("# not markdown\n")
    (root / "node_modules" / "pkg" / "README.md").write_text("# Vendored\n")
    (root / ".git" / "description.md").write_text("# Git internals\n")
    return root


def indexed_paths(db: Database) -> list[str]:
    return [summary.file_path for summary in db.list_documents()]


class TestScanning:
    def test_finds_markdown_recursively_and_prunes_vendored_trees(self, docs: Path) -> None:
        found = [str(path.relative_to(docs)) for path in iter_markdown_files(docs)]
        assert found == [
            "README.md",
            "guides/legacy.markdown",
            "guides/setup.md",
            "guides/deep/NOTES.MD",
        ]

    def test_missing_directory(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        with pytest.raises(IndexingError, match="does not exist"):
            Indexer(db, fake_embedder).index_directory(tmp_path / "nope")

    def test_file_instead_of_directory(
        self, db: Database, fake_embedder: FakeEmbedder, docs: Path
    ) -> None:
        with pytest.raises(IndexingError, match="Not a directory"):
            Indexer(db, fake_embedder).index_directory(docs / "README.md")

    def test_dimension_mismatch_is_detected_up_front(self, db: Database) -> None:
        with pytest.raises(IndexingError, match="dimensional"):
            Indexer(db, FakeEmbedder(dimension=8))


class TestIncrementalIndexing:
    def test_first_run_indexes_everything(
        self, db: Database, fake_embedder: FakeEmbedder, docs: Path
    ) -> None:
        report = Indexer(db, fake_embedder).index_directory(docs)
        assert (report.files_scanned, report.files_indexed) == (4, 4)
        assert (report.files_unchanged, report.files_purged) == (0, 0)
        assert report.sections_indexed == db.count_rows("sections") == 5
        assert db.count_rows("sections_vec") == 5
        assert report.errors == ()
        assert report.elapsed_seconds >= 0
        assert "4 scanned, 4 (re)indexed" in report.summary()

    def test_stored_hash_is_the_files_sha256(
        self, db: Database, fake_embedder: FakeEmbedder, docs: Path
    ) -> None:
        Indexer(db, fake_embedder).index_directory(docs)
        document = db.get_document(str(docs / "README.md"))
        assert document is not None
        expected = hashlib.sha256((docs / "README.md").read_bytes()).hexdigest()
        assert document.content_hash == expected == hash_bytes((docs / "README.md").read_bytes())
        assert document.title == "Readme"
        assert document.last_modified == int((docs / "README.md").stat().st_mtime)

    def test_breadcrumb_is_prepended_to_the_embedded_text(
        self, db: Database, fake_embedder: FakeEmbedder, docs: Path
    ) -> None:
        Indexer(db, fake_embedder).index_directory(docs)
        embedded = [text for call in fake_embedder.document_calls for text in call]
        assert "Readme > Usage: run it" in embedded  # every passage carries its breadcrumb
        # The section's own text is no longer embedded separately: its vector is the mean
        # of its passages, which the model cannot truncate and costs one call fewer.
        assert "Readme > Usage\n\nrun it" not in embedded

    def test_unchanged_files_are_skipped_without_embedding(
        self, db: Database, fake_embedder: FakeEmbedder, docs: Path
    ) -> None:
        indexer = Indexer(db, fake_embedder)
        indexer.index_directory(docs)
        calls_after_first_run = len(fake_embedder.document_calls)
        report = indexer.index_directory(docs)
        assert (report.files_indexed, report.files_unchanged) == (0, 4)
        assert report.sections_indexed == 0
        assert len(fake_embedder.document_calls) == calls_after_first_run

    def test_touching_a_file_without_changing_bytes_does_not_reindex(
        self, db: Database, fake_embedder: FakeEmbedder, docs: Path
    ) -> None:
        indexer = Indexer(db, fake_embedder)
        indexer.index_directory(docs)
        (docs / "README.md").touch()
        assert indexer.index_directory(docs).files_indexed == 0

    def test_only_the_modified_file_is_reindexed(
        self, db: Database, fake_embedder: FakeEmbedder, docs: Path
    ) -> None:
        indexer = Indexer(db, fake_embedder)
        indexer.index_directory(docs)
        before = db.get_document(str(docs / "guides" / "setup.md"))
        (docs / "guides" / "setup.md").write_text("# Setup\n\nnew steps\n\n## Extra\n\nmore\n")
        report = indexer.index_directory(docs)
        assert (report.files_indexed, report.files_unchanged) == (1, 3)
        after = db.get_document(str(docs / "guides" / "setup.md"))
        assert before is not None and after is not None
        assert after.id == before.id
        assert after.content_hash != before.content_hash
        assert [s.heading_path for s in db.get_sections(after.id)] == ["Setup", "Setup > Extra"]
        assert db.fts_search('"steps"', 5) != []
        assert db.count_rows("sections") == db.count_rows("sections_vec") == 6

    def test_deleted_files_are_purged_with_their_sections_and_vectors(
        self, db: Database, fake_embedder: FakeEmbedder, docs: Path
    ) -> None:
        indexer = Indexer(db, fake_embedder)
        indexer.index_directory(docs)
        (docs / "README.md").unlink()
        report = indexer.index_directory(docs)
        assert report.files_purged == 1
        assert str(docs / "README.md") not in indexed_paths(db)
        assert db.count_rows("sections") == db.count_rows("sections_vec") == 3
        assert db.count_rows("sections_fts") == 3
        assert db.fts_search('"hello"', 5) == []

    def test_purge_is_scoped_to_the_indexed_directory(
        self, db: Database, fake_embedder: FakeEmbedder, docs: Path, tmp_path: Path
    ) -> None:
        other = tmp_path / "other"
        other.mkdir()
        (other / "keep.md").write_text("# Keep\n")
        indexer = Indexer(db, fake_embedder)
        indexer.index_directory(other)
        report = indexer.index_directory(docs)
        assert report.files_purged == 0
        assert str(other / "keep.md") in indexed_paths(db)

    def test_indexing_a_subdirectory_leaves_siblings_alone(
        self, db: Database, fake_embedder: FakeEmbedder, docs: Path
    ) -> None:
        indexer = Indexer(db, fake_embedder)
        indexer.index_directory(docs)
        report = indexer.index_directory(docs / "guides")
        assert (report.files_scanned, report.files_unchanged, report.files_purged) == (3, 3, 0)
        assert len(indexed_paths(db)) == 4

    def test_empty_and_heading_less_files_are_indexed_without_error(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        root = tmp_path / "edge"
        root.mkdir()
        (root / "empty.md").write_text("")
        (root / "plain.md").write_text("no headings here\n")
        report = Indexer(db, fake_embedder).index_directory(root)
        assert report.errors == ()
        counts = {Path(d.file_path).name: d.section_count for d in db.list_documents()}
        assert counts == {"empty.md": 0, "plain.md": 1}
        titles = {Path(d.file_path).name: d.title for d in db.list_documents()}
        assert titles == {"empty.md": "empty", "plain.md": "plain"}
        assert Indexer(db, fake_embedder).index_directory(root).files_unchanged == 2

    def test_invalid_utf8_is_decoded_with_replacement(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        root = tmp_path / "bytes"
        root.mkdir()
        (root / "latin.md").write_bytes(b"# Caf\xe9\n\nbody\n")
        report = Indexer(db, fake_embedder).index_directory(root)
        assert report.errors == ()
        assert db.list_documents()[0].title == "Caf�"

    def test_changing_the_embedding_model_discards_stale_vectors(
        self, db: Database, docs: Path
    ) -> None:
        Indexer(db, FakeEmbedder(model_name="model-a")).index_directory(docs)
        assert db.get_meta("embedding_model") == "model-a"
        report = Indexer(db, FakeEmbedder(model_name="model-b")).index_directory(docs)
        assert report.files_indexed == 4  # hashes matched, but every vector was stale
        assert db.get_meta("embedding_model") == "model-b"
        assert db.count_rows("sections_vec") == 5


class TestFailureIsolation:
    def test_one_bad_file_does_not_abort_the_run(self, db: Database, docs: Path) -> None:
        class FlakyEmbedder(FakeEmbedder):
            def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
                if any("steps" in text for text in texts):
                    raise EmbeddingError("simulated model failure")
                return super().embed_documents(texts)

        report = Indexer(db, FlakyEmbedder()).index_directory(docs)
        assert report.files_indexed == 3
        assert [Path(e.file_path).name for e in report.errors] == ["setup.md"]
        assert "simulated model failure" in report.errors[0].message
        assert "ERROR" in report.summary()
        assert str(docs / "guides" / "setup.md") not in indexed_paths(db)

    def test_a_failing_file_is_retried_on_the_next_run(self, db: Database, docs: Path) -> None:
        class FailsOnce(FakeEmbedder):
            failed = False

            def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
                if not self.failed and any("steps" in text for text in texts):
                    self.failed = True
                    raise EmbeddingError("transient")
                return super().embed_documents(texts)

        indexer = Indexer(db, FailsOnce())
        assert len(indexer.index_directory(docs).errors) == 1
        second = indexer.index_directory(docs)
        assert (second.files_indexed, second.files_unchanged, len(second.errors)) == (1, 3, 0)

    def test_a_previously_indexed_file_that_now_fails_is_not_purged(
        self, db: Database, docs: Path
    ) -> None:
        Indexer(db, FakeEmbedder()).index_directory(docs)
        (docs / "guides" / "setup.md").write_text("# Setup\n\nchanged steps\n")

        class AlwaysFails(FakeEmbedder):
            def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
                raise EmbeddingError("down")

        report = Indexer(db, AlwaysFails()).index_directory(docs)
        assert report.files_purged == 0
        assert len(report.errors) == 1
        assert str(docs / "guides" / "setup.md") in indexed_paths(db)  # stale beats missing


class TestSectionVectorsCoverTheWholeSection:
    """A section's own vector used to be a separate embedding of all its text.

    The model truncates at 512 tokens, and 126 of the 1,589 sections in the vendored
    corpus are longer than that - the largest half again over - so the tail never reached
    the section-level signal. The vector is the mean of the section's passages now, which
    nothing truncates, and which costs one embedding call fewer per section rather than
    one more.
    """

    def long_section(self) -> str:
        body = "\n\n".join(
            f"Paragraph {n} describes a distinct rule about retries and deadlines."
            for n in range(40)
        )
        return f"# Doc\n\n## Retry policy\n\n{body}\n"

    def test_the_whole_section_reaches_the_embedder(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        (tmp_path / "long.md").write_text(self.long_section(), encoding="utf-8")
        Indexer(db, fake_embedder).index_directory(tmp_path)
        embedded = " ".join(text for call in fake_embedder.document_calls for text in call)
        assert "Paragraph 0 " in embedded
        assert "Paragraph 39 " in embedded, "the tail of a long section never reached the model"

    def test_the_section_text_is_not_embedded_a_second_time(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """Pooling replaces that call; embedding both would cost more, not less."""
        (tmp_path / "long.md").write_text(self.long_section(), encoding="utf-8")
        Indexer(db, fake_embedder).index_directory(tmp_path)
        embedded = [text for call in fake_embedder.document_calls for text in call]
        assert all(text.startswith("Doc > Retry policy: ") for text in embedded)

    def test_the_section_vector_is_the_centroid_of_its_passages(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """Not the first passage, not a truncation: the mean of all of them.

        Searching with that centroid must land on the section at distance zero, which one
        passage's vector cannot satisfy for a section of forty.
        """
        import math

        (tmp_path / "long.md").write_text(self.long_section(), encoding="utf-8")
        Indexer(db, fake_embedder).index_directory(tmp_path)
        draft = next(
            section
            for section in MarkdownParser()
            .parse(self.long_section(), fallback_title="Doc")
            .sections
            if section.units
        )
        passages = fake_embedder.embed_documents(list(draft.unit_texts))
        totals = [sum(values) for values in zip(*passages, strict=True)]
        norm = math.sqrt(sum(value * value for value in totals))
        centroid = [value / norm for value in totals]
        nearest = db.vec_search(centroid, 1)
        assert nearest and nearest[0][1] < 1e-6, f"section vector is not the centroid: {nearest}"

    def test_passages_that_cancel_produce_no_vector_rather_than_noise(self) -> None:
        """`norm == 0.0` was the wrong test.

        Opposing passages cancel to float residue near 1e-16, and dividing that by its
        own magnitude yields a full-length vector pointing in an arbitrary direction -
        which then answers arbitrary queries with this section.
        """
        from markdown_memory.indexer import _mean_vector

        assert _mean_vector([[1.0, 0.0], [-1.0, 0.0]]) is None
        assert _mean_vector([[1.0, 0.0], [-1.0, 1e-17]]) is None
        assert _mean_vector([]) is None

    def test_cancelling_passages_do_not_cost_the_file_its_place(self) -> None:
        """Storage rejects a body-bearing section with no vector, and rejects the whole
        file with it. Passages that cancel must not be able to do that."""
        from markdown_memory.indexer import _section_vector

        assert _section_vector([[1.0, 0.0], [-1.0, 0.0]]) == [1.0, 0.0]
        assert _section_vector([]) is None

    def test_a_single_passage_pools_to_itself(self) -> None:
        from markdown_memory.indexer import _mean_vector

        pooled = _mean_vector([[0.6, 0.8]])
        assert pooled is not None
        assert pooled == pytest.approx([0.6, 0.8])

    def test_a_section_with_no_body_still_has_no_vector(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """Pooling nothing must give nothing, not a zero vector pointing nowhere."""
        (tmp_path / "stub.md").write_text(
            "# Doc\n\n## Empty\n\n### Child\n\ntext\n", encoding="utf-8"
        )
        Indexer(db, fake_embedder).index_directory(tmp_path)
        document = db.get_document(str(tmp_path / "stub.md"))
        assert document is not None
        sections = db.get_sections(document.id)
        with_body = db.sections_with_passages([section.id for section in sections])
        assert len(with_body) < len(sections), "the fixture has no heading-only section"
        assert db.count_rows("sections_vec") == len(with_body)
