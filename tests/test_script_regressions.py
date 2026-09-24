"""Regressions in the developer scripts: the retrieval evaluation gate.

One test per defect found in code review; each fails when its fix is reverted.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import threading
from collections.abc import Sequence
from pathlib import Path

import pytest
from fakes import FakeEmbedder
from helpers import store

from markdown_memory.config import ServerConfig, resolve_config
from markdown_memory.db import Database
from markdown_memory.models import (
    FileFailure,
    IndexReport,
    SearchResult,
)
from markdown_memory.server import MarkdownMemoryService


class TestEvalScript:
    @pytest.fixture
    def evaluation(self) -> object:
        import importlib.util

        path = Path(__file__).parent.parent / "scripts" / "eval_retrieval.py"
        spec = importlib.util.spec_from_file_location("eval_retrieval_under_test", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_ndcg_is_bounded_when_parts_share_a_label(
        self, evaluation: object, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        body = "\n\n".join(
            f"capacity planning paragraph {n} " + "sizing words " * 30 for n in range(12)
        )
        root = tmp_path / "docs"
        root.mkdir()
        (root / "a.md").write_text(f"# Guide\n\n## Capacity Planning\n\n{body}\n")
        service = MarkdownMemoryService(
            ServerConfig(db_path=tmp_path / "e.db", docs_dir=root), embedder=fake_embedder
        )
        try:
            service.index_directory()
            cases = [{"query": "capacity planning sizing", "expected": "Guide > Capacity Planning"}]
            scores = evaluation.evaluate(service, cases)  # type: ignore[attr-defined]
        finally:
            service.close()
        assert scores.top1 == 1.0
        assert 0.0 < scores.ndcg5 <= 1.0  # three "(Part n)" hits used to score 2.13

    def test_exported_embedder_cannot_switch_the_gate_off(self, evaluation: object) -> None:
        source = Path(evaluation.__file__).read_text(encoding="utf-8")  # type: ignore[attr-defined]
        assert "default=DEFAULT_EMBEDDER" in source
        assert "base.embedder" not in source  # the environment must not pick the scored model
        assert "GATES NOT CHECKED" in source


class _EvalStubService:
    """Enough of MarkdownMemoryService for eval_retrieval.main(); scoring is stubbed out."""

    def __init__(self, config: ServerConfig) -> None:
        self.embedder = FakeEmbedder(model_name="stub")
        self.db = None  # only ever handed to the cache probes, which the tests stub out
        self.closed = False
        self.report = IndexReport(
            directory="stub", files_scanned=0, files_indexed=0, files_unchanged=0, files_purged=0,
            sections_indexed=0, passages_indexed=0, elapsed_seconds=0.0,
        )  # fmt: skip

    def index_directory(self, directory: str | None = None) -> IndexReport:
        return self.report

    def search_docs(self, query: str, limit: int) -> list[SearchResult]:
        return []

    def close(self) -> None:
        self.closed = True


class TestBaselineIsNotRecordedForAFailingRun(TestEvalScript):
    def run_main(
        self, evaluation: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, top1: float
    ) -> tuple[int, dict[str, object]]:
        baseline = tmp_path / "baseline.json"
        original = json.loads(
            (Path(evaluation.__file__).parent / "eval_data/baseline.json").read_text()  # type: ignore[attr-defined]
        )
        baseline.write_text(json.dumps(original))
        scores = evaluation.Scores(  # type: ignore[attr-defined]
            top1=top1, top3=top1, top5=top1, any_valid_top1=top1, ndcg5=top1,
            median_ms=1.0, p95_ms=1.0, misses=(),
        )  # fmt: skip
        monkeypatch.setattr(evaluation, "BASELINE", baseline)
        monkeypatch.setattr(evaluation, "evaluate", lambda service, cases: scores)
        monkeypatch.setattr(sys, "argv", ["eval_retrieval.py", "--update-baseline"])
        # Keep the run off the real index cache: it belongs to the user's machine, and a
        # real evaluation may be holding its lock.
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        stub = _EvalStubService(ServerConfig(db_path=tmp_path / "x.db", docs_dir=tmp_path))
        monkeypatch.setattr(evaluation, "open_service", lambda *_args: (stub, False))
        try:
            code = evaluation.main()  # type: ignore[attr-defined]
        finally:
            logging.disable(logging.NOTSET)  # main() disables logging process-wide
        return code, json.loads(baseline.read_text())["embeddinggemma"]

    def test_a_regressed_run_leaves_the_frozen_baseline_alone(
        self, evaluation: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:  # fmt: skip
        code, recorded = self.run_main(evaluation, tmp_path, monkeypatch, top1=0.5)
        assert code == 1
        assert "REGRESSION" in capsys.readouterr().out
        assert recorded["held_out/paraphrase"]["top1"] != 0.5  # still the frozen numbers

    def test_a_passing_run_records_the_new_numbers(
        self, evaluation: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        code, recorded = self.run_main(evaluation, tmp_path, monkeypatch, top1=1.0)
        assert code == 0
        assert recorded["held_out/paraphrase"]["top1"] == 1.0


class TestEvalIndexCache:
    """The cached evaluation index must never answer for a corpus it was not built from.

    A gate that silently scores a stale index reports "no regression" for a change it
    never measured, which is the one failure mode a benchmark cannot recover from.
    """

    @pytest.fixture
    def corpus(self, tmp_path: Path) -> Path:
        root = tmp_path / "corpus"
        (root / "nested").mkdir(parents=True)
        (root / "guide.md").write_text(
            "# Guide\n\n## Storage\n\n<!-- editorial: first draft -->\n\n"
            "HELIOS_WAL_SEGMENT_MB sets the segment size.\n",
            encoding="utf-8",
        )
        (root / "nested" / "other.md").write_text("# Other\n\nunrelated prose\n", encoding="utf-8")
        return root

    def test_a_changed_corpus_file_changes_the_key(self, corpus: Path) -> None:
        import eval_cache

        before = eval_cache.corpus_digest(corpus)
        (corpus / "guide.md").write_text("# Guide\n\n## Storage\n\nrewritten\n", encoding="utf-8")
        assert eval_cache.corpus_digest(corpus) != before

    def test_a_changed_chunking_constant_changes_the_key(
        self, corpus: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import eval_cache

        before = eval_cache.build_key(corpus, "embeddinggemma").digest
        monkeypatch.setattr(eval_cache, "MAX_UNIT_CHARS", eval_cache.MAX_UNIT_CHARS + 1)
        assert eval_cache.build_key(corpus, "embeddinggemma").digest != before

    def test_two_embedders_do_not_share_one_index(self, corpus: Path) -> None:
        import eval_cache

        gemma = eval_cache.build_key(corpus, "embeddinggemma").digest
        assert eval_cache.build_key(corpus, "bge-small").digest != gemma

    def test_moving_text_between_units_changes_the_fingerprint(self, corpus: Path) -> None:
        """Counts alone would miss this: same sections, same unit count, different text."""
        import eval_cache

        (corpus / "guide.md").write_text(
            "# Guide\n\n## Storage\n\nfirst half\n\nsecond half\n", encoding="utf-8"
        )
        before = eval_cache.parse_fingerprint(corpus)
        (corpus / "guide.md").write_text(
            "# Guide\n\n## Storage\n\nfirst\n\nhalf second half\n", encoding="utf-8"
        )
        assert eval_cache.parse_fingerprint(corpus) != before

    def test_text_no_passage_carries_still_changes_the_fingerprint(self, corpus: Path) -> None:
        """FTS5 indexes the whole section, so section content is part of the artifact.

        An HTML comment never becomes a passage, and rewriting one in place moves no line
        numbers - so only hashing the section's own content can notice it.
        """
        import eval_cache

        before = eval_cache.parse_fingerprint(corpus)
        guide = corpus / "guide.md"
        guide.write_text(guide.read_text().replace("first draft", "final cut!"), encoding="utf-8")
        assert eval_cache.parse_fingerprint(corpus) != before

    def test_a_change_in_how_passages_are_cut_changes_the_fingerprint(
        self, corpus: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The files are untouched; only the splitting rule moved.

        This is the change the fingerprint exists for - a section's text is identical but
        the passages embedded from it are not - and section content alone cannot see it.
        """
        import eval_cache

        from markdown_memory import parser as parser_module

        before = eval_cache.parse_fingerprint(corpus)
        monkeypatch.setattr(parser_module, "MAX_UNIT_CHARS", 12)
        assert eval_cache.parse_fingerprint(corpus) != before

    def test_a_corpus_that_moved_during_indexing_is_not_cached(self, corpus: Path) -> None:
        """A build over a moving corpus describes no single state of it.

        Worse, once the edit is reverted, that database validates cleanly forever.
        """
        import eval_cache

        fingerprint = eval_cache.parse_fingerprint(corpus)
        eval_cache.confirm_stable(corpus, fingerprint)  # unchanged: no complaint
        guide = corpus / "guide.md"
        guide.write_text(guide.read_text() + "\n## Added\n\nmid-build\n", encoding="utf-8")
        with pytest.raises(eval_cache.StaleCacheError, match="changed while"):
            eval_cache.confirm_stable(corpus, fingerprint)

    def test_a_partial_build_is_neither_scored_nor_cached(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`IndexReport.errors` means files are missing from the index.

        Scoring it reports a number for a system that was never built, and caching it
        keeps reporting that number on every later run.
        """
        import eval_cache
        import eval_retrieval

        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        monkeypatch.setattr(eval_retrieval, "create_embedder", lambda *a, **k: FakeEmbedder())
        failed = IndexReport(
            directory="stub", files_scanned=2, files_indexed=1, files_unchanged=0,
            files_purged=0, sections_indexed=1, passages_indexed=1, elapsed_seconds=0.0,
            errors=(FileFailure(file_path="broken.md", message="Embedding failed"),),
        )  # fmt: skip
        stub = _EvalStubService(ServerConfig(db_path=tmp_path / "x.db", docs_dir=tmp_path))
        stub.report = failed
        monkeypatch.setattr(eval_retrieval, "_service", lambda *a, **k: stub)
        arguments = argparse.Namespace(embedder="embeddinggemma", rebuild=True)
        base = ServerConfig(db_path=tmp_path / "x.db", docs_dir=tmp_path)
        probes = [eval_cache.Probe("body number 1", "Doc > S1: body number 1")]
        # Without these, a build that is wrongly allowed through dies in the cache probes
        # instead of reaching the recording step - and a test that dies proves nothing
        # about whether the partial index would have been scored and cached.
        monkeypatch.setattr(eval_cache, "check_vectors", lambda *a, **k: None)
        monkeypatch.setattr(eval_cache, "confirm_stable", lambda *a, **k: None)

        with pytest.raises(SystemExit, match="indexing failed"):
            eval_retrieval.open_service(arguments, base, probes)
        assert not list((tmp_path / "cache").rglob("*.meta.json"))
        assert stub.closed

    def test_a_cache_without_metadata_is_refused(self, tmp_path: Path) -> None:
        import eval_cache

        db_path = tmp_path / "eval.db"
        db_path.write_bytes(b"")
        with pytest.raises(eval_cache.StaleCacheError):
            eval_cache.validate(db_path, "fingerprint", eval_cache.CacheKey("c", "k", "e", "s"))

    def test_a_cache_built_from_another_corpus_is_refused(self, tmp_path: Path) -> None:
        import eval_cache

        db_path = tmp_path / "eval.db"
        db_path.write_bytes(b"")
        key = eval_cache.CacheKey("c", "k", "e", "s")
        eval_cache.record(db_path, key, "old-fingerprint")
        with pytest.raises(eval_cache.StaleCacheError):
            eval_cache.validate(db_path, "new-fingerprint", key)
        eval_cache.validate(db_path, "old-fingerprint", key)  # the matching one is accepted

    def test_a_cache_directory_copied_from_another_key_is_refused(self, tmp_path: Path) -> None:
        """The directory name is a 64-bit truncation; the recorded key is not.

        A cache copied between machines or keys would otherwise be read as whatever key
        its directory name happens to spell.
        """
        import eval_cache

        db_path = tmp_path / "eval.db"
        db_path.write_bytes(b"")
        eval_cache.record(db_path, eval_cache.CacheKey("corpus-a", "k", "e", "s"), "same")
        with pytest.raises(eval_cache.StaleCacheError, match="different corpus"):
            eval_cache.validate(db_path, "same", eval_cache.CacheKey("corpus-b", "k", "e", "s"))

    def test_metadata_is_written_atomically(self, tmp_path: Path) -> None:
        """The metadata file is the commit marker: half of one must never exist."""
        import eval_cache

        db_path = tmp_path / "eval.db"
        db_path.write_bytes(b"")
        eval_cache.record(db_path, eval_cache.CacheKey("c", "k", "e", "s"), "f")
        assert not list(tmp_path.glob("*.tmp"))
        source = (Path(__file__).parent.parent / "scripts" / "eval_cache.py").read_text()
        assert "os.replace(temporary, target)" in source

    def test_a_reinstalled_model_invalidates_the_cache(self, tmp_path: Path) -> None:
        """The constants name the revision requested; only the files say what is on disk."""
        import eval_cache

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.md").write_text("# A\n\nbody\n", encoding="utf-8")
        cache_dir = tmp_path / "models"
        models = eval_cache.gemma_model_dir(cache_dir) / "onnx"
        models.mkdir(parents=True)
        weights = models / "model_quantized.onnx"
        weights.write_bytes(b"first")
        before = eval_cache.build_key(corpus, "embeddinggemma", model_cache_dir=cache_dir).digest
        weights.write_bytes(b"a different download entirely")
        after = eval_cache.build_key(corpus, "embeddinggemma", model_cache_dir=cache_dir).digest
        assert before != after

    def test_discard_removes_the_write_ahead_log_too(self, tmp_path: Path) -> None:
        """A surviving -wal would restore rows into the rebuilt index."""
        import eval_cache

        db_path = tmp_path / "eval.db"
        for suffix in ("", "-wal", "-shm"):
            Path(str(db_path) + suffix).write_bytes(b"x")
        eval_cache.record(db_path, eval_cache.CacheKey("c", "k", "e", "s"), "f")
        eval_cache.discard(db_path)
        assert not any(Path(str(db_path) + s).exists() for s in ("", "-wal", "-shm"))

    def test_a_second_evaluation_refuses_to_start(self, tmp_path: Path) -> None:
        """Concurrent runs share a CPU, and the latency each reports is then meaningless."""
        import eval_cache

        with contextlib.ExitStack() as held:
            held.enter_context(eval_cache.lock(tmp_path))
            with pytest.raises(eval_cache.BusyError), eval_cache.lock(tmp_path):
                pass
        with eval_cache.lock(tmp_path):  # released, so the next run may start
            pass

    def test_pruning_keeps_only_the_current_index(self, tmp_path: Path) -> None:
        import eval_cache

        for name in ("keepme", "staleone", "staletwo"):
            (tmp_path / name).mkdir()
            (tmp_path / name / "eval.db").write_bytes(b"x")
        eval_cache.prune(tmp_path, "keepme")
        assert sorted(entry.name for entry in tmp_path.iterdir()) == ["keepme"]

    def test_the_probes_are_spread_through_the_corpus(self, corpus: Path) -> None:
        """One probe only tells you about one passage.

        A build that went wrong partway through leaves the first passage perfect and
        everything after it wrong.
        """
        import eval_cache

        probes = eval_cache.probe_passages(corpus)
        assert len(probes) > 1
        assert len({probe.text for probe in probes}) == len(probes)

    def test_a_named_pipe_in_the_corpus_does_not_hang_the_cache(self, corpus: Path) -> None:
        """`iter_markdown_files` yields whatever is named *.md; reading a FIFO blocks."""
        import eval_cache

        if not hasattr(os, "mkfifo"):
            pytest.skip("no FIFOs on this platform")
        os.mkfifo(corpus / "pipe.md")
        finished: list[str] = []
        worker = threading.Thread(
            target=lambda: finished.append(eval_cache.corpus_digest(corpus)), daemon=True
        )
        worker.start()
        worker.join(timeout=20)
        assert not worker.is_alive(), "corpus_digest blocked on a FIFO"
        assert finished

    def test_a_matching_index_passes_the_vector_probe(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        import eval_cache

        store(db, fake_embedder, "doc.md")
        eval_cache.check_vectors(
            db,
            fake_embedder,
            [eval_cache.Probe(text="body number 1", embedding_text="Doc > S1: body number 1")],
        )

    def test_an_index_whose_vectors_came_from_another_model_is_refused(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        """The check the identifier canary could not do.

        Search fuses a keyword ranking with the vector ranking, so a canary query comes
        back correctly from FTS5 alone however wrong the vectors are. Only re-embedding a
        stored passage reads them.
        """
        import eval_cache

        store(db, fake_embedder, "doc.md")
        with pytest.raises(eval_cache.StaleCacheError, match="vectors"):
            eval_cache.check_vectors(
                db,
                _ReversedEmbedder(),
                [eval_cache.Probe("body number 1", "Doc > S1: body number 1")],
            )

    def test_a_probe_matched_to_the_wrong_passage_is_refused(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        """Distance alone is not enough: two passages can sit at the same point."""
        import eval_cache

        store(db, fake_embedder, "doc.md")
        # The same words in another order: identical bag-of-words vector, different text.
        with pytest.raises(eval_cache.StaleCacheError):
            eval_cache.check_vectors(
                db, fake_embedder, [eval_cache.Probe("number body 1", "Doc > S1: 1 number body")]
            )

    def test_an_index_without_passage_vectors_is_refused(
        self, db: Database, fake_embedder: FakeEmbedder
    ) -> None:
        import eval_cache

        with pytest.raises(eval_cache.StaleCacheError, match="no passage vectors"):
            eval_cache.check_vectors(
                db, fake_embedder, [eval_cache.Probe("anything", "Doc > S1: anything")]
            )


class _ReversedEmbedder(FakeEmbedder):
    """Another model, in the only way the storage layer can tell: different vectors."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(reversed(vector)) for vector in super().embed_documents(texts)]


class TestEveryMutationStillPointsAtCode:
    """The harness quotes source it does not own, and source moves underneath it.

    A refactor renamed nothing and broke five mutations at once: each anchor still read
    like the code, but no longer matched a character of it, so the mutation was never
    applied, the tests ran against an unmutated copy, and the sweep called it a survivor.
    A sweep takes forty minutes and reads like a hole in the suite. This reads like what
    it is, in a second, on the commit that moved the line.
    """

    @pytest.fixture
    def harness(self) -> object:
        import importlib.util

        path = Path(__file__).parent.parent / "scripts" / "mutation_check.py"
        spec = importlib.util.spec_from_file_location("mutation_check_under_test", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_every_anchor_matches_exactly_one_line_of_its_module(self, harness: object) -> None:
        root = Path(__file__).parent.parent
        stale: list[str] = []
        for mutation in harness.MUTATIONS:  # type: ignore[attr-defined]
            directory = "src/markdown_memory" if mutation.area == "src" else "scripts"
            source = (root / directory / mutation.module).read_text(encoding="utf-8")
            found = harness._occurrences(source, mutation.old)  # type: ignore[attr-defined]
            if len(found) != 1:
                stale.append(f"{mutation.name}: {len(found)} matches in {mutation.module}")
            elif mutation.old.startswith(" ") and source[found[0] - 1] != "\n":
                stale.append(f"{mutation.name}: matches mid-line in {mutation.module}")
        assert stale == [], "these mutations no longer name any code: " + "; ".join(stale)

    def test_every_selector_names_a_test_that_exists(self, harness: object) -> None:
        """A renamed test makes pytest exit 5, which the sweep used to read as a catch."""
        suite = "\n".join(
            path.read_text(encoding="utf-8") for path in Path(__file__).parent.glob("test_*.py")
        )
        # ``tests`` is a ``-k`` expression: one name, or several joined by and/or/not, and
        # a name may be a class as readily as a function.
        missing = [
            f"{mutation.name} -> {name}"
            for mutation in harness.MUTATIONS  # type: ignore[attr-defined]
            for name in mutation.tests.replace("(", " ").replace(")", " ").split()
            if name not in {"and", "or", "not"}
            and f"def {name}(" not in suite
            and f"class {name}" not in suite
        ]
        assert missing == [], "these mutations select no test: " + "; ".join(missing)


class TestTheReindexScriptTargetsTheDirectoryItWasGiven:
    """It verified one project's index while writing to another's.

    `ServerConfig.from_env()` derives the default database from the *environment's*
    documentation root. The script replaced `docs_dir` with the directory on its command
    line and kept that `db_path`, so `reindex_docs.py /other/docs` re-indexed one tree into
    another tree's database - the exact failure its own skill warns about ("use the same
    database the MCP server uses, or you will verify a different index than the one being
    searched"), committed by the tool meant to prevent it.
    """

    @pytest.fixture
    def roots(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
        alpha, beta = tmp_path / "alpha" / "docs", tmp_path / "beta" / "docs"
        for root in (alpha, beta):
            root.mkdir(parents=True)
        monkeypatch.setenv("MARKDOWN_MEMORY_DOCS_DIR", str(alpha))
        monkeypatch.delenv("MARKDOWN_MEMORY_DB", raising=False)
        monkeypatch.delenv("MARKDOWN_MEMORY_EXCLUDE", raising=False)
        return alpha, beta

    def test_naming_a_directory_rekeys_the_database(self, roots: tuple[Path, Path]) -> None:
        alpha, beta = roots
        assert resolve_config(docs_dir=beta).db_path != ServerConfig.from_env().db_path
        assert resolve_config(docs_dir=alpha).db_path == ServerConfig.from_env().db_path

    def test_an_explicitly_configured_database_still_wins(
        self, roots: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Re-keying must not overrule a database someone named on purpose."""
        _, beta = roots
        chosen = tmp_path / "chosen.db"
        monkeypatch.setenv("MARKDOWN_MEMORY_DB", str(chosen))
        assert resolve_config(docs_dir=beta).db_path == chosen
        flag = tmp_path / "flag.db"
        assert resolve_config(db=flag, docs_dir=beta).db_path == flag

    def test_exclusions_are_inherited_rather_than_dropped(
        self, roots: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A forced re-index used to pull in trees the server itself never indexes.

        The script built its own `ServerConfig` without `exclude`, which defaults to empty,
        so it wrote documents the running server would never have written - and then
        pronounced that index verified.
        """
        _, beta = roots
        monkeypatch.setenv("MARKDOWN_MEMORY_EXCLUDE", "vendor,tests/fixtures")
        assert resolve_config(docs_dir=beta).exclude == ("vendor", "tests/fixtures")

    def test_the_script_resolves_its_configuration_the_same_way(
        self, roots: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The script under test, not a re-statement of it: run `main` with the model stubbed."""
        import importlib.util

        _, beta = roots
        monkeypatch.setenv("MARKDOWN_MEMORY_EXCLUDE", "vendor")
        path = Path(__file__).parent.parent / "scripts" / "reindex_docs.py"
        spec = importlib.util.spec_from_file_location("reindex_docs_under_test", path)
        assert spec is not None and spec.loader is not None
        script = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = script
        spec.loader.exec_module(script)

        seen: list[ServerConfig] = []

        class _StopError(Exception):
            pass

        def _capture(config: ServerConfig) -> object:
            seen.append(config)
            raise _StopError

        monkeypatch.setattr(script, "MarkdownMemoryService", _capture)
        monkeypatch.setattr(sys, "argv", ["reindex_docs.py", str(beta)])
        with pytest.raises(_StopError):
            script.main()
        assert seen[0].db_path == resolve_config(docs_dir=beta).db_path
        assert seen[0].db_path != ServerConfig.from_env().db_path
        assert seen[0].exclude == ("vendor",)
