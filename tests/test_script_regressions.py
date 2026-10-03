"""Regressions in the developer scripts: the retrieval evaluation gate.

One test per defect found in code review; each fails when its fix is reverted.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import sys
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeEmbedder
from helpers import store

from markdown_memory import model_cache
from markdown_memory.config import ServerConfig, resolve_config
from markdown_memory.db import Database
from markdown_memory.models import (
    FileFailure,
    IndexReport,
    SearchPage,
    SearchResult,
    estimate_tokens,
)
from markdown_memory.server import MarkdownMemoryService, create_server


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
        self.searched: list[str] = []
        self.report = IndexReport(
            directory="stub", files_scanned=0, files_indexed=0, files_unchanged=0, files_purged=0,
            sections_indexed=0, passages_indexed=0, elapsed_seconds=0.0,
        )  # fmt: skip

    def index_directory(self, directory: str | None = None) -> IndexReport:
        return self.report

    def search_docs(self, query: str, limit: int) -> list[SearchResult]:
        self.searched.append(query)
        return []

    def search_page(self, query: str, limit: int = 5) -> SearchPage:
        self.searched.append(query)
        return SearchPage((), "no_match")

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
        # The stub holds no corpus to resolve labels in or to cost; both have their own tests.
        monkeypatch.setattr(evaluation, "resolve_answers", lambda *_args: {})
        monkeypatch.setattr(evaluation, "measure_costs", lambda *_args: {})
        monkeypatch.setattr(evaluation, "_payloads", lambda _service, qs: tuple(0 for _ in qs))
        try:
            code = evaluation.main()  # type: ignore[attr-defined]
        finally:
            logging.disable(logging.NOTSET)  # main() disables logging process-wide
        fixture = json.loads((evaluation.DATA / "queries.json").read_text())  # type: ignore[attr-defined]
        unanswerable = {
            c["query"] for split in ("dev", "held_out") for c in fixture[split]["no_answer"]
        }
        # The no-answer pass ran rather than failing quietly inside its informational guard.
        assert unanswerable and unanswerable <= set(stub.searched)
        recorded = json.loads(baseline.read_text())["embeddinggemma"]
        # ...and never reached the baseline: only the four answerable sets are recorded.
        assert set(recorded) == {
            "dev/paraphrase", "dev/identifier", "held_out/paraphrase", "held_out/identifier"
        }  # fmt: skip
        return code, recorded

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


_COST_CORPUS = {
    "a.md": "# Guide\n\nintro words\n\n## Setup\n\ninstall the tool first\n\n"
    "### Nested\n\nnested setup detail\n\n## Big\n\n"
    + "\n\n".join(f"capacity paragraph {n} " + "sizing words " * 30 for n in range(12))
    + "\n",
    "b.md": "# Guide\n\n## Setup\n\nanother setup entirely\n",
}


def _queries(*labels: str) -> dict[str, dict[str, list[dict[str, object]]]]:
    cases: list[dict[str, object]] = [{"query": label, "expected": label} for label in labels]
    return {
        "dev": {"paraphrase": cases, "identifier": [], "no_answer": []},
        "held_out": {"paraphrase": [], "identifier": [], "no_answer": []},
    }


def _passing_scores(evaluation: object) -> Any:
    return evaluation.Scores(  # type: ignore[attr-defined]
        top1=1.0, top3=1.0, top5=1.0, any_valid_top1=1.0, ndcg5=1.0,
        median_ms=1.0, p95_ms=1.0, misses=(),
    )  # fmt: skip


class TestTheCostReport:
    """#40: what the default call costs beside the section that answers - reported, never gated."""

    @pytest.fixture
    def evaluation(self) -> object:
        # By name, not by path: the mutation check points `pythonpath` at its mutated copy.
        import eval_retrieval

        return eval_retrieval

    @pytest.fixture
    def service(
        self, tmp_path: Path, fake_embedder: FakeEmbedder
    ) -> Iterator[MarkdownMemoryService]:
        root = tmp_path / "docs"
        root.mkdir()
        for name, text in _COST_CORPUS.items():
            (root / name).write_text(text, encoding="utf-8")
        service = MarkdownMemoryService(
            ServerConfig(db_path=tmp_path / "c.db", docs_dir=root), embedder=fake_embedder
        )
        service.index_directory()
        yield service
        service.close()

    @staticmethod
    def resolve(evaluation: object, service: MarkdownMemoryService, *labels: str) -> dict[str, Any]:
        try:
            return evaluation.resolve_answers(service, _queries(*labels))  # type: ignore[attr-defined,no-any-return]
        except SystemExit as exc:
            raise AssertionError(f"a sound label was refused: {exc}") from exc

    def test_the_payload_is_the_text_block_the_default_call_sends(
        self, evaluation: object, service: MarkdownMemoryService
    ) -> None:
        label = "Guide > Big"
        answers = self.resolve(evaluation, service, label)
        cost = evaluation.measure_costs(service, _queries(label), answers)["dev/paraphrase"]  # type: ignore[attr-defined]
        outcome = asyncio.run(
            create_server(service=service).call_tool("search_docs", {"query": label})
        )
        assert cost.payloads == (estimate_tokens(outcome.content[0].text),)  # type: ignore[union-attr]
        assert cost.ratios == [cost.payloads[0] / answers[label].tokens]

    def test_a_split_section_answers_with_its_parts_together(
        self, evaluation: object, service: MarkdownMemoryService
    ) -> None:
        answer = self.resolve(evaluation, service, "Guide > Big")["Guide > Big"]
        (big,) = [
            node
            for node in service.get_document_outline(answer.file_path)[0].children
            if node.heading_path == "Guide > Big"
        ]
        assert big.part_count > 1, "the fixture no longer splits, so this proves nothing"
        whole = service.read_section(answer.file_path, "Guide > Big")
        assert "(Part" not in answer.heading_path
        assert answer.tokens == estimate_tokens(whole)

    def test_a_qualified_label_resolves_in_the_file_it_names(
        self, evaluation: object, service: MarkdownMemoryService
    ) -> None:
        answer = self.resolve(evaluation, service, "b.md::Guide > Setup")["b.md::Guide > Setup"]
        assert Path(answer.file_path).name == "b.md"

    def test_a_nested_heading_resolves(
        self, evaluation: object, service: MarkdownMemoryService
    ) -> None:
        answer = self.resolve(evaluation, service, "Guide > Setup > Nested")[
            "Guide > Setup > Nested"
        ]
        assert Path(answer.file_path).name == "a.md"

    @pytest.mark.parametrize("label", ["Guide > Nowhere", "Guide > Setup", "c.md::Guide > Setup"])
    def test_a_label_naming_no_section_or_two_is_a_broken_fixture(
        self, evaluation: object, service: MarkdownMemoryService, label: str
    ) -> None:
        try:
            evaluation.resolve_answers(service, _queries(label))  # type: ignore[attr-defined]
        except SystemExit as exc:
            assert "fixture" in str(exc)
        else:
            raise AssertionError(f"{label!r} was taken for one section")

    @staticmethod
    def isolate(evaluation: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
        """main() on a stub service and a one-case fixture of its own: no checkout data read."""
        data = tmp_path / "eval_data"
        data.mkdir()
        (data / "queries.json").write_text(json.dumps(_queries("q")), encoding="utf-8")
        monkeypatch.setattr(evaluation, "DATA", data)
        monkeypatch.setattr(evaluation, "BASELINE", data / "baseline.json")
        monkeypatch.setattr(evaluation, "_probes", lambda corpus: ())
        monkeypatch.setattr(sys, "argv", ["eval_retrieval.py"])
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        stub = _EvalStubService(ServerConfig(db_path=tmp_path / "x.db", docs_dir=tmp_path))
        monkeypatch.setattr(evaluation, "open_service", lambda *_args: (stub, False))
        return stub

    def test_a_broken_fixture_stops_the_run_before_any_search(
        self, evaluation: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub = self.isolate(evaluation, tmp_path, monkeypatch)
        passing = _passing_scores(evaluation)
        monkeypatch.setattr(evaluation, "evaluate", lambda service, cases: passing)

        def broken(*_args: object) -> dict[str, object]:
            raise SystemExit("fixture: 'X' names 0 sections, not one")

        monkeypatch.setattr(evaluation, "resolve_answers", broken)
        try:
            evaluation.main()  # type: ignore[attr-defined]
        except SystemExit:
            pass
        else:
            raise AssertionError("a broken fixture was scored")
        finally:
            logging.disable(logging.NOTSET)
        assert stub.searched == [], "the fixture was checked only after searching"

    def test_a_cost_pass_that_fails_does_not_decide_the_exit_code(
        self,
        evaluation: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        passing = _passing_scores(evaluation)
        self.isolate(evaluation, tmp_path, monkeypatch)
        monkeypatch.setattr(evaluation, "evaluate", lambda service, cases: passing)
        monkeypatch.setattr(evaluation, "resolve_answers", lambda *_args: {})

        def failing(*_args: object) -> dict[str, object]:
            raise RuntimeError("search_docs failed for 'q'")

        monkeypatch.setattr(evaluation, "measure_costs", failing)
        try:
            code = evaluation.main()  # type: ignore[attr-defined]
        except RuntimeError as exc:
            raise AssertionError("an informational table took the run down") from exc
        finally:
            logging.disable(logging.NOTSET)
        assert code == 0
        assert "cost: not measured (search_docs failed for 'q')" in capsys.readouterr().out

    def test_show_costs_lists_every_query(
        self, evaluation: object, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cost = evaluation.Cost(queries=("q1", "q2"), payloads=(300, 90), answers=(30, 45))  # type: ignore[attr-defined]
        evaluation.print_costs({"dev/paraphrase": cost}, per_query=True)  # type: ignore[attr-defined]
        listed = [line for line in capsys.readouterr().out.splitlines() if "cost [" in line]
        assert len(listed) == 2 and listed[0].endswith("q1") and "10.0x" in listed[0]

    def test_a_recorded_baseline_holds_no_cost(
        self, evaluation: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.isolate(evaluation, tmp_path, monkeypatch)
        passing = _passing_scores(evaluation)
        monkeypatch.setattr(evaluation, "evaluate", lambda service, cases: passing)
        monkeypatch.setattr(evaluation, "resolve_answers", lambda *_args: {})
        cost = evaluation.Cost(queries=("q",), payloads=(300,), answers=(30,))  # type: ignore[attr-defined]
        monkeypatch.setattr(evaluation, "measure_costs", lambda *_args: {"dev/paraphrase": cost})
        monkeypatch.setattr(sys, "argv", ["eval_retrieval.py", "--update-baseline"])
        try:
            assert evaluation.main() == 0  # type: ignore[attr-defined]
        finally:
            logging.disable(logging.NOTSET)
        recorded = json.loads((tmp_path / "eval_data" / "baseline.json").read_text())
        fields = set(recorded["embeddinggemma"]["dev/paraphrase"])
        assert fields == {"top1", "top3", "top5", "any_valid_top1", "ndcg5", "median_ms", "p95_ms"}

    def test_p95_is_the_same_rule_for_latency_and_cost(self, evaluation: object) -> None:
        assert evaluation._p95([float(n) for n in range(1, 21)]) == 19.0  # type: ignore[attr-defined]


class TestTheRealDocsCorpus:
    """#75: corpus_v2 is scored by the same script, apart from the gate and its baseline."""

    @pytest.fixture
    def evaluation(self) -> object:
        import eval_retrieval

        return eval_retrieval

    @staticmethod
    def service(tmp_path: Path, fake_embedder: FakeEmbedder, root: Path) -> MarkdownMemoryService:
        service = MarkdownMemoryService(
            ServerConfig(db_path=tmp_path / "v.db", docs_dir=root), embedder=fake_embedder
        )
        service.index_directory()
        return service

    def test_a_label_names_a_file_by_its_whole_path_from_the_corpus_root(
        self, evaluation: object, tmp_path: Path, fake_embedder: FakeEmbedder
    ) -> None:
        root = tmp_path / "docs"
        for folder in ("x", "p/x"):
            (root / folder).mkdir(parents=True)
            (root / folder / "README.md").write_text(f"# Guide\n\n## Setup\n\n{folder} setup\n")
        service = self.service(tmp_path, fake_embedder, root)
        try:
            try:
                answers = evaluation.resolve_answers(
                    service, _queries("x/README.md::Guide > Setup")
                )  # type: ignore[attr-defined]
            except SystemExit as exc:
                raise AssertionError(f"a whole path was taken for two files: {exc}") from exc
            with pytest.raises(SystemExit, match="names 2 sections"):
                evaluation.resolve_answers(service, _queries("README.md::Guide > Setup"))  # type: ignore[attr-defined]
        finally:
            service.close()
        assert answers["x/README.md::Guide > Setup"].file_path == str(root / "x" / "README.md")

    def test_a_hit_answers_to_its_file_name_and_its_whole_path_only(
        self, evaluation: object, tmp_path: Path
    ) -> None:
        hit = SearchResult(
            section_id=1,
            file_path=str(tmp_path / "gh" / "docs" / "README.md"),
            document_title="Guide",
            heading_title="Setup",
            heading_path="Guide > Setup (Part 2)",
            content="",
            start_line=1,
            end_line=2,
            score=1.0,
            fts_rank=1,
            vec_rank=None,
        )
        assert evaluation._labels(hit, str(tmp_path)) == {  # type: ignore[attr-defined]
            "Guide > Setup",
            "README.md::Guide > Setup",
            "gh/docs/README.md::Guide > Setup",
        }

    def test_the_churn_report_ranks_by_the_same_labels_as_the_gate(
        self, tmp_path: Path, fake_embedder: FakeEmbedder
    ) -> None:
        import cross_host_churn

        root = tmp_path / "docs"
        (root / "guides").mkdir(parents=True)
        (root / "guides" / "a.md").write_text("# Guide\n\n## Setup\n\ninstall the tool\n")
        service = self.service(tmp_path, fake_embedder, root)
        try:
            ranked = cross_host_churn._rank(service, "install the tool")
        finally:
            service.close()
        assert "guides/a.md::Guide > Setup" in ranked[0]

    def test_every_v2_label_names_exactly_one_section(
        self, evaluation: object, tmp_path: Path, fake_embedder: FakeEmbedder
    ) -> None:
        corpus = evaluation.corpora()["v2"]  # type: ignore[attr-defined]
        queries = json.loads(corpus.queries.read_text(encoding="utf-8"))
        for split in ("dev", "held_out"):
            assert [len(queries[split][kind]) for kind in ("paraphrase", "identifier")] == [20, 20]
        graded = [
            label
            for split in ("dev", "held_out")
            for kind in ("paraphrase", "identifier")
            for case in queries[split][kind]
            for label in case.get("also_valid", {})
        ]
        service = self.service(tmp_path, fake_embedder, corpus.root)
        try:
            evaluation.resolve_answers(service, queries)  # type: ignore[attr-defined]
            evaluation.resolve_answers(service, _queries(*graded))  # type: ignore[attr-defined]
        finally:
            service.close()

    def test_v2_keeps_its_own_index_and_baseline(self, evaluation: object) -> None:
        import eval_cache

        v1, v2 = (evaluation.corpora()[name] for name in ("v1", "v2"))  # type: ignore[attr-defined]
        assert v1.cache() == eval_cache.cache_root()
        assert v2.cache() != v1.cache() and v2.cache().parent == v1.cache().parent
        assert (v1.baseline_key("embeddinggemma"), v2.baseline_key("embeddinggemma")) == (
            "embeddinggemma",
            "embeddinggemma@v2",
        )


class TestTheNoAnswerStratum:
    """#41: queries the corpus cannot answer, scored by abstention - reported, never gated."""

    @pytest.fixture
    def evaluation(self) -> object:
        # By name, not by path: the mutation check points `pythonpath` at its mutated copy.
        import eval_retrieval

        return eval_retrieval

    @staticmethod
    def with_cases(*cases: dict[str, object]) -> dict[str, dict[str, list[dict[str, object]]]]:
        queries = _queries()
        queries["dev"]["no_answer"] = list(cases)
        return queries

    def test_resolving_labels_skips_cases_that_have_none(
        self, evaluation: object, tmp_path: Path, fake_embedder: FakeEmbedder
    ) -> None:
        root = tmp_path / "docs"
        root.mkdir()
        (root / "a.md").write_text("# Guide\n\n## Setup\n\ninstall it\n", encoding="utf-8")
        service = MarkdownMemoryService(
            ServerConfig(db_path=tmp_path / "r.db", docs_dir=root), embedder=fake_embedder
        )
        try:
            service.index_directory()
            queries = self.with_cases({"query": "ECONNRESET", "shape": "identifier"})
            queries["dev"]["paraphrase"] = [{"query": "setup", "expected": "Guide > Setup"}]
            answers = evaluation.resolve_answers(service, queries)  # type: ignore[attr-defined]
        finally:
            service.close()
        assert set(answers) == {"Guide > Setup"}

    def test_the_default_call_is_measured_per_shape(
        self, evaluation: object, tmp_path: Path, fake_embedder: FakeEmbedder
    ) -> None:
        root = tmp_path / "docs"
        root.mkdir()
        (root / "a.md").write_text("# Guide\n\n## Setup\n\ninstall it\n", encoding="utf-8")
        service = MarkdownMemoryService(
            ServerConfig(db_path=tmp_path / "m.db", docs_dir=root), embedder=fake_embedder
        )
        try:
            service.index_directory()
            measured = evaluation.measure_no_answer(  # type: ignore[attr-defined]
                service,
                self.with_cases(
                    {"query": "ECONNRESET", "shape": "identifier"},
                    {"query": "install", "shape": "question"},
                ),
            )
        finally:
            service.close()
        identifier, question = measured["dev/identifier"], measured["dev/question"]
        assert identifier.queries == ("ECONNRESET",) and identifier.no_match == (True,)
        assert identifier.hits == (0,), "#38: an identifier nothing contains abstains"
        assert question.queries == ("install",) and question.no_match == (False,)
        assert measured["held_out/identifier"].queries == ()
        # Costed like any call: an abstention still pays for its keyword_message.
        assert identifier.payloads[0] > 0 and len(question.payloads) == 1

    def test_the_report_counts_an_empty_page_as_abstaining(
        self, evaluation: object, capsys: pytest.CaptureFixture[str]
    ) -> None:
        no_answer = evaluation.NoAnswer  # type: ignore[attr-defined]
        empty = no_answer(queries=(), hits=(), no_match=(), payloads=())
        measured = {
            "dev/identifier": no_answer(
                queries=("A", "B"), hits=(0, 3), no_match=(True, True), payloads=(40, 90)
            ),
            "dev/question": no_answer(
                queries=("c d",), hits=(5,), no_match=(False,), payloads=(120,)
            ),
            "held_out/identifier": empty,
            "held_out/question": empty,
        }
        evaluation.print_no_answer(measured, show_misses=True)  # type: ignore[attr-defined]
        out = capsys.readouterr().out
        assert "dev identifier           2        50%      100%       65" in out
        assert "dev all                  3        33%       67%       90" in out
        assert "held_out" not in out.split("miss")[0], "a set with no cases prints no rate"
        assert "miss [dev/no_answer identifier] 3 hits: B" in out
        assert "hits: A" not in out, "an abstention is not a miss"

    def test_the_fixture_guard_finds_a_query_the_corpus_contains(
        self, evaluation: object, tmp_path: Path
    ) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.md").write_text("Set `enable_tls_v2` and enable_tls.\n", encoding="utf-8")
        check = evaluation.check_no_answer  # type: ignore[attr-defined]
        with pytest.raises(SystemExit, match="is in the corpus"):
            check(self.with_cases({"query": "--enable-tls", "shape": "identifier"}), corpus)
        # A longer identifier that contains it is not the same token run.
        check(self.with_cases({"query": "--enable-tls-v", "shape": "identifier"}), corpus)

    def test_the_fixture_guard_rejects_a_malformed_case(
        self, evaluation: object, tmp_path: Path
    ) -> None:
        check = evaluation.check_no_answer  # type: ignore[attr-defined]
        for case in (
            {"query": "X", "shape": "identifier", "expected": "Some > Section"},
            {"query": "X", "shape": "paraphrase"},
            {"query": "--", "shape": "identifier"},
        ):
            with pytest.raises(SystemExit, match="malformed"):
                check(self.with_cases(case), tmp_path)

    def test_the_fixture_guard_reads_no_corpus_without_cases(
        self, evaluation: object, tmp_path: Path
    ) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "gone.md").symlink_to(tmp_path / "nowhere.md")  # unreadable if read at all
        evaluation.check_no_answer(_queries(), corpus)  # type: ignore[attr-defined]

    def test_a_no_answer_query_the_corpus_contains_stops_the_run_before_any_search(
        self, evaluation: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub = TestTheCostReport.isolate(evaluation, tmp_path, monkeypatch)
        queries = self.with_cases({"query": "ECONNRESET", "shape": "identifier"})
        (evaluation.DATA / "queries.json").write_text(json.dumps(queries))  # type: ignore[attr-defined]
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.md").write_text("# Errors\n\nECONNRESET means the peer hung up.\n")
        monkeypatch.setattr(evaluation, "CORPUS", corpus)
        monkeypatch.setattr(evaluation, "resolve_answers", lambda *_args: {})
        passing = _passing_scores(evaluation)
        monkeypatch.setattr(evaluation, "evaluate", lambda service, cases: passing)
        try:
            with pytest.raises(SystemExit, match="is in the corpus"):
                evaluation.main()  # type: ignore[attr-defined]
        finally:
            logging.disable(logging.NOTSET)
        assert stub.searched == [], "the fixture was checked only after searching"

    def test_a_no_answer_pass_that_fails_does_not_decide_the_exit_code(
        self,
        evaluation: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        TestTheCostReport.isolate(evaluation, tmp_path, monkeypatch)
        passing = _passing_scores(evaluation)
        monkeypatch.setattr(evaluation, "evaluate", lambda service, cases: passing)
        monkeypatch.setattr(evaluation, "resolve_answers", lambda *_args: {})

        def broken(*_args: object) -> dict[str, object]:
            raise RuntimeError("no-answer pass exploded")

        monkeypatch.setattr(evaluation, "measure_no_answer", broken)
        try:
            code = evaluation.main()  # type: ignore[attr-defined]
        finally:
            logging.disable(logging.NOTSET)
        assert code == 0
        assert "no-answer stratum: not measured" in capsys.readouterr().out


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
        # The graph this version actually loads, not a name frozen into the test: a file
        # nothing reads would prove the key digests *something*, not the weights.
        weights = eval_cache.gemma_model_dir(cache_dir) / model_cache.GEMMA_MODEL_FILE
        weights.parent.mkdir(parents=True)
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


class TestLiveTestScript:
    """`live_test.py` is a gate: a check in it that cannot fail is a gate that is not there."""

    @pytest.fixture
    def live(self) -> Any:
        # By name, not by path: the mutation check points `pythonpath` at its mutated copy.
        import live_test

        return live_test

    @staticmethod
    def client(status: dict[str, Any]) -> Any:
        from mcp.types import CallToolResult, TextContent

        class Client:
            async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
                text = (
                    "Indexed: 6 scanned, 6 (re)indexed"
                    if name == "index_directory"
                    else json.dumps({"documents": [], "index_status": status})
                )
                return CallToolResult(content=[TextContent(type="text", text=text)])

        return Client()

    async def test_a_root_that_does_not_vouch_for_itself_fails_the_run(
        self, live: Any, tmp_path: Path
    ) -> None:
        clean = {
            "root": str(tmp_path.resolve()),
            "coverage": "verified",
            "failures": [],
            "changed_files": 0,
            "indexing": False,
            "gitignore": "no_repository",
            "message": None,
        }
        vouch = "a freshly indexed root vouches for itself"
        for wrong in ({"coverage": "unknown"}, {"changed_files": 2}, {"gitignore": "unavailable"}):
            run = live.LiveTest(self.client(clean | wrong), tmp_path, {})
            with pytest.raises(live.CheckFailedError, match=vouch):
                await run.indexing()
        # A clean status passes that check; the run then stops at the next one, documents.
        run = live.LiveTest(self.client(clean), tmp_path, {})
        with pytest.raises(live.CheckFailedError, match="reports 6 documents"):
            await run.indexing()
