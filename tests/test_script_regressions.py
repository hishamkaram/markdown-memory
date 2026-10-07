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
            median_ms=1.0, p95_ms=1.0,
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
        median_ms=1.0, p95_ms=1.0,
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
            kinds = ("paraphrase", "identifier", "mixed")
            assert [len(queries[split][kind]) for kind in kinds] == [20, 20, 20]
        graded = [
            label
            for split in ("dev", "held_out")
            for kind in ("paraphrase", "identifier", "mixed")
            for case in queries[split][kind]
            for label in case.get("also_valid", {})
        ]
        service = self.service(tmp_path, fake_embedder, corpus.root)
        try:
            evaluation.resolve_answers(service, queries)  # type: ignore[attr-defined]
            evaluation.resolve_answers(service, _queries(*graded))  # type: ignore[attr-defined]
        finally:
            service.close()

    def test_no_v2_held_out_answer_is_a_dev_answer(self, evaluation: object) -> None:
        """#100: a held-out question answered by a section dev already tunes on measures dev."""
        corpus = evaluation.corpora()["v2"]  # type: ignore[attr-defined]
        queries = json.loads(corpus.queries.read_text(encoding="utf-8"))
        tuned = {
            label
            for cases in queries["dev"].values()
            for case in cases
            if "expected" in case
            for label in (case["expected"], *case.get("also_valid", {}))
        }
        shared = [
            case["expected"]
            for kind in ("paraphrase", "identifier")
            for case in queries["held_out"][kind]
            if case["expected"] in tuned
        ]
        assert not shared, f"{len(shared)} held-out answer(s) are dev answers too"

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


class TestTheExcerptHarness:
    """`scripts/eval_excerpts.py` (#76): what the judges see, and how their verdicts score."""

    @staticmethod
    def page(top: dict[str, Any], *pointers: dict[str, Any]) -> str:
        return json.dumps(
            {"results": [top, *pointers], "keyword_match": "matched", "index_status": {}},
            separators=(",", ":"),
        )

    TOP = {"file_path": "a.md", "heading_path": "A > B", "lines": "3-5", "tokens": 90}

    def test_an_excerpt_and_its_section_are_shown_in_a_seeded_random_order(self) -> None:
        import random

        import eval_excerpts

        text = self.page(dict(self.TOP, excerpt=True, content="the passage"))
        whole = self.page(dict(self.TOP, content="# B\n\nthe section"))
        orders = set()
        for seed in range(20):
            item = eval_excerpts.make_item(
                f"v2-dev-paraphrase-{seed:02d}",
                "q",
                text,
                whole,
                "# B\n\nthe section",
                random.Random(seed),
            )
            assert item is not None
            excerpt, section = item.key["excerpt"], item.key["section"]
            assert {excerpt, section} == {"A", "B"}
            assert item.judge["texts"][excerpt] == "the passage"
            assert item.judge["texts"][section] == "# B\n\nthe section"
            assert "excerpt" not in json.dumps(item.judge)  # nothing tells the judge which
            orders.add(excerpt)
        assert orders == {"A", "B"}
        again = eval_excerpts.make_item(
            "x", "q", text, whole, "# B\n\nthe section", random.Random(3)
        )
        first = eval_excerpts.make_item(
            "x", "q", text, whole, "# B\n\nthe section", random.Random(3)
        )
        assert again is not None and first is not None and again.key == first.key

    def test_a_whole_section_is_shown_once_and_its_baseline_is_its_own_payload(self) -> None:
        import random

        import eval_excerpts

        text = self.page(dict(self.TOP, content="# B\n\nthe section"))
        item = eval_excerpts.make_item(
            "v1-dev-identifier-00", "q", text, text, "# B\n\nthe section", random.Random(0)
        )
        assert item is not None
        assert item.judge["texts"] == {"A": "# B\n\nthe section"}
        assert item.key["excerpt"] is None and item.key["section"] == "A"
        assert item.key["payload_tokens"] == item.key["section_payload_tokens"]

    def test_the_baseline_is_the_whole_section_response_for_the_same_top_hit(self) -> None:
        """Priced from what the server sends with the excerpt off, `lines` and all."""
        import random

        import eval_excerpts

        pointer = {"file_path": "b.md", "heading_path": "C", "lines": "1-2", "tokens": 7}
        section = "# B\n\n" + "a long section " * 40
        text = self.page(dict(self.TOP, excerpt=True, content="the passage"), pointer)
        whole = self.page(dict(self.TOP, lines="1-40", content=section), pointer)
        item = eval_excerpts.make_item(
            "v2-dev-paraphrase-00", "q", text, whole, section, random.Random(0)
        )
        assert item is not None
        assert item.key["section_payload_tokens"] == estimate_tokens(whole)
        assert item.key["payload_tokens"] == estimate_tokens(text)
        assert item.key["read_tokens"] == estimate_tokens(section)
        elsewhere = self.page(dict(self.TOP, heading_path="A > C", content=section), pointer)
        with pytest.raises(RuntimeError, match="another top hit"):
            eval_excerpts.make_item("x", "q", text, elsewhere, section, random.Random(0))

    def test_a_key_names_the_section_its_query_was_answered_by(self) -> None:
        import random

        import eval_excerpts

        text = self.page(dict(self.TOP, content="s"))
        item = eval_excerpts.make_item(
            "v2-dev-paraphrase-00", "q", text, text, "s", random.Random(0)
        )
        assert item is not None and item.key["cluster"] == "a.md :: A > B"

    @staticmethod
    def key(item_id: str, excerpt: str | None, cluster: str = "", **tokens: int) -> dict[str, Any]:
        section = "A" if excerpt is None else ("B" if excerpt == "A" else "A")
        return {
            "id": item_id,
            "cluster": cluster or item_id,
            "excerpt": excerpt,
            "section": section,
            "payload_tokens": tokens.get("payload", 100),
            "section_payload_tokens": tokens.get("baseline", 200),
            "read_tokens": tokens.get("read", 150),
        }

    def test_a_judge_sees_one_text_the_section_or_the_excerpt(self) -> None:
        import eval_excerpts

        keys = [
            self.key("v2-dev-paraphrase-00", "A", "CHANGELOG.md :: 15.2.0"),
            self.key("v2-dev-paraphrase-01", None, "a.md :: A > B"),
        ]
        judged = [
            {"id": "v2-dev-paraphrase-00", "query": "q0", "texts": {"A": "ex", "B": "whole"}},
            {"id": "v2-dev-paraphrase-01", "query": "q1", "texts": {"A": "whole 1"}},
        ]
        release = {"file_path": "CHANGELOG.md", "heading_path": "15.2.0"}
        assert eval_excerpts.judge_texts(judged, keys, "section") == [
            {"id": "v2-dev-paraphrase-00", "query": "q0", "hit": release, "texts": {"A": "whole"}},
            {
                "id": "v2-dev-paraphrase-01",
                "query": "q1",
                "hit": {"file_path": "a.md", "heading_path": "A > B"},
                "texts": {"A": "whole 1"},
            },
        ]
        # The release an excerpt answers is named by its heading path, as the agent sees it.
        assert eval_excerpts.judge_texts(judged, keys, "excerpt") == [
            {"id": "v2-dev-paraphrase-00", "query": "q0", "hit": release, "texts": {"A": "ex"}},
        ]

    def test_a_verdict_is_a_strict_majority_and_a_split_is_not_a_yes(self) -> None:
        import eval_excerpts

        def judges(*verdicts: str) -> list[dict[str, str]]:
            return [{"x": verdict} for verdict in verdicts]

        assert eval_excerpts.majority_yes("x", judges("yes", "yes", "partial"))
        assert not eval_excerpts.majority_yes("x", judges("yes", "partial", "no"))
        assert not eval_excerpts.majority_yes("x", judges("yes", "partial"))
        assert eval_excerpts.majority_yes("x", judges("yes", "yes"))

    def test_a_lost_excerpt_costs_the_read_and_a_whole_section_is_kept_as_sent(self) -> None:
        import eval_excerpts

        lost = self.key("v2-held_out-paraphrase-00", "A")
        judges = [{lost["id"]: "partial"}, {lost["id"]: "yes"}, {lost["id"]: "no"}]
        assert eval_excerpts.outcome(lost, judges) == eval_excerpts.Outcome(
            False, 250, 200, True, lost["id"]
        )
        whole = self.key("v2-held_out-paraphrase-01", None, payload=200)
        assert eval_excerpts.outcome(whole, []) == eval_excerpts.Outcome(
            True, 200, 200, False, whole["id"]
        )

    def test_the_wilson_bound_is_one_sided_at_95_percent(self) -> None:
        import eval_excerpts

        assert eval_excerpts.wilson_lower(20, 20) == pytest.approx(0.8808, abs=1e-4)
        assert eval_excerpts.wilson_lower(114, 120) == pytest.approx(0.9062, abs=1e-4)
        assert eval_excerpts.wilson_lower(0, 0) == 0.0

    def test_the_cost_bound_resamples_sections_not_queries(self) -> None:
        import eval_excerpts

        def outcomes(clusters: list[str]) -> list[Any]:
            costs = [100, 300] * (len(clusters) // 2)
            return [
                eval_excerpts.Outcome(True, cost, 200, True, cluster)
                for cost, cluster in zip(costs, clusters, strict=True)
            ]

        # Eight queries of one section are one observation: nothing to resample.
        assert eval_excerpts.cost_upper(outcomes(["s"] * 8), "x") == 1.0
        spread = eval_excerpts.cost_upper(outcomes([f"s{n}" for n in range(8)]), "x")
        assert 1.0 < spread <= 1.5
        assert spread == eval_excerpts.cost_upper(outcomes([f"s{n}" for n in range(8)]), "x")

    def test_the_gates_count_only_frozen_items_and_hold_only_on_v2(self) -> None:
        import eval_excerpts

        def keys(corpus: str, total: int, payload: int) -> list[dict[str, Any]]:
            return [
                self.key(f"{corpus}-sealed-paraphrase-{n:03d}", "A", payload=payload)
                for n in range(total)
            ]

        def run(kept: int, total: int, payload: int, extra: int = 0, v1: int = 0) -> bool:
            items = keys("v2", total + extra, payload) + keys("v1", v1, 200)
            ids = [k["id"] for k in items]
            verdict = {item_id: "yes" if n < kept else "no" for n, item_id in enumerate(ids)}
            eligible = ids[:total] + ids[total + extra :]
            return eval_excerpts.score(items, eligible, [verdict] * 3)[1]

        assert run(124, 124, 100)
        assert run(118, 124, 100)  # 95.2%, Wilson bound 90.9%
        assert not run(117, 124, 100)  # 94.4%
        assert not run(20, 20, 100)  # 100%, but a bound of 88.1% on twenty items
        assert run(124, 124, 180)  # 90% of the baseline: a material saving, target missed
        assert not run(124, 124, 182)  # 91%
        assert run(124, 124, 100, extra=30)  # items outside the frozen list do not count
        assert run(124, 124, 100, v1=20)  # v1 is reported: its lost excerpts gate nothing
        assert not eval_excerpts.score(keys("v1", 20, 100), [], [{}] * 3)[1]

    def test_the_cost_target_is_reported_but_does_not_gate(self) -> None:
        import eval_excerpts

        items = [self.key(f"v2-sealed-paraphrase-{n:03d}", "A", payload=170) for n in range(124)]
        verdict = {k["id"]: "yes" for k in items}
        lines, passed = eval_excerpts.score(items, [k["id"] for k in items], [verdict] * 3)
        assert passed and any("target 80% / 85%: missed" in line for line in lines)

    def test_a_malformed_missing_or_repeated_verdict_stops_the_score(self, tmp_path: Path) -> None:
        import eval_excerpts

        ids = ["v2-dev-paraphrase-00"]

        def read(answer: object) -> dict[str, str]:
            path = tmp_path / "judge.json"
            path.write_text(answer if isinstance(answer, str) else json.dumps(answer))
            return eval_excerpts.read_verdicts(path, ids)

        with pytest.raises(SystemExit, match="malformed"):
            read([{"id": ids[0], "A": "maybe"}])
        with pytest.raises(SystemExit, match="malformed"):
            read([{"id": ids[0], "A": "yes", "B": "no"}])
        with pytest.raises(SystemExit, match="malformed"):
            read([{"id": ids[0], "A": "yes"}, {"id": ids[0], "A": "no"}])
        with pytest.raises(SystemExit, match="no verdict"):
            read([])
        with pytest.raises(SystemExit, match="not given"):
            read([{"id": ids[0], "A": "yes"}, {"id": "another", "A": "yes"}])
        answer = [{"id": ids[0], "A": "yes", "note": "x"}]
        assert read("```json\n" + json.dumps(answer) + "\n```") == {ids[0]: "yes"}

    def test_a_build_that_drifted_from_the_frozen_denominator_is_not_scored(self) -> None:
        import eval_excerpts

        keys = [self.key("v2-sealed-paraphrase-00", "A", "a.md :: A")]
        frozen = {"eligible": [keys[0]["id"]], "clusters": {keys[0]["id"]: "a.md :: A"}}
        assert eval_excerpts.drifted(keys, frozen) == ""
        assert "missing" in eval_excerpts.drifted([], frozen)
        assert "repeated" in eval_excerpts.drifted(keys * 2, frozen)
        moved = [self.key("v2-sealed-paraphrase-00", "A", "a.md :: B")]
        assert "another section" in eval_excerpts.drifted(moved, frozen)

    def test_the_prompt_asks_for_the_letter_the_score_reads(self) -> None:
        import eval_excerpts

        assert '{"id": <id>, "A": "yes|partial|no"}' in eval_excerpts.JUDGE_PROMPT


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


def _paired_fixture(
    held_out_label: str = "a.md::Guide > Big", held_out_absent: str = "zzheld absent"
) -> dict[str, object]:
    """Two splits whose held-out queries all carry `zzheld`, so any leak can be searched for."""
    return {
        "dev": {
            "paraphrase": [{"query": "install the tool first", "expected": "a.md::Guide > Setup"}],
            "identifier": [
                {"query": "nested setup detail", "expected": "a.md::Guide > Setup > Nested"}
            ],
            "no_answer": [{"query": "zebra quantum", "shape": "question"}],
        },
        "held_out": {
            "paraphrase": [{"query": "zzheld capacity sizing", "expected": held_out_label}],
            "identifier": [{"query": "zzheld sizing words", "expected": held_out_label}],
            "no_answer": [{"query": held_out_absent, "shape": "identifier"}],
        },
    }


class _RecordingService(MarkdownMemoryService):
    """The real service, noting every query that reached the index - by any path."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.searched: list[str] = []

    def search_docs(self, query: str, limit: int = 5) -> list[SearchResult]:
        self.searched.append(query)
        return super().search_docs(query, limit)

    def search_page(self, query: str, limit: int = 5) -> SearchPage:
        self.searched.append(query)
        return super().search_page(query, limit)


class TestTheDevOnlyRun:
    """#94: tuning reads dev; `--split dev` must not search, resolve, record or show held-out."""

    @pytest.fixture
    def evaluation(self) -> object:
        import eval_retrieval  # by name: mutation_check points `pythonpath` at its copy

        return eval_retrieval

    @pytest.fixture
    def run_main(
        self,
        evaluation: Any,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fake_embedder: FakeEmbedder,
    ) -> Iterator[Any]:
        root = tmp_path / "docs"
        root.mkdir()
        for name, text in _COST_CORPUS.items():
            (root / name).write_text(text, encoding="utf-8")
        data = tmp_path / "eval_data"
        data.mkdir()
        service = _RecordingService(
            ServerConfig(db_path=tmp_path / "e.db", docs_dir=root), embedder=fake_embedder
        )
        service.index_directory()
        monkeypatch.setattr(evaluation, "DATA", data)
        monkeypatch.setattr(evaluation, "CORPUS", root)
        monkeypatch.setattr(evaluation, "BASELINE", data / "baseline.json")
        monkeypatch.setattr(evaluation, "_probes", lambda corpus: ())
        monkeypatch.setattr(evaluation, "open_service", lambda *_args: (service, False))
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

        def run(*argv: str, fixture: dict[str, object] | None = None) -> tuple[int, Any]:
            (data / "queries.json").write_text(json.dumps(fixture or _paired_fixture()))
            monkeypatch.setattr(sys, "argv", ["eval_retrieval.py", *argv])
            try:
                return evaluation.main(), service
            finally:
                logging.disable(logging.NOTSET)

        yield run
        service.close()

    def test_a_dev_run_never_reaches_held_out(
        self, run_main: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        record = tmp_path / "dev.json"
        # A held-out label naming no section, or a held-out "unanswerable" query the corpus
        # answers, stops a full run; a dev run must not even look.
        broken = _paired_fixture(held_out_label="a.md::Nowhere", held_out_absent="sizing words")
        try:
            code, service = run_main("--split", "dev", "--record", str(record), fixture=broken)
        except SystemExit as stopped:
            raise AssertionError(f"a dev run validated a held-out case: {stopped}") from stopped
        out = capsys.readouterr().out
        assert code == 0
        assert "GATES NOT CHECKED" in out and "--split dev" in out
        assert "held_out" not in out and "zzheld" not in out
        assert service.searched and not [q for q in service.searched if "zzheld" in q]
        # The scoring pass, the page pass and the cost pass's MCP calls all reached dev.
        assert service.searched.count("install the tool first") >= 3
        written = json.loads(record.read_text())
        assert written["splits"] == ["dev"]
        assert all(key.startswith("dev|") for key in written["cases"])

    @pytest.mark.parametrize(
        ("spoilt", "message"),
        [({"held_out_label": "a.md::Nowhere"}, "Nowhere"),
         ({"held_out_absent": "sizing words"}, "sizing words")],
    )  # fmt: skip
    def test_a_full_run_still_validates_every_case(
        self, run_main: Any, spoilt: dict[str, str], message: str
    ) -> None:
        with pytest.raises(SystemExit, match=message):
            run_main(fixture=_paired_fixture(**spoilt))

    @pytest.mark.parametrize("override", [["--split", "dev"], ["--queries", "other.json"]])
    def test_a_report_only_run_cannot_write_the_baseline(
        self, run_main: Any, override: list[str]
    ) -> None:
        with pytest.raises(SystemExit) as stopped:
            run_main(*override, "--update-baseline")
        assert stopped.value.code == 2

    def test_another_label_file_checks_no_gate(
        self, run_main: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        other = tmp_path / "other.json"
        other.write_text(json.dumps(_paired_fixture()))
        code, _ = run_main("--queries", str(other))
        assert code == 0
        assert "GATES NOT CHECKED: report-only run (--queries" in capsys.readouterr().out

    def test_a_record_holds_every_case_and_the_identity_of_what_ran(
        self, run_main: Any, tmp_path: Path
    ) -> None:
        record = tmp_path / "all.json"
        code, _ = run_main("--record", str(record))
        written = json.loads(record.read_text())
        assert code in (0, 1)  # the fake embedder may miss a floor; the record stands either way
        assert written["schema"] == 1 and written["splits"] == ["dev", "held_out"]
        for field in ("queries_sha256", "corpus_sha256", "parse_fingerprint", "evaluator"):
            assert isinstance(written[field], str) and len(written[field]) == 64
        assert len(written["code"]["sha256"]) == 64
        assert set(written["counts"].values()) == {1}
        assert len(written["cases"]) == 6
        case = written["cases"]["dev|paraphrase|install the tool first"]
        assert case["expected"] == "a.md::Guide > Setup" and case["also_valid"] == {}
        assert {"rank", "any_valid", "ndcg5", "top", "hits", "keyword_match"} <= set(case)
        assert set(written["cases"]["held_out|no_answer|zzheld absent"]) == {
            "shape", "hits", "keyword_match"
        }  # fmt: skip

    def test_a_mixed_stratum_is_scored_and_recorded(
        self, run_main: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """#78: a question naming an identifier is its own stratum, compared like the others."""
        fixture: Any = _paired_fixture()
        fixture["dev"]["mixed"] = [
            {"query": "how do I run setup first", "expected": "a.md::Guide > Setup"}
        ]
        fixture["held_out"]["mixed"] = [{"query": "zzheld sizing", "expected": "a.md::Guide > Big"}]
        record = tmp_path / "dev.json"
        code, _ = run_main("--split", "dev", "--record", str(record), fixture=fixture)
        written = json.loads(record.read_text())
        assert code == 0 and "dev mixed" in capsys.readouterr().out
        assert written["counts"]["dev|mixed"] == 1
        assert written["cases"]["dev|mixed|how do I run setup first"]["expected"] == (
            "a.md::Guide > Setup"
        )

    def test_a_fixture_without_a_mixed_stratum_is_scored_as_before(
        self, run_main: Any, tmp_path: Path
    ) -> None:
        record = tmp_path / "dev.json"
        try:
            run_main("--split", "dev", "--record", str(record))
        except (KeyError, SystemExit) as crash:
            raise AssertionError(
                f"v1's queries, without a mixed stratum, broke: {crash!r}"
            ) from crash
        assert sorted(json.loads(record.read_text())["counts"]) == [
            "dev|identifier", "dev|no_answer", "dev|paraphrase"
        ]  # fmt: skip

    def test_a_run_that_fails_the_floors_is_still_recorded(
        self, run_main: Any, evaluation: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = evaluation.evaluate

        def missing(service: Any, cases: list[dict[str, object]]) -> Any:
            import dataclasses

            return dataclasses.replace(real(service, cases), top1=0.0, top3=0.0, top5=0.0)

        monkeypatch.setattr(evaluation, "evaluate", missing)
        record = tmp_path / "regressed.json"
        code, _ = run_main("--record", str(record))
        assert code == 1
        assert json.loads(record.read_text())["cases"]

    def test_a_failed_record_leaves_the_last_one_alone(
        self, run_main: Any, evaluation: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        record = tmp_path / "dev.json"
        record.write_text("previous")

        def failing(*_args: object) -> dict[str, object]:
            raise RuntimeError("search_page failed")

        monkeypatch.setattr(evaluation, "record_cases", failing)
        with pytest.raises(RuntimeError):
            run_main("--split", "dev", "--record", str(record))
        assert record.read_text() == "previous"
        assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []

    def test_a_case_named_twice_is_not_recorded(self, run_main: Any, tmp_path: Path) -> None:
        fixture = _paired_fixture()
        dev = fixture["dev"]
        assert isinstance(dev, dict)
        dev["paraphrase"] = dev["paraphrase"] * 2
        with pytest.raises(ValueError, match="share the key"):
            run_main("--split", "dev", "--record", str(tmp_path / "x.json"), fixture=fixture)
        assert not (tmp_path / "x.json").exists()

    def test_a_non_finite_score_is_not_recorded(self, evaluation: Any) -> None:
        outcome = evaluation.Outcome(query="q", rank=1, any_valid=True, ndcg5=float("nan"), top=())
        scores = evaluation.Scores(
            top1=1.0, top3=1.0, top5=1.0, any_valid_top1=1.0, ndcg5=1.0,
            median_ms=1.0, p95_ms=1.0, cases=(outcome,),
        )  # fmt: skip
        queries = {"dev": {"paraphrase": [{"query": "q", "expected": "X"}], "identifier": [],
                           "no_answer": []}}  # fmt: skip
        stub = _EvalStubService(ServerConfig(db_path=Path("unused.db"), docs_dir=Path(".")))
        try:
            evaluation.record_cases(
                stub, queries, ("dev",), {"dev/paraphrase": scores, "dev/identifier": scores}
            )
        except ValueError as refused:
            assert "nan" in str(refused)
        else:
            raise AssertionError("a NaN nDCG@5 was recorded")


def _case(rank: int | None = 1, *, any_valid: bool = True, ndcg5: float = 1.0, hits: int = 5,
          keyword: str = "matched", expected: str = "a.md::A") -> dict[str, object]:  # fmt: skip
    return {
        "expected": expected, "also_valid": {}, "rank": rank, "any_valid": any_valid,
        "ndcg5": ndcg5, "top": [], "hits": hits, "keyword_match": keyword,
    }  # fmt: skip


def _record(cases: dict[str, dict[str, object]] | None = None, **header: object) -> dict[str, Any]:
    cases = cases if cases is not None else {
        "dev|paraphrase|p": _case(), "dev|identifier|i": _case(),
        "dev|no_answer|n": {"shape": "question", "hits": 0, "keyword_match": "no_match"},
    }  # fmt: skip
    counts: dict[str, int] = {
        f"dev|{kind}": 0 for kind in ("paraphrase", "identifier", "no_answer")
    }
    for key in cases:
        stratum = "|".join(key.split("|", 2)[:2])
        counts[stratum] = counts.get(stratum, 0) + 1
    return {
        "schema": 1, "corpus": "v1", "preset": "embeddinggemma", "splits": ["dev"],
        "queries": "q.json", "queries_sha256": "q" * 64, "corpus_sha256": "c" * 64,
        "parse_fingerprint": "f" * 64, "evaluator": "e" * 64,
        "code": {"package": "src", "revision": "r", "sha256": "a" * 64},
        "counts": counts, "cases": cases, **header,
    }  # fmt: skip


class TestThePairedRule:
    """#94: two revisions held to a per-query rule - a total cannot hide one query's loss."""

    @pytest.fixture
    def compare(self, tmp_path: Path) -> Any:
        import eval_compare  # by name: mutation_check points `pythonpath` at its copy

        def run(base: dict[str, Any], candidate: dict[str, Any], *flags: str) -> int:
            paths = []
            for name, record in (("base.json", base), ("candidate.json", candidate)):
                path = tmp_path / name
                path.write_text(record if isinstance(record, str) else json.dumps(record))
                paths.append(str(path))
            try:
                return int(eval_compare.main([*paths, *flags]))
            except (AttributeError, KeyError, TypeError, ValueError) as crash:
                # A malformed record must be refused with exit 2, never crash the comparator.
                raise AssertionError(f"the comparator crashed: {crash!r}") from crash

        return run

    @staticmethod
    def changed(**case: Any) -> dict[str, Any]:
        candidate = _record()
        candidate["code"] = {"package": "src", "revision": "s", "sha256": "b" * 64}
        candidate["cases"]["dev|paraphrase|p"] = _case(**case)
        return candidate

    @pytest.mark.parametrize(
        ("before", "after", "reason"),
        [
            ({"rank": 1}, {"rank": 2}, "rank 1->2"),
            ({"rank": 3}, {"rank": None}, "rank 3->None"),
            ({"any_valid": True}, {"any_valid": False}, "any-valid@1 lost"),
            ({"ndcg5": 0.5}, {"ndcg5": 0.5 - 2e-9}, "nDCG@5"),
            ({"rank": None, "hits": 5}, {"rank": None, "hits": 0}, "new empty page"),
        ],
    )
    def test_one_query_worse_rejects(
        self, compare: Any, capsys: pytest.CaptureFixture[str],
        before: dict[str, Any], after: dict[str, Any], reason: str,
    ) -> None:  # fmt: skip
        base = _record()
        base["cases"]["dev|paraphrase|p"] = _case(**before)
        assert compare(base, self.changed(**after)) == 1
        out = capsys.readouterr().out
        assert "REJECT" in out and reason in out and "dev|paraphrase|p" in out

    def test_a_drop_within_the_tolerance_is_not_a_loss(self, compare: Any) -> None:
        base = _record()
        base["cases"]["dev|paraphrase|p"] = _case(ndcg5=0.5)
        assert compare(base, self.changed(ndcg5=0.5 - 1e-10)) == 0

    def test_a_no_answer_page_must_not_change(
        self, compare: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        candidate = _record()
        candidate["cases"]["dev|no_answer|n"]["keyword_match"] = "matched"
        assert compare(_record(), candidate) == 1
        assert "no-answer page changed" in capsys.readouterr().out

    def test_a_loss_among_gains_still_rejects(
        self, compare: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cases = {f"dev|paraphrase|p{n}": _case(rank=2) for n in range(5)}
        cases["dev|identifier|i"] = _case(rank=1)
        base = _record(cases)
        better = {key: _case(rank=1) for key in cases}
        better["dev|identifier|i"] = _case(rank=2)
        assert compare(base, _record(better)) == 1
        out = capsys.readouterr().out
        assert "wins 5, losses 0" in out and "dev|identifier|i: rank 1->2" in out

    def test_an_answerable_keyword_change_is_shown_not_vetoed(
        self, compare: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert compare(_record(), self.changed(keyword="no_match")) == 0
        assert "keyword_match matched->no_match" in capsys.readouterr().out

    def test_a_label_change_needs_an_explicit_review(
        self, compare: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        candidate = self.changed(expected="a.md::Moved")
        assert compare(_record(), candidate) == 2
        assert compare(_record(), candidate, "--allow-label-changes") == 0
        assert "not an identical-definition comparison" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "spoil",
        [
            "duplicate key",
            "NaN",
            "1e400",
            "boolean rank",
            "rank 0",
            "rank 21",
            "missing case",
            "count mismatch",
            "foreign split",
            "empty stratum",
            "schema 2",
            "boolean schema",
            "missing rank",
            "missing top",
            "unknown shape",
            "array root",
        ],
    )
    def test_a_malformed_record_is_not_compared(self, compare: Any, spoil: str) -> None:
        record: Any = _record()
        cases = record["cases"]
        if spoil == "boolean rank":
            cases["dev|paraphrase|p"]["rank"] = True
        elif spoil in ("rank 0", "rank 21"):
            cases["dev|paraphrase|p"]["rank"] = int(spoil.split()[1])
        elif spoil == "missing case":
            del cases["dev|identifier|i"]
        elif spoil == "count mismatch":
            record["counts"]["dev|identifier"] = 2
        elif spoil == "foreign split":
            cases["held_out|identifier|i"] = _case()
        elif spoil == "empty stratum":
            del cases["dev|identifier|i"]
            record["counts"]["dev|identifier"] = 0
        elif spoil == "schema 2":
            record["schema"] = 2
        elif spoil == "boolean schema":
            record["schema"] = True
        elif spoil in ("missing rank", "missing top"):
            del cases["dev|paraphrase|p"][spoil.split()[1]]
        elif spoil == "unknown shape":
            cases["dev|no_answer|n"]["shape"] = "riddle"
        text = json.dumps(record) if spoil != "array root" else json.dumps([record])
        if spoil == "duplicate key":
            text = text.replace('"rank": 1,', '"rank": 1, "rank": 2,', 1)
        elif spoil in ("NaN", "1e400"):
            text = text.replace('"ndcg5": 1.0', f'"ndcg5": {spoil}', 1)
        assert compare(_record(), text) == 2
        assert compare(text, _record()) == 2
        assert compare(text, text) == 2, "two records spoilt alike were compared"

    @pytest.mark.parametrize("field", ["corpus_sha256", "evaluator", "preset"])
    def test_records_of_different_things_are_not_compared(self, compare: Any, field: str) -> None:
        assert compare(_record(), _record(**{field: "other"})) == 2

    @staticmethod
    def with_mixed(rank: int | None = 1) -> dict[str, Any]:
        record = _record()
        record["cases"]["dev|mixed|m"] = _case(rank)
        record["counts"]["dev|mixed"] = 1
        return record

    def test_a_mixed_query_worse_rejects(self, compare: Any) -> None:
        assert compare(self.with_mixed(1), self.with_mixed(1)) == 0
        assert compare(self.with_mixed(1), self.with_mixed(2)) == 1

    def test_records_without_a_mixed_stratum_still_compare(self, compare: Any) -> None:
        assert compare(_record(), _record()) == 0

    def test_a_declared_mixed_stratum_must_hold_queries(self, compare: Any) -> None:
        empty = _record()
        empty["counts"]["dev|mixed"] = 0
        assert compare(empty, empty) == 2

    def test_a_mixed_stratum_on_one_side_only_is_not_compared(self, compare: Any) -> None:
        assert compare(self.with_mixed(), _record()) == 2

    def test_the_same_code_twice_is_a_repeatability_check(
        self, compare: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert compare(_record(), _record()) == 0
        out = capsys.readouterr().out
        assert "same-code comparison: a repeatability check" in out
        assert compare(_record(), _record(queries_sha256="z" * 64)) == 0
        out = capsys.readouterr().out
        assert "same-code comparison" in out and "repeatability" not in out


class TestTheCacheNamesWhatBuiltIt:
    """#94: the eval cache keys on the imported code and the corpus's location."""

    @pytest.fixture
    def corpus(self, tmp_path: Path) -> Path:
        root = tmp_path / "corpus"
        root.mkdir()
        (root / "guide.md").write_text("# Guide\n\n## Storage\n\nsegment size\n", encoding="utf-8")
        return root

    def test_another_revision_on_pythonpath_keys_its_own_index(
        self, corpus: Path, tmp_path: Path
    ) -> None:
        import shutil
        import subprocess

        import eval_cache

        import markdown_memory

        copy = tmp_path / "other" / "markdown_memory"
        shutil.copytree(Path(markdown_memory.__file__).parent, copy)
        # Indexing code only: the parse fingerprint cannot see this difference.
        with (copy / "indexer.py").open("a", encoding="utf-8") as indexer:
            indexer.write("\n# another revision\n")
        program = (
            "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
            "import eval_cache; print(eval_cache.build_key(Path(sys.argv[2]), 'bge-small').digest)"
        )
        scripts = str(Path(eval_cache.__file__).parent)

        def key(pythonpath: str | None) -> str:
            env = {**os.environ}
            env.pop("PYTHONPATH", None)
            if pythonpath:
                env["PYTHONPATH"] = pythonpath
            return subprocess.run(
                [sys.executable, "-c", program, scripts, str(corpus)],
                capture_output=True, text=True, check=True, env=env,
            ).stdout.strip()  # fmt: skip

        assert key(str(copy.parent)) != key(None)

    def test_the_same_corpus_at_two_roots_is_two_indexes(
        self, corpus: Path, tmp_path: Path
    ) -> None:
        import shutil

        import eval_cache

        other = shutil.copytree(corpus, tmp_path / "worktree" / "corpus")
        assert eval_cache.corpus_digest(other) == eval_cache.corpus_digest(corpus)
        assert (
            eval_cache.build_key(other, "bge-small").digest
            != eval_cache.build_key(corpus, "bge-small").digest
        )

    def test_an_entry_from_the_old_identity_is_not_reused(self, tmp_path: Path) -> None:
        import eval_cache

        db_path = tmp_path / "eval.db"
        db_path.write_bytes(b"")
        key = eval_cache.CacheKey("c", "k", "e", "s")
        eval_cache.record(db_path, key, "same")
        meta = db_path.with_suffix(".meta.json")
        meta.write_text(json.dumps({**json.loads(meta.read_text()), "version": 1}))
        with pytest.raises(eval_cache.StaleCacheError, match="version 1"):
            eval_cache.validate(db_path, "same", key)


def _entry(root: Path, name: str, mtime: int, *, complete: bool = True) -> Path:
    """A cached index directory as `record` leaves it, dated ``mtime`` seconds."""
    entry = root / (name * 16)[:16]
    entry.mkdir(parents=True)
    (entry / "eval.db").write_bytes(b"x")
    if complete:
        (entry / "eval.meta.json").write_text("{}")
    os.utime(entry, ns=(mtime * 10**9, mtime * 10**9))
    return entry


class TestTheCacheKeepsRecentIndexes:
    """#95: alternating presets or revisions must not rebuild an index it just built."""

    @pytest.fixture
    def cache(self) -> Any:
        import eval_cache  # by name: mutation_check points `pythonpath` at its copy

        return eval_cache

    @staticmethod
    def left(root: Path) -> list[str]:
        return sorted(entry.name[0] for entry in root.iterdir() if entry.name != "eval.lock")

    def test_the_current_index_and_the_three_most_recent_others_stay(
        self, cache: Any, tmp_path: Path
    ) -> None:
        for name, mtime in (("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5), ("f", 6)):
            _entry(tmp_path, name, mtime)
        (tmp_path / "eval.lock").write_text("")
        cache.prune(tmp_path, "a" * 16)
        assert self.left(tmp_path) == ["a", "d", "e", "f"]
        assert (tmp_path / "eval.lock").exists()

    def test_the_current_index_stays_whatever_its_date(self, cache: Any, tmp_path: Path) -> None:
        far = 4_000_000_000  # 2096: dated in the future, so they outrank anything touched now
        for name in "bcde":
            _entry(tmp_path, name, far)
        _entry(tmp_path, "a", 1)
        cache.prune(tmp_path, "a" * 16)
        assert "a" in self.left(tmp_path) and len(self.left(tmp_path)) == 4

    def test_equal_dates_evict_in_name_order(self, cache: Any, tmp_path: Path) -> None:
        for name in "abcde":
            _entry(tmp_path, name, 7)
        _entry(tmp_path, "z", 1)
        cache.prune(tmp_path, "z" * 16)
        assert self.left(tmp_path) == ["c", "d", "e", "z"]

    def test_a_symlink_is_neither_counted_followed_nor_removed(
        self, cache: Any, tmp_path: Path
    ) -> None:
        root = tmp_path / "cache"
        outside = _entry(tmp_path / "elsewhere", "x", 1)
        for name, mtime in (("a", 1), ("b", 2), ("c", 3), ("d", 4)):
            _entry(root, name, mtime)
        (root / ("9" * 16)).symlink_to(outside)
        cache.prune(root, "a" * 16)
        assert (root / ("9" * 16)).is_symlink() and (outside / "eval.db").exists()
        assert self.left(root) == ["9", "a", "b", "c", "d"]  # the link took no slot

    def test_only_cache_entries_are_considered(self, cache: Any, tmp_path: Path) -> None:
        for name, mtime in (("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5)):
            _entry(tmp_path, name, mtime)
        (tmp_path / "notes").mkdir()  # not named like a digest: not this cache's to delete
        cache.prune(tmp_path, "a" * 16)
        assert (tmp_path / "notes").is_dir()

    def test_an_unfinished_build_is_swept(self, cache: Any, tmp_path: Path) -> None:
        _entry(tmp_path, "a", 1)
        _entry(tmp_path, "b", 9, complete=False)
        cache.drop_incomplete(tmp_path, "a" * 16)
        assert self.left(tmp_path) == ["a"]

    def test_a_reused_index_outlives_one_left_alone(self, cache: Any, tmp_path: Path) -> None:
        for name, mtime in (("a", 1), ("b", 2), ("c", 3), ("d", 4)):
            _entry(tmp_path, name, mtime)
        cache.prune(tmp_path, "a" * 16)  # "a" is used again: now the most recent
        _entry(tmp_path, "e", 2_000_000_000)
        cache.prune(tmp_path, "e" * 16)
        assert self.left(tmp_path) == ["a", "c", "d", "e"]  # "b" was the least recent

    def test_another_corpus_directory_is_never_touched(self, cache: Any, tmp_path: Path) -> None:
        for name, mtime in (("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5)):
            _entry(tmp_path / "eval", name, mtime)
            _entry(tmp_path / "eval-v2", name, mtime)
        cache.prune(tmp_path / "eval", "e" * 16)
        assert len(self.left(tmp_path / "eval-v2")) == 5


class TestEvaluationsReuseWhatTheyBuilt:
    """#95 end to end: real indexes, real validation, a fake embedder - counting builds."""

    @pytest.fixture
    def harness(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
        import shutil

        import eval_cache
        import eval_retrieval

        import markdown_memory

        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        monkeypatch.setattr(eval_retrieval, "create_embedder", lambda *a, **k: FakeEmbedder())
        corpus_root = tmp_path / "corpus"
        corpus_root.mkdir()
        (corpus_root / "guide.md").write_text(
            "# Guide\n\n## Storage\n\nsegment size and retention\n\n## Network\n\nports\n"
        )
        corpus = eval_retrieval.Corpus("v1", corpus_root, tmp_path / "queries.json")
        base = ServerConfig(db_path=tmp_path / "unused.db", docs_dir=corpus_root,
                            model_cache_dir=tmp_path / "models")  # fmt: skip
        probes = eval_cache.probe_passages(corpus_root)
        builds: list[str] = []
        real_index = eval_retrieval.MarkdownMemoryService.index_directory

        def counted(service: Any, *args: Any, **kwargs: Any) -> Any:
            builds.append(str(service.config.db_path))
            return real_index(service, *args, **kwargs)

        monkeypatch.setattr(eval_retrieval.MarkdownMemoryService, "index_directory", counted)
        revisions: dict[str, Path] = {}

        def open_cell(preset: str, revision: str) -> bool:
            """Open one cell; True when it had to be built. Each revision name is its own
            index identity: a copy of the package whose `indexer.py` differs."""
            if revision not in revisions:
                copy = tmp_path / revision / "markdown_memory"
                shutil.copytree(Path(markdown_memory.__file__).parent, copy)
                with (copy / "indexer.py").open("a", encoding="utf-8") as indexer:
                    indexer.write(f"\n# {revision}\n")
                revisions[revision] = copy
            monkeypatch.setattr(eval_cache, "SOURCE", revisions[revision])
            arguments = argparse.Namespace(embedder=preset, rebuild=False)
            service, built = eval_retrieval.open_service(arguments, base, probes, corpus)
            service.close()
            return bool(built)

        return argparse.Namespace(open=open_cell, builds=builds, root=corpus.cache())

    CELLS = [(preset, revision) for revision in ("base", "candidate")
             for preset in ("embeddinggemma", "bge-small")]  # fmt: skip

    def test_four_cells_are_built_once_and_then_reused(self, harness: Any) -> None:
        assert all(harness.open(*cell) for cell in self.CELLS)
        assert len(harness.builds) == 4
        assert not any(harness.open(*cell) for cell in self.CELLS)
        assert len(harness.builds) == 4, "revisiting a retained cell rebuilt it"

    def test_a_fifth_identity_evicts_the_least_recently_used(self, harness: Any) -> None:
        for cell in self.CELLS:
            harness.open(*cell)
        harness.open(*self.CELLS[0])  # used again: the second cell is now the least recent
        harness.open("embeddinggemma", "third")
        first, second = (Path(build).parent for build in harness.builds[:2])
        assert first.is_dir(), "a cell used a moment ago was evicted"
        assert not second.exists(), "building a fifth cell kept the least recent one"

    def test_a_reuse_trims_the_cache_too(self, harness: Any) -> None:
        import shutil

        for cell in self.CELLS:
            harness.open(*cell)
        # A fifth complete entry, as a run with a larger limit or a killed prune leaves it.
        shutil.copytree(Path(harness.builds[1]).parent, harness.root / ("f" * 16))
        os.utime(harness.root / ("f" * 16), ns=(1, 1))
        assert harness.open(*self.CELLS[0]) is False
        assert len([entry for entry in harness.root.iterdir() if entry.is_dir()]) == 4

    def test_failed_builds_never_pile_up(
        self, harness: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import eval_retrieval

        for cell in self.CELLS:
            harness.open(*cell)
        failed = IndexReport(
            directory="x", files_scanned=1, files_indexed=0, files_unchanged=0, files_purged=0,
            sections_indexed=0, passages_indexed=0, elapsed_seconds=0.0,
            errors=(FileFailure(file_path="guide.md", message="Embedding failed"),),
        )  # fmt: skip
        monkeypatch.setattr(
            eval_retrieval.MarkdownMemoryService, "index_directory", lambda *a, **k: failed
        )
        for attempt in range(3):  # three different identities, each failing
            with pytest.raises(SystemExit, match="indexing failed"):
                harness.open("embeddinggemma", f"broken{attempt}")
            for entry in harness.root.iterdir():  # as a killed build would leave it
                if entry.is_dir() and not (entry / "eval.meta.json").exists():
                    (entry / "eval.db-wal").write_bytes(b"x" * 1024)
        entries = [entry for entry in harness.root.iterdir() if entry.is_dir()]
        unfinished = [entry for entry in entries if not (entry / "eval.meta.json").exists()]
        assert len(unfinished) == 1, "each attempt sweeps the failure before it"
        assert len(entries) == 5  # the four complete cells survived every failure

    def test_a_stale_index_is_rebuilt_without_being_told(self, harness: Any) -> None:
        harness.open("embeddinggemma", "base")
        harness.open("bge-small", "base")
        meta = Path(harness.builds[0]).parent / "eval.meta.json"  # the embeddinggemma cell
        meta.write_text(json.dumps({**json.loads(meta.read_text()), "parse_fingerprint": "x"}))
        assert harness.open("embeddinggemma", "base") is True
        assert harness.open("embeddinggemma", "base") is False
        assert harness.open("bge-small", "base") is False  # the sibling survived


class TestEveryModuleIsKeyedOrExcused:
    """#99: a module that shapes the index and is missing from the key is a false cache hit."""

    @pytest.fixture
    def cache(self) -> Any:
        import eval_cache  # by name: mutation_check points `pythonpath` at its copy

        return eval_cache

    @pytest.fixture
    def key_after(self, cache: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
        import shutil

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "guide.md").write_text(
            "# Guide\n\n## Storage\n\nsegment size\n", encoding="utf-8"
        )
        before = cache.build_key(corpus, "bge-small").digest

        def run(module: str) -> tuple[str, str]:
            """The key before and after one line is added to ``module`` of the imported package."""
            copy = tmp_path / module / "markdown_memory"
            shutil.copytree(cache.SOURCE, copy)
            with (copy / module).open("a", encoding="utf-8") as source:
                source.write("\n# another revision of this module\n")
            monkeypatch.setattr(cache, "SOURCE", copy)
            return before, cache.build_key(corpus, "bge-small").digest

        return run

    def test_every_package_module_is_classified(self, cache: Any) -> None:
        present = {path.name for path in cache.SOURCE.glob("*.py")}
        keyed, excused = set(cache.INDEX_SOURCES), set(cache.NOT_INDEX_SOURCES)
        assert keyed.isdisjoint(excused)
        assert keyed | excused == present, sorted(present ^ (keyed | excused))
        assert all(reason.strip() for reason in cache.NOT_INDEX_SOURCES.values())

    @pytest.mark.parametrize("module", ["server.py", "headings.py", "exceptions.py"])
    def test_a_change_to_what_builds_the_index_changes_the_key(
        self, key_after: Any, module: str
    ) -> None:
        before, after = key_after(module)
        assert before != after, f"a change to {module} would reuse an index built without it"

    def test_a_ranking_change_keeps_the_key(self, key_after: Any) -> None:
        before, after = key_after("search.py")
        assert before == after, "a ranking change would rebuild the evaluation index"
