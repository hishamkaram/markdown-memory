"""Regressions in the developer scripts: the retrieval evaluation gate.

One test per defect found in code review; each fails when its fix is reverted.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest
from fakes import FakeEmbedder

from markdown_memory.db import Database
from markdown_memory.models import (
    IndexReport,
    SearchResult,
)
from markdown_memory.server import MarkdownMemoryService, ServerConfig


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

    def index_directory(self, directory: str | None = None) -> IndexReport:
        return IndexReport(
            directory="stub", files_scanned=0, files_indexed=0, files_unchanged=0, files_purged=0,
            sections_indexed=0, passages_indexed=0, elapsed_seconds=0.0,
        )  # fmt: skip

    def search_docs(self, query: str, limit: int) -> list[SearchResult]:
        return []

    def close(self) -> None:
        return None


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
        monkeypatch.setattr(evaluation, "MarkdownMemoryService", _EvalStubService)
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
