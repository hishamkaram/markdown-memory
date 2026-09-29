"""The background runner: when it starts a run, when it does not, and how it stops."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import pytest
from fakes import FakeEmbedder

import markdown_memory.server as server_module
from markdown_memory.autoindex import AutoIndexer
from markdown_memory.config import ServerConfig, resolve_config
from markdown_memory.exceptions import IndexBusyError, IndexCancelled
from markdown_memory.indexer import Indexer
from markdown_memory.models import FileFailure, IndexReport, IndexStatus
from markdown_memory.server import MarkdownMemoryService


def _report(errors: int = 0) -> IndexReport:
    return IndexReport(
        directory="/docs",
        files_scanned=1,
        files_indexed=1,
        files_unchanged=0,
        files_purged=0,
        sections_indexed=1,
        passages_indexed=1,
        elapsed_seconds=0.0,
        errors=tuple(FileFailure(file_path=f"/docs/{n}.md", message="no") for n in range(errors)),
    )


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _Harness:
    """A runner over scripted runs: each run returns (or raises) the next outcome."""

    def __init__(self, *outcomes: IndexReport | Exception) -> None:
        self.outcomes = list(outcomes)
        self.runs = 0
        self.after = IndexStatus(verified=True)
        self.clock = _Clock()
        self.runner = AutoIndexer(self._run, lambda: self.after, clock=self.clock)

    def _run(self, _should_stop: Callable[[], bool]) -> IndexReport:
        self.runs += 1
        outcome = self.outcomes.pop(0) if self.outcomes else _report()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def settle(self) -> None:
        deadline = time.monotonic() + 5
        while self.runner.active:
            assert time.monotonic() < deadline, "the run never finished"
            time.sleep(0.005)

    def consider(self, changed: int = 0, mismatch: str | None = None) -> bool:
        started = self.runner.consider(
            IndexStatus(verified=True, changed_files=changed, weights_mismatch=mismatch)
        )
        self.settle()
        return started


@pytest.fixture
def harness() -> _Harness:
    h = _Harness()
    assert h.runner.request()  # the run at start
    h.settle()
    return h


class TestWhatStartsARun:
    def test_an_edit_a_search_sees_starts_a_run_once_the_gap_has_passed(
        self, harness: _Harness
    ) -> None:
        harness.clock.now += 9.9
        assert not harness.consider(changed=1), "a burst of saves is one run, not one each"
        harness.clock.now += 0.1
        assert harness.consider(changed=1)
        assert harness.runs == 2

    def test_a_file_the_last_run_could_not_index_does_not_start_one_per_search(self) -> None:
        """An unreadable file stays "changed" on every sweep. Measured against zero, each
        search after the gap walked the whole tree again, for ever."""
        h = _Harness(_report(errors=1))
        h.after = IndexStatus(verified=False, changed_files=1)
        h.runner.request()
        h.settle()
        h.clock.now += 60
        assert not h.consider(changed=1)
        assert h.consider(changed=2), "a new edit still counts"

    def test_an_edit_made_while_a_clean_run_ran_starts_the_next_one(self) -> None:
        """A clean run resets the baseline to zero rather than to what it measured: a file
        edited while it ran is still changed afterwards and must not become the norm."""
        h = _Harness(_report())
        h.after = IndexStatus(verified=True, changed_files=1)  # edited mid-run
        h.runner.request()
        h.settle()
        h.clock.now += 10
        assert h.consider(changed=1)

    @pytest.mark.parametrize(
        "interruption",
        [IndexBusyError("another process"), IndexCancelled("stopped")],
        ids=["busy", "cancelled"],
    )
    def test_a_run_that_did_not_happen_keeps_the_baseline(self, interruption: Exception) -> None:
        h = _Harness(_report(errors=1), interruption)
        h.after = IndexStatus(verified=False, changed_files=1)
        h.runner.request()
        h.settle()
        h.clock.now += 10
        h.after = IndexStatus(verified=False, changed_files=2)  # what it would have measured
        assert h.consider(changed=2)  # the interrupted run
        h.clock.now += 10
        assert not h.consider(changed=1), "the baseline absorbed what nobody indexed"
        assert h.consider(changed=2), "an edit the interrupted run never saw still counts"

    def test_the_walk_that_finds_new_files_is_due_after_the_interval(
        self, harness: _Harness
    ) -> None:
        harness.clock.now += 299
        assert not harness.consider()
        harness.clock.now += 1
        assert harness.consider()

    def test_a_weights_mismatch_starts_one_run_not_one_per_search(self, harness: _Harness) -> None:
        harness.after = IndexStatus(verified=False, weights_mismatch="drift")
        assert harness.consider(mismatch="drift"), "repaired at once, not after the walk"
        assert not harness.consider(mismatch="drift"), "one this run could not repair"
        assert harness.consider(mismatch="another drift")


class TestOneRunAtATime:
    def test_nothing_starts_while_a_run_is_running_or_after_stop(self) -> None:
        release = threading.Event()
        runs = 0

        def run(_should_stop: Callable[[], bool]) -> IndexReport:
            nonlocal runs
            runs += 1
            release.wait(5)
            return _report()

        runner = AutoIndexer(run, lambda: IndexStatus(verified=True))
        assert runner.request()
        assert not runner.request()
        assert not runner.consider(IndexStatus(verified=True, changed_files=9))
        release.set()
        runner.stop()
        assert not runner.active
        assert not runner.request(), "a stopped runner starts nothing"
        assert runs == 1

    def test_stop_asks_the_run_to_stop_and_waits_for_it(self) -> None:
        asked = threading.Event()

        def run(should_stop: Callable[[], bool]) -> IndexReport:
            while not should_stop():
                time.sleep(0.001)
            asked.set()
            raise IndexCancelled("stopped")

        runner = AutoIndexer(run, lambda: IndexStatus(verified=True))
        runner.request()
        runner.stop()
        assert asked.is_set()
        assert not runner.active


def _tree(root: Path, names: Sequence[str]) -> Path:
    root.mkdir()
    for name in names:
        (root / f"{name}.md").write_text(f"# {name}\n\nbody of {name}\n")
    return root


class TestCancellation:
    def test_a_stop_lands_between_documents_and_the_next_run_resumes(self, tmp_path: Path) -> None:
        """The stop is its own exception, not a `MarkdownMemoryError`: the per-file handler
        catches those and moves on to the next file, so a stop raised as one would have
        been recorded as a failure and the run would have carried on."""
        from markdown_memory.db import Database

        root = _tree(tmp_path / "docs", ("a", "b", "c"))
        with Database(tmp_path / "index.db") as db:
            indexer = Indexer(db, FakeEmbedder(), workers=1)
            with pytest.raises(IndexCancelled):
                indexer.index_directory(root, should_stop=lambda: db.count_rows("documents") >= 1)
            assert db.count_rows("documents") == 1
            assert db.failure_paths(str(root)) == []
            assert not db.index_status(str(root)).verified

            report = indexer.index_directory(root)
            assert (report.files_indexed, report.files_unchanged) == (2, 1)
            assert db.index_status(str(root)).verified

    def test_a_stop_asked_before_the_run_starts_embeds_nothing(self, tmp_path: Path) -> None:
        from markdown_memory.db import Database

        root = _tree(tmp_path / "docs", ("a", "b"))
        embedder = FakeEmbedder()
        with Database(tmp_path / "index.db") as db, pytest.raises(IndexCancelled):
            Indexer(db, embedder, workers=1).index_directory(root, should_stop=lambda: True)
        assert embedder.document_calls == [], "a model's worth of work for nothing"


class _StopWhileEmbedding(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.asked = threading.Event()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.asked.set()  # the stop arrives while this file is being embedded
        return super().embed_documents(texts)


class TestAStopDuringEmbedding:
    def test_the_file_being_embedded_is_not_written_and_not_called_a_failure(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Embedding one file takes seconds; a stop asked meanwhile writes nothing more,
        and is not swallowed by the per-file handler as that file's failure."""
        from markdown_memory.db import Database

        root = _tree(tmp_path / "docs", ("a", "b"))
        embedder = _StopWhileEmbedding()
        with Database(tmp_path / "index.db") as db:
            with pytest.raises(IndexCancelled):
                Indexer(db, embedder, workers=1).index_directory(
                    root, should_stop=embedder.asked.is_set
                )
            assert db.count_rows("documents") == 0
        assert "Failed to index" not in caplog.text


class _Slow(FakeEmbedder):
    """Embeds only once released, so a test can close the service mid-run."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.entered.set()
        self.release.wait(5)
        return super().embed_documents(texts)


@pytest.fixture
def slow_service(tmp_path: Path) -> Iterator[tuple[MarkdownMemoryService, _Slow]]:
    root = _tree(tmp_path / "docs", ("a", "b", "c", "d"))
    embedder = _Slow()
    service = MarkdownMemoryService(
        ServerConfig(db_path=tmp_path / "index.db", docs_dir=root, index_workers=1),
        embedder=embedder,
    )
    yield service, embedder
    embedder.release.set()
    service.close()


class TestTheService:
    def test_a_service_built_directly_starts_nothing(self, tmp_path: Path) -> None:
        root = _tree(tmp_path / "docs", ("a",))
        service = MarkdownMemoryService(
            ServerConfig(db_path=tmp_path / "index.db", docs_dir=root), embedder=FakeEmbedder()
        )
        try:
            status = service.index_status()
            assert not status.indexing
            assert "mdmem-autoindex" not in {thread.name for thread in threading.enumerate()}
            assert service.list_documents() == []
        finally:
            service.close()

    def test_close_stops_the_run_before_the_database_closes(
        self, slow_service: tuple[MarkdownMemoryService, _Slow], caplog: pytest.LogCaptureFixture
    ) -> None:
        service, embedder = slow_service
        service.start_auto_index()
        assert embedder.entered.wait(5)
        closer = threading.Thread(target=service.close)
        closer.start()
        embedder.release.set()
        closer.join(10)
        assert not closer.is_alive()
        assert "Automatic index run failed" not in caplog.text
        assert "mdmem-autoindex" not in {thread.name for thread in threading.enumerate()}

    def test_the_status_says_a_run_is_in_progress_instead_of_asking_for_one(
        self, slow_service: tuple[MarkdownMemoryService, _Slow]
    ) -> None:
        service, embedder = slow_service
        service.start_auto_index()
        assert embedder.entered.wait(5)
        status = service.index_status()
        assert status.indexing and status.to_dict()["indexing"] is True
        assert "Run index_directory" not in (status.message() or "")
        assert "in progress" in (status.message() or "")
        embedder.release.set()

    def test_a_search_after_an_edit_brings_the_index_up_to_date(self, tmp_path: Path) -> None:
        root = _tree(tmp_path / "docs", ("a",))
        service = MarkdownMemoryService(
            ServerConfig(db_path=tmp_path / "index.db", docs_dir=root), embedder=FakeEmbedder()
        )
        try:
            service.start_auto_index()
            auto = service._auto
            assert auto is not None
            _wait(auto)
            assert [d.title for d in service.list_documents()] == ["a"]

            (root / "b.md").write_text("# b\n\nbody of b\n")
            auto._clock = lambda: time.monotonic() + 3600  # the walk interval has passed
            service.index_status()  # what every search_docs asks
            _wait(auto)
            assert sorted(d.title for d in service.list_documents()) == ["a", "b"]
        finally:
            service.close()


def _wait(runner: AutoIndexer) -> None:
    deadline = time.monotonic() + 5
    while runner.active:
        assert time.monotonic() < deadline, "the run never finished"
        time.sleep(0.005)


class TestConfiguration:
    @pytest.mark.parametrize("value", ["0", "false", "OFF", "no"])
    def test_the_environment_turns_it_off(
        self, value: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MARKDOWN_MEMORY_AUTO_INDEX", value)
        assert not ServerConfig.from_env().auto_index
        assert resolve_config(auto_index=True).auto_index, "an explicit choice wins"

    def test_it_is_on_unless_turned_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MARKDOWN_MEMORY_AUTO_INDEX", raising=False)
        assert ServerConfig.from_env().auto_index
        assert not resolve_config(auto_index=False).auto_index

    @pytest.mark.parametrize(
        ("argv", "env", "started"),
        [([], None, True), (["--no-auto-index"], None, False), ([], "0", False)],
        ids=["default", "flag", "environment"],
    )
    def test_only_the_stdio_server_starts_it_and_only_when_asked_to(
        self,
        argv: list[str],
        env: str | None,
        started: bool,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[MarkdownMemoryService] = []

        class _FakeServer:
            def run(self, transport: str) -> None:
                assert transport == "stdio"

        monkeypatch.setattr(server_module, "create_server", lambda config, service: _FakeServer())
        monkeypatch.setattr(server_module, "_warm_up", lambda embedder: None)
        monkeypatch.setattr(server_module, "configure_logging", lambda level=None: None)
        monkeypatch.setattr(
            MarkdownMemoryService, "start_auto_index", lambda self: calls.append(self)
        )
        monkeypatch.setenv("MARKDOWN_MEMORY_DB", str(tmp_path / "env.db"))
        monkeypatch.setenv("MARKDOWN_MEMORY_DOCS_DIR", str(tmp_path))
        if env is None:
            monkeypatch.delenv("MARKDOWN_MEMORY_AUTO_INDEX", raising=False)
        else:
            monkeypatch.setenv("MARKDOWN_MEMORY_AUTO_INDEX", env)
        server_module.main(argv)
        assert bool(calls) is started
