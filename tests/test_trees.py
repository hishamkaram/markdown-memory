"""Answering an agent from its own work tree (#69).

An agent in a linked worktree shares the server of the checkout it was started for, and was
answered from that checkout's docs - another branch's. Every tool now takes `cwd`, and a path
in another work tree of the same repository is answered from that tree's own copy of the
docs root, indexed into a database of its own, with the vectors the configured root already
holds reused for every passage whose embedded text is identical.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeEmbedder
from mcp import Client

import markdown_memory.server as server_module
import markdown_memory.trees as trees_module
from markdown_memory.config import (
    ServerConfig,
    _project_database,
    resolve_config,
    tree_database,
)
from markdown_memory.db import Database
from markdown_memory.exceptions import DatabaseError, WorkTreeError
from markdown_memory.indexer import Indexer
from markdown_memory.models import IndexStatus, SearchPage
from markdown_memory.server import MarkdownMemoryService, create_server
from markdown_memory.trees import Tree, counterpart, work_tree

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

PRIMARY_TEXT = "The primary checkout deploys with blue lanterns."
BRANCH_TEXT = "The branch deploys with green turbines."
SHARED = "# Shared\n\n## Setup\n\nInstall the toolchain and run the bootstrap script.\n"


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "-c", "user.email=t@t", "-c", "user.name=t", *args],
        text=True,
        stderr=subprocess.STDOUT,
    )


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository whose docs a linked worktree changes: one edit, one new, one deleted."""
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    write(root / "docs" / "guide.md", f"# Guide\n\n## Deploy\n\n{PRIMARY_TEXT}\n")
    write(root / "docs" / "shared.md", SHARED)
    write(root / "docs" / "gone.md", "# Gone\n\nOnly the primary checkout has this page.\n")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "docs")
    return root


@pytest.fixture
def worktree(repo: Path) -> Path:
    """Nested where Claude Code puts them: under `.claude/worktrees/` of the primary."""
    tree = repo / ".claude" / "worktrees" / "w"
    git(repo, "worktree", "add", "-q", "-b", "w", str(tree))
    write(tree / "docs" / "guide.md", f"# Guide\n\n## Deploy\n\n{BRANCH_TEXT}\n")
    write(tree / "docs" / "new.md", "# New\n\nA page only the branch has.\n")
    (tree / "docs" / "gone.md").unlink()
    return tree


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(weights="fake-weights")


@pytest.fixture
def primary(tmp_path: Path, repo: Path, embedder: FakeEmbedder) -> Iterator[MarkdownMemoryService]:
    service = MarkdownMemoryService(
        ServerConfig(db_path=tmp_path / "index" / "primary.db", docs_dir=repo / "docs"),
        embedder=embedder,
    )
    service.index_directory()
    yield service
    service.close()


async def call(client: Client, name: str, **arguments: Any) -> Any:
    outcome = await client.call_tool(name, arguments)
    assert not outcome.is_error, outcome.content
    content = outcome.structured_content
    assert content is not None
    return content["result"] if set(content) == {"result"} else content


async def error(client: Client, name: str, **arguments: Any) -> str:
    outcome = await client.call_tool(name, arguments)
    assert outcome.is_error
    return str(outcome.content[0].text)  # type: ignore[union-attr]


def paths(service: MarkdownMemoryService) -> set[str]:
    return {summary.file_path for summary in service.db.list_documents()}


class TestWorkTree:
    def test_a_linked_worktree_shares_the_repository_of_its_primary(
        self, repo: Path, worktree: Path
    ) -> None:
        home, other = work_tree(repo / "docs"), work_tree(worktree / "docs" / "guide.md")
        assert home == Tree(repo.resolve(), (repo / ".git").resolve())
        assert other == Tree(worktree.resolve(), (repo / ".git").resolve())

    def test_a_directory_in_no_repository_is_in_no_work_tree(self, tmp_path: Path) -> None:
        (tmp_path / "plain").mkdir()
        assert work_tree(tmp_path / "plain") is None

    def test_a_bare_repository_has_no_work_tree(self, tmp_path: Path) -> None:
        git(tmp_path, "init", "-q", "--bare", "bare.git")
        assert work_tree(tmp_path / "bare.git") is None

    def test_a_worktree_whose_admin_directory_is_gone_is_an_error_not_a_guess(
        self, repo: Path, worktree: Path
    ) -> None:
        shutil.rmtree(repo / ".git" / "worktrees" / "w")
        with pytest.raises(WorkTreeError, match="not a git repository"):
            work_tree(worktree)

    def test_no_git_on_path_means_no_work_tree(
        self, worktree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PATH", "")
        assert work_tree(worktree) is None

    def test_a_git_dir_inherited_from_a_hook_does_not_misroute(
        self, tmp_path: Path, repo: Path, worktree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        other = tmp_path / "other"
        other.mkdir()
        git(other, "init", "-q")
        monkeypatch.setenv("GIT_DIR", str(other / ".git"))
        assert work_tree(worktree) == Tree(worktree.resolve(), (repo / ".git").resolve())

    def test_the_docs_root_maps_to_the_same_place_in_the_other_tree(
        self, repo: Path, worktree: Path
    ) -> None:
        home, other = work_tree(repo), work_tree(worktree)
        assert home is not None and other is not None
        assert counterpart((repo / "docs").resolve(), home, other) == (worktree / "docs").resolve()

    def test_a_tree_without_the_docs_directory_is_refused_with_the_reason(
        self, repo: Path, worktree: Path
    ) -> None:
        shutil.rmtree(worktree / "docs")
        home, other = work_tree(repo), work_tree(worktree)
        assert home is not None and other is not None
        with pytest.raises(WorkTreeError, match="has no docs directory"):
            counterpart((repo / "docs").resolve(), home, other)


class TestRouting:
    async def test_cwd_in_a_worktree_answers_from_that_worktree(
        self, primary: MarkdownMemoryService, worktree: Path, repo: Path
    ) -> None:
        async with Client(create_server(service=primary)) as client:
            await call(client, "index_directory", cwd=str(worktree))
            branch = await call(client, "search_docs", query="deploys", cwd=str(worktree))
            main = await call(client, "search_docs", query="deploys")
        assert BRANCH_TEXT in branch["results"][0]["content"]
        assert branch["index_status"]["root"] == str((worktree / "docs").resolve())
        assert PRIMARY_TEXT in main["results"][0]["content"]
        assert main["index_status"]["root"] == str((repo / "docs").resolve())

    async def test_cwd_in_the_primary_or_anywhere_else_answers_from_the_configured_root(
        self, primary: MarkdownMemoryService, repo: Path, tmp_path: Path
    ) -> None:
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        foreign = tmp_path / "foreign"
        foreign.mkdir()
        git(foreign, "init", "-q")
        async with Client(create_server(service=primary)) as client:
            for cwd in (repo, repo / "docs", elsewhere, foreign):
                page = await call(client, "search_docs", query="deploys", cwd=str(cwd))
                assert page["index_status"]["root"] == primary.root
                assert page["results"] and PRIMARY_TEXT in page["results"][0]["content"]

    async def test_a_worktree_path_routes_there_without_cwd(
        self, primary: MarkdownMemoryService, worktree: Path
    ) -> None:
        guide = str((worktree / "docs" / "guide.md").resolve())
        async with Client(create_server(service=primary)) as client:
            await call(client, "index_directory", directory=str(worktree / "docs"))
            text = await call(client, "read_section", file_path=guide, heading_path="Deploy")
            outline = await call(client, "get_document_outline", file_path=guide)
            listed = await call(client, "list_documents", directory=str(worktree / "docs"))
        assert BRANCH_TEXT in text
        assert [node["heading_path"] for node in outline] == ["Guide"]
        assert listed["index_status"]["root"] == str((worktree / "docs").resolve())
        # The P0 both reviews found in v2: indexing a worktree never writes the primary's.
        assert not any(".claude" in path for path in paths(primary))

    async def test_a_relative_path_is_resolved_where_the_agent_is(
        self, primary: MarkdownMemoryService, worktree: Path
    ) -> None:
        # A second `docs/guide.md` below, so the path is not also a unique suffix.
        write(worktree / "docs" / "archive" / "docs" / "guide.md", "# Old\n\n## Deploy\n\nOld.\n")
        async with Client(create_server(service=primary)) as client:
            await call(client, "index_directory", cwd=str(worktree))
            text = await call(
                client,
                "read_section",
                file_path="docs/guide.md",
                heading_path="Deploy",
                cwd=str(worktree),
            )
        assert BRANCH_TEXT in text

    async def test_the_two_indexes_never_hold_each_others_paths(
        self, primary: MarkdownMemoryService, worktree: Path, repo: Path
    ) -> None:
        created: list[MarkdownMemoryService] = []
        original = server_module.MarkdownMemoryService

        class Tracked(original):  # type: ignore[valid-type,misc]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                created.append(self)

        server_module.MarkdownMemoryService = Tracked  # type: ignore[misc]
        try:
            async with Client(create_server(service=primary)) as client:
                await call(client, "index_directory", cwd=str(worktree))
                tree_paths = paths(created[0])
                primary.index_directory()  # the root's run purges nothing of the tree's
                assert paths(created[0]) == tree_paths
        finally:
            server_module.MarkdownMemoryService = original  # type: ignore[misc]
        docs = (worktree / "docs").resolve()
        assert tree_paths == {str(docs / name) for name in ("guide.md", "shared.md", "new.md")}
        assert all(path.startswith(str((repo / "docs").resolve())) for path in paths(primary))
        with pytest.raises(DatabaseError, match="closed"):  # closed with the session
            created[0].list_documents()

    async def test_a_server_started_in_a_worktree_answers_for_the_main_checkout(
        self, tmp_path: Path, repo: Path, worktree: Path, embedder: FakeEmbedder
    ) -> None:
        service = MarkdownMemoryService(
            ServerConfig(db_path=tmp_path / "w.db", docs_dir=worktree / "docs"), embedder=embedder
        )
        try:
            async with Client(create_server(service=service)) as client:
                await call(client, "index_directory", cwd=str(repo))
                page = await call(client, "search_docs", query="deploys", cwd=str(repo))
        finally:
            service.close()
        assert PRIMARY_TEXT in page["results"][0]["content"]
        assert page["index_status"]["root"] == str((repo / "docs").resolve())

    async def test_what_cannot_be_served_is_refused_with_the_reason(
        self, primary: MarkdownMemoryService, repo: Path, worktree: Path, tmp_path: Path
    ) -> None:
        async with Client(create_server(service=primary)) as client:
            missing = await error(client, "search_docs", query="x", cwd=str(tmp_path / "nope"))
            relative = await error(client, "search_docs", query="x", cwd="docs")
            shutil.rmtree(worktree / "docs")
            no_docs = await error(client, "search_docs", query="x", cwd=str(worktree))
            shutil.rmtree(repo / ".git" / "worktrees" / "w")
            broken = await error(client, "search_docs", query="x", cwd=str(worktree))
        assert "absolute path of an existing directory" in missing
        assert "absolute path of an existing directory" in relative
        assert "has no docs directory" in no_docs and "Call without `cwd`" in no_docs
        assert "not a git repository" in broken

    async def test_one_server_answers_for_a_bounded_number_of_trees(
        self, primary: MarkdownMemoryService, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(trees_module, "MAX_TREES", 1)
        for name in ("a", "b"):
            git(repo, "worktree", "add", "-q", "-b", name, str(repo.parent / name))
        async with Client(create_server(service=primary)) as client:
            await call(client, "list_documents", cwd=str(repo.parent / "a"))
            refused = await error(client, "list_documents", cwd=str(repo.parent / "b"))
        assert "already answers for 1 other work trees" in refused

    async def test_a_relative_directory_is_the_agents_when_it_exists_there(
        self, primary: MarkdownMemoryService, worktree: Path
    ) -> None:
        async with Client(create_server(service=primary)) as client:
            report = await call(client, "index_directory", directory="docs", cwd=str(worktree))
            listed = await call(client, "list_documents", directory="docs", cwd=str(worktree))
        docs = str((worktree / "docs").resolve())
        assert f"Indexed {docs} " in report
        assert {document["file_path"] for document in listed["documents"]} == {
            f"{docs}/{name}" for name in ("guide.md", "shared.md", "new.md")
        }

    async def test_a_submodule_is_its_own_repository_and_changes_nothing(
        self, tmp_path: Path, primary: MarkdownMemoryService, repo: Path
    ) -> None:
        library = tmp_path / "library"
        library.mkdir()
        git(library, "init", "-q")
        git(library, "commit", "-q", "--allow-empty", "-m", "init")
        git(repo, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(library), "lib")
        async with Client(create_server(service=primary)) as client:
            page = await call(client, "search_docs", query="deploys", cwd=str(repo / "lib"))
        assert page["index_status"]["root"] == primary.root

    async def test_another_trees_run_waits_its_turn_behind_the_roots(
        self, primary: MarkdownMemoryService, worktree: Path
    ) -> None:
        async with Client(create_server(service=primary)) as client:
            with primary.run_lock:
                refused = await error(client, "index_directory", cwd=str(worktree))
            await call(client, "index_directory", cwd=str(worktree))
        assert "Another index run is in progress" in refused

    async def test_a_trees_background_run_is_armed_not_started_by_the_call_that_made_it(
        self, primary: MarkdownMemoryService, worktree: Path
    ) -> None:
        requests: list[bool] = []
        original = server_module.MarkdownMemoryService

        class Tracked(original):  # type: ignore[valid-type,misc]
            def start_auto_index(self, *, request: bool = True) -> None:
                requests.append(request)
                super().start_auto_index(request=request)

        primary.start_auto_index(request=False)
        server_module.MarkdownMemoryService = Tracked  # type: ignore[misc]
        try:
            async with Client(create_server(service=primary)) as client:
                await call(client, "index_directory", cwd=str(worktree))
        finally:
            server_module.MarkdownMemoryService = original  # type: ignore[misc]
        assert requests == [False]


class TestAFreshTree:
    """Found end to end, under Claude Code and Codex alike: a worktree's first search reads an
    index its first run has barely begun, and was told its terms are not documented."""

    def test_a_miss_while_an_unvouched_tree_is_indexing_says_so(self) -> None:
        page = SearchPage((), "no_match")
        building = IndexStatus(verified=False, indexing=True)
        assert "being indexed right now" in (page.keyword_message(building) or "")
        assert "not in the indexed documentation" in (page.keyword_message() or "")
        whole = IndexStatus(verified=True, indexing=True)  # a refresh of a tree walked whole
        assert "not in the indexed documentation" in (page.keyword_message(whole) or "")
        assert SearchPage((), "matched").keyword_message(building) is None

    async def test_search_docs_tells_the_agent_to_wait_rather_than_conclude(
        self, primary: MarkdownMemoryService, worktree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            MarkdownMemoryService,
            "index_status",
            lambda self, directory=None: IndexStatus(verified=False, indexing=True),
        )
        async with Client(create_server(service=primary)) as client:
            page = await call(client, "search_docs", query="zebra quartz", cwd=str(worktree))
        assert page["results"] == []
        assert "search again once index_status.indexing is false" in page["keyword_message"]


class TestProvider:
    def test_concurrent_first_calls_for_one_tree_share_one_service(
        self, primary: MarkdownMemoryService, worktree: Path
    ) -> None:
        provider = server_module._ServiceProvider(None, primary)
        provider.session_started()
        got: list[MarkdownMemoryService] = []
        threads = [
            threading.Thread(target=lambda: got.append(provider.get(str(worktree))))
            for _ in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        try:
            assert len({id(service) for service in got}) == 1
            assert got[0] is not primary
            # One index run at a time across every tree the server answers for.
            assert got[0].run_lock is primary.run_lock
            assert got[0].embedder is primary.embedder
        finally:
            provider.session_ended()

    def test_only_a_database_the_environment_derived_counts_as_derived(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MARKDOWN_MEMORY_DOCS_DIR", str(tmp_path))
        monkeypatch.delenv("MARKDOWN_MEMORY_DB", raising=False)
        assert not ServerConfig.from_env().db_explicit
        assert not resolve_config().db_explicit
        assert resolve_config(db=tmp_path / "x.db").db_explicit
        monkeypatch.setenv("MARKDOWN_MEMORY_DB", str(tmp_path / "y.db"))
        assert ServerConfig.from_env().db_explicit and resolve_config().db_explicit

    def test_a_trees_database_is_beside_a_chosen_one_and_derived_otherwise(
        self, tmp_path: Path
    ) -> None:
        docs = tmp_path / "w" / "docs"
        chosen = ServerConfig(db_path=tmp_path / "dbs" / "index.db", docs_dir=tmp_path)
        derived = ServerConfig(db_path=tmp_path / "x.db", docs_dir=tmp_path, db_explicit=False)
        sibling = tree_database(chosen, docs)
        assert sibling.parent == tmp_path / "dbs" and sibling.name.startswith("index-docs-")
        assert sibling != chosen.db_path
        assert tree_database(derived, docs) == _project_database(docs)


class TestReuse:
    async def test_unchanged_passages_are_not_embedded_again(
        self, primary: MarkdownMemoryService, worktree: Path, embedder: FakeEmbedder
    ) -> None:
        embedder.document_calls.clear()
        async with Client(create_server(service=primary)) as client:
            await call(client, "index_directory", cwd=str(worktree))
        embedded = [text for batch in embedder.document_calls for text in batch]
        assert any(BRANCH_TEXT in text for text in embedded)
        assert any("only the branch has" in text for text in embedded)
        assert not any("bootstrap script" in text for text in embedded)

    async def test_a_reused_index_is_the_index_a_cold_run_builds(
        self, tmp_path: Path, primary: MarkdownMemoryService, worktree: Path
    ) -> None:
        async with Client(create_server(service=primary)) as client:
            await call(client, "index_directory", cwd=str(worktree))
        warm_path = tree_database(primary.config, (worktree / "docs").resolve())
        cold = MarkdownMemoryService(
            ServerConfig(db_path=tmp_path / "cold.db", docs_dir=worktree / "docs"),
            embedder=FakeEmbedder(weights="fake-weights"),
        )
        try:
            cold.index_directory()
            with Database(warm_path) as warm:
                assert warm.integrity_problems() == []
                for summary in cold.db.list_documents():
                    texts = [
                        f"{section.heading_path}: {unit}"
                        for section in cold.db.get_sections(
                            cold.db.get_document(summary.file_path).id  # type: ignore[union-attr]
                        )
                        for unit in _units(cold.db, section.id)
                    ]
                    expected = cold.db.passage_vectors(summary.file_path, texts, "fake-weights")
                    got = warm.passage_vectors(summary.file_path, texts, "fake-weights")
                    assert set(got) == set(expected) == set(texts)
                    for text, vector in expected.items():
                        pairs = zip(got[text], vector, strict=True)
                        assert max(abs(a - b) for a, b in pairs) < 1e-6
        finally:
            cold.close()

    def test_only_identical_input_under_identical_weights_and_format_is_reused(
        self, primary: MarkdownMemoryService, repo: Path
    ) -> None:
        shared = str((repo / "docs" / "shared.md").resolve())
        text = "Shared > Setup: Install the toolchain and run the bootstrap script."
        db = primary.db
        assert set(db.passage_vectors(shared, [text], "fake-weights")) == {text}
        assert db.passage_vectors(shared, [text], "other-weights") == {}
        assert db.passage_vectors(shared, ["Elsewhere > Setup: " + text[16:]], "fake-weights") == {}
        with db.transaction() as conn:
            conn.execute("UPDATE documents SET vector_format = 1 WHERE file_path = ?", (shared,))
        assert db.passage_vectors(shared, [text], "fake-weights") == {}

    def test_a_vector_that_does_not_decode_to_a_usable_one_is_embedded_instead(
        self, primary: MarkdownMemoryService, repo: Path
    ) -> None:
        shared = str((repo / "docs" / "shared.md").resolve())
        text = "Shared > Setup: Install the toolchain and run the bootstrap script."
        with primary.db.transaction() as conn:
            conn.execute(
                "UPDATE units_vec SET embedding = ?",
                (bytes(4 * primary.db.embedding_dim),),
            )
        assert primary.db.passage_vectors(shared, [text], "fake-weights") == {}

    async def test_a_model_that_names_its_weights_once_loaded_still_reuses(
        self, tmp_path: Path, repo: Path, worktree: Path
    ) -> None:
        class LoadsToKnow(FakeEmbedder):
            def __init__(self) -> None:
                super().__init__()
                self.loaded = False

            @property
            def weights_revision(self) -> str | None:
                return "fake-weights" if self.loaded else None

            def warm_up(self) -> None:
                self.loaded = True

        embedder = LoadsToKnow()
        embedder.loaded = True
        primary = MarkdownMemoryService(
            ServerConfig(db_path=tmp_path / "p.db", docs_dir=repo / "docs"), embedder=embedder
        )
        try:
            primary.index_directory()
            embedder.loaded = False  # a fresh server: nothing has loaded the model yet
            embedder.document_calls.clear()
            async with Client(create_server(service=primary)) as client:
                await call(client, "index_directory", cwd=str(worktree))
        finally:
            primary.close()
        embedded = [text for batch in embedder.document_calls for text in batch]
        assert embedded and not any("bootstrap script" in text for text in embedded)

    def test_weights_nobody_can_name_are_never_matched(self, tmp_path: Path) -> None:
        asked: list[str] = []
        docs = write(tmp_path / "docs" / "a.md", "# A\n\nSome text.\n").parent
        with Database(tmp_path / "i.db") as db:
            Indexer(
                db,
                FakeEmbedder(weights=None),
                reuse=lambda path, texts, weights: asked.append(path) or {},
            ).index_directory(docs)
        assert asked == []


def _units(db: Database, section_id: int) -> list[str]:
    with db.transaction() as conn:
        rows = conn.execute(
            "SELECT content FROM units WHERE section_id = ? ORDER BY ordinal", (section_id,)
        ).fetchall()
    return [str(row[0]) for row in rows]
