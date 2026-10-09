"""Why a document is not in the index, told to the agent that asked for it (#70).

Every miss used to say "Run index_directory" - advice that cannot help a path outside the
root, a file that does not exist, or one the root's own runs leave out and would purge again.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fakes import FakeEmbedder
from mcp import Client

from markdown_memory.config import ServerConfig
from markdown_memory.models import IndexStatus
from markdown_memory.server import SERVER_INSTRUCTIONS, MarkdownMemoryService, create_server

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def git(root: Path, *args: str) -> None:
    subprocess.check_output(
        ["git", "-C", str(root), "-c", "user.email=t@t", "-c", "user.name=t", *args],
        stderr=subprocess.STDOUT,
    )


def write(path: Path, text: str = "# T\n\n## Part\n\ntext\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    write(root / ".gitignore", "gen/\nnode_modules/\n-dashed.md\n:(top)magic.md\n")
    write(root / "a.md")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "docs")
    git(root, "worktree", "add", "-q", "-b", "w", str(root / ".claude" / "worktrees" / "w"))
    write(root / "gen" / "out.md")
    write(root / "gen" / "deep" / "er.md")  # a pathspec this deep made git fail
    write(root / "node_modules" / "pkg" / "readme.md")
    write(root / "vendor" / "v.md")
    write(root / "notes.txt", "plain text\n")
    write(root / "-dashed.md")  # read by git as an option, unless after `--`
    write(root / ":(top)magic.md")  # read by git as pathspec magic, unless literal
    (root / "broken.md").symlink_to(root / "nowhere.md")
    return root


def make(tmp_path: Path, repo: Path, *, gitignore: bool = True) -> MarkdownMemoryService:
    service = MarkdownMemoryService(
        ServerConfig(
            db_path=tmp_path / f"i-{gitignore}.db",
            docs_dir=repo,
            exclude=("vendor",),
            gitignore=gitignore,
        ),
        embedder=FakeEmbedder(),
    )
    service.index_directory()
    write(repo / "later.md")  # in scope, written after the run
    return service


@pytest.fixture
def service(tmp_path: Path, repo: Path) -> Iterator[MarkdownMemoryService]:
    instance = make(tmp_path, repo)
    yield instance
    instance.close()


async def failures(service: MarkdownMemoryService, file_path: str, **extra: Any) -> list[str]:
    """The error of an outline and of a read of ``file_path``, which must agree."""
    said = []
    async with Client(create_server(service=service)) as client:
        for name, arguments in (
            ("get_document_outline", {"file_path": file_path}),
            ("read_section", {"file_path": file_path, "heading_path": "Part"}),
        ):
            outcome = await client.call_tool(name, arguments | extra)
            assert outcome.is_error, outcome.content
            text = str(outcome.content[0].text)  # type: ignore[union-attr]
            said.append(text.split(f"tool {name}: ", 1)[-1])
    assert said[0] == said[1]
    return said


class TestTheReasonIsGiven:
    @pytest.mark.parametrize(
        ("relative", "expected"),
        [
            ("missing.md", "does not exist"),
            ("broken.md", "does not exist"),
            ("gen", "is not a Markdown document"),
            ("notes.txt", "is not a Markdown document"),
            ("node_modules/pkg/readme.md", "which every walk of the root skips"),
            ("vendor/v.md", "left out of its index by an exclusion pattern"),
            ("gen/out.md", "left out of its index by git ignoring it"),
            ("gen/deep/er.md", "left out of its index by git ignoring it"),
            ("-dashed.md", "left out of its index by git ignoring it"),
            (":(top)magic.md", "left out of its index by git ignoring it"),
            ("later.md", "is not indexed. Run index_directory"),
        ],
    )
    async def test_for_a_path_the_agent_spelled(
        self, service: MarkdownMemoryService, repo: Path, relative: str, expected: str
    ) -> None:
        said = (await failures(service, str(repo / relative)))[0]
        assert expected in said, said
        if relative == "node_modules/pkg/readme.md":
            assert f"index_directory('{(repo / 'node_modules').resolve()}')" in said

    async def test_a_linked_worktree_named_relative_to_the_root_is_pointed_at_cwd(
        self, service: MarkdownMemoryService, repo: Path
    ) -> None:
        said = (await failures(service, ".claude/worktrees/w/a.md"))[0]
        worktree = (repo / ".claude" / "worktrees" / "w").resolve()
        assert f"is inside the linked worktree {worktree}" in said
        assert "pass that worktree as cwd" in said

    async def test_its_absolute_path_is_its_own_index_where_indexing_is_the_answer(
        self, service: MarkdownMemoryService, repo: Path
    ) -> None:
        """#69 routes it to the worktree's own index, so there the old advice is right."""
        said = (await failures(service, str(repo / ".claude" / "worktrees" / "w" / "a.md")))[0]
        assert "is not indexed. Run index_directory" in said

    async def test_outside_the_root_says_so_and_points_at_cwd(
        self, service: MarkdownMemoryService, tmp_path: Path
    ) -> None:
        elsewhere = write(tmp_path / "elsewhere" / "x.md")
        said = (await failures(service, str(elsewhere)))[0]
        assert "is outside" in said and "pass that worktree as cwd" in said
        assert "index_directory" not in said

    async def test_a_relative_path_is_judged_where_it_points(
        self, service: MarkdownMemoryService, repo: Path
    ) -> None:
        said = (await failures(service, "docs/typo.md", cwd=str(repo)))[0]
        assert f"'{repo / 'docs' / 'typo.md'}' does not exist" in said

    async def test_the_candidate_that_exists_is_the_one_judged(
        self, service: MarkdownMemoryService, tmp_path: Path
    ) -> None:
        # Not under cwd, but under the docs root: that file is what the agent meant.
        said = (await failures(service, "gen/out.md", cwd=str(tmp_path)))[0]
        assert "left out of its index by git ignoring it" in said
        # Not under the docs root, which is tried first (#76), but under cwd: that one.
        write(tmp_path / "notes" / "loose.md")
        said = (await failures(service, "notes/loose.md", cwd=str(tmp_path)))[0]
        assert "is outside" in said and "does not exist" not in said

    async def test_a_link_is_judged_where_it_is_not_where_it_points(
        self, service: MarkdownMemoryService, repo: Path, tmp_path: Path
    ) -> None:
        (repo / "link.md").symlink_to(write(tmp_path / "elsewhere" / "target.md"))
        said = (await failures(service, str(repo / "link.md")))[0]
        assert "is not indexed. Run index_directory" in said  # a walk indexes the link

    async def test_a_nested_clones_own_ignore_rules_are_not_the_roots(
        self, service: MarkdownMemoryService, repo: Path
    ) -> None:
        nested = repo / "nested"
        nested.mkdir()
        git(nested, "init", "-q")
        write(nested / ".gitignore", "*.md\n")
        said = (await failures(service, str(write(nested / "page.md"))))[0]
        assert "is not indexed. Run index_directory" in said  # a walk of the root indexes it

    async def test_a_root_git_ignores_whole_ignores_nothing_below_it(
        self, tmp_path: Path, repo: Path
    ) -> None:
        service = MarkdownMemoryService(
            ServerConfig(db_path=tmp_path / "gen.db", docs_dir=repo / "gen"),
            embedder=FakeEmbedder(),
        )
        try:
            service.index_directory()
            said = (await failures(service, str(write(repo / "gen" / "fresh.md"))))[0]
        finally:
            service.close()
        assert "is not indexed. Run index_directory" in said

    async def test_a_bare_name_is_a_suffix_nothing_matched(
        self, service: MarkdownMemoryService
    ) -> None:
        said = (await failures(service, "nothing-like-this.md"))[0]
        assert said.startswith("No indexed document matches 'nothing-like-this.md'")

    async def test_ignored_output_is_not_called_ignored_when_git_is_not_asked(
        self, tmp_path: Path, repo: Path
    ) -> None:
        service = make(tmp_path, repo, gitignore=False)
        try:
            service.index_directory()  # a run that applies no ignore rules indexes gen/
            (repo / "gen" / "second.md").write_text("# S\n\n## Part\n\nx\n", encoding="utf-8")
            said = (await failures(service, str(repo / "gen" / "second.md")))[0]
        finally:
            service.close()
        assert "is not indexed. Run index_directory" in said

    async def test_a_run_in_progress_is_named_instead_of_asking_for_one(
        self, service: MarkdownMemoryService, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(service, "_auto", SimpleNamespace(active=True))
        said = (await failures(service, str(repo / "later.md")))[0]
        assert "a run is indexing" in said and "Run index_directory" not in said


class TestGitStateIsSpelledOut:
    def test_unavailable_names_a_command_the_agent_can_run(self) -> None:
        message = IndexStatus(verified=True, gitignore="unavailable", root="/r").message() or ""
        assert "`git -C /r status`" in message and "server log" not in message

    async def test_every_value_is_defined_where_agents_read(self, tmp_path: Path) -> None:
        service = MarkdownMemoryService(
            ServerConfig(db_path=tmp_path / "d.db", docs_dir=tmp_path), embedder=FakeEmbedder()
        )
        try:
            tools = {t.name: t for t in await create_server(service=service).list_tools()}
        finally:
            service.close()
        values = ("applied", "off", "no_repository", "unavailable", "unknown")
        for name in ("search_docs", "list_documents"):
            described = tools[name].description or ""
            assert all(f"`{value}`" in described for value in values), name
        assert all(value in SERVER_INSTRUCTIONS for value in values)
        assert "unknown - no run has recorded" in SERVER_INSTRUCTIONS

    async def test_the_first_thing_an_agent_reads_is_when_to_search(self, tmp_path: Path) -> None:
        """#126: Codex keeps the instructions' first 512 characters for choosing a server, and
        Claude sees only the instructions and tool names until it loads a tool's schema."""
        lead = (
            "Search this project's Markdown documentation before grepping or reading .md files: "
            "search_docs(query) returns the best-matching section - or an excerpt of the "
            "passage that matched - with pointers to the next best, and read_section fetches "
            "any one of them by heading. Use it for every question the documentation might "
            "answer, with exact identifiers (flags, env vars, config keys) or plain words; for "
            "an identifier, it also says when no indexed section contains it."
        )
        assert len(lead) <= 512
        assert SERVER_INSTRUCTIONS.startswith(lead + " ")
        service = MarkdownMemoryService(
            ServerConfig(db_path=tmp_path / "d.db", docs_dir=tmp_path), embedder=FakeEmbedder()
        )
        try:
            tools = {t.name: t for t in await create_server(service=service).list_tools()}
        finally:
            service.close()
        # The SDK sends the docstring with its line breaks and indentation.
        assert " ".join((tools["search_docs"].description or "").split()).startswith(
            "Search this project's Markdown documentation for the best-matching section - call "
            "it first for any question the docs might answer, before grep or reading .md files."
        )
