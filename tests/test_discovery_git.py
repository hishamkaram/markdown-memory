"""Running git where it calls a checkout's repository bare (#70).

Some agent worktree tooling sets `core.bare = true` on a repository's main checkout. Git then
refuses every work-tree command there with "this operation must be run in a work tree", and
markdown-memory, which asks git what it ignores and which work tree a path is in, indexed
ignored output and answered a worktree's questions from the main checkout without saying so.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import pytest
from fakes import FakeEmbedder

import markdown_memory.discovery as discovery
from markdown_memory.config import ServerConfig
from markdown_memory.discovery import git_ignored
from markdown_memory.server import MarkdownMemoryService, _ServiceProvider
from markdown_memory.trees import Tree, work_tree

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "-c", "user.email=t@t", "-c", "user.name=t",
         "-c", "protocol.file.allow=always", *args],
        text=True,
        stderr=subprocess.STDOUT,
    )  # fmt: skip


def write(path: Path, text: str = "# T\n\ntext\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _forget_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(discovery, "_warned", set())


@pytest.fixture
def main(tmp_path: Path) -> Path:
    """A main checkout whose repository says it is bare, with a linked worktree and output git
    ignores below the docs root."""
    root = tmp_path / "main"
    root.mkdir()
    git(root, "init", "-q")
    write(root / "docs" / "guide.md")
    write(root / ".gitignore", "docs/gen/\n")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "docs")
    write(root / "docs" / "gen" / "out.md")
    git(root, "worktree", "add", "-q", "-b", "w", str(root / ".claude" / "worktrees" / "w"))
    git(root, "config", "core.bare", "true")
    return root


class TestABareConfiguredCheckout:
    def test_git_still_says_what_it_ignores(self, main: Path) -> None:
        assert git_ignored(main / "docs") == discovery.GitIgnore("applied", frozenset({"gen"}))

    def test_git_still_says_whether_one_file_is_ignored(self, main: Path) -> None:
        docs = main / "docs"
        assert discovery._git_ignores(docs / "gen" / "out.md", docs)
        assert not discovery._git_ignores(docs / "guide.md", docs)

    def test_the_checkout_is_still_a_work_tree(self, main: Path) -> None:
        assert work_tree(main / "docs") == Tree(main.resolve(), (main / ".git").resolve())

    def test_a_symlink_into_it_finds_the_same_checkout(self, main: Path, tmp_path: Path) -> None:
        (tmp_path / "link").symlink_to(main / "docs")
        assert work_tree(tmp_path / "link") == Tree(main.resolve(), (main / ".git").resolve())

    def test_it_is_logged_once_with_the_fix_nobody_ran(
        self, main: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="markdown_memory.discovery"):
            git_ignored(main / "docs")
            work_tree(main / "docs")
        warnings = [r.getMessage() for r in caplog.records if "core.bare" in r.getMessage()]
        assert len(warnings) == 1
        assert f"git -C {main.resolve()} config core.bare false" in warnings[0]
        assert git(main, "config", "--get", "core.bare").strip() == "true"  # left as it was

    def test_a_cwd_in_its_worktree_is_answered_from_the_worktree(self, main: Path) -> None:
        """#69's routing, which this setting turned off without a word."""
        service = MarkdownMemoryService(
            ServerConfig(db_path=main.parent / "i.db", docs_dir=main / "docs"),
            embedder=FakeEmbedder(),
        )
        provider = _ServiceProvider(None, service)
        provider.session_started()
        try:
            tree = provider.get(str(main / ".claude" / "worktrees" / "w"))
            assert tree.root == str((main / ".claude" / "worktrees" / "w" / "docs").resolve())
        finally:
            provider.session_ended()
            service.close()


class TestOnlyACheckoutIsReadAsOne:
    def test_a_real_bare_repository_is_still_in_no_work_tree(self, tmp_path: Path) -> None:
        git(tmp_path, "init", "-q", "--bare", "real.git")
        assert work_tree(tmp_path / "real.git") is None
        assert git_ignored(tmp_path / "real.git").state == "unavailable"

    def test_inside_the_git_directory_is_in_no_work_tree(self, main: Path) -> None:
        assert work_tree(main / ".git" / "refs") is None

    def test_a_linked_worktree_of_it_needs_no_help(self, main: Path) -> None:
        tree = main / ".claude" / "worktrees" / "w"
        assert work_tree(tree) == Tree(tree.resolve(), (main / ".git").resolve())

    def test_a_submodule_configured_bare_is_read_from_its_own_directory(
        self, tmp_path: Path
    ) -> None:
        library = tmp_path / "library"
        library.mkdir()
        git(library, "init", "-q")
        git(library, "commit", "-q", "--allow-empty", "-m", "init")
        top = tmp_path / "top"
        top.mkdir()
        git(top, "init", "-q")
        git(top, "submodule", "add", "-q", str(library), "lib")
        write(top / "lib" / ".gitignore", "gen/\n")
        write(top / "lib" / "gen" / "x.md")
        git(top / "lib", "config", "core.bare", "true")
        assert git_ignored(top / "lib") == discovery.GitIgnore("applied", frozenset({"gen"}))
        assert work_tree(top / "lib") == Tree(
            (top / "lib").resolve(), (top / ".git" / "modules" / "lib").resolve()
        )

    def test_a_separate_git_dir_checkout_configured_bare(self, tmp_path: Path) -> None:
        checkout = tmp_path / "checkout"
        git(tmp_path, "init", "-q", f"--separate-git-dir={tmp_path / 'store'}", str(checkout))
        write(checkout / ".gitignore", "gen/\n")
        write(checkout / "gen" / "x.md")
        git(checkout, "config", "core.bare", "true")
        assert git_ignored(checkout) == discovery.GitIgnore("applied", frozenset({"gen"}))
        assert work_tree(checkout) == Tree(checkout.resolve(), (tmp_path / "store").resolve())
