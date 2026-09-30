"""What a run of one documentation root owns: its own files, not other checkouts, not what git
ignores, and each file once however many symlinks point at it.

Found in the field: a monorepo with 38 git worktrees under `.claude/worktrees/`, rules
symlinked from `.claude/` into `.agents/`, and gitignored generated output. The index came out
mostly stale copies and duplicates, and the real `docs/` had not been reached minutes later.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fakes import FakeEmbedder

from markdown_memory import discovery
from markdown_memory.config import ServerConfig, resolve_config
from markdown_memory.db import Database
from markdown_memory.exceptions import DatabaseError, IndexCancelled
from markdown_memory.indexer import Indexer
from markdown_memory.server import MarkdownMemoryService

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _write(path: Path, text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or f"# {path.stem}\n\nbody of {path.stem}\n", encoding="utf-8")
    return path


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.email=t@t", "-c", "user.name=t", *args],
        check=True,
        capture_output=True,
    )


def _repo(root: Path, ignore: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")
    (root / ".gitignore").write_text(ignore, encoding="utf-8")
    return root


def _indexed(db: Database, root: Path) -> list[str]:
    return sorted(os.path.relpath(path, root) for path in db.document_hashes(str(root)))


class TestOtherCheckouts:
    def test_a_worktree_and_a_nested_clone_are_left_out_and_their_copies_purged(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        _write(tmp_path / "docs" / "a.md")
        worktree = _write(tmp_path / ".claude" / "worktrees" / "wt" / "docs" / "a.md").parents[1]
        clone = _write(tmp_path / "vendor-clone" / "README.md").parent
        Indexer(db, fake_embedder).index_directory(tmp_path)
        assert len(_indexed(db, tmp_path)) == 3  # what 0.1.1 left behind

        (worktree / ".git").write_text("gitdir: /elsewhere\n", encoding="utf-8")  # a file
        (clone / ".git").mkdir()  # a directory
        report = Indexer(db, fake_embedder).index_directory(tmp_path)
        assert _indexed(db, tmp_path) == ["docs/a.md"]
        assert report.files_purged == 2

    def test_a_checkout_pointed_at_directly_is_still_indexed(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        worktree = tmp_path / "wt"
        _write(worktree / "guide.md")
        (worktree / ".git").write_text("gitdir: /elsewhere\n", encoding="utf-8")
        Indexer(db, fake_embedder).index_directory(worktree)
        assert _indexed(db, worktree) == ["guide.md"]


@needs_git
class TestGitIgnore:
    def test_what_git_ignores_is_left_out_and_purged_but_tracked_files_stay(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        root = _repo(tmp_path / "repo", "gen/\nnotes.md\n[[]id]/\nforced.md\n")
        _write(root / "docs" / "a.md")
        _write(root / "gen" / "api.md")
        _write(root / "notes.md")
        _write(root / "[id]" / "route.md")  # a literal name, not a character class
        _write(root / "forced.md")
        _git(root, "add", "-f", "forced.md")  # tracked, so git does not call it ignored
        Indexer(db, fake_embedder, gitignore=False).index_directory(root)
        assert len(_indexed(db, root)) == 5

        report = Indexer(db, fake_embedder).index_directory(root)
        assert _indexed(db, root) == ["docs/a.md", "forced.md"]
        assert report.files_purged == 3

    def test_a_directory_that_became_an_ignored_symlink_takes_its_rows_with_it(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """git names an ignored symlinked directory as a file (`vendor`), not `vendor/`."""
        root = _repo(tmp_path / "repo", "vendor\n")
        _write(root / "docs" / "a.md")
        _write(root / "vendor" / "old.md")
        Indexer(db, fake_embedder, gitignore=False).index_directory(root)
        shutil.rmtree(root / "vendor")
        elsewhere = _write(tmp_path / "elsewhere" / "old.md").parent
        (root / "vendor").symlink_to(elsewhere, target_is_directory=True)

        Indexer(db, fake_embedder).index_directory(root)
        assert _indexed(db, root) == ["docs/a.md"]

    def test_a_root_git_ignores_is_indexed_when_asked_for(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        root = _repo(tmp_path / "repo", "gen/\n")
        generated = _write(root / "gen" / "api.md").parent
        Indexer(db, fake_embedder).index_directory(generated)
        assert _indexed(db, generated) == ["api.md"]

    def test_explicitly_indexed_dependency_docs_survive_a_root_run(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """`node_modules` is pruned, not disowned, even when git ignores it too."""
        root = _repo(tmp_path / "repo", "node_modules/\n")
        _write(root / "docs" / "a.md")
        package = _write(root / "node_modules" / "pkg" / "README.md").parent
        Indexer(db, fake_embedder).index_directory(package)
        Indexer(db, fake_embedder).index_directory(root)
        assert _indexed(db, root) == ["docs/a.md", "node_modules/pkg/README.md"]


class TestWithoutGit:
    """The fallback: whatever keeps git from answering, the walk is what it was before."""

    def test_outside_a_repository_nothing_changes(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        assert discovery.git_ignored(tmp_path) is None
        _write(tmp_path / "gen" / "api.md")
        Indexer(db, fake_embedder).index_directory(tmp_path)
        assert _indexed(db, tmp_path) == ["gen/api.md"]

    @pytest.mark.parametrize(
        "failure",
        [FileNotFoundError("git"), subprocess.TimeoutExpired("git", 10)],
        ids=["not-installed", "hung"],
    )
    def test_git_missing_or_hung_is_no_answer_not_a_failed_run(
        self,
        db: Database,
        fake_embedder: FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        failure: Exception,
    ) -> None:
        def refuse(*args: object, **kwargs: object) -> bytes:
            raise failure

        monkeypatch.setattr(subprocess, "check_output", refuse)
        _write(tmp_path / "a.md")
        try:
            report = Indexer(db, fake_embedder).index_directory(tmp_path)
        except (OSError, subprocess.SubprocessError) as exc:
            raise AssertionError("asking git took the whole run down") from exc
        assert report.errors == ()
        assert _indexed(db, tmp_path) == ["a.md"]

    def test_the_server_passes_the_switch_to_its_indexer(self, tmp_path: Path) -> None:
        config = ServerConfig(db_path=tmp_path / "i.db", docs_dir=tmp_path, gitignore=False)
        service = MarkdownMemoryService(config, embedder=FakeEmbedder())
        try:
            assert service._indexer._gitignore is False
        finally:
            service.close()

    def test_it_can_be_switched_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MARKDOWN_MEMORY_GITIGNORE", "0")
        assert not ServerConfig.from_env().gitignore
        assert resolve_config(gitignore=True).gitignore, "an explicit choice wins"
        monkeypatch.delenv("MARKDOWN_MEMORY_GITIGNORE")
        assert ServerConfig.from_env().gitignore


class TestSymlinkedFiles:
    def test_a_link_to_a_file_already_indexed_is_not_indexed_again(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        rule = _write(tmp_path / ".agents" / "rules" / "hitl.md")
        link = tmp_path / ".claude" / "rules" / "hitl.md"
        link.parent.mkdir(parents=True)
        link.symlink_to(os.path.relpath(rule, link.parent))
        # While the target is excluded, the link is the only way to that text: kept.
        Indexer(db, fake_embedder, exclude=(".agents",)).index_directory(tmp_path)
        assert _indexed(db, tmp_path) == [".claude/rules/hitl.md"]

        report = Indexer(db, fake_embedder).index_directory(tmp_path)
        assert _indexed(db, tmp_path) == [".agents/rules/hitl.md"]
        assert report.files_purged == 1

    def test_links_to_what_is_not_indexed_here_are_kept(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        root = tmp_path / "root"
        _write(root / "content.txt", "# Text\n\nnot markdown by name\n")
        (root / "alias.md").symlink_to("content.txt")
        outside = _write(tmp_path / "outside" / "shared.md")
        (root / "shared.md").symlink_to(outside)
        Indexer(db, fake_embedder).index_directory(root)
        assert _indexed(db, root) == ["alias.md", "shared.md"]

    @pytest.mark.skipif(sys.platform == "darwin", reason="APFS refuses names that are not UTF-8")
    def test_a_link_to_a_file_that_cannot_be_indexed_stands_in_for_it(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """The real name is not UTF-8, so it fails; the link is the only way to its text."""
        real = os.fsdecode(b"\xff-notes.md")
        (tmp_path / real).write_text("# Notes\n\nbody\n", encoding="utf-8")
        (tmp_path / "notes.md").symlink_to(real)
        report = Indexer(db, fake_embedder).index_directory(tmp_path)
        assert _indexed(db, tmp_path) == ["notes.md"]
        assert len(report.errors) == 1

    def test_the_walk_keeps_its_order(self, tmp_path: Path) -> None:
        paths = [_write(tmp_path / name) for name in ("b.md", "a.md", "c.md")]
        (tmp_path / "z.md").symlink_to("a.md")
        assert discovery.without_aliases([*paths, tmp_path / "z.md"]) == paths


class TestDanglingLinks:
    def test_a_dangling_link_is_a_failure_every_run_not_a_crash_on_the_second(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        _write(tmp_path / "a.md")
        (tmp_path / "broken.md").symlink_to("missing.md")
        for _ in range(2):
            report = Indexer(db, fake_embedder).index_directory(tmp_path)
            assert [Path(f.file_path).name for f in report.errors] == ["broken.md"]
        assert len(db.index_status(str(tmp_path)).failures) == 1

    def test_recording_a_failure_twice_restates_it(self, db: Database, tmp_path: Path) -> None:
        path = str(tmp_path / "x.md")
        db.record_failures([], {path: "first"})
        try:
            db.record_failures([], {path: "second"})
        except DatabaseError as exc:
            raise AssertionError("a row recorded twice failed the whole run") from exc
        assert [f.message for f in db.index_status(str(tmp_path)).failures] == ["second"]


class TestTheWalk:
    def test_the_projects_own_docs_come_before_dot_directories(self, tmp_path: Path) -> None:
        _write(tmp_path / ".agents" / "rule.md")
        _write(tmp_path / "docs" / "a.md")
        walked = [
            p.relative_to(tmp_path).as_posix() for p in discovery.iter_markdown_files(tmp_path)
        ]
        assert walked == ["docs/a.md", ".agents/rule.md"]

    def test_a_stop_is_honoured_during_the_walk(self, tmp_path: Path) -> None:
        _write(tmp_path / "a.md")
        with pytest.raises(IndexCancelled):
            list(discovery.iter_markdown_files(tmp_path, should_stop=lambda: True))
