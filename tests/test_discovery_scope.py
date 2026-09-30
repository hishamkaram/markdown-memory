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
from markdown_memory.models import IndexStatus
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


def _linked_worktree(directory: Path, repository: Path, *, relative: bool = False) -> None:
    """What `git worktree add` leaves: a `.git` file naming a gitdir that has a `commondir`."""
    gitdir = repository / ".git" / "worktrees" / directory.name
    gitdir.mkdir(parents=True)
    (gitdir / "commondir").write_text("../..\n", encoding="utf-8")
    named = os.path.relpath(gitdir, directory) if relative else str(gitdir)
    (directory / ".git").write_text(f"gitdir: {named}\n", encoding="utf-8")


class TestOtherCheckouts:
    @pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
    def test_a_linked_worktree_is_left_out_and_its_copies_purged(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path, relative: bool
    ) -> None:
        _write(tmp_path / "docs" / "a.md")
        worktree = _write(tmp_path / ".claude" / "worktrees" / "wt" / "docs" / "a.md").parents[1]
        Indexer(db, fake_embedder).index_directory(tmp_path)
        assert len(_indexed(db, tmp_path)) == 2  # what 0.1.1 left behind

        _linked_worktree(worktree, tmp_path / "elsewhere", relative=relative)
        report = Indexer(db, fake_embedder).index_directory(tmp_path)
        assert _indexed(db, tmp_path) == ["docs/a.md"]
        assert report.files_purged == 1
        assert db.index_status(str(tmp_path)).verified, "the upgrade did not converge"

    @pytest.mark.parametrize(
        "shape",
        [
            "submodule",
            "clone",
            "garbled",
            "empty",
            "dangling",
            "commondir-directory",
            "no-path",
            "no-space",
            "bare-path",
            "leading-space",
        ],
    )
    def test_a_checkout_not_known_to_be_a_copy_is_indexed(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path, shape: str
    ) -> None:
        """A `docs/` submodule or a hub of clones is content; 0.1.2 dropped it without a word."""
        _write(tmp_path / "docs" / "a.md")
        nested = _write(tmp_path / "sub" / "guide.md").parent
        modules = tmp_path / ".git" / "modules" / "sub"
        if shape == "submodule":  # its gitdir is a whole repository: no `commondir`
            modules.mkdir(parents=True)
            (nested / ".git").write_text("gitdir: ../.git/modules/sub\n", encoding="utf-8")
        elif shape == "clone":
            (nested / ".git").mkdir()
        elif shape == "garbled":
            (nested / ".git").write_bytes(b"\xff\x00 not a gitfile\n")
        elif shape == "empty":
            (nested / ".git").write_bytes(b"")
        elif shape in ("no-path", "no-space", "bare-path", "leading-space"):  # not a gitfile
            (nested / "commondir").write_text("..\n", encoding="utf-8")
            (modules / "commondir").parent.mkdir(parents=True)
            (modules / "commondir").write_text("../..\n", encoding="utf-8")
            named = {
                "no-path": "gitdir: ",
                "no-space": f"gitdir:{modules}",
                "leading-space": f" gitdir: {modules}",
            }.get(shape, str(modules))
            named += "\n"
            (nested / ".git").write_text(named, encoding="utf-8")
        elif shape == "dangling":  # a worktree whose gitdir was pruned away
            (nested / ".git").write_text(f"gitdir: {tmp_path / 'gone'}\n", encoding="utf-8")
        else:
            (modules / "commondir").mkdir(parents=True)
            (nested / ".git").write_text(f"gitdir: {modules}\n", encoding="utf-8")
        Indexer(db, fake_embedder).index_directory(tmp_path)
        assert _indexed(db, tmp_path) == ["docs/a.md", "sub/guide.md"]

    def test_a_checkout_pointed_at_directly_is_still_indexed(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        worktree = tmp_path / "wt"
        _write(worktree / "guide.md")
        _linked_worktree(worktree, tmp_path / "repo")
        Indexer(db, fake_embedder).index_directory(worktree)
        assert _indexed(db, worktree) == ["guide.md"]

    @needs_git
    def test_a_real_git_worktree_is_left_out(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        root = _repo(tmp_path / "repo", "")
        _write(root / "docs" / "a.md")
        _git(root, "add", "-A")
        _git(root, "commit", "-qm", "one")
        _git(root, "worktree", "add", "-q", "wt", "-b", "wt")  # below the root, not ignored
        Indexer(db, fake_embedder).index_directory(root)
        assert _indexed(db, root) == ["docs/a.md"]


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


def _git_answers(monkeypatch: pytest.MonkeyPatch, failure: Exception) -> list[object]:
    calls: list[object] = []

    def refuse(*args: object, **kwargs: object) -> bytes:
        calls.append(args)
        raise failure

    monkeypatch.setattr(subprocess, "check_output", refuse)
    return calls


class TestWhatGitSaid:
    """Whatever keeps git from answering, the walk is what it was before git was asked - and
    a repository git could not be asked about is said to be one."""

    @needs_git
    def test_outside_a_repository_nothing_changes(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        assert discovery.git_ignored(tmp_path).state == "no_repository"
        _write(tmp_path / "gen" / "api.md")
        Indexer(db, fake_embedder).index_directory(tmp_path)
        assert _indexed(db, tmp_path) == ["gen/api.md"]
        status = db.index_status(str(tmp_path))
        assert (status.gitignore, status.message()) == ("no_repository", None)

    @pytest.mark.parametrize(
        ("failure", "repository", "state"),
        [
            (FileNotFoundError("git"), True, "unavailable"),
            (FileNotFoundError("git"), False, "no_repository"),
            (subprocess.TimeoutExpired("git", 10), True, "unavailable"),
            (
                subprocess.CalledProcessError(
                    128, "git", stderr=b"fatal: detected dubious ownership in repository at '/r'\n"
                ),
                True,
                "unavailable",
            ),
            (
                subprocess.CalledProcessError(
                    128,
                    "git",
                    stderr=b"fatal: not a git repository (or any of the parent "
                    b"directories): .git\n",
                ),
                False,
                "no_repository",
            ),
            (
                subprocess.CalledProcessError(
                    128, "git", stderr=b"fatal: not a git repository: /r/.git/worktrees/gone\n"
                ),
                True,
                "unavailable",
            ),
        ],
        ids=[
            "not-installed",
            "not-installed-no-repo",
            "hung",
            "refused",
            "not-a-repo",
            "bad-gitfile",
        ],
    )
    def test_a_failed_git_is_no_answer_and_says_so_only_for_a_repository(
        self,
        db: Database,
        fake_embedder: FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        failure: Exception,
        repository: bool,
        state: str,
    ) -> None:
        _git_answers(monkeypatch, failure)
        if repository:
            (tmp_path / ".git").mkdir()
        else:  # whatever lies above the test's temporary directory is not the test's
            monkeypatch.setattr(discovery, "_in_repository", lambda root: False)
        _write(tmp_path / "a.md")
        try:
            report = Indexer(db, fake_embedder).index_directory(tmp_path)
        except (OSError, subprocess.SubprocessError) as exc:
            raise AssertionError("asking git took the whole run down") from exc
        assert (report.errors, _indexed(db, tmp_path)) == ((), ["a.md"])
        status = db.index_status(str(tmp_path))
        assert status.verified
        assert status.gitignore == state
        said = [note for note in report.notes if "git could not list" in note]
        message = status.message() or ""
        if state == "unavailable":
            assert len(said) == 1, "a manual run was not told"
            assert "git could not list" in message, "the agent was not told"
        else:
            assert said == [], "a folder that is no repository was warned about"
            assert message == ""

    def test_switched_off_git_is_never_asked(
        self,
        db: Database,
        fake_embedder: FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = _git_answers(monkeypatch, FileNotFoundError("git"))
        (tmp_path / ".git").mkdir()
        _write(tmp_path / "a.md")
        Indexer(db, fake_embedder, gitignore=False).index_directory(tmp_path)
        assert calls == []
        assert db.index_status(str(tmp_path)).gitignore == "off"

    def test_the_state_is_the_last_finished_walks(
        self,
        db: Database,
        fake_embedder: FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _git_answers(monkeypatch, subprocess.TimeoutExpired("git", 10))
        (tmp_path / ".git").mkdir()
        _write(tmp_path / "a.md")
        assert db.index_status(str(tmp_path)).gitignore == "unknown"
        Indexer(db, fake_embedder).index_directory(tmp_path)
        assert db.index_status(str(tmp_path)).gitignore == "unavailable"
        db.revoke_coverage()  # the index was discarded: nothing walked what is there now
        assert db.index_status(str(tmp_path)).gitignore == "unknown"

    def test_only_a_whole_tree_mentions_git(self) -> None:
        """Missing files outrank extra ones: the unverified messages say more that matters."""
        assert "git could not list" in (
            IndexStatus(verified=True, gitignore="unavailable").message() or ""
        )
        assert IndexStatus(verified=True, gitignore="applied").message() is None
        assert "git" not in (IndexStatus(verified=False, gitignore="unavailable").message() or "")

    @needs_git
    def test_a_git_hook_s_environment_does_not_choose_the_repository(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _repo(tmp_path / "root", "gen/\n")
        _write(root / "gen" / "api.md")
        other = _repo(tmp_path / "other", "")
        monkeypatch.setenv("GIT_DIR", str(other / ".git"))
        monkeypatch.setenv("GIT_WORK_TREE", str(other))
        assert discovery.git_ignored(root).ignored == frozenset({"gen"})

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


@needs_git
class TestHostileNames:
    """git's paths come back NUL-separated and byte-exact (`-z`), so no name is special."""

    def test_a_newline_in_an_ignored_name_is_still_ignored(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        root = _repo(tmp_path / "repo", "gen*/\nnote*.md\n")
        _write(root / "docs" / "a.md")
        _write(root / "gen\nout" / "api.md")
        _write(root / "note\nb.md")
        Indexer(db, fake_embedder).index_directory(root)
        assert _indexed(db, root) == ["docs/a.md"]

    @pytest.mark.skipif(sys.platform == "darwin", reason="APFS refuses names that are not UTF-8")
    def test_an_ignored_name_that_is_not_utf8_is_neither_indexed_nor_a_failure(
        self, db: Database, fake_embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        root = _repo(tmp_path / "repo", "*-gen.md\n")
        _write(root / "docs" / "a.md")
        (root / os.fsdecode(b"\xff-gen.md")).write_text("# Gen\n\nbody\n", encoding="utf-8")
        report = Indexer(db, fake_embedder).index_directory(root)
        assert _indexed(db, root) == ["docs/a.md"]
        assert report.errors == ()
        assert db.index_status(str(root)).verified


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
