"""Which work tree a path belongs to, and which documentation root answers for it there.

An agent in a linked worktree - `claude -w`, Codex in a worktree, a subagent isolated in
one - shares this server with the checkout it was started for, and was answered from that
checkout's documentation: another branch's. Every tool therefore takes the agent's working
directory, and a path in another work tree of the *same repository* is answered from that
tree's own copy of the docs root, indexed into its own database.

Git is asked, never parsed. Where a worktree's administrative files live, whether `gitdir`
and `commondir` are relative, what a submodule or a bare repository looks like: git is the
authority on its own layout, and is already what decides what is ignored.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from markdown_memory import discovery
from markdown_memory.exceptions import WorkTreeError

#: Other work trees one server will serve at once, beside its configured root. Each holds a
#: database and a search pool open until the session ends; nothing is evicted, because
#: closing a service under a read in flight is a race no cap is worth.
MAX_TREES = 8


#: What git says when a path is in no work tree at all: outside every repository, or in a
#: bare one, or inside a `.git` directory. Such a call is answered from the configured root.
_NOT_A_WORK_TREE = ("not a git repository (or any", "must be run in a work tree")


@dataclass(frozen=True, slots=True)
class Tree:
    """A work tree, and the repository it is a checkout of (its common git directory)."""

    top: Path
    common_dir: Path


def work_tree(path: Path) -> Tree | None:
    """The work tree holding ``path`` (absolute); ``None`` when it is in none, or no git.

    Any other answer from git - a worktree whose administrative directory was removed, a
    repository git refuses as unsafe - is an error rather than ``None``: falling back to
    the configured root would answer from another checkout without saying so, which is the
    failure this module exists to end.
    """
    # A file, or a file that is gone, is asked about through the nearest directory that
    # exists: a deleted document still names the tree it was indexed from.
    directory = path
    while not directory.is_dir():
        if directory.parent == directory:
            raise WorkTreeError(f"No directory of {discovery._printable(str(path))} exists")
        directory = directory.parent
    try:
        listed = subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "--path-format=absolute",
             "--show-toplevel", "--git-common-dir"],
            stdin=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=10,
            env=discovery.git_environment(),
        )  # fmt: skip
    except FileNotFoundError:
        return None
    except subprocess.CalledProcessError as exc:
        reason = os.fsdecode(exc.stderr or b"").strip()
        if any(marker in reason for marker in _NOT_A_WORK_TREE):
            return None
        first = reason.splitlines()[0] if reason else f"exit {exc.returncode}"
        raise WorkTreeError(f"git could not say which work tree holds {path}: {first}") from None
    except (OSError, subprocess.SubprocessError) as exc:
        raise WorkTreeError(f"git could not say which work tree holds {path}: {exc}") from exc
    lines = os.fsdecode(listed).splitlines()
    if len(lines) != 2:
        raise WorkTreeError(f"git gave an unexpected answer for {path}: {lines!r}")
    return Tree(Path(lines[0]).resolve(), Path(lines[1]).resolve())


def counterpart(docs_dir: Path, home: Tree, other: Tree) -> Path:
    """Where ``other`` keeps what ``docs_dir`` holds in ``home``: the same path inside it."""
    try:
        relative = docs_dir.relative_to(home.top)
    except ValueError:
        raise WorkTreeError(
            f"The documentation root {docs_dir} is not inside its work tree {home.top}, so "
            f"it has no counterpart in {other.top}."
        ) from None
    docs = other.top / relative
    if not docs.is_dir():
        raise WorkTreeError(
            f"{other.top} is a work tree of the same repository, but it has no {relative} "
            f"directory to answer from. Call without `cwd` to search {docs_dir}."
        )
    return docs.resolve()
