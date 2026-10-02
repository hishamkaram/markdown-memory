"""Finding the Markdown files to index, and reading them safely.

Everything here is about the filesystem and nothing about embedding: which paths a walk
should yield, which it must skip, and how to read one without being taken somewhere else
by a symlink or blocked forever by a FIFO.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import stat
import subprocess
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

from markdown_memory.exceptions import IndexCancelled

logger = logging.getLogger(__name__)

#: What ``_printable`` leaves where it could not decode a byte of a file name.
_UNDECODABLE = "�"


MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})


MAX_FILE_BYTES = 10 * 1024 * 1024


_SKIPPED_DIRECTORIES = frozenset(
    {
        ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__",
        ".mypy_cache", ".ruff_cache", ".pytest_cache", ".tox", "site-packages",
    }
)  # fmt: skip


def read_regular_file(path: Path) -> bytes | None:
    """The file's bytes, or ``None`` if what opened is not a regular file after all.

    `stat` and `open` are two moments, and between them a path can become a FIFO - at
    which point a blocking open waits for a writer that may never come, holding whatever
    the caller was holding: an indexing worker, or the lock a freshness sweep runs under.
    Opening without blocking and asking the descriptor itself what it is closes that
    window; `O_NONBLOCK` is then cleared, because it is only the open that must not block.

    At most ``MAX_FILE_BYTES + 1`` bytes, so the caller can tell "too large" from "exactly
    at the cap" without trusting `st_size`, which the file is free to disagree with.
    """
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return None
        os.set_blocking(descriptor, True)
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return handle.read(MAX_FILE_BYTES + 1)
    finally:
        os.close(descriptor)


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, slots=True)
class Scope:
    """What a run of one documentation root owns, and so what it may index and purge.

    Out of scope: what the operator excluded, what git ignores below the root, and anything
    inside a linked worktree below it. Each is a statement that the path is not this
    project's documentation, which is what an exclusion already meant here: its documents
    leave the index, rather than being kept the way a pruned `node_modules` is, where the
    walk merely learned nothing. A submodule or a nested clone is not a copy of anything and
    stays, unless git ignores it.
    """

    root: Path
    exclude: tuple[str, ...] = ()
    #: Paths git ignores, relative to the root, as `_printable` spells them, no trailing `/`.
    ignored: frozenset[str] = frozenset()

    def excludes(self, path: Path | str) -> bool:
        target = Path(path)
        if self.exclude and _is_excluded(target, self.root, self.exclude):
            return True
        try:
            parts = target.relative_to(self.root).parts
        except ValueError:
            return False
        # Every ancestor, not only the path: git reports an ignored directory that is now a
        # symlink as a *file*, and the rows recorded under it while it was real must go too.
        if self.ignored and any(
            _printable("/".join(parts[:end])) in self.ignored for end in range(1, len(parts) + 1)
        ):
            return True
        return _in_linked_worktree(target, self.root)


#: Variables that would point git at some other repository than the root's own: a server
#: started from a git hook inherits them.
_GIT_LOCATION_VARIABLES = ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE")


@dataclass(frozen=True, slots=True)
class GitIgnore:
    """What git said about a root: whether its ignore rules scope this run, and what they cover.

    ``state`` is `applied`, `off` (switched off), `no_repository` (nothing to ignore - the walk
    is already right) or `unavailable` (a repository git could not be asked about, so what it
    ignores is indexed as well; ``cause`` says why).
    """

    state: str
    ignored: frozenset[str] = frozenset()
    cause: str | None = None


def git_ignored(root: Path) -> GitIgnore:
    """What git ignores below ``root``.

    One `git ls-files` for the whole run: git's own ignore rules, global excludes and
    `.git/info/exclude` included, and nothing tracked, since only untracked paths are listed.
    Ignored directories come back collapsed (`coverage/`), so the set stays small. When git
    cannot answer, the walk proceeds exactly as it did before git was asked - never that the
    run fails - but a repository git could not be asked about is said so, because its
    generated output is then indexed. Input and output are captured: stdout is the JSON-RPC
    channel, and stdin is the client's.
    """
    env = git_environment()
    try:
        listed = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "--others", "--ignored", "--exclude-standard",
             "--directory", "-z"],
            stdin=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=10,
            env=env,
        )  # fmt: skip
    except FileNotFoundError:
        if not _in_repository(root):
            return GitIgnore("no_repository")
        return _unavailable(root, "git is not installed or not on PATH")
    except subprocess.CalledProcessError as exc:
        reason = os.fsdecode(exc.stderr or b"").strip()
        # Not "not a git repository: <path>", which is a `.git` that names nothing usable.
        if exc.returncode == 128 and "not a git repository (or any" in reason:
            return GitIgnore("no_repository")
        return _unavailable(root, reason.splitlines()[0] if reason else f"exit {exc.returncode}")
    except (OSError, subprocess.SubprocessError) as exc:
        return _unavailable(root, str(exc))
    # A root git ignores comes back as `./`, which names no path below it: whoever asked
    # for an ignored directory asked on purpose, and its contents are indexed.
    entries = (os.fsdecode(raw).rstrip("/") for raw in listed.split(b"\0"))
    return GitIgnore("applied", frozenset(_printable(entry) for entry in entries if entry))


def git_environment() -> dict[str, str]:
    """The environment git is run in: its own repository's, never a hook's, and in C.

    C because callers match git's messages, and no prompt because nobody is there to answer.
    """
    env = {name: value for name, value in os.environ.items() if name not in _GIT_LOCATION_VARIABLES}
    return env | {"GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"}


def _unavailable(root: Path, cause: str) -> GitIgnore:
    cause = _printable(cause)
    logger.warning("Not using .gitignore for %s: %s", _printable(str(root)), cause)
    return GitIgnore("unavailable", cause=cause)


def _in_repository(root: Path) -> bool:
    """Whether a `.git` sits at or above ``root`` - asked only when git itself is missing."""
    return any(os.path.lexists(directory / ".git") for directory in (root, *root.parents))


def iter_markdown_files(
    directory: Path,
    on_error: Callable[[OSError], None] | None = None,
    exclude: Sequence[str] = (),
    *,
    scope: Scope | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Iterator[Path]:
    """Yield Markdown files beneath ``directory`` in a stable order, pruning vendored trees.

    ``on_error`` receives the ``OSError`` for every sub-directory that cannot be listed
    (``os.walk`` would otherwise skip it silently).

    ``exclude`` holds glob patterns matched against each path *relative to* ``directory``
    (``scripts/eval_data/*``, ``**/vendor/**``, ``CHANGELOG.md``). A repository that keeps
    fixtures, vendored documentation or a test corpus in-tree would otherwise index them
    as if they were its own documentation. A matching directory is pruned, so its subtree
    costs nothing to skip. ``scope`` replaces it with everything a run owns (`Scope`).

    Dot-directories come last: a project's own `docs/` should not wait behind `.claude/`.
    ``should_stop`` is asked at every entry, so a stop is honoured during the walk too.
    """
    owned = scope if scope is not None else Scope(directory, tuple(exclude))
    for root, dirnames, filenames in os.walk(directory, followlinks=False, onerror=on_error):
        here = Path(root)
        dirnames[:] = sorted(
            (
                name
                for name in dirnames
                if name not in _SKIPPED_DIRECTORIES and not owned.excludes(here / name)
            ),
            key=lambda name: (name.startswith("."), name),
        )
        for filename in sorted(filenames):
            if should_stop is not None and should_stop():
                raise IndexCancelled("index run stopped by its owner")
            path = here / filename
            if path.suffix.lower() in MARKDOWN_SUFFIXES and not owned.excludes(path):
                yield path


def without_aliases(paths: Sequence[Path]) -> list[Path]:
    """``paths`` without the symlinks that point at another file on the list, in walk order.

    Identity, not spelling: a link is an alias when its target is the same file
    (`st_dev`, `st_ino`) as a regular file this run will index - which survives `..`, chains,
    a case-insensitive filesystem and a symlinked root, where comparing paths would not. A
    link whose target is excluded, ignored, not Markdown, outside the root or missing stays
    under its own name, as before: it is then the only way that text is reached, or the
    failure a broken link deserves.
    """
    originals: set[tuple[int, int]] = set()
    for path in paths:
        with contextlib.suppress(OSError):
            info = os.lstat(path)
            if stat.S_ISREG(info.st_mode) and _encodable(str(path)):
                originals.add((info.st_dev, info.st_ino))
    kept = []
    for path in paths:
        if os.path.islink(path):
            try:
                target = os.stat(path)
            except OSError:
                kept.append(path)
                continue
            if (target.st_dev, target.st_ino) in originals:
                continue
        kept.append(path)
    return kept


def _behind_symlink(root: Path, path: str) -> bool:
    """True when reaching ``path`` from ``root`` passes through a symlinked directory.

    The walk sets ``followlinks=False``, so it never descends into one - it has no idea
    what is in there, which is the same position an unreadable directory leaves it in. A
    run that treats "I did not look" as "there is nothing there" purges documents that are
    still on disk and still readable at that path, and clears failures it never rechecked.
    """
    current = root
    for part in os.path.relpath(path, root).split(os.sep)[:-1]:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _certainly_gone(root: Path, path: str) -> bool:
    """True when ``path`` is observably absent, rather than merely out of the walk's sight.

    Everything else here refuses to draw conclusions from what the walk did not visit, and
    that refusal has one consequence nobody wanted: a row about a pruned, excluded or
    symlink-shadowed path can never be retired, because the walk that would retire it never
    goes there. The file is then deleted and the row outlives it - the root it sits under
    reports itself incomplete forever, naming a path that no longer exists, and no run can
    ever change that answer.

    Not looking is not evidence. Looking at that one path is: `lstat` separates "there is
    nothing here" (`ENOENT`) from "I am not allowed to know" (`EACCES`, a loop, a dead
    mount), and only the first retires anything. That is a direct observation of one name,
    not an inference from a walk's silence, which is why it is safe where the walk is not.

    Two things make that observation worthless, and both answer `ENOENT` about a path that
    was never the file's. A name that is not valid UTF-8 reached the row through
    `_printable`, which substitutes U+FFFD for the bytes it could not decode - the stored
    string is a rendering of the name, not the name, and nothing is at it. (Testing that
    the string survives `_printable` does not find this: the replacement character is
    itself valid UTF-8 and round-trips.) And a path reached through a symlink answers for
    the link's target: replace a real directory with a broken link and every document under
    it reports `ENOENT` while the files sit untouched wherever they were moved to. Neither
    is evidence of a deletion.
    """
    if _UNDECODABLE in path:
        return False
    if _behind_symlink(root, path) or os.path.islink(path):
        return False
    try:
        os.lstat(path)
    except FileNotFoundError:
        return True
    except OSError:  # no permission, symlink loop, unreachable mount: no evidence either way
        return False
    return False


def _is_shadowing_symlink(path: str) -> bool:
    """True when ``path`` is a symlink the walk sees the name of but never follows.

    `os.walk(followlinks=False)` lists a symlinked directory among its parent's names and
    stops there, so a failure recorded against that directory cannot be rechecked by any
    walk of the tree above it. `_behind_symlink` cannot answer this: it tests the
    components *before* the last one, which is right for a file inside a linked tree and
    blind to the linked directory itself. A symlink to anything else - a file, or nothing at
    all - is listed among the walk's files and reached every run, so only a linked
    directory is out of reach. (Calling a dangling link unreachable kept its failure row
    from ever being cleared, and the next run's insert of the same row failed the run.)
    """
    return os.path.islink(path) and os.path.isdir(path)


def _is_excluded(path: Path, root: Path, patterns: Sequence[str]) -> bool:
    """True when ``path`` matches a pattern, tested against its path relative to ``root``.

    A pattern with no ``/`` matches a *name* anywhere in the tree, the way ``.gitignore``
    treats one: ``eval_data`` excludes ``scripts/eval_data/corpus/a.md``. Requiring the
    full relative path there was a trap - the pattern looked right, matched nothing, and
    the files were indexed silently. A pattern that does contain ``/`` is anchored at the
    root and matched with ``fnmatchcase`` (never the platform's case folding), plus an
    implied ``/*`` so naming a directory covers its subtree.
    """
    if not patterns:
        return False
    try:
        relative = path.relative_to(root)
    except ValueError:  # outside the root: nothing to match against
        return False
    text = relative.as_posix()
    parts = relative.parts
    for pattern in patterns:
        if "/" in pattern:
            anchored = pattern.rstrip("/")
            if fnmatchcase(text, anchored) or fnmatchcase(text, anchored + "/*"):
                return True
        elif any(fnmatchcase(part, pattern) for part in parts):
            return True
    return False


def _printable(path: str) -> str:
    """``path`` safe to log and to send as JSON (undecodable bytes become U+FFFD)."""
    return os.fsencode(path).decode("utf-8", errors="replace")


def _is_walkable(relative_directories: Sequence[str]) -> bool:
    return not any(name in _SKIPPED_DIRECTORIES for name in relative_directories)


def _in_linked_worktree(path: Path, root: Path) -> bool:
    """True when ``path``, or a directory between it and ``root``, is a linked worktree.

    A worktree is always a copy of some other state of a repository - the monorepo that
    prompted this kept 38 of them under `.claude/worktrees/`. ``root`` itself is never asked,
    so pointing `index_directory` at a worktree still indexes it.
    """
    current = path
    while current != root and current.parent != current:
        if _is_linked_worktree(current):
            return True
        current = current.parent
    return False


def _is_linked_worktree(directory: Path) -> bool:
    """Git's own marker: `.git` is a file naming a gitdir, and that gitdir has a `commondir`.

    Positive evidence only. A `.git` directory (a clone, an old-form submodule), a submodule's
    `.git` file (its gitdir has no `commondir`), a `.git` file that is garbled or points
    nowhere: none of them is known to be a copy, so none of them is left out on that account.
    `git worktree add` writes an absolute gitdir; a relative one is relative to the directory.
    """
    marker = directory / ".git"
    try:
        if not stat.S_ISREG(os.lstat(marker).st_mode):
            return False
        with open(marker, "rb") as handle:
            head = handle.read(4096)
    except OSError:
        return False
    first = os.fsdecode(head).splitlines()[0] if head else ""
    named = first.removeprefix("gitdir: ")  # as git's own reader: that prefix, nothing trimmed
    if named == first or not named:
        return False
    return os.path.isfile(directory / named / "commondir")  # an absolute `named` replaces


def _encodable(path: str) -> bool:
    try:
        path.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True
