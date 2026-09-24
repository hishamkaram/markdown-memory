"""Finding the Markdown files to index, and reading them safely.

Everything here is about the filesystem and nothing about embedding: which paths a walk
should yield, which it must skip, and how to read one without being taken somewhere else
by a symlink or blocked forever by a FIFO.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable, Iterator, Sequence
from fnmatch import fnmatchcase
from pathlib import Path

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


def iter_markdown_files(
    directory: Path,
    on_error: Callable[[OSError], None] | None = None,
    exclude: Sequence[str] = (),
) -> Iterator[Path]:
    """Yield Markdown files beneath ``directory`` in a stable order, pruning vendored trees.

    ``on_error`` receives the ``OSError`` for every sub-directory that cannot be listed
    (``os.walk`` would otherwise skip it silently).

    ``exclude`` holds glob patterns matched against each path *relative to* ``directory``
    (``scripts/eval_data/*``, ``**/vendor/**``, ``CHANGELOG.md``). A repository that keeps
    fixtures, vendored documentation or a test corpus in-tree would otherwise index them
    as if they were its own documentation. A matching directory is pruned, so its subtree
    costs nothing to skip.
    """
    for root, dirnames, filenames in os.walk(directory, followlinks=False, onerror=on_error):
        here = Path(root)
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in _SKIPPED_DIRECTORIES
            and not _is_excluded(here / name, directory, exclude)
        )
        for filename in sorted(filenames):
            path = here / filename
            if path.suffix.lower() in MARKDOWN_SUFFIXES and not _is_excluded(
                path, directory, exclude
            ):
                yield path


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
    blind to the linked directory itself. A symlink to a *file* is walked and indexed like
    any other file, so only one that does not resolve to a file is out of reach.
    """
    return os.path.islink(path) and not os.path.isfile(path)


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
