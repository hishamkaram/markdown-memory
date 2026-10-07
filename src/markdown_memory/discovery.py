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
import shlex
import stat
import subprocess
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePath

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
    ignores is indexed as well; ``cause`` says why). A status reads `unknown` until a finished
    run has recorded one of these.
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
    try:
        listed = git(
            root, "ls-files", "--others", "--ignored", "--exclude-standard", "--directory", "-z"
        )
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


#: What git says in a checkout whose repository is configured bare (`core.bare = true`, which
#: some agent worktree tooling sets on the main checkout); the second is a submodule's, whose
#: config also names its work tree.
_BARE_CHECKOUT = ("must be run in a work tree", "unable to set up work tree using invalid config")


#: Checkouts already warned about, so a server running for days logs each one once.
_warned: set[str] = set()


_warned_lock = threading.Lock()


def git(directory: Path, *args: str) -> bytes:
    """Run git in ``directory`` and return what it printed; `CalledProcessError` if it failed.

    The one way this package runs git: input and output captured (stdout is the JSON-RPC
    channel, stdin the client's), its own repository's environment, a timeout. A checkout
    whose repository says it is bare is read with the checkout as the work tree, which is
    what git itself does once the setting is gone. Nothing in git's config is changed.
    """
    try:
        return _run_git(directory, args)
    except subprocess.CalledProcessError as exc:
        said = os.fsdecode(exc.stderr or b"")
        if exc.returncode != 128 or not any(marker in said for marker in _BARE_CHECKOUT):
            raise
        checkout = _bare_checkout(directory)
        if checkout is None:
            raise
        with _warned_lock:
            first = str(checkout) not in _warned
            _warned.add(str(checkout))
        if first:
            shown = _printable(str(checkout))
            logger.warning(
                "git treats the repository of %s as bare (core.bare = true); it was read with "
                "%s as its work tree. Other git commands there fail until `%s`, unless "
                "something set it on purpose.",
                shown,
                shown,
                shlex.join(["git", "-C", shown, "config", "core.bare", "false"]),
            )
        return _run_git(directory, args, work_tree=checkout)


def _run_git(directory: Path, args: Sequence[str], work_tree: Path | None = None) -> bytes:
    command = ["git", "-C", str(directory)]
    if work_tree is not None:
        command.append(f"--work-tree={work_tree}")
    return subprocess.check_output(
        [*command, *args],
        stdin=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=10,
        env=git_environment(),
    )


def _bare_checkout(directory: Path) -> Path | None:
    """The checkout around ``directory`` when git calls its repository bare, else ``None``.

    Git is asked first: a real bare repository, or a path inside a git directory, answers
    true and is left alone. Otherwise the work tree is the nearest directory with a `.git`
    entry, which is where git itself starts - its existence is checked, nothing is read.
    """
    try:
        inside = _run_git(directory, ("rev-parse", "--is-inside-git-dir")).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if inside != b"false":
        return None
    start = directory.resolve()
    return next(
        (candidate for candidate in (start, *start.parents) if os.path.lexists(candidate / ".git")),
        None,
    )


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


def _directory_order(name: str) -> tuple[bool, str]:
    """Where a sub-directory goes among its siblings in the walk: by name, dot-directories last."""
    return (name.startswith("."), name)


def walk_order(file_path: str) -> tuple[tuple[int, bool, str], ...]:
    """Where `iter_markdown_files` reaches ``file_path``: a directory's own files by name, then
    its sub-directories, each walked whole, in `_directory_order`.

    Search breaks an exact tie with it (#103). Section ids follow this order on a fresh build,
    but an edited document is stored again under new ids, so a tie decided by id goes to
    whichever document was edited least recently. The stored spelling is compared, never a
    resolved path: that is the walk's own.
    """
    *directories, name = PurePath(file_path).parts
    return (*((1, *_directory_order(part)) for part in directories), (0, False, name))


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
            key=_directory_order,
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
    return _linked_worktree_of(path, root) is not None


def _linked_worktree_of(path: Path, root: Path) -> Path | None:
    current = path
    while current != root and current.parent != current:
        if _is_linked_worktree(current):
            return current
        current = current.parent
    return None


def unindexed_reason(path: Path, root: Path, exclude: Sequence[str], gitignore: bool) -> str | None:
    """Why the file at ``path`` is not in the index of ``root``; ``None`` when nothing says.

    For an agent that named a document the index does not hold, and was told only "Run
    index_directory" - advice that cannot help a path outside the root, or one the root's own
    runs leave out and would purge again. The first rule that holds answers, in the order a
    walk decides: what is on disk, then where it is, then the root's own rules. Only the
    error path asks, so one `git check-ignore` here costs nothing anyone waits on.
    """
    shown = _printable(str(path))
    if not path.exists():
        return f"'{shown}' does not exist."
    if path.is_dir() or path.suffix.lower() not in MARKDOWN_SUFFIXES:
        return f"'{shown}' is not a Markdown document (.md or .markdown)."
    # The directory resolved, the name kept: a link is indexed under its own path, wherever
    # it points, so it is judged there too.
    real = path.parent.resolve() / path.name
    if real != root and root not in real.parents:
        return (
            f"'{shown}' is outside {_printable(str(root))}, the documentation root this server "
            "indexes. If it is in another worktree of this repository, pass that worktree as cwd."
        )
    parts = real.relative_to(root).parts[:-1]
    skipped = next((end for end in range(len(parts)) if parts[end] in _SKIPPED_DIRECTORIES), None)
    if skipped is not None:
        directory = _printable(str(root.joinpath(*parts[: skipped + 1])))
        return (
            f"'{shown}' is inside {directory}, which every walk of the root skips; "
            f"index_directory('{directory}') indexes it."
        )
    worktree = _linked_worktree_of(real, root)
    if worktree is not None:
        return (
            f"'{shown}' is inside the linked worktree {_printable(str(worktree))}, which the "
            "root's index leaves out: pass that worktree as cwd, or this file's absolute path, "
            "to read the worktree's own index."
        )
    if _is_excluded(real, root, exclude):
        why = "an exclusion pattern (the server's --exclude)"
    elif gitignore and _git_ignores(real, root):
        why = "git ignoring it (the server's --no-gitignore turns that off)"
    else:
        return None
    return (
        f"'{shown}' is under {_printable(str(root))} but left out of its index by {why}: "
        "a run of the root leaves it out, and removes any copy indexed here."
    )


def _git_ignores(path: Path, root: Path) -> bool:
    """Whether a walk of ``root`` leaves ``path`` out because git ignores it.

    The walk's own question (`git_ignored`), narrowed to the path's top component: a deeper
    pathspec below an ignored directory makes git fail ("directory entry not superset of
    prefix"), and `check-ignore` would also read a nested clone's `.gitignore`, which the walk
    does not, and call everything below a root git ignores ignored, which the walk indexes
    anyway (`./`). Anything git cannot answer counts as no.
    """
    relative = path.relative_to(root)
    try:
        listed = git(
            root, "--literal-pathspecs",  # a name such as `:(top)x.md` is not pathspec magic
            "ls-files", "--others", "--ignored", "--exclude-standard", "--directory", "-z",
            "--", relative.parts[0],
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return False
    # A root git ignores whole comes back as `./`, which names no path below it.
    entries = {os.fsdecode(raw).rstrip("/") for raw in listed.split(b"\0")}
    parts = relative.parts
    return any("/".join(parts[:end]) in entries for end in range(1, len(parts) + 1))


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
