"""Where the server reads its settings from, and in what order.

One precedence for every entry point: an explicit argument beats the environment, which
beats the project default. ``resolve_config`` is the only place that order lives.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from markdown_memory.embedders import DEFAULT_EMBEDDER
from markdown_memory.exceptions import ConfigurationError
from markdown_memory.indexer import DEFAULT_INDEX_WORKERS

ENV_DB_PATH = "MARKDOWN_MEMORY_DB"


ENV_DOCS_DIR = "MARKDOWN_MEMORY_DOCS_DIR"


ENV_MODEL_CACHE = "MARKDOWN_MEMORY_MODEL_CACHE"


ENV_EMBEDDER = "MARKDOWN_MEMORY_EMBEDDER"


ENV_LOG_LEVEL = "MARKDOWN_MEMORY_LOG_LEVEL"


ENV_EXCLUDE = "MARKDOWN_MEMORY_EXCLUDE"


ENV_INDEX_WORKERS = "MARKDOWN_MEMORY_INDEX_WORKERS"


ENV_AUTO_INDEX = "MARKDOWN_MEMORY_AUTO_INDEX"
ENV_GITIGNORE = "MARKDOWN_MEMORY_GITIGNORE"


# Claude Code exports this to every stdio MCP server it spawns, set to the project root.
# `.mcp.json` cannot interpolate it - measured on Claude Code 2.1.278, `${CLAUDE_PROJECT_DIR}`
# and `${workspaceFolder}` are both reported as "Missing environment variables" and passed
# through literally - so a project-scoped config uses relative paths and the server resolves
# them here instead.
ENV_PROJECT_DIR = "CLAUDE_PROJECT_DIR"


def _xdg_dir(variable: str, fallback: str) -> Path:
    configured = os.environ.get(variable, "").strip()
    return Path(configured) if configured else Path.home() / fallback


def _project_database(docs_dir: Path) -> Path:
    """Where one documentation root's index lives when nothing configured it.

    Keyed on the documentation root, never on the working directory. The working directory
    belongs to whoever launched the server, so two servers started from one directory for
    two different projects would share a database - which is the cross-project leak this
    default exists to close, arrived at from the other side. Search is scoped to the docs
    root, but a document stays resolvable across a whole database by path or unique suffix,
    so sharing the file is enough to leak one project's documentation into another's answers.

    Kept out of the project too. A database inside the repository is committed by accident,
    deleted by `git clean -xdf`, rebuilt per worktree, unwritable when the checkout is
    read-only, and - on a network share - sits where SQLite's WAL cannot take the locks it
    needs. The name carries the root's own basename so a person can tell the indexes apart,
    and a digest of its resolved path so two projects called `docs` cannot collide.
    """
    try:
        resolved = docs_dir.expanduser().resolve()
    except (OSError, RuntimeError):  # symlink loop, or a path the OS will not resolve
        resolved = docs_dir.expanduser().absolute()
    digest = hashlib.sha256(os.fsencode(str(resolved))).hexdigest()[:12]
    label = re.sub(r"[^A-Za-z0-9_.-]", "-", resolved.name) or "root"
    return (
        _xdg_dir("XDG_DATA_HOME", ".local/share")
        / "markdown-memory"
        / "projects"
        / f"{label}-{digest}"
        / "index.db"
    )


@dataclass(slots=True, frozen=True)
class ServerConfig:
    """Runtime configuration, resolved from the environment (CLI flags override)."""

    db_path: Path
    docs_dir: Path
    embedder: str = DEFAULT_EMBEDDER
    model_cache_dir: Path | None = None
    # Glob patterns, relative to the docs root, that indexing must not descend into: a
    # repository's own fixtures, vendored documentation or test corpus are not its docs.
    exclude: tuple[str, ...] = ()
    #: Files embedded at the same time while indexing.
    index_workers: int = DEFAULT_INDEX_WORKERS
    #: Whether the stdio server keeps its own docs root indexed in the background.
    auto_index: bool = True
    #: Whether what git ignores below the docs root is left out of the index.
    gitignore: bool = True
    #: Whether `db_path` was chosen rather than derived from the docs root. Chosen unless
    #: the environment resolution derived it: a path written into a config by hand is one
    #: somebody picked, and other work trees' indexes go beside it rather than elsewhere.
    db_explicit: bool = True

    @classmethod
    def from_env(cls) -> ServerConfig:
        root = _project_root()
        db_path = _configured_path(ENV_DB_PATH, root)
        docs_dir = _configured_path(ENV_DOCS_DIR, root)
        model_cache = _configured_path(ENV_MODEL_CACHE, root)
        return cls(
            # One index per documentation root, rather than one for the whole machine.
            # Isolation should not depend on the user having set an environment variable.
            # The model cache below stays shared on purpose: 218 MB of read-only weights,
            # identical everywhere, and copying it per project would be pure waste.
            db_path=(db_path if db_path else _project_database(docs_dir if docs_dir else root)),
            docs_dir=docs_dir if docs_dir else root,
            embedder=os.environ.get(ENV_EMBEDDER, "").strip() or DEFAULT_EMBEDDER,
            model_cache_dir=(
                model_cache
                if model_cache
                else _xdg_dir("XDG_CACHE_HOME", ".cache") / "markdown-memory" / "models"
            ),
            exclude=parse_exclusions(os.environ.get(ENV_EXCLUDE, "")),
            index_workers=_positive_int(ENV_INDEX_WORKERS, DEFAULT_INDEX_WORKERS),
            auto_index=_switched_on(ENV_AUTO_INDEX),
            gitignore=_switched_on(ENV_GITIGNORE),
            db_explicit=db_path is not None,
        )


def resolve_config(
    *,
    db: Path | None = None,
    docs_dir: Path | None = None,
    embedder: str | None = None,
    exclude: Sequence[str] = (),
    auto_index: bool | None = None,
    gitignore: bool | None = None,
) -> ServerConfig:
    """Environment configuration with explicit overrides laid over it.

    Naming a different documentation root re-keys the database, because the default is
    keyed on that root: taking `ServerConfig.from_env().db_path` as the fallback reads a
    path derived from the *environment's* root, and two callers pointed at different roots
    from one directory would land in the launcher's single database - exactly the
    cross-project leak keying was added to close. An explicitly configured database still
    wins, from the argument or the environment, in that order.

    Every entry point that takes overrides resolves them here - the server's own flags
    and the scripts alike - so that precedence is written once and cannot drift between
    them. The paths that accept none (the in-process service, `eval_retrieval.py`) go
    straight to `ServerConfig.from_env`, which is the same answer with nothing laid over
    it.
    """
    base = ServerConfig.from_env()
    root = docs_dir.expanduser() if docs_dir else base.docs_dir
    configured_db = _configured_path(ENV_DB_PATH, _project_root())
    return ServerConfig(
        db_path=(
            db.expanduser() if db else configured_db if configured_db else _project_database(root)
        ),
        docs_dir=root,
        embedder=embedder or base.embedder,
        model_cache_dir=base.model_cache_dir,
        exclude=tuple(exclude) or base.exclude,
        index_workers=base.index_workers,
        auto_index=base.auto_index if auto_index is None else auto_index,
        gitignore=base.gitignore if gitignore is None else gitignore,
        db_explicit=bool(db or configured_db),
    )


def tree_database(config: ServerConfig, docs_dir: Path) -> Path:
    """The database another work tree's copy of the docs root is indexed into.

    Never the configured root's own file: the root's runs purge every row outside it, which
    is what used to undo a worktree indexed into the same database. By default it is the
    file a server configured for that tree would use. A database placed explicitly says
    where indexes may live - a mounted volume, a writable directory - so the tree's goes
    beside it, named after the tree the way the default is.
    """
    default = _project_database(docs_dir)
    if not config.db_explicit:
        return default
    configured = config.db_path
    return configured.with_name(f"{configured.stem}-{default.parent.name}{configured.suffix}")


def _config_from_cli(arguments: argparse.Namespace) -> ServerConfig:
    """The command line laid over the environment."""
    return resolve_config(
        db=arguments.db,
        docs_dir=arguments.docs_dir,
        embedder=arguments.embedder,
        exclude=arguments.exclude,
        auto_index=False if getattr(arguments, "no_auto_index", False) else None,
        gitignore=False if getattr(arguments, "no_gitignore", False) else None,
    )


def _switched_on(variable: str) -> bool:
    """On unless the variable says otherwise: `0`, `false`, `off` or `no`."""
    return os.environ.get(variable, "").strip().lower() not in {"0", "false", "off", "no"}


def _project_root() -> Path:
    """The directory a relative configured path is relative to.

    Claude Code sets the working directory of a project-scoped server to the project root
    as well, so the fallback agrees with the export in that case; it differs only for a
    server started by hand from somewhere else.
    """
    exported = os.environ.get(ENV_PROJECT_DIR, "").strip()
    return Path(exported).expanduser() if exported else Path.cwd()


def _positive_int(variable: str, default: int) -> int:
    """A count read from the environment; anything that is not one keeps the default."""
    value = os.environ.get(variable, "").strip()
    return int(value) if value.isdigit() and int(value) > 0 else default


def _configured_path(variable: str, root: Path) -> Path | None:
    """One configured path, resolved against ``root`` when it is relative.

    An unexpanded `${...}` is rejected rather than used as a directory name: Claude Code
    loads a config whose variables it could not expand and passes the literal text through,
    which would otherwise index a directory named `${workspaceFolder}` and report success
    over zero files.
    """
    value = os.environ.get(variable, "").strip()
    if not value:
        return None
    # A directory really named `docs/${version}` is allowed: if the literal path exists,
    # it is a path, not a variable nobody expanded.
    if "${" in value and not Path(value).expanduser().exists():
        raise ConfigurationError(
            f"{variable} is set to {value!r}, which still contains an unexpanded variable. "
            "Claude Code expands only environment variables in .mcp.json - not "
            "${workspaceFolder} or ${CLAUDE_PROJECT_DIR} - so write the path relative to the "
            "project root instead (for example '.markdown-memory/index.db')."
        )
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path)


def parse_exclusions(value: str) -> tuple[str, ...]:
    """Split a configured exclusion list on commas; blanks and stray ``./`` dropped.

    Comma only: a colon separator would split a pattern that contains one, and silently
    excluding the wrong thing is worse than not accepting the separator.
    """
    patterns = []
    for part in value.split(","):
        # One leading "./" only: `lstrip("./")` would eat the dot of `.hidden` and
        # exclude a `hidden` directory instead of the one that was named.
        cleaned = part.strip().removeprefix("./").rstrip("/")
        if cleaned:
            patterns.append(cleaned)
    return tuple(patterns)
