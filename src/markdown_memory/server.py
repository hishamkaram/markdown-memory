"""MCP server: configuration, the application service, tool handlers and the entrypoint.

stdout belongs to the JSON-RPC transport. Every log record goes to ``sys.stderr``;
nothing in this package calls ``print``.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import logging
import os
import re
import sys
import threading
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import ParamSpec, TypeVar

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from markdown_memory.db import Database
from markdown_memory.exceptions import (
    ConfigurationError,
    DocumentNotFoundError,
    IndexingError,
    MarkdownMemoryError,
    SearchError,
    SectionNotFoundError,
)
from markdown_memory.indexer import (
    DEFAULT_EMBEDDER,
    Embedder,
    EmbeddingGemmaEmbedder,
    FastEmbedEmbedder,
    Indexer,
    create_embedder,
    parse_exclusions,
)
from markdown_memory.models import (
    PATH_SEPARATOR,
    Document,
    DocumentSummary,
    IndexReport,
    IndexStatus,
    JsonDict,
    OutlineNode,
    SearchResult,
    Section,
    estimate_tokens,
)
from markdown_memory.parser import join_parts
from markdown_memory.search import HybridSearcher

logger = logging.getLogger(__name__)

ENV_DB_PATH = "MARKDOWN_MEMORY_DB"
ENV_DOCS_DIR = "MARKDOWN_MEMORY_DOCS_DIR"
ENV_MODEL_CACHE = "MARKDOWN_MEMORY_MODEL_CACHE"
ENV_EMBEDDER = "MARKDOWN_MEMORY_EMBEDDER"
ENV_LOG_LEVEL = "MARKDOWN_MEMORY_LOG_LEVEL"
ENV_EXCLUDE = "MARKDOWN_MEMORY_EXCLUDE"
# Claude Code exports this to every stdio MCP server it spawns, set to the project root.
# `.mcp.json` cannot interpolate it - measured on Claude Code 2.1.278, `${CLAUDE_PROJECT_DIR}`
# and `${workspaceFolder}` are both reported as "Missing environment variables" and passed
# through literally - so a project-scoped config uses relative paths and the server resolves
# them here instead.
ENV_PROJECT_DIR = "CLAUDE_PROJECT_DIR"

_P = ParamSpec("_P")
_R = TypeVar("_R")

_PATH_SEPARATOR_PATTERN = re.compile(r"\s*>\s*")
_MAX_LISTED_PATHS = 40

SERVER_INSTRUCTIONS = (
    "Markdown documentation memory. Workflow: index_directory once, then search_docs to "
    "locate relevant sections, or get_document_outline followed by read_section to fetch "
    "one heading's text. Prefer these tools over reading whole Markdown files."
)


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

    @classmethod
    def from_env(cls) -> ServerConfig:
        root = _project_root()
        db_path = _configured_path(ENV_DB_PATH, root)
        docs_dir = _configured_path(ENV_DOCS_DIR, root)
        model_cache = _configured_path(ENV_MODEL_CACHE, root)
        return cls(
            # One index per documentation root, rather than one for the whole machine.
            # Isolation should not depend on the user having set an environment variable.
            # The model cache below stays shared on purpose: 330 MB of read-only weights,
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
        )


def _project_root() -> Path:
    """The directory a relative configured path is relative to.

    Claude Code sets the working directory of a project-scoped server to the project root
    as well, so the fallback agrees with the export in that case; it differs only for a
    server started by hand from somewhere else.
    """
    exported = os.environ.get(ENV_PROJECT_DIR, "").strip()
    return Path(exported).expanduser() if exported else Path.cwd()


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


class MarkdownMemoryService:
    """Application service behind the MCP tools; returns typed domain models."""

    def __init__(self, config: ServerConfig, embedder: Embedder | None = None) -> None:
        self._config = config
        self._embedder: Embedder = embedder or create_embedder(
            config.embedder, cache_dir=config.model_cache_dir
        )
        self._db = Database(config.db_path, embedding_dim=self._embedder.dimension)
        self._indexer = Indexer(self._db, self._embedder, exclude=config.exclude)
        # Resolved, because indexing resolves: a document under a symlinked or relative
        # docs root is stored by its real path, and a scope spelled any other way filters
        # every one of them out and returns nothing. Resolved ONCE, and reused: resolving
        # again per call lets a retargeted symlink answer from one tree while reporting on
        # another, which is a lie told with two correct halves.
        self._root = str(_absolute(self._config.docs_dir, SearchError))
        self._searcher = HybridSearcher(self._db, self._embedder, scope=self._root)

    @property
    def db(self) -> Database:
        return self._db

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    def close(self) -> None:
        self._searcher.close()
        self._db.close()

    # ------------------------------------------------------------------ operations

    def index_directory(self, directory: str | None = None) -> IndexReport:
        return self._indexer.index_directory(self._resolve_directory(directory))

    def list_documents(self, directory: str = "") -> list[DocumentSummary]:
        # An empty argument means "this project", not "everything this database holds":
        # the default database is shared by every project on the machine.
        scope = self._resolve_directory(directory or None)
        return self._db.list_documents(str(self._within_root(scope, IndexingError)))

    def get_document_outline(self, file_path: str) -> list[OutlineNode]:
        document = self._resolve_document(file_path)
        return build_outline(self._db.get_sections(document.id))

    def read_section(
        self, file_path: str, heading_path: str, *, include_subsections: bool = False
    ) -> str:
        document = self._resolve_document(file_path)
        sections = self._db.get_sections(document.id)
        matched = select_sections(sections, heading_path, include_subsections=include_subsections)
        return join_parts(matched)

    def search_docs(self, query: str, limit: int = 5) -> list[SearchResult]:
        return self._searcher.search(query, limit)

    def index_status(self, directory: str | None = None) -> IndexStatus:
        """What can honestly be said about answers drawn from this server's documents.

        Indexing reports its own failures, but almost nothing calls indexing: an agent
        opens a session, searches, and is served from whatever the index happens to hold.
        Until this is asked at the point of use, a root that lost files to a permissions
        error - or was never indexed at all - answers with confidence and no caveat.

        Coverage is always the configured docs root's, because that is the tree every
        answer is drawn from; `directory` only narrows which failures are worth naming.
        """
        if directory is None:
            return self._db.index_status(self._root)
        # Narrowed in one read, not composed from two: coverage stays the root's - that is
        # the tree every answer is drawn from - while the failures and stale documents
        # named are the ones that live here. Resolved against the root this service was
        # built for, never against the configured path again, or a retargeted symlink
        # pairs this root's certificate with another tree's failures.
        scope = self._within_root(
            _absolute(Path(self._root) / directory.strip(), SearchError), SearchError
        )
        return self._db.index_status(self._root, str(scope))

    # ------------------------------------------------------------------ resolution

    def _resolve_directory(self, directory: str | None) -> Path:
        if directory is None or not directory.strip():
            return _absolute(self._config.docs_dir, IndexingError)
        path = _user_path(directory.strip(), IndexingError)
        if not path.is_absolute():
            path = self._config.docs_dir / path
        return _absolute(path, IndexingError)

    def _within_root(self, resolved: Path, error: type[MarkdownMemoryError]) -> Path:
        """Refuse to *answer about* a directory outside the tree this server serves.

        `..`, an absolute path and a symlink each reach out of the root, and each was
        obeyed: `list_documents` handed back another project's file paths, and a status
        lookup paired this root's certificate with that tree's failures - `coverage:
        verified` beside a non-empty failure list, which the envelope promises cannot
        happen. Search has been scoped to the root ever since it answered one project's
        question from another's documentation; these are the two other ways in.

        `index_directory` is deliberately not scoped this way. It is an instruction rather
        than a question - go and index that tree - and it keys the tree it walked under its
        own root, so nothing it writes is attributed here.
        """
        root = Path(self._root)
        if resolved != root and root not in resolved.parents:
            raise error(
                f"{str(resolved)!r} is outside this server's documentation root "
                f"({self._root}); name a directory inside it."
            )
        return resolved

    def _resolve_document(self, file_path: str) -> Document:
        """Find an indexed document by absolute path, relative path, or unique path suffix."""
        requested = file_path.strip()
        if not requested:
            raise DocumentNotFoundError("file_path must not be empty")
        path = _user_path(requested, DocumentNotFoundError)
        candidates = [path]
        if not path.is_absolute():
            candidates = [self._config.docs_dir / path]
            try:
                candidates.append(Path.cwd() / path)
            except OSError:  # the working directory was deleted under the server
                logger.debug("Working directory is gone; not resolving %s against it", requested)
        for candidate in candidates:
            document = self._db.get_document(str(_absolute(candidate, DocumentNotFoundError)))
            if document is not None:
                return document
        matches = self._db.find_documents_by_suffix(requested)
        if len(matches) == 1:
            return matches[0]
        if matches:
            listing = ", ".join(match.file_path for match in matches[:_MAX_LISTED_PATHS])
            raise DocumentNotFoundError(f"'{requested}' is ambiguous; it matches: {listing}")
        raise DocumentNotFoundError(
            f"'{requested}' is not indexed. Run index_directory, then list_documents "
            "to see the available paths."
        )


# ---------------------------------------------------------------------- pure helpers


def _user_path(text: str, error: type[MarkdownMemoryError]) -> Path:
    """``Path(text).expanduser()``; pathlib's RuntimeError/ValueError become ``error``."""
    try:
        text.encode("utf-8")  # a lone surrogate cannot be bound as a SQLite parameter
        return Path(text).expanduser()
    except (RuntimeError, ValueError) as exc:  # unknown ~user, embedded NUL, bad encoding
        raise error(f"Invalid path {text!r}: {exc}") from exc


def _absolute(path: Path, error: type[MarkdownMemoryError]) -> Path:
    try:
        resolved = path.expanduser().resolve()
        str(resolved).encode("utf-8")
    except (OSError, RuntimeError, ValueError) as exc:  # symlink loop, NUL, bad encoding
        raise error(f"Invalid path {str(path)!r}: {exc}") from exc
    return resolved


def normalize_heading_path(heading_path: str) -> str:
    """Canonicalise breadcrumb spacing: ``A>B`` and ``A  >  B`` both become ``A > B``."""
    return PATH_SEPARATOR.join(
        segment.strip() for segment in _PATH_SEPARATOR_PATTERN.split(heading_path.strip())
    )


def _casefolded(heading_path: str) -> str:
    return normalize_heading_path(heading_path).casefold()


# Progressively looser ways to compare a requested path with a stored one. The same
# transformation is applied to BOTH sides, so a title containing '>' ("Step 1 -> Step 2",
# "Result<T, E>") still matches itself, and an exact-case request beats a case-folded one
# (sibling headings "Setup" and "SETUP" stay individually addressable).
_MATCH_KEYS: tuple[Callable[[str], str], ...] = (str, normalize_heading_path, _casefolded)


def resolve_heading_path(candidates: Sequence[str], requested: str) -> str:
    """The one stored path that ``requested`` designates.

    Whole-path matches are tried before trailing fragments (``Child > Subchild`` or the
    bare heading title); within each, stricter comparisons come first. The first tier
    with any match decides: one match wins, several are reported as ambiguous.
    """
    for as_suffix in (False, True):
        for key in _MATCH_KEYS:
            wanted = key(requested)
            if as_suffix:
                ending = PATH_SEPARATOR + wanted
                matches = [path for path in candidates if key(path).endswith(ending)]
            else:
                matches = [path for path in candidates if key(path) == wanted]
            if len(matches) == 1:
                return matches[0]
            if matches:
                raise SectionNotFoundError(
                    f"'{requested}' is ambiguous; use one of (exact spelling): "
                    + " | ".join(matches[:_MAX_LISTED_PATHS])
                )
    available = " | ".join(candidates[:_MAX_LISTED_PATHS]) or "(document has no sections)"
    raise SectionNotFoundError(f"No section '{requested}'. Available heading paths: {available}")


def select_sections(
    sections: Sequence[Section], heading_path: str, *, include_subsections: bool = False
) -> list[Section]:
    """Sections addressed by ``heading_path``, in source order.

    ``heading_path`` may name a whole section (every part of it is returned) or a single
    ``(Part n)`` of an oversized one. ``include_subsections`` adds the section's
    descendants; it has no meaning for a single part and is ignored there.
    """
    requested = heading_path.strip()
    if not requested:
        raise SectionNotFoundError("heading_path must not be empty")
    base_paths = list(dict.fromkeys(section.base_path for section in sections))
    parts = {s.heading_path: s for s in sections if s.part_index > 0}
    part_paths = [path for path in parts if path not in base_paths]
    chosen = resolve_heading_path([*base_paths, *part_paths], requested)
    if chosen not in base_paths:
        return [parts[chosen]]

    selected: list[Section] = []
    level: int | None = None
    for section in sections:
        if section.base_path == chosen:
            level = section.heading_level
            selected.append(section)
        elif level is not None:
            # Descendants are the sections that follow until a heading at the same or a
            # shallower level; walking the order is immune to '>' inside titles.
            if not include_subsections or level < 1 or section.heading_level <= level:
                break
            selected.append(section)
    return selected


def build_outline(sections: Sequence[Section]) -> list[OutlineNode]:
    """Nest a document's sections into a hierarchical table of contents."""

    @dataclass(slots=True)
    class _Pending:
        title: str
        level: int
        path: str
        start_line: int
        end_line: int
        tokens: int
        parts: int
        children: list[_Pending]

    roots: list[_Pending] = []
    stack: list[_Pending] = []
    by_path: dict[str, _Pending] = {}
    for section in sections:
        existing = by_path.get(section.base_path)
        if existing is not None:  # a further part of an oversized section
            existing.end_line = max(existing.end_line, section.end_line)
            existing.tokens += estimate_tokens(section.content)
            existing.parts += 1
            continue
        node = _Pending(
            title=section.heading_title,
            level=section.heading_level,
            path=section.base_path,
            start_line=section.start_line,
            end_line=section.end_line,
            tokens=estimate_tokens(section.content),
            parts=1,
            children=[],
        )
        by_path[section.base_path] = node
        if node.level < 1:  # the preamble is a sibling of the headings, never their parent
            roots.append(node)
            continue
        while stack and stack[-1].level >= node.level:
            stack.pop()
        (stack[-1].children if stack else roots).append(node)
        stack.append(node)

    def freeze(node: _Pending) -> OutlineNode:
        return OutlineNode(
            heading_title=node.title,
            heading_level=node.level,
            heading_path=node.path,
            start_line=node.start_line,
            end_line=node.end_line,
            token_estimate=node.tokens,
            part_count=node.parts,
            children=tuple(freeze(child) for child in node.children),
        )

    return [freeze(node) for node in roots]


# ---------------------------------------------------------------------- MCP wiring


def anticipated_errors(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Report domain failures to the model verbatim.

    The SDK hides the message of any exception other than ``ToolError`` behind a generic
    "Error executing tool" (treating it as a crash). Domain errors carry actionable text
    - available heading paths, "run index_directory first" - that the agent needs to see.
    """

    @functools.wraps(function)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return function(*args, **kwargs)
        except MarkdownMemoryError as exc:
            raise ToolError(str(exc)) from exc

    return wrapper


class _ServiceProvider:
    """Hands the tools their service and owns its lifetime when nobody else does.

    A service passed in by the caller is never closed here. One the server creates itself
    lives as long as any session is open, and is created again on demand afterwards, so
    the same server object can serve several sessions (or concurrent ones) in a row.
    """

    def __init__(self, config: ServerConfig | None, service: MarkdownMemoryService | None) -> None:
        self._config = config
        self._external = service
        self._owned: MarkdownMemoryService | None = None
        self._sessions = 0
        self._lock = threading.Lock()

    def get(self) -> MarkdownMemoryService:
        if self._external is not None:
            return self._external
        with self._lock:
            if self._owned is None:
                self._owned = MarkdownMemoryService(self._config or ServerConfig.from_env())
            return self._owned

    def session_started(self) -> None:
        with self._lock:
            self._sessions += 1

    def session_ended(self) -> None:
        with self._lock:
            self._sessions -= 1
            if self._sessions > 0 or self._owned is None:
                return
            owned, self._owned = self._owned, None
        owned.close()


def create_server(
    config: ServerConfig | None = None, *, service: MarkdownMemoryService | None = None
) -> MCPServer[None]:
    """Build the MCP server and register its tools."""
    services = _ServiceProvider(config, service)

    @asynccontextmanager
    async def lifespan(_server: MCPServer[None]) -> AsyncIterator[None]:
        services.session_started()
        try:
            yield
        finally:
            services.session_ended()

    server: MCPServer[None] = MCPServer(
        "markdown-memory", instructions=SERVER_INSTRUCTIONS, lifespan=lifespan
    )

    @server.tool()
    @anticipated_errors
    def index_directory(directory: str | None = None) -> str:
        """Scan a directory tree for Markdown files and (re)index new or changed ones.

        Unchanged files are skipped via SHA-256 hashes; files deleted from disk are purged.
        Omit `directory` to index the server's configured documentation root.
        """
        return services.get().index_directory(directory).summary()

    @server.tool()
    @anticipated_errors
    def list_documents(directory: str = "") -> JsonDict:
        """List indexed documents (path, title, section count), optionally under `directory`.

        Returns `{"documents": [...], "index_status": {...}}`. `index_status.coverage` is
        "verified" only when a full index run of this documentation root finished and read
        every file it found; otherwise it is "unknown" and `index_status.message` says why.
        """
        service = services.get()
        scope = directory if directory.strip() else None
        return {
            "documents": [summary.to_dict() for summary in service.list_documents(directory)],
            "index_status": service.index_status(scope).to_dict(),
        }

    @server.tool()
    @anticipated_errors
    def get_document_outline(file_path: str) -> list[JsonDict]:
        """Hierarchical table of contents of one document: heading paths, line ranges and
        token estimates. Costs a few hundred tokens; use it to pick a section to read."""
        return [node.to_dict() for node in services.get().get_document_outline(file_path)]

    @server.tool()
    @anticipated_errors
    def read_section(file_path: str, heading_path: str, include_subsections: bool = False) -> str:
        """Return the verbatim Markdown of one section, addressed by its breadcrumb
        (`Root > Child > Subchild`, as shown by get_document_outline). By default only the
        section's own text is returned; set `include_subsections` to append its children."""
        return services.get().read_section(
            file_path, heading_path, include_subsections=include_subsections
        )

    @server.tool()
    @anticipated_errors
    def search_docs(query: str, limit: int = 5) -> JsonDict:
        """Hybrid search (BM25 keywords + semantic vectors, fused with RRF) over all indexed
        sections. Works for exact identifiers (flags, env vars) and for conceptual questions.

        Returns `{"results": [...], "index_status": {...}}`, `results` holding at most
        `limit` sections. When `index_status.coverage` is "unknown", what you searched is
        missing part of its documentation, or was never indexed end to end: an answer drawn
        from it may be confidently incomplete, and `index_status.message` says what to run.
        """
        service = services.get()
        return {
            "results": [result.to_dict() for result in service.search_docs(query, limit)],
            "index_status": service.index_status().to_dict(),
        }

    return server


def configure_logging(level: str | None = None) -> None:
    """Route all logging to stderr; stdout is reserved for JSON-RPC frames."""
    name = (level or os.environ.get(ENV_LOG_LEVEL, "") or "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, name, logging.INFO),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )


def _warm_up(embedder: Embedder) -> None:
    if not isinstance(embedder, EmbeddingGemmaEmbedder | FastEmbedEmbedder):
        return
    try:
        embedder.warm_up()
    except MarkdownMemoryError:
        logger.warning("Embedding model warm-up failed; it will be retried on first use")


def main(argv: Sequence[str] | None = None) -> None:
    """Console entrypoint: serve MCP over stdio."""
    parser = argparse.ArgumentParser(
        prog="markdown-memory", description="Markdown documentation memory MCP server (stdio)."
    )
    parser.add_argument("--db", type=Path, help=f"SQLite database path (env {ENV_DB_PATH})")
    parser.add_argument("--docs-dir", type=Path, help=f"Default docs root (env {ENV_DOCS_DIR})")
    parser.add_argument("--log-level", help=f"Logging level (env {ENV_LOG_LEVEL})")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GLOB",
        help=f"Skip paths matching this glob, relative to the docs root; repeatable "
        f"(env {ENV_EXCLUDE}, comma or colon separated)",
    )
    parser.add_argument(
        "--embedder",
        choices=("embeddinggemma", "bge-small"),
        help=f"Embedding model preset (env {ENV_EMBEDDER}; default {DEFAULT_EMBEDDER})",
    )
    arguments = parser.parse_args(argv)

    configure_logging(arguments.log_level)
    base = ServerConfig.from_env()
    db_path: Path | None = arguments.db
    docs_dir: Path | None = arguments.docs_dir
    config = ServerConfig(
        db_path=db_path.expanduser() if db_path else base.db_path,
        docs_dir=docs_dir.expanduser() if docs_dir else base.docs_dir,
        embedder=arguments.embedder or base.embedder,
        model_cache_dir=base.model_cache_dir,
        exclude=tuple(arguments.exclude) or base.exclude,
    )
    try:
        service = MarkdownMemoryService(config)
    except MarkdownMemoryError:
        logger.exception("Cannot start markdown-memory")
        raise SystemExit(1) from None
    logger.info("markdown-memory serving; db=%s docs_dir=%s", config.db_path, config.docs_dir)
    threading.Thread(
        target=_warm_up, args=(service.embedder,), name="mdmem-warmup", daemon=True
    ).start()
    try:
        create_server(config, service=service).run("stdio")
    finally:
        service.close()


if __name__ == "__main__":
    main()
