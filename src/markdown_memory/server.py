"""MCP server: the application service, tool handlers and the entrypoint.

Configuration lives in ``config.py``, the freshness sweep in ``freshness.py`` and
heading resolution in ``headings.py``.

stdout belongs to the JSON-RPC transport. Every log record goes to ``sys.stderr``;
nothing in this package calls ``print``.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import logging
import os
import sys
import threading
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import ParamSpec, TypeVar

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from markdown_memory import __version__, headings
from markdown_memory.autoindex import AutoIndexer
from markdown_memory.config import (
    ENV_AUTO_INDEX,
    ENV_DB_PATH,
    ENV_DOCS_DIR,
    ENV_EMBEDDER,
    ENV_EXCLUDE,
    ENV_LOG_LEVEL,
    ServerConfig,
    _config_from_cli,
)
from markdown_memory.db import Database
from markdown_memory.embedders import (
    DEFAULT_EMBEDDER,
    Embedder,
    create_embedder,
)
from markdown_memory.exceptions import (
    DocumentNotFoundError,
    IndexingError,
    MarkdownMemoryError,
    SearchError,
)
from markdown_memory.freshness import FreshnessSweep
from markdown_memory.indexer import Indexer
from markdown_memory.models import (
    Document,
    DocumentSummary,
    IndexReport,
    IndexStatus,
    JsonDict,
    OutlineNode,
    SearchResult,
)
from markdown_memory.parser import join_parts
from markdown_memory.search import HybridSearcher

logger = logging.getLogger(__name__)


_P = ParamSpec("_P")


_R = TypeVar("_R")


SERVER_INSTRUCTIONS = (
    "Markdown documentation memory. The server keeps its documentation root indexed by "
    "itself (unless started with --no-auto-index); call index_directory only when "
    "index_status says so. Use search_docs to locate relevant sections, or "
    "get_document_outline followed by read_section to fetch one heading's text. Prefer "
    "these tools over reading whole Markdown files."
)


class MarkdownMemoryService:
    """Application service behind the MCP tools; returns typed domain models."""

    def __init__(self, config: ServerConfig, embedder: Embedder | None = None) -> None:
        self._config = config
        self._embedder: Embedder = embedder or create_embedder(
            config.embedder, cache_dir=config.model_cache_dir
        )
        self._db = Database(config.db_path, embedding_dim=self._embedder.dimension)
        self._indexer = Indexer(
            self._db, self._embedder, workers=config.index_workers, exclude=config.exclude
        )
        # Resolved, because indexing resolves: a document under a symlinked or relative
        # docs root is stored by its real path, and a scope spelled any other way filters
        # every one of them out and returns nothing. Resolved ONCE, and reused: resolving
        # again per call lets a retargeted symlink answer from one tree while reporting on
        # another, which is a lie told with two correct halves.
        self._root = str(headings._absolute(self._config.docs_dir, SearchError))
        self._freshness = FreshnessSweep(self._db)
        self._searcher = HybridSearcher(self._db, self._embedder, scope=self._root)
        #: Only the stdio server starts one (`main`): a service built by a test or a
        #: script does exactly what it is asked and nothing in the background.
        self._auto: AutoIndexer | None = None

    @property
    def db(self) -> Database:
        return self._db

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    def close(self) -> None:
        # The background run first, and waited for: it writes through the database that
        # is about to close, and a stop lands between two documents.
        if self._auto is not None:
            self._auto.stop()
        self._searcher.close()
        self._db.close()

    def start_auto_index(self, *, request: bool = True) -> None:
        """Keep the docs root indexed from now on, as searches see change.

        `request=False` arms the runner without starting a run: the first search then finds
        it has never run and starts the catch-up itself. That is what the stdio server does,
        so nothing loads the model before the client's handshake has been answered.
        """
        if self._auto is None:
            self._auto = AutoIndexer(
                lambda should_stop: self.index_directory(None, should_stop=should_stop),
                self._root_status,
            )
        if request:
            self._auto.request()

    # ------------------------------------------------------------------ operations

    def index_directory(
        self, directory: str | None = None, should_stop: Callable[[], bool] | None = None
    ) -> IndexReport:
        try:
            return self._indexer.index_directory(
                self._resolve_directory(directory), should_stop=should_stop
            )
        finally:
            # Whatever just happened, the sweep's answer is about the tree as it was
            # before it: a run that refreshed the files it named would otherwise keep
            # being reported as stale for the rest of the window, which is exactly the
            # moment an agent looks. A run that failed partway invalidates it too - some
            # of it may have been written.
            self._freshness.invalidate()

    def list_documents(self, directory: str = "") -> list[DocumentSummary]:
        # An empty argument means "this project", not "everything this database holds".
        # The default database is per-root now, but a configured MARKDOWN_MEMORY_DB can
        # still be shared, and a database outlives the root it was first keyed to.
        scope = self._resolve_directory(directory or None)
        return self._db.list_documents(str(self._within_root(scope, IndexingError)))

    def get_document_outline(self, file_path: str) -> list[OutlineNode]:
        document = self._resolve_document(file_path)
        return headings.build_outline(self._db.get_sections(document.id))

    def read_section(
        self, file_path: str, heading_path: str, *, include_subsections: bool = False
    ) -> str:
        document = self._resolve_document(file_path)
        sections = self._db.get_sections(document.id)
        matched = headings.select_sections(
            sections, heading_path, include_subsections=include_subsections
        )
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
            status = self._root_status()
            # The root's own status is what a search reads, so it is also what decides
            # whether the background run is due - there is no other look at the disk.
            if self._auto is not None:
                self._auto.consider(status)
            return self._with_indexing(status)
        # Narrowed in one read, not composed from two: coverage stays the root's - that is
        # the tree every answer is drawn from - while the failures and stale documents
        # named are the ones that live here. Resolved against the root this service was
        # built for, never against the configured path again, or a retargeted symlink
        # pairs this root's certificate with another tree's failures.
        scope = self._within_root(
            headings._absolute(Path(self._root) / directory.strip(), SearchError), SearchError
        )
        return self._with_indexing(
            self._with_freshness(self._db.index_status(self._root, str(scope)), str(scope))
        )

    def _root_status(self) -> IndexStatus:
        return self._with_freshness(self._db.index_status(self._root), self._root)

    def _with_indexing(self, status: IndexStatus) -> IndexStatus:
        active = self._auto is not None and self._auto.active
        return dataclasses.replace(status, indexing=active) if active else status

    def _with_freshness(self, status: IndexStatus, scope: str) -> IndexStatus:
        """Add what only the filesystem knows: which indexed files have moved on.

        The database's own status is one SQLite snapshot and deliberately says nothing
        about the disk, so this is composed here rather than there. It asks only about
        rows the index holds - a file nobody has indexed yet is found by walking the tree,
        which is the expensive half of indexing and not something a search should pay for.
        """
        return dataclasses.replace(status, changed_files=self._freshness.changed_files(scope))

    # ------------------------------------------------------------------ resolution

    def _resolve_directory(self, directory: str | None) -> Path:
        """Resolve against the root this server settled on, never the configured spelling.

        The docs root is resolved once at construction precisely so a retargeted symlink
        cannot make the server answer from one tree while reporting on another. Resolving
        the configured path again here reopened that door from the other side: indexing
        followed the link to its new target and wrote rows the frozen root can never see,
        `list_documents()` with no argument then resolved outside its own root and raised,
        and a restart keyed a different database and read as never indexed.
        """
        if directory is None or not directory.strip():
            return Path(self._root)
        path = headings._user_path(directory.strip(), IndexingError)
        if not path.is_absolute():
            path = Path(self._root) / path
        return headings._absolute(path, IndexingError)

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
        path = headings._user_path(requested, DocumentNotFoundError)
        candidates = [path]
        if not path.is_absolute():
            candidates = [self._config.docs_dir / path]
            try:
                candidates.append(Path.cwd() / path)
            except OSError:  # the working directory was deleted under the server
                logger.debug("Working directory is gone; not resolving %s against it", requested)
        for candidate in candidates:
            resolved = headings._absolute(candidate, DocumentNotFoundError)
            document = self._db.get_document(str(resolved))
            if document is not None:
                return document
        matches = self._db.find_documents_by_suffix(requested)
        if len(matches) == 1:
            return matches[0]
        if matches:
            listing = ", ".join(match.file_path for match in matches[: headings._MAX_LISTED_PATHS])
            raise DocumentNotFoundError(f"'{requested}' is ambiguous; it matches: {listing}")
        raise DocumentNotFoundError(
            f"'{requested}' is not indexed. Run index_directory, then list_documents "
            "to see the available paths."
        )


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
        "markdown-memory", version=__version__, instructions=SERVER_INSTRUCTIONS, lifespan=lifespan
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

        Returns `{"documents": [...], "index_status": {...}}`. `index_status.changed_files`
        counts indexed documents that no longer match the index; it is independent of
        coverage, so read `index_status.message` whenever either is set.
        `index_status.coverage` is
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
        `limit` sections. `index_status.changed_files` counts indexed documents that no
        longer match the index - a hit may quote text that is no longer there - and is
        independent of coverage: it can be non-zero while coverage reads "verified", so
        read `index_status.message` whenever either is set.
        When `index_status.coverage` is "unknown", what you searched is
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


def _download_model(config: ServerConfig) -> None:
    """Fetch and load the configured embedder, so no client waits on the first download."""
    try:
        create_embedder(config.embedder, cache_dir=config.model_cache_dir).warm_up()
    except MarkdownMemoryError:
        logger.exception("Cannot download the embedding model")
        raise SystemExit(1) from None
    logger.info("Embedding model %s is ready in %s", config.embedder, config.model_cache_dir)


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
        f"(env {ENV_EXCLUDE}, comma separated)",
    )
    parser.add_argument(
        "--no-auto-index",
        action="store_true",
        help=f"Do not keep the docs root indexed in the background (env {ENV_AUTO_INDEX}=0)",
    )
    parser.add_argument(
        "--embedder",
        choices=("embeddinggemma", "bge-small"),
        help=f"Embedding model preset (env {ENV_EMBEDDER}; default {DEFAULT_EMBEDDER})",
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download and load the configured embedding model, then exit",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    arguments = parser.parse_args(argv)

    configure_logging(arguments.log_level)
    config = _config_from_cli(arguments)
    if arguments.download_model:
        _download_model(config)
        return
    try:
        service = MarkdownMemoryService(config)
    except MarkdownMemoryError:
        logger.exception("Cannot start markdown-memory")
        raise SystemExit(1) from None
    logger.info("markdown-memory serving; db=%s docs_dir=%s", config.db_path, config.docs_dir)
    # Nothing heavy starts here. Loading the model holds the GIL for seconds, and a client
    # waits on the handshake with a short timeout - Codex's is 10 s. The first search loads
    # it instead, and starts the catch-up index run on its way out.
    if config.auto_index:
        service.start_auto_index(request=False)
    try:
        create_server(config, service=service).run("stdio")
    finally:
        service.close()


if __name__ == "__main__":
    main()
