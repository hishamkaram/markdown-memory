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
import json
import logging
import os
import sys
import threading
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import ParamSpec, TypeVar

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import JsonValue

from markdown_memory import __version__, discovery, headings, trees
from markdown_memory.autoindex import AutoIndexer
from markdown_memory.config import (
    ENV_AUTO_INDEX,
    ENV_DB_PATH,
    ENV_DOCS_DIR,
    ENV_EMBEDDER,
    ENV_EXCLUDE,
    ENV_GITIGNORE,
    ENV_LOG_LEVEL,
    ServerConfig,
    _config_from_cli,
    tree_database,
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
    WorkTreeError,
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
    SearchPage,
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
    "get_document_outline followed by read_section to fetch one heading's text. search_docs "
    "returns the best section in full - or, marked excerpt: true, the passage that matched "
    "and its neighbours, when read_section(file_path, heading_path) returns the full "
    "section at the tokens it states - and pointers to the rest: follow a pointer with "
    "read_section, heading_path verbatim, only when the first is not enough. file_path is "
    "relative to index_status.root (absolute when another work tree answered): pass it "
    "verbatim, with the same cwd. When its "
    "keyword_match is no_match, no indexed section contains the searched terms: an identifier "
    "lookup then returns no results, and must be reported as not in the indexed docs - "
    "unless index_status.indexing is true and coverage unknown, when it may only not be "
    "indexed yet. "
    "Pass your working directory as cwd on every call: in a git worktree the answers then "
    "come from that worktree's own copy of the docs, and index_status.root names the tree "
    "that answered. index_status.gitignore is applied, off, no_repository, unavailable (git "
    "could not be asked; the message says how to see why) or unknown - no run has recorded "
    "git's state yet, which is not a failure. Prefer these tools over reading whole Markdown "
    "files."
)


class MarkdownMemoryService:
    """Application service behind the MCP tools; returns typed domain models."""

    def __init__(
        self,
        config: ServerConfig,
        embedder: Embedder | None = None,
        *,
        run_lock: threading.Lock | None = None,
        donor: MarkdownMemoryService | None = None,
    ) -> None:
        self._config = config
        self._embedder: Embedder = embedder or create_embedder(
            config.embedder, cache_dir=config.model_cache_dir
        )
        # Resolved, because indexing resolves: a document under a symlinked or relative
        # docs root is stored by its real path, and a scope spelled any other way filters
        # every one of them out and returns nothing. Resolved ONCE, and reused: resolving
        # again per call lets a retargeted symlink answer from one tree while reporting on
        # another, which is a lie told with two correct halves.
        self._root = str(headings._absolute(self._config.docs_dir, SearchError))
        self._db = Database(config.db_path, embedding_dim=self._embedder.dimension)
        self._run_lock = run_lock or threading.Lock()
        self._indexer = Indexer(
            self._db,
            self._embedder,
            workers=config.index_workers,
            exclude=config.exclude,
            gitignore=config.gitignore,
            # Another work tree of the repository mostly holds the same text as the root
            # it was branched from, and that root's vectors for it are already stored.
            reuse=(
                None
                if donor is None
                else functools.partial(_donated, donor.db, Path(donor.root), Path(self._root))
            ),
            run_lock=self._run_lock,
        )
        self._freshness = FreshnessSweep(self._db)
        self._searcher = HybridSearcher(self._db, self._embedder, scope=self._root)
        #: Only the stdio server starts one (`main`): a service built by a test or a
        #: script does exactly what it is asked and nothing in the background.
        self._auto: AutoIndexer | None = None

    @property
    def db(self) -> Database:
        return self._db

    @property
    def config(self) -> ServerConfig:
        return self._config

    @property
    def root(self) -> str:
        return self._root

    @property
    def run_lock(self) -> threading.Lock:
        return self._run_lock

    @property
    def auto_indexing(self) -> bool:
        return self._auto is not None

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

    def get_document_outline(self, file_path: str, *, cwd: str = "") -> list[OutlineNode]:
        document = self._resolve_document(file_path, cwd)
        return headings.build_outline(self._db.get_sections(document.id))

    def read_section(
        self, file_path: str, heading_path: str, *, include_subsections: bool = False, cwd: str = ""
    ) -> str:
        document = self._resolve_document(file_path, cwd)
        sections = self._db.get_sections(document.id)
        matched = headings.select_sections(
            sections, heading_path, include_subsections=include_subsections
        )
        return join_parts(matched)

    def search_docs(self, query: str, limit: int = 5) -> list[SearchResult]:
        return self._searcher.search(query, limit)

    def search_page(self, query: str, limit: int = 5) -> SearchPage:
        return self._searcher.search_page(query, limit)

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
        # And which tree answered: one server can serve several work trees of a repository.
        active = self._auto is not None and self._auto.active
        return dataclasses.replace(status, indexing=active, root=self._root)

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

    def _resolve_document(self, file_path: str, cwd: str = "") -> Document:
        """Find an indexed document by absolute path, relative path, or unique path suffix.

        A relative path is tried under the root first: it is how this server sends paths, and
        one it sent must come back to the same document whatever the agent's directory holds.
        Only then is it the agent's (`cwd`), or failing that the server's own working directory.
        Each candidate is looked up as spelled before its symlinks are followed: a link the
        index kept is stored under its own name.
        """
        requested = file_path.strip()
        if not requested:
            raise DocumentNotFoundError("file_path must not be empty")
        path = headings._user_path(requested, DocumentNotFoundError)
        candidates = [path]
        if not path.is_absolute() and cwd.strip():
            candidates = [Path(self._root) / path, Path(cwd.strip()).expanduser() / path]
        elif not path.is_absolute():
            candidates = [Path(self._root) / path]
            try:
                candidates.append(Path.cwd() / path)
            except OSError:  # the working directory was deleted under the server
                logger.debug("Working directory is gone; not resolving %s against it", requested)
        for candidate in dict.fromkeys(candidates):
            resolved = str(headings._absolute(candidate, DocumentNotFoundError))
            spelled = os.path.abspath(candidate)
            document = self._db.get_document(spelled) or self._db.get_document(resolved)
            if document is not None:
                return document
        matches = self._db.find_documents_by_suffix(requested)
        if len(matches) == 1:
            return matches[0]
        if matches:
            listing = ", ".join(match.file_path for match in matches[: headings._MAX_LISTED_PATHS])
            raise DocumentNotFoundError(f"'{requested}' is ambiguous; it matches: {listing}")
        raise DocumentNotFoundError(self._not_indexed(requested, path, candidates))

    def _not_indexed(self, requested: str, path: Path, candidates: Sequence[Path]) -> str:
        """What to tell an agent whose document is not in the index, and what to do instead.

        A bare name was a suffix lookup, and nothing on disk is what it meant. A path the
        agent spelled is judged where it points: the candidate that exists, else the first.
        """
        advice = "Run index_directory, then list_documents to see the available paths."
        if not (path.is_absolute() or "/" in requested or requested.startswith(".")):
            return f"No indexed document matches '{requested}'. {advice}"
        target = next((c for c in candidates if os.path.lexists(c)), candidates[0])
        reason = discovery.unindexed_reason(
            Path(os.path.abspath(target)),
            Path(self._root),
            self._config.exclude,
            self._config.gitignore,
        )
        if reason is not None:
            return reason
        shown = discovery._printable(os.path.abspath(target))
        if self._auto is not None and self._auto.active:
            return (
                f"'{shown}' is not indexed yet: a run is indexing {self._root} now; try again "
                "when index_status.indexing is false."
            )
        return f"'{shown}' is not indexed. {advice}"


# ---------------------------------------------------------------------- MCP wiring


def _donated(
    donor: Database,
    donor_root: Path,
    root: Path,
    file_path: str,
    texts: Sequence[str],
    weights: str,
) -> Mapping[str, Sequence[float]]:
    """The donor root's stored vectors for the same file there, by the texts they embed."""
    try:
        relative = Path(file_path).relative_to(root)
    except ValueError:  # indexed from outside this tree: there is no same file to ask about
        return {}
    return donor.passage_vectors(str(donor_root / relative), texts, weights)


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
    """Hands each tool call the service for its work tree, and owns their lifetime.

    The configured service may be passed in, and is then never closed here. One the server
    creates itself lives as long as any session is open, and is created again on demand
    afterwards, so the same server object can serve several sessions (or concurrent ones) in
    a row. The services of other work trees are always the provider's own: one per tree,
    created on its first call and closed with the session, never evicted while one is open.
    """

    def __init__(self, config: ServerConfig | None, service: MarkdownMemoryService | None) -> None:
        self._config = config
        self._external = service
        self._owned: MarkdownMemoryService | None = None
        self._trees: dict[str, MarkdownMemoryService] = {}
        self._home: dict[str, trees.Tree | None] = {}
        self._sessions = 0
        self._lock = threading.Lock()

    def get(self, cwd: str = "", path: str | None = None) -> MarkdownMemoryService:
        """The service that answers for ``path``, else for ``cwd``, else the configured one.

        An absolute path decides first, so a pointer a worktree search returned - sent absolute,
        see `relative_root` - is read from that worktree with or without `cwd`, and a worktree
        named to `index_directory` is never written into the configured root's database. A
        tree that is not another checkout of the configured root's repository is answered
        from the configured root, as before.
        """
        configured = self._configured()
        probe = _probe(cwd, path)
        if probe is None:
            return configured
        other = trees.work_tree(probe)
        home = self._home_tree(configured)
        if other is None or home is None or other.common_dir != home.common_dir:
            return configured
        if other.top == home.top:
            return configured
        return self._tree_service(configured, trees.counterpart(Path(configured.root), home, other))

    def relative_root(self, service: MarkdownMemoryService) -> str | None:
        """The root ``service``'s paths are sent relative to, or None to send them absolute (#76).

        Only the configured root's: a relative path cannot say which work tree it belongs to,
        so one from another tree, followed without `cwd`, would be read from the configured
        checkout's copy of the same file. Another tree's answers keep their absolute paths.
        """
        return service.root if service is self._configured() else None

    def _configured(self) -> MarkdownMemoryService:
        if self._external is not None:
            return self._external
        with self._lock:
            if self._owned is None:
                self._owned = MarkdownMemoryService(self._config or ServerConfig.from_env())
            return self._owned

    def _home_tree(self, configured: MarkdownMemoryService) -> trees.Tree | None:
        """The configured root's own work tree, asked once: it does not move under a server."""
        with self._lock:
            if configured.root not in self._home:
                try:
                    self._home[configured.root] = trees.work_tree(Path(configured.root))
                except WorkTreeError as exc:
                    logger.warning("Serving %s alone: %s", configured.root, exc)
                    self._home[configured.root] = None
            return self._home[configured.root]

    def _tree_service(self, configured: MarkdownMemoryService, docs: Path) -> MarkdownMemoryService:
        with self._lock:
            tree = self._trees.get(str(docs))
            if tree is not None:
                return tree
            if len(self._trees) >= trees.MAX_TREES:
                raise WorkTreeError(
                    f"This server already answers for {trees.MAX_TREES} other work trees "
                    f"besides {configured.root}; {docs} would be one more. Call without `cwd` "
                    "to search the configured root, or restart the server."
                )
            database = tree_database(configured.config, docs)
            if os.path.realpath(database) == os.path.realpath(configured.config.db_path):
                raise WorkTreeError(f"{docs} would share the configured root's database")
            tree = MarkdownMemoryService(
                dataclasses.replace(configured.config, docs_dir=docs, db_path=database),
                embedder=configured.embedder,
                run_lock=configured.run_lock,
                donor=configured,
            )
            # Armed, not started: the first search starts it, as for the configured root,
            # and an explicit index_directory that created this tree is not then refused
            # as busy by a run of its own making.
            if configured.auto_indexing:
                tree.start_auto_index(request=False)
            self._trees[str(docs)] = tree
            return tree

    def session_started(self) -> None:
        with self._lock:
            self._sessions += 1

    def session_ended(self) -> None:
        with self._lock:
            self._sessions -= 1
            if self._sessions > 0:
                return
            # Before the configured service: their runs read its database for vectors.
            closing = list(self._trees.values())
            self._trees = {}
            owned, self._owned = self._owned, None
        for service in closing:
            service.close()
        if owned is not None:
            owned.close()


def _anchored(directory: str | None, cwd: str) -> str | None:
    """A relative directory the agent names is its own, when it exists where the agent is.

    Otherwise it keeps meaning what it always did: relative to the documentation root.
    """
    if directory is None or not directory.strip() or not cwd.strip():
        return directory
    named = headings._user_path(directory.strip(), WorkTreeError)
    here = headings._user_path(cwd.strip(), WorkTreeError) / named
    return str(here) if not named.is_absolute() and here.is_dir() else directory


def _probe(cwd: str, path: str | None) -> Path | None:
    """What decides the work tree of a call: an absolute path it names, else its `cwd`."""
    if path is not None and path.strip():
        named = headings._user_path(path.strip(), WorkTreeError)
        if named.is_absolute():
            return named
    if not cwd.strip():
        return None
    directory = headings._user_path(cwd.strip(), WorkTreeError)
    if not directory.is_absolute() or not directory.is_dir():
        raise WorkTreeError(
            f"cwd must be the absolute path of an existing directory, not {cwd.strip()!r}"
        )
    return directory


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

    # Every result goes out once, as text: a tool with an output schema is sent twice, as
    # indented text and as `structuredContent`, and a client that shows the model the whole
    # result (Codex) pays for both. Claude Code, which shows the structured copy, sees the
    # same compact JSON either way.
    @server.tool(structured_output=False)
    @anticipated_errors
    def index_directory(directory: str | None = None, cwd: str = "") -> str:
        """Scan a directory tree for Markdown files and (re)index new or changed ones.

        Unchanged files are skipped via SHA-256 hashes; files deleted from disk are purged.
        Omit `directory` to index the documentation root that answers for `cwd`.
        `cwd` is your working directory: pass it on every call. In a git worktree the answer then
        comes from that worktree's own copy of the docs.
        """
        directory = _anchored(directory, cwd)
        return services.get(cwd, directory).index_directory(directory).summary()

    @server.tool(structured_output=False)
    @anticipated_errors
    def list_documents(directory: str = "", cwd: str = "") -> str:
        """List indexed documents (path, title, section count), optionally under `directory`.

        Returns JSON `{"documents": [...], "index_status": {...}}`. `index_status` is the one
        search_docs describes: `root`, `coverage`, `changed_files`, `message`, and `gitignore`
        (`applied`, `off`, `no_repository`, `unavailable` or `unknown`). Each `file_path` is
        relative to `index_status.root` (absolute when another worktree answered), as in
        search_docs.
        `cwd` is your working directory: pass it on every call. In a git worktree the answer then
        comes from that worktree's own copy of the docs.
        """
        directory = _anchored(directory, cwd) or ""
        service = services.get(cwd, directory)
        scope = directory if directory.strip() else None
        root = services.relative_root(service)
        return _json(
            {
                "documents": [
                    _relative(summary.to_dict(), root)
                    for summary in service.list_documents(directory)
                ],
                "index_status": _relative_failures(service.index_status(scope).to_dict(), root),
            }
        )

    @server.tool(structured_output=False)
    @anticipated_errors
    def get_document_outline(file_path: str, cwd: str = "") -> str:
        """Hierarchical table of contents of one document, as JSON: heading paths, line ranges and
        token estimates. Costs a few hundred tokens; use it to pick a section to read.
        `cwd` is your working directory: pass it on every call. In a git worktree the answer then
        comes from that worktree's own copy of the docs.
        """
        service = services.get(cwd, file_path)
        return _json([node.to_dict() for node in service.get_document_outline(file_path, cwd=cwd)])

    @server.tool(structured_output=False)
    @anticipated_errors
    def read_section(
        file_path: str, heading_path: str, include_subsections: bool = False, cwd: str = ""
    ) -> str:
        """Return the verbatim Markdown of one section, addressed by its breadcrumb
        (`Root > Child > Subchild`, as shown by get_document_outline). By default only the
        section's own text is returned; set `include_subsections` to append its children.
        `cwd` is your working directory: pass it on every call. In a git worktree the answer then
        comes from that worktree's own copy of the docs.
        """
        return services.get(cwd, file_path).read_section(
            file_path, heading_path, include_subsections=include_subsections, cwd=cwd
        )

    @server.tool(structured_output=False)
    @anticipated_errors
    def search_docs(query: str, limit: int = 5, cwd: str = "") -> str:
        """Hybrid search (BM25 keywords + semantic vectors, fused with RRF) over all indexed
        sections. Works for exact identifiers (flags, env vars) and for conceptual questions.

        Returns JSON `{"results": [...], "keyword_match": ..., "index_status": {...}}`, at most
        `limit` hits, best first. The first carries the section's full `content`, or - marked
        `excerpt: true`, with `lines` naming them - the passage that matched and its neighbours,
        verbatim; `read_section(file_path, heading_path)` then returns the full section at its
        `tokens`. The rest are pointers - `file_path`, `heading_path`, `lines`, `tokens` (what
        reading it costs) and, when a passage won the vector ranking, a `matched_passage`
        preview of it. Follow one with
        `read_section(file_path, heading_path)` only when the first hit does not answer, passing
        `heading_path` verbatim: for a `(Part n)` of a split section the base path returns every
        part, and the pointer carries a `part_preview` of how its part begins.
        `keyword_match` other than "matched" comes with a `keyword_message`, and the hits are
        semantic neighbours only. Only "no_match" means no indexed section contains the searched
        terms; a query made only of identifiers then returns no results at all, as the
        identifier is not in the indexed documentation - whatever its neighbours would say.
        `file_path` is relative to `index_status.root`, the documentation root that answered
        (absolute when another work tree answered): pass it verbatim, with the same `cwd`.
        `coverage` "unknown" means
        what you searched is missing part of its documentation, or was never indexed end to end:
        an answer drawn from it may be confidently incomplete. `changed_files` counts indexed
        documents that no longer match the index - a hit may quote text that is no longer there -
        and can be non-zero while coverage reads "verified". Read `message` whenever either is
        set: it says what to run. `gitignore` is what git said when a run of the root last
        finished: `applied`, `off` (switched off), `no_repository`, `unavailable` (git could not
        be asked, so ignored files may be indexed; the message says how to see why) or `unknown`
        (no run has recorded it yet - not a failure, and nothing to do).
        `cwd` is your working directory: pass it on every call. In a git worktree the answer then
        comes from that worktree's own copy of the docs.
        """
        service = services.get(cwd)
        root = services.relative_root(service)
        page = service.search_page(query, limit)
        payload: JsonDict = {
            "results": [
                _relative(hit.to_dict() if rank == 0 else hit.to_pointer(), root)
                for rank, hit in enumerate(page.results)
            ],
            "keyword_match": page.keyword_match,
        }
        status = service.index_status()
        message = page.keyword_message(status)
        if message is not None:
            payload["keyword_message"] = message
        payload["index_status"] = _relative_failures(status.to_dict(), root)
        return _json(payload)

    return server


def _relative(item: JsonDict, root: str | None) -> JsonDict:
    """``item`` with its ``file_path`` relative to ``root`` (POSIX separators), when under it.

    The wire is the only place a path is shortened (#76): everything inside the server - and
    every script that calls the service directly - keeps the absolute path it stored.
    """
    path = item.get("file_path")
    if root is None or not isinstance(path, str) or not Path(path).is_relative_to(root):
        return item
    return {**item, "file_path": Path(path).relative_to(root).as_posix()}


def _relative_failures(status: JsonDict, root: str | None) -> JsonDict:
    """An `index_status` whose failures name files relative to ``root``; ``root`` stays absolute."""
    failures = status.get("failures")
    if root is None or not isinstance(failures, list):
        return status
    shown = [
        _relative(failure, root) if isinstance(failure, dict) else failure for failure in failures
    ]
    return {**status, "failures": shown}


def _json(value: JsonValue) -> str:
    """A tool result as compact JSON: indentation is paid for by the reader, on every call."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


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
        "--no-gitignore",
        action="store_true",
        help=f"Index what git ignores below the docs root too (env {ENV_GITIGNORE}=0)",
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
