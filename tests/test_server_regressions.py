"""Server regressions: heading-path resolution, service ownership, the CLI entry point.

One test per defect found in code review; each fails when its fix is reverted.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from fakes import FakeEmbedder
from mcp import Client
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

import markdown_memory.server as server_module
from markdown_memory.exceptions import (
    ConfigurationError,
    DatabaseError,
    DocumentNotFoundError,
    IndexingError,
    SearchError,
)
from markdown_memory.models import (
    OutlineNode,
)
from markdown_memory.search import HybridSearcher
from markdown_memory.server import MarkdownMemoryService, ServerConfig, create_server

ARROWS = """# API

## Step 1 -> Step 2

arrow body

### Detail

detail body

## x >= 5

compare body

## `Result<T, E>` handling

generic body

## Setup

lower body

## SETUP

upper body
"""

ArrowService = tuple[MarkdownMemoryService, "MCPServer[None]"]


@pytest.fixture
def arrow_service(tmp_path: Path) -> Iterator[ArrowService]:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "api.md").write_text(ARROWS, encoding="utf-8")
    body = "\n\n".join(f"Paragraph {n}. " + "words and more words " * 25 for n in range(10))
    (docs / "big.md").write_text(f"# Big\n\n## Huge\n\n{body}\n\n### Child\n\nchild body\n")
    service = MarkdownMemoryService(
        ServerConfig(db_path=tmp_path / "s.db", docs_dir=docs), embedder=FakeEmbedder()
    )
    service.index_directory()
    yield service, create_server(service=service)
    service.close()


class TestHeadingPathResolution:
    def test_paths_shown_by_the_outline_can_always_be_read(
        self, arrow_service: ArrowService
    ) -> None:
        service, _ = arrow_service

        def walk(nodes: Sequence[OutlineNode]) -> Iterator[str]:
            for node in nodes:
                yield node.heading_path
                yield from walk(node.children)

        listed = list(walk(service.get_document_outline("api.md")))
        assert "API > Step 1 -> Step 2" in listed
        assert "API > Result<T, E> handling" in listed
        for heading_path in listed:
            assert service.read_section("api.md", heading_path)

    def test_titles_containing_a_greater_than_sign(self, arrow_service: ArrowService) -> None:
        service, _ = arrow_service
        assert "arrow body" in service.read_section("api.md", "API > Step 1 -> Step 2")
        assert "arrow body" in service.read_section("api.md", "Step 1 -> Step 2")
        assert "arrow body" in service.read_section("api.md", "api>step 1 -> step 2")
        assert "compare body" in service.read_section("api.md", "API > x >= 5")
        assert "generic body" in service.read_section("api.md", "Result<T, E> handling")
        assert "detail body" in service.read_section("api.md", "API > Step 1 -> Step 2 > Detail")
        subtree = service.read_section("api.md", "Step 1 -> Step 2", include_subsections=True)
        assert "arrow body" in subtree and "detail body" in subtree
        assert "compare body" not in subtree

    def test_siblings_differing_only_in_case(self, arrow_service: ArrowService) -> None:
        service, _ = arrow_service
        assert "lower body" in service.read_section("api.md", "API > Setup")
        assert "upper body" in service.read_section("api.md", "API > SETUP")
        assert "upper body" in service.read_section("api.md", "SETUP")
        with pytest.raises(Exception, match="ambiguous") as raised:
            service.read_section("api.md", "api > setup")
        assert "API > Setup | API > SETUP" in str(raised.value)

    def test_part_path_with_include_subsections(self, arrow_service: ArrowService) -> None:
        service, _ = arrow_service
        plain = service.read_section("big.md", "Big > Huge (Part 2)")
        assert plain == service.read_section(
            "big.md", "Big > Huge (Part 2)", include_subsections=True
        )
        whole = service.read_section("big.md", "Big > Huge", include_subsections=True)
        assert plain in whole and whole.endswith("child body")

    async def test_unusable_paths_are_anticipated_errors(self, arrow_service: ArrowService) -> None:
        _, server = arrow_service
        for arguments in (
            {"file_path": "~no_such_user_zz/x.md"},
            {"file_path": "bad\x00name.md"},
        ):
            with pytest.raises(ToolError) as raised:
                await server.call_tool("get_document_outline", arguments)
            assert not isinstance(raised.value, UnexpectedToolError), arguments
        for directory in ("~no_such_user_zz/docs", "bad\x00dir"):
            with pytest.raises(ToolError) as raised:
                await server.call_tool("index_directory", {"directory": directory})
            assert not isinstance(raised.value, UnexpectedToolError), directory


class TestServiceOwnership:
    async def test_lifespan_leaves_a_callers_service_open(self, tmp_path: Path) -> None:
        service = MarkdownMemoryService(
            ServerConfig(db_path=tmp_path / "own.db", docs_dir=tmp_path), embedder=FakeEmbedder()
        )
        server: MCPServer[None] = create_server(service=service)
        try:
            for _ in range(2):  # a second session on the same server must still work
                async with Client(server) as client:
                    outcome = await client.call_tool("list_documents", {})
                    assert not outcome.is_error
            assert service.list_documents() == []
        finally:
            service.close()

    async def test_server_closes_a_service_it_created_itself(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created: list[MarkdownMemoryService] = []

        class Tracked(MarkdownMemoryService):
            def __init__(self, config: ServerConfig) -> None:
                super().__init__(config, embedder=FakeEmbedder())
                created.append(self)

        monkeypatch.setattr(server_module, "MarkdownMemoryService", Tracked)
        server = create_server(ServerConfig(db_path=tmp_path / "owned.db", docs_dir=tmp_path))
        async with Client(server) as client:
            assert not (await client.call_tool("list_documents", {})).is_error
        with pytest.raises(DatabaseError, match="closed"):
            created[0].list_documents()


class TestMainEntrypoint:
    @pytest.fixture
    def harness(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, object]:
        seen: dict[str, object] = {}

        class FakeServer:
            def run(self, transport: str) -> None:
                seen["transport"] = transport

        def fake_create_server(
            config: ServerConfig, *, service: MarkdownMemoryService
        ) -> FakeServer:
            seen["config"] = config
            seen["service"] = service
            return FakeServer()

        monkeypatch.setattr(server_module, "create_server", fake_create_server)
        monkeypatch.setattr(server_module, "_warm_up", lambda embedder: None)
        monkeypatch.setattr(
            server_module, "configure_logging", lambda level=None: seen.update(level=level)
        )
        monkeypatch.setenv("MARKDOWN_MEMORY_DB", str(tmp_path / "env.db"))
        monkeypatch.setenv("MARKDOWN_MEMORY_DOCS_DIR", str(tmp_path / "env-docs"))
        return seen

    def test_environment_configures_the_server(
        self, harness: dict[str, object], tmp_path: Path
    ) -> None:
        server_module.main([])
        config = harness["config"]
        assert isinstance(config, ServerConfig)
        assert config.db_path == tmp_path / "env.db"
        assert config.docs_dir == tmp_path / "env-docs"
        assert harness["transport"] == "stdio"
        assert harness["level"] is None

    def test_flags_override_the_environment(
        self, harness: dict[str, object], tmp_path: Path
    ) -> None:
        server_module.main(
            ["--db", str(tmp_path / "flag.db"), "--docs-dir", str(tmp_path), "--log-level", "debug"]
        )
        config = harness["config"]
        assert isinstance(config, ServerConfig)
        assert (config.db_path, config.docs_dir) == (tmp_path / "flag.db", tmp_path)
        assert harness["level"] == "debug"
        assert (tmp_path / "flag.db").exists()

    def test_service_is_closed_when_the_server_stops(self, harness: dict[str, object]) -> None:
        server_module.main([])
        service = harness["service"]
        assert isinstance(service, MarkdownMemoryService)
        with pytest.raises(DatabaseError, match="closed"):
            service.list_documents()

    def test_unusable_database_exits_with_status_one(
        self, harness: dict[str, object], tmp_path: Path
    ) -> None:
        blocker = tmp_path / "file"
        blocker.write_text("x")
        with pytest.raises(SystemExit) as raised:
            server_module.main(["--db", str(blocker / "nested" / "index.db")])
        assert raised.value.code == 1
        assert "service" not in harness


class TestServerLifecycleRound2:
    def test_relative_lookup_survives_a_deleted_working_directory(
        self, arrow_service: ArrowService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service, _ = arrow_service
        doomed = tmp_path / "doomed"
        doomed.mkdir()
        monkeypatch.chdir(doomed)
        doomed.rmdir()
        try:
            with pytest.raises(OSError):
                Path.cwd()
            assert "lower body" in service.read_section("api.md", "API > Setup")
        finally:
            os.chdir(tmp_path)  # give monkeypatch a directory it can restore from

    async def test_server_that_owns_its_service_serves_many_sessions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created: list[MarkdownMemoryService] = []

        class Tracked(MarkdownMemoryService):
            def __init__(self, config: ServerConfig) -> None:
                super().__init__(config, embedder=FakeEmbedder())
                created.append(self)

        monkeypatch.setattr(server_module, "MarkdownMemoryService", Tracked)
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.md").write_text("# A\n\nsearchable body text\n")
        server = create_server(ServerConfig(db_path=tmp_path / "multi.db", docs_dir=docs))
        for session in range(3):
            async with Client(server) as client:
                assert not (await client.call_tool("index_directory", {})).is_error
                found = await client.call_tool("search_docs", {"query": "searchable body"})
                assert not found.is_error, session
        assert len(created) == 3  # one per session, each closed when its session ended
        for service in created:
            with pytest.raises(DatabaseError, match="closed"):
                service.list_documents()

    async def test_overlapping_sessions_share_one_owned_service(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created: list[MarkdownMemoryService] = []

        class Tracked(MarkdownMemoryService):
            def __init__(self, config: ServerConfig) -> None:
                super().__init__(config, embedder=FakeEmbedder())
                created.append(self)

        monkeypatch.setattr(server_module, "MarkdownMemoryService", Tracked)
        server = create_server(ServerConfig(db_path=tmp_path / "overlap.db", docs_dir=tmp_path))
        async with Client(server) as first:
            assert not (await first.call_tool("list_documents", {})).is_error
            async with Client(server) as second:
                assert not (await second.call_tool("list_documents", {})).is_error
            # the inner session ended, the outer one must still work
            assert not (await first.call_tool("list_documents", {})).is_error
        assert len(created) == 1


class TestProjectScopedConfiguration:
    """`.mcp.json` cannot interpolate the project root, so the server resolves it.

    Measured on Claude Code 2.1.278: `${workspaceFolder}` and `${CLAUDE_PROJECT_DIR}` are
    both reported as missing environment variables and passed through as literal text,
    while the spawned server does receive `CLAUDE_PROJECT_DIR` and a working directory of
    the project root.
    """

    @pytest.fixture(autouse=True)
    def clean_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for variable in (
            server_module.ENV_DB_PATH,
            server_module.ENV_DOCS_DIR,
            server_module.ENV_MODEL_CACHE,
            server_module.ENV_EXCLUDE,
            server_module.ENV_PROJECT_DIR,
        ):
            monkeypatch.delenv(variable, raising=False)

    def test_the_docs_root_defaults_to_the_project_claude_code_reports(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(server_module.ENV_PROJECT_DIR, str(tmp_path / "project"))
        monkeypatch.chdir(tmp_path)
        assert ServerConfig.from_env().docs_dir == tmp_path / "project"

    def test_a_relative_database_lands_in_the_project_not_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A project-scoped config has to spell the database relatively; it must not
        follow a working directory that moved."""
        project = tmp_path / "project"
        project.mkdir()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.setenv(server_module.ENV_PROJECT_DIR, str(project))
        monkeypatch.setenv(server_module.ENV_DB_PATH, ".markdown-memory/index.db")
        monkeypatch.chdir(elsewhere)
        assert ServerConfig.from_env().db_path == project / ".markdown-memory/index.db"

    def test_the_working_directory_is_used_when_no_project_is_exported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(server_module.ENV_DOCS_DIR, "docs")
        config = ServerConfig.from_env()
        assert config.docs_dir == tmp_path / "docs"

    def test_an_absolute_path_is_left_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(server_module.ENV_PROJECT_DIR, str(tmp_path / "project"))
        monkeypatch.setenv(server_module.ENV_DOCS_DIR, str(tmp_path / "absolute"))
        assert ServerConfig.from_env().docs_dir == tmp_path / "absolute"

    @pytest.mark.parametrize(
        "value", ["${workspaceFolder}", "${CLAUDE_PROJECT_DIR}/docs", "${MISSING}/index.db"]
    )
    def test_an_unexpanded_variable_is_refused_instead_of_indexed(
        self, value: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Claude Code loads a config it could not expand and passes the text through.

        Treating that as a directory name indexes nothing and reports success, which is
        indistinguishable from a project with no documentation.
        """
        monkeypatch.setenv(server_module.ENV_DOCS_DIR, value)
        with pytest.raises(ConfigurationError) as raised:
            ServerConfig.from_env()
        assert server_module.ENV_DOCS_DIR in str(raised.value)
        assert "relative to the project root" in str(raised.value)

    def test_a_directory_really_named_like_a_variable_is_allowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The rejection is for text nobody expanded, not for an unusual name."""
        odd = tmp_path / "${version}"
        odd.mkdir()
        monkeypatch.setenv(server_module.ENV_DOCS_DIR, str(odd))
        assert ServerConfig.from_env().docs_dir == odd

    def test_the_committed_config_points_at_the_project_it_ships_with(self) -> None:
        """The config in this repository must survive the rules measured above."""
        import json

        config = json.loads((Path(__file__).parent.parent / ".mcp.json").read_text())
        environment = config["mcpServers"]["markdown-memory"]["env"]
        assert not any("${" in value for value in environment.values())
        # The database is keyed on the docs root now, so the shipped config sets no path at
        # all. Any path it does set must stay relative: the server resolves one against the
        # project root, and an absolute path in a committed config belongs to one machine.
        for name in (server_module.ENV_DB_PATH, server_module.ENV_DOCS_DIR):
            if name in environment:
                assert not Path(environment[name]).is_absolute()


class TestOneDatabaseManyProjects:
    """The default database is shared by every project on the machine.

    Found in design review: the docs root defaults to the current project, the database
    defaults to a single path under the user's data directory, and search had no root
    filter - so an agent working in one project got confident answers out of another
    project's documentation.
    """

    @staticmethod
    def tree(root: Path, name: str, body: str) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{name}.md").write_text(f"# {name}\n\n## Retry policy\n\n{body}\n")
        return root

    @pytest.fixture
    def shared(self, tmp_path: Path, fake_embedder: FakeEmbedder) -> tuple[Path, Path, Path]:
        alpha = self.tree(tmp_path / "alpha", "alpha", "alpha service retries twice")
        beta = self.tree(tmp_path / "beta", "beta", "beta service retries twice")
        return tmp_path / "shared.db", alpha, beta

    def service(self, db_path: Path, docs: Path, embedder: FakeEmbedder) -> MarkdownMemoryService:
        service = MarkdownMemoryService(ServerConfig(db_path=db_path, docs_dir=docs), embedder)
        service.index_directory()
        return service

    def test_one_project_never_answers_with_another_project_s_documentation(
        self, shared: tuple[Path, Path, Path], fake_embedder: FakeEmbedder
    ) -> None:
        db_path, alpha, beta = shared
        first = self.service(db_path, alpha, fake_embedder)
        second = self.service(db_path, beta, fake_embedder)
        try:
            for service, own, other in ((first, "alpha", "beta"), (second, "beta", "alpha")):
                results = service.search_docs("retries twice", 5)
                assert results, f"{own} found nothing in its own documentation"
                paths = {Path(result.file_path).name for result in results}
                assert paths == {f"{own}.md"}, f"{own} saw {other}: {paths}"
        finally:
            first.close()
            second.close()

    def test_listing_documents_shows_this_project_only(
        self, shared: tuple[Path, Path, Path], fake_embedder: FakeEmbedder
    ) -> None:
        db_path, alpha, beta = shared
        first = self.service(db_path, alpha, fake_embedder)
        second = self.service(db_path, beta, fake_embedder)
        try:
            listed = {Path(doc.file_path).name for doc in first.list_documents()}
            assert listed == {"alpha.md"}
        finally:
            first.close()
            second.close()

    def test_the_vector_side_keeps_looking_past_a_crowded_neighbour(
        self, tmp_path: Path, fake_embedder: FakeEmbedder
    ) -> None:
        """A vec0 KNN query applies its own k before anything can filter by root.

        With eighty closer sections next door, one page of neighbours is entirely
        someone else's documentation, and this project's own section is never reached.
        The keyword side cannot cover for it: a query that lands only in the vector
        index would come back empty.
        """
        db_path = tmp_path / "shared.db"
        # This project's passage carries unrelated words too, so every neighbour passage
        # sits closer to the query than it does.
        mine = self.tree(tmp_path / "mine", "mine", "retry backoff policy plus local detail here")
        crowd = tmp_path / "crowd"
        crowd.mkdir()
        # More passages than one widened fetch returns (limit 1 -> 4 sections -> 40
        # passages), so reaching this project's section takes another round.
        for index in range(60):
            body = "\n\n".join(["retry backoff policy"] * 3)
            (crowd / f"doc{index}.md").write_text(f"# Doc {index}\n\n## Retry policy\n\n{body}\n")
        neighbour = MarkdownMemoryService(
            ServerConfig(db_path=db_path, docs_dir=crowd), fake_embedder
        )
        neighbour.index_directory()
        neighbour.close()
        service = self.service(db_path, mine, fake_embedder)
        try:
            searcher = HybridSearcher(service.db, fake_embedder, scope=str(mine))
            best, _ = searcher._nearest(fake_embedder.embed_query("retry backoff policy"), 1)
            assert best, "the vector search never reached this project's own section"
        finally:
            service.close()

    def test_a_crowded_neighbour_does_not_empty_the_page(
        self, tmp_path: Path, fake_embedder: FakeEmbedder
    ) -> None:
        """Scoping filters after each index applied its own limit.

        Without over-fetching, a larger root next door fills the candidate list and the
        scoped search hands back a short page - or nothing at all.
        """
        db_path = tmp_path / "shared.db"
        mine = self.tree(tmp_path / "mine", "mine", "retry backoff policy")
        crowd = tmp_path / "crowd"
        crowd.mkdir()
        # Every neighbour matches the query better than the one document that belongs to
        # this project, and there are eighty of them.
        for index in range(80):
            (crowd / f"doc{index}.md").write_text(
                f"# Doc {index}\n\n## Retry policy\n\n"
                "retry backoff policy retry backoff policy retry backoff policy\n"
            )
        neighbour = MarkdownMemoryService(
            ServerConfig(db_path=db_path, docs_dir=crowd), fake_embedder
        )
        neighbour.index_directory()
        neighbour.close()
        service = self.service(db_path, mine, fake_embedder)
        try:
            results = service.search_docs("retry backoff policy", 5)
            assert [Path(result.file_path).name for result in results] == ["mine.md"]
        finally:
            service.close()


class TestTheAnswerSaysWhenItIsIncomplete:
    """Indexing reports its own failures, but almost nothing calls indexing.

    An agent opens a session and searches; it is served from whatever the index holds.
    Until the question is asked at the point of use, a root that lost files to a
    permissions error - or was never indexed at all - answers with confidence and no
    caveat, and the agent concludes the documentation does not cover the thing it could
    not read.
    """

    @staticmethod
    def broken_tree(root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / "good.md").write_text("# Good\n\nretry backoff policy documented here\n")
        broken = root / "broken.md"
        broken.write_text("# Broken\n\nbody\n")
        broken.chmod(0o000)
        return broken

    @staticmethod
    def service(tmp_path: Path, docs: Path, embedder: FakeEmbedder) -> MarkdownMemoryService:
        return MarkdownMemoryService(
            ServerConfig(db_path=tmp_path / "i.db", docs_dir=docs), embedder
        )

    def test_a_search_says_the_index_is_missing_files(
        self, tmp_path: Path, fake_embedder: FakeEmbedder
    ) -> None:
        docs = tmp_path / "docs"
        broken = self.broken_tree(docs)
        service = self.service(tmp_path, docs, fake_embedder)
        try:
            service.index_directory()
        finally:
            broken.chmod(0o644)
        try:
            status = service.index_status()
            assert not status.verified
            assert [failure.file_path for failure in status.failures] == [str(broken)]
            # a fresh service over the same database - the agent's usual case, where
            # nothing re-indexes - must still say it
            second = self.service(tmp_path, docs, fake_embedder)
            try:
                assert not second.index_status().verified
            finally:
                second.close()
        finally:
            service.close()

    def test_a_whole_index_says_nothing(self, tmp_path: Path, fake_embedder: FakeEmbedder) -> None:
        """A caveat on every answer would be ignored by the time it mattered."""
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "good.md").write_text("# Good\n\nall readable\n")
        service = self.service(tmp_path, docs, fake_embedder)
        try:
            service.index_directory()
            status = service.index_status()
            assert status.verified
            assert status.failures == ()
            assert status.message() is None
        finally:
            service.close()

    def test_a_root_nobody_indexed_does_not_claim_to_be_whole(
        self, tmp_path: Path, fake_embedder: FakeEmbedder
    ) -> None:
        """The commonest way to answer from a tree you have not read: never read it.

        Nothing failed, so there is nothing to report file by file. Only the certificate
        can tell "indexed and clean" from "never indexed", and they are not the same
        answer to an agent about to trust what comes back.
        """
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "good.md").write_text("# Good\n\nnever indexed\n")
        service = self.service(tmp_path, docs, fake_embedder)
        try:
            status = service.index_status()
            assert not status.verified
            assert status.failures == ()
            assert status.message() is not None
        finally:
            service.close()

    def test_a_failure_below_the_docs_root_is_not_silent(
        self, tmp_path: Path, fake_embedder: FakeEmbedder
    ) -> None:
        """Indexing is often pointed at one subdirectory; searching is not."""
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "index.md").write_text("# Home\n\nwelcome\n")
        broken = self.broken_tree(docs / "api")
        service = self.service(tmp_path, docs, fake_embedder)
        try:
            # Index the whole root first, so the certificate has something to lose and the
            # assertion below cannot pass merely because nothing ever walked this tree.
            broken.chmod(0o644)
            service.index_directory()
            assert service.index_status().verified
            broken.chmod(0o000)

            # A readable file alongside the broken one, so the run actually writes and
            # the root's certificate is genuinely retracted - otherwise this passes on the
            # failure row alone and says nothing about the retraction it is named for.
            (docs / "api" / "also.md").write_text("# Also\n\nreadable\n")
            service.index_directory("api")
            status = service.index_status()
            assert not status.verified, "the subdirectory failed silently"
            assert [f.file_path for f in status.failures] == [str(broken)]
            assert not service._db.index_status(str(docs)).verified
        finally:
            broken.chmod(0o644)
            service.close()

    def test_fixing_a_file_and_reindexing_its_directory_clears_the_failure(
        self, tmp_path: Path, fake_embedder: FakeEmbedder
    ) -> None:
        """The other direction: a warning that outlives its problem gets ignored."""
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "index.md").write_text("# Home\n\nwelcome\n")
        broken = self.broken_tree(docs / "api")
        service = self.service(tmp_path, docs, fake_embedder)
        try:
            service.index_directory()
            assert service.index_status().failures
            broken.chmod(0o644)
            service.index_directory("api")
            assert service.index_status().failures == (), "the warning outlived the problem"
        finally:
            broken.chmod(0o644)
            service.close()

    def test_listing_one_directory_names_only_its_own_failures(
        self, tmp_path: Path, fake_embedder: FakeEmbedder
    ) -> None:
        """Coverage is the root's; which failures are worth naming is the caller's scope."""
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "index.md").write_text("# Home\n\nwelcome\n")
        broken = self.broken_tree(docs / "api")
        service = self.service(tmp_path, docs, fake_embedder)
        try:
            service.index_directory()
            assert service.index_status("api").failures
            elsewhere = service.index_status(str(docs / "other"))
            assert elsewhere.failures == ()
            assert not elsewhere.verified, "coverage is still the root's, and the root is not"
        finally:
            broken.chmod(0o644)
            service.close()

    def test_a_narrowed_status_is_never_internally_contradictory(
        self, tmp_path: Path, fake_embedder: FakeEmbedder
    ) -> None:
        """Coverage and the failures named have to come from one snapshot.

        Composed from two reads, a status can carry a failure it has just been handed
        while still calling the tree verified - each half true when it was taken, the pair
        never true at once, and the answer says "verified" with the contradiction attached.
        """
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "index.md").write_text("# Home\n\nwelcome\n")
        broken = self.broken_tree(docs / "api")
        service = self.service(tmp_path, docs, fake_embedder)
        try:
            service.index_directory()
            for scope in (None, "api", "."):
                status = service.index_status(scope)
                assert not (status.verified and status.failures), (
                    f"status for {scope!r} called a tree whole while naming what is wrong"
                )
        finally:
            broken.chmod(0o644)
            service.close()

    def test_a_subdirectory_is_not_blamed_for_the_root_s_stale_vectors(
        self, tmp_path: Path, fake_embedder: FakeEmbedder
    ) -> None:
        """Coverage is the root's; what is *named* has to be what lives here.

        Filtering the failures to the requested directory but not the stale-vector count
        tells a clean subdirectory that documents it does not contain were indexed by an
        older format - a caveat about somewhere else, attached to its answers.
        """
        docs = tmp_path / "docs"
        (docs / "api").mkdir(parents=True)
        (docs / "old.md").write_text("# Old\n\nbody\n")
        (docs / "api" / "a.md").write_text("# A\n\nalpha body\n")
        service = self.service(tmp_path, docs, fake_embedder)
        try:
            service.index_directory()
            with service._db.transaction() as conn:  # what a pre-pooling release left
                conn.execute(
                    "UPDATE documents SET vector_format = 1 WHERE file_path LIKE '%old.md'"
                )
            assert service.index_status().stale_vectors == 1
            assert service.index_status("api").stale_vectors == 0, "api wore the root's staleness"
        finally:
            service.close()

    def test_status_and_search_always_describe_the_same_tree(
        self, tmp_path: Path, fake_embedder: FakeEmbedder
    ) -> None:
        """One resolution, used twice - not two resolutions that can drift apart.

        The searcher fixes its scope when the service is built. A status lookup that
        resolves the configured path again on every call answers from the tree it was
        built for while reporting on whatever the path points at now, and both halves are
        individually correct.
        """
        first, second = tmp_path / "real-a", tmp_path / "real-b"
        for root, token in ((first, "alpha-only-token"), (second, "beta-only-token")):
            root.mkdir()
            (root / "guide.md").write_text(f"# Guide\n\n{token} documented here\n")
        # The second tree is damaged, so the two describe themselves differently and this
        # test can tell which one the status is actually about.
        broken = second / "broken.md"
        broken.write_text("# Broken\n\nbody\n")
        broken.chmod(0o000)
        link = tmp_path / "docs"
        link.symlink_to(first, target_is_directory=True)
        service = self.service(tmp_path, link, fake_embedder)
        try:
            service.index_directory()
            link.unlink()
            link.symlink_to(second, target_is_directory=True)  # retargeted underneath it
            service.index_directory()

            hits = service.search_docs("alpha-only-token", 5)
            assert {Path(hit.file_path).parent.name for hit in hits} == {"real-a"}
            assert service.index_status().verified, (
                "status described the retargeted tree while the search answered from the "
                "one this service was built for"
            )
        finally:
            broken.chmod(0o644)
            service.close()

    def test_a_symlinked_docs_root_still_answers(
        self, tmp_path: Path, fake_embedder: FakeEmbedder
    ) -> None:
        """The scope a search filters by has to be the path the documents are stored under."""
        real = tmp_path / "real_docs"
        real.mkdir()
        (real / "guide.md").write_text("# Guide\n\nretry backoff policy documented here\n")
        link = tmp_path / "docs"
        link.symlink_to(real, target_is_directory=True)
        service = self.service(tmp_path, link, fake_embedder)
        try:
            service.index_directory()
            assert service.search_docs("retry backoff policy", 5)
        finally:
            service.close()


class TestADirectoryArgumentCannotLeaveTheRoot:
    """Search was scoped to the docs root; every other way in was not.

    `search_docs` has filtered by the resolved root ever since it answered one project's
    question out of another project's documentation. The `directory` argument reached past
    it three ways - an absolute path, `..`, and a symlink pointing out of the tree - and
    `list_documents` obeyed all three, handing back another project's file paths. The
    status lookup was worse than a leak: coverage stayed the configured root's while the
    failures came from wherever the symlink landed, so the envelope returned
    `coverage: "verified"` beside a non-empty failure list and a null message, which its
    own contract says cannot happen.
    """

    @staticmethod
    def service(tmp_path: Path, embedder: FakeEmbedder) -> tuple[MarkdownMemoryService, Path]:
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "good.md").write_text("# Good\n\nreadable\n")
        outside = tmp_path / "other"
        outside.mkdir()
        (outside / "secret.md").write_text("# Secret\n\nanother project\n")
        service = MarkdownMemoryService(
            ServerConfig(db_path=tmp_path / "index.db", docs_dir=docs), embedder
        )
        service.index_directory()
        return service, outside

    def test_every_spelling_of_outside_is_refused(
        self, tmp_path: Path, fake_embedder: FakeEmbedder
    ) -> None:
        service, outside = self.service(tmp_path, fake_embedder)
        (tmp_path / "docs" / "api").symlink_to(outside)
        try:
            for spelling in ("api", str(outside), "../other"):
                with pytest.raises(IndexingError, match="outside this server"):
                    service.list_documents(spelling)
            with pytest.raises(SearchError, match="outside this server"):
                service.index_status("api")
        finally:
            service.close()

    def test_a_directory_inside_the_root_still_answers(
        self, tmp_path: Path, fake_embedder: FakeEmbedder
    ) -> None:
        """The containment check must not cost the ordinary case its answer."""
        service, _ = self.service(tmp_path, fake_embedder)
        nested = tmp_path / "docs" / "api"
        nested.mkdir()
        (nested / "ref.md").write_text("# Ref\n\nbody\n")
        try:
            service.index_directory()
            assert [d.file_path for d in service.list_documents("api")] == [str(nested / "ref.md")]
            assert service.index_status("api").verified
        finally:
            service.close()


class TestEachProjectKeepsItsOwnIndex:
    """The default database was shared by every project on the machine.

    Search has been scoped to the docs root since it once answered one project's question
    out of another's documentation, but scoping is a filter over a shared file, not
    isolation: a document stays resolvable across the whole database by path or unique
    suffix, and one project's failure rows and coverage certificate sat beside another's.
    The safe arrangement existed - set MARKDOWN_MEMORY_DB - but it was opt-in, so anyone
    who simply ran the server got the unsafe one.

    The index is keyed on the documentation root, and the first attempt keyed it on the
    working directory instead. That closed nothing for a launcher that starts both servers
    from one directory - a CI runner, an editor daemon, a shell that never changed
    directory - because both projects landed in one database again.
    """

    @staticmethod
    def project(root: Path, name: str) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{name}.md").write_text(f"# {name}\n\n## Retry policy\n\n{name} body\n")
        return root

    def test_one_working_directory_two_projects_two_databases(
        self, tmp_path: Path, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        alpha = self.project(tmp_path / "alpha", "alpha")
        beta = self.project(tmp_path / "beta", "beta")
        launcher = tmp_path / "runner"
        launcher.mkdir()
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
        monkeypatch.delenv(server_module.ENV_DB_PATH, raising=False)
        monkeypatch.delenv(server_module.ENV_PROJECT_DIR, raising=False)
        # Neither server is started from its own project: the cwd is the launcher's.
        monkeypatch.chdir(launcher)

        databases = []
        for root in (alpha, beta):
            monkeypatch.setenv(server_module.ENV_DOCS_DIR, str(root))
            config = ServerConfig.from_env()
            databases.append(config.db_path)
            service = MarkdownMemoryService(config, fake_embedder)
            try:
                service.index_directory()
            finally:
                service.close()

        assert databases[0] != databases[1], "two projects shared one database"
        assert not (launcher / ".markdown-memory").exists(), "an index was written to the cwd"
        for root in (alpha, beta):
            assert list(root.iterdir()) == [root / f"{root.name}.md"], (
                "an index was written into the project"
            )

        monkeypatch.setenv(server_module.ENV_DOCS_DIR, str(beta))
        service = MarkdownMemoryService(ServerConfig.from_env(), fake_embedder)
        try:
            assert [Path(d.file_path).name for d in service.list_documents()] == ["beta.md"]
            # Resolution by path or unique suffix is database-wide, so a shared file hands
            # this over however search is scoped.
            with pytest.raises(DocumentNotFoundError):
                service.get_document_outline(str(alpha / "alpha.md"))
        finally:
            service.close()

    def test_two_roots_of_the_same_name_do_not_collide(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The label is only for humans; the digest is what keeps them apart."""
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
        monkeypatch.delenv(server_module.ENV_DB_PATH, raising=False)
        monkeypatch.delenv(server_module.ENV_PROJECT_DIR, raising=False)
        first = tmp_path / "one" / "docs"
        second = tmp_path / "two" / "docs"
        paths = []
        for root in (first, second):
            root.mkdir(parents=True)
            monkeypatch.setenv(server_module.ENV_DOCS_DIR, str(root))
            paths.append(ServerConfig.from_env().db_path)
        assert paths[0] != paths[1], "two roots named 'docs' shared one index"
        assert all(p.parent.name.startswith("docs-") for p in paths)


class TestTheCommandLineRekeysTheDatabase:
    """`--docs-dir` names a different project, so the default database must follow it.

    The command line was laid over a configuration whose database path had already been
    derived from the *environment's* docs root. Two servers launched from one directory
    with different `--docs-dir` therefore shared the launcher's single database and could
    resolve each other's documents - the cross-project leak that keying the database on
    the documentation root exists to close, reached through the one path that skipped it.
    """

    @staticmethod
    def config(docs: Path | None, db: Path | None = None) -> ServerConfig:
        arguments = argparse.Namespace(docs_dir=docs, db=db, embedder=None, exclude=[])
        return server_module._config_from_cli(arguments)

    @pytest.fixture(autouse=True)
    def _launcher(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
        for name in (
            server_module.ENV_DB_PATH,
            server_module.ENV_DOCS_DIR,
            server_module.ENV_PROJECT_DIR,
        ):
            monkeypatch.delenv(name, raising=False)
        launcher = tmp_path / "launcher"
        launcher.mkdir()
        monkeypatch.chdir(launcher)
        yield

    def test_two_docs_dir_flags_do_not_share_the_launcher_s_database(self, tmp_path: Path) -> None:
        alpha, beta = tmp_path / "alpha", tmp_path / "beta"
        for root in (alpha, beta):
            root.mkdir()
        assert self.config(alpha).db_path != self.config(beta).db_path, (
            "two --docs-dir projects shared the launcher's database"
        )

    def test_an_explicit_database_still_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Keying is the default, not a policy: whoever names a database gets it."""
        alpha = tmp_path / "alpha"
        alpha.mkdir()
        flag = tmp_path / "flag.db"
        assert self.config(alpha, flag).db_path == flag, "--db stopped winning"
        monkeypatch.setenv(server_module.ENV_DB_PATH, str(tmp_path / "from_env.db"))
        assert self.config(alpha).db_path == tmp_path / "from_env.db", (
            "MARKDOWN_MEMORY_DB stopped winning"
        )
        assert self.config(alpha, flag).db_path == flag, "the flag lost to the environment"
