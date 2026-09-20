"""Server regressions: heading-path resolution, service ownership, the CLI entry point.

One test per defect found in code review; each fails when its fix is reverted.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from fakes import FakeEmbedder
from mcp import Client
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

import markdown_memory.server as server_module
from markdown_memory.exceptions import ConfigurationError, DatabaseError
from markdown_memory.models import (
    OutlineNode,
)
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
        assert not Path(environment["MARKDOWN_MEMORY_DB"]).is_absolute()
