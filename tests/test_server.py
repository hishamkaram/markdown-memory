"""MCP tool tests: every tool is invoked through the server, exactly as a client would."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeEmbedder
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

from markdown_memory.exceptions import SectionNotFoundError
from markdown_memory.models import PREAMBLE_TITLE, Section
from markdown_memory.server import (
    MarkdownMemoryService,
    ServerConfig,
    build_outline,
    configure_logging,
    create_server,
    normalize_heading_path,
    select_sections,
)


@pytest.fixture
def docs_dir(tmp_path: Path, fixtures_dir: Path) -> Path:
    root = tmp_path / "docs"
    (root / "nested").mkdir(parents=True)
    for name in ("clean_doc.md", "messy_doc.md"):
        (root / name).write_text(
            (fixtures_dir / name).read_text(encoding="utf-8"), encoding="utf-8"
        )
    (root / "nested" / "clean_doc.md").write_text("# Nested Copy\n\nbody\n", encoding="utf-8")
    return root


@pytest.fixture
def service(tmp_path: Path, docs_dir: Path) -> Iterator[MarkdownMemoryService]:
    config = ServerConfig(db_path=tmp_path / "server.db", docs_dir=docs_dir)
    instance = MarkdownMemoryService(config, embedder=FakeEmbedder())
    yield instance
    instance.close()


@pytest.fixture
def server(service: MarkdownMemoryService) -> MCPServer[None]:
    return create_server(service=service)


async def call(server: MCPServer[None], name: str, **arguments: Any) -> Any:
    """Invoke a tool and return its structured result (the ``result`` wrapper removed).

    A tool that already returns an object is its own structured content; only a bare list
    or scalar is wrapped in ``result``.
    """
    outcome = await server.call_tool(name, arguments)
    assert not outcome.is_error
    assert outcome.structured_content is not None
    content = outcome.structured_content
    return content["result"] if set(content) == {"result"} else content


async def test_the_five_tools_are_registered_with_typed_schemas(server: MCPServer[None]) -> None:
    tools = {tool.name: tool for tool in await server.list_tools()}
    assert set(tools) == {
        "index_directory",
        "list_documents",
        "get_document_outline",
        "read_section",
        "search_docs",
    }
    search_schema = tools["search_docs"].input_schema
    assert search_schema["required"] == ["query"]
    assert search_schema["properties"]["limit"] == {
        "default": 5,
        "title": "Limit",
        "type": "integer",
    }
    assert tools["read_section"].input_schema["required"] == ["file_path", "heading_path"]
    assert tools["index_directory"].input_schema.get("required", []) == []
    for tool in tools.values():
        assert tool.description
        assert tool.output_schema is not None


async def test_index_directory_defaults_to_the_configured_root(server: MCPServer[None]) -> None:
    summary = await call(server, "index_directory")
    assert "3 scanned, 3 (re)indexed, 0 unchanged, 0 purged" in summary
    again = await call(server, "index_directory")
    assert "0 (re)indexed, 3 unchanged" in again


async def test_index_directory_accepts_relative_and_absolute_paths(
    server: MCPServer[None], docs_dir: Path
) -> None:
    assert "1 scanned" in await call(server, "index_directory", directory="nested")
    assert "1 scanned" in await call(server, "index_directory", directory=str(docs_dir / "nested"))


async def test_list_documents(server: MCPServer[None], docs_dir: Path) -> None:
    await call(server, "index_directory")
    answer = await call(server, "list_documents")
    assert set(answer) == {"documents", "index_status"}
    assert answer["index_status"]["coverage"] == "verified"
    assert answer["index_status"]["failures"] == []
    assert answer["index_status"]["message"] is None
    documents = answer["documents"]
    assert [Path(d["file_path"]).name for d in documents] == [
        "clean_doc.md",
        "messy_doc.md",
        "clean_doc.md",
    ]
    clean = documents[0]
    assert clean["title"] == "Orbit Gateway"
    assert clean["section_count"] == 11
    assert set(clean) == {"file_path", "title", "section_count", "last_modified"}
    nested = await call(server, "list_documents", directory="nested")
    assert [d["title"] for d in nested["documents"]] == ["Nested Copy"]
    missing = await call(server, "list_documents", directory=str(docs_dir / "missing"))
    assert missing["documents"] == []


async def test_outline_is_hierarchical_with_lines_and_token_estimates(
    server: MCPServer[None], docs_dir: Path, clean_doc: str
) -> None:
    await call(server, "index_directory")
    outline = await call(server, "get_document_outline", file_path=str(docs_dir / "clean_doc.md"))
    assert [node["title"] for node in outline] == [PREAMBLE_TITLE, "Orbit Gateway"]
    root = outline[1]
    assert root["level"] == 1
    assert [child["title"] for child in root["children"]] == [
        "Installation",
        "Configuration",
        "Operations",
        "License",
    ]
    configuration = root["children"][1]
    assert [child["heading_path"] for child in configuration["children"]] == [
        "Orbit Gateway > Configuration > Environment Variables",
        "Orbit Gateway > Configuration > Command Line Flags",
    ]
    env = configuration["children"][0]
    start, end = (int(number) for number in env["lines"].split("-"))
    assert clean_doc.split("\n")[start - 1] == "### Environment Variables"
    assert env["tokens"] == -(-len("\n".join(clean_doc.split("\n")[start - 1 : end])) // 4)
    assert "children" not in env
    assert "content" not in json.dumps(outline)  # the outline never carries section bodies


async def test_outline_costs_a_fraction_of_the_document(
    server: MCPServer[None], docs_dir: Path, messy_doc: str
) -> None:
    await call(server, "index_directory")
    outline = await call(server, "get_document_outline", file_path="messy_doc.md")
    assert len(json.dumps(outline)) < len(messy_doc) / 4


async def test_outline_collapses_parts_of_an_oversized_section(server: MCPServer[None]) -> None:
    await call(server, "index_directory")
    outline = await call(server, "get_document_outline", file_path="messy_doc.md")

    def find(nodes: list[dict[str, Any]], title: str) -> dict[str, Any] | None:
        for node in nodes:
            if node["title"] == title:
                return node
            found = find(node.get("children", []), title)
            if found:
                return found
        return None

    wall = find(outline, "Wall Of Text")
    assert wall is not None
    assert wall["parts"] >= 2
    assert wall["heading_path"] == "Messy Service Notes > Back To Level Two > Wall Of Text"
    assert wall["tokens"] > 800
    deep = find(outline, "Deeply Nested Without Parents")
    assert deep is not None and deep["level"] == 4
    recovered = find(outline, "Section After Unclosed Fence")
    assert recovered is not None
    assert [c["title"] for c in recovered["children"]] == ["Child After Recovery"]


async def test_read_section_returns_exact_boundaries(
    server: MCPServer[None], clean_doc: str
) -> None:
    await call(server, "index_directory")
    text = await call(
        server,
        "read_section",
        file_path="nested/../clean_doc.md",  # relative paths are normalised against the root
        heading_path="Orbit Gateway > Configuration > Command Line Flags",
    )
    lines = clean_doc.split("\n")
    start = lines.index("### Command Line Flags")
    end = lines.index("## Operations")
    assert text == "\n".join(lines[start:end]).rstrip("\n")
    assert text.startswith("### Command Line Flags")
    assert text.endswith("waits for in-flight requests on shutdown.")
    assert "ORBIT_LISTEN_ADDR" not in text  # previous section
    assert "Health Checks" not in text  # next section


async def test_read_section_excludes_children_unless_asked(
    server: MCPServer[None], docs_dir: Path, clean_doc: str
) -> None:
    await call(server, "index_directory")
    path = str(docs_dir / "clean_doc.md")
    own = await call(
        server, "read_section", file_path=path, heading_path="Orbit Gateway > Operations"
    )
    assert own == "## Operations"
    full = await call(
        server,
        "read_section",
        file_path=path,
        heading_path="Orbit Gateway > Operations",
        include_subsections=True,
    )
    lines = clean_doc.split("\n")
    expected = "\n".join(lines[lines.index("## Operations") : lines.index("## License")])
    assert full == expected.rstrip("\n")


async def test_read_section_reassembles_oversized_sections_verbatim(
    server: MCPServer[None], docs_dir: Path, messy_doc: str
) -> None:
    await call(server, "index_directory")
    path = str(docs_dir / "messy_doc.md")
    base = "Messy Service Notes > Back To Level Two > Many Paragraphs"
    whole = await call(server, "read_section", file_path=path, heading_path=base)
    lines = messy_doc.split("\n")
    start = lines.index("### Many Paragraphs")
    end = lines.index("## Duplicate")
    assert whole == "\n".join(lines[start:end]).rstrip("\n")
    part_two = await call(server, "read_section", file_path=path, heading_path=f"{base} (Part 2)")
    assert part_two.startswith("Paragraph ")
    assert part_two in whole
    assert len(part_two) < len(whole)


async def test_read_section_is_forgiving_about_spacing_case_and_short_paths(
    server: MCPServer[None], docs_dir: Path
) -> None:
    await call(server, "index_directory")
    path = str(docs_dir / "clean_doc.md")
    expected = await call(
        server,
        "read_section",
        file_path=path,
        heading_path="Orbit Gateway > Operations > Health Checks",
    )
    for variant in (
        "orbit gateway>operations  >  health checks",
        "Health Checks",
        "Operations > Health Checks",
    ):
        assert await call(server, "read_section", file_path=path, heading_path=variant) == expected
    preamble = await call(server, "read_section", file_path=path, heading_path=PREAMBLE_TITLE)
    assert preamble.startswith("[![CI]")


async def test_duplicate_headings_are_individually_addressable(
    server: MCPServer[None], docs_dir: Path
) -> None:
    await call(server, "index_directory")
    path = str(docs_dir / "messy_doc.md")
    first = await call(
        server, "read_section", file_path=path, heading_path="Messy Service Notes > Duplicate"
    )
    second = await call(
        server, "read_section", file_path=path, heading_path="Messy Service Notes > Duplicate [2]"
    )
    assert "first duplicate body" in first and "second" not in first
    assert "second duplicate body" in second


async def test_unknown_section_lists_the_available_paths(
    server: MCPServer[None], docs_dir: Path
) -> None:
    await call(server, "index_directory")
    with pytest.raises(ToolError) as raised:
        await server.call_tool(
            "read_section",
            {"file_path": str(docs_dir / "clean_doc.md"), "heading_path": "Orbit Gateway > Nope"},
        )
    assert not isinstance(raised.value, UnexpectedToolError)  # anticipated, message preserved
    assert "No section 'Orbit Gateway > Nope'" in str(raised.value)
    assert "Orbit Gateway > Installation" in str(raised.value)


async def test_unindexed_and_ambiguous_files_are_reported(server: MCPServer[None]) -> None:
    with pytest.raises(ToolError, match="not indexed"):
        await server.call_tool("get_document_outline", {"file_path": "clean_doc.md"})
    await call(server, "index_directory")
    # docs/clean_doc.md resolves exactly against the docs root even though a nested twin exists
    outline = await call(server, "get_document_outline", file_path="clean_doc.md")
    assert outline[1]["title"] == "Orbit Gateway"
    with pytest.raises(ToolError, match="not indexed"):
        await server.call_tool("get_document_outline", {"file_path": "absent.md"})


async def test_suffix_resolution_and_ambiguity(tmp_path: Path, docs_dir: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    config = ServerConfig(db_path=tmp_path / "suffix.db", docs_dir=elsewhere)
    instance = MarkdownMemoryService(config, embedder=FakeEmbedder())
    try:
        instance.index_directory(str(docs_dir))
        assert instance.get_document_outline("messy_doc.md")[0].heading_title.startswith(
            "[Overview"
        )
        assert (
            instance.get_document_outline("nested/clean_doc.md")[0].heading_title == "Nested Copy"
        )
        server = create_server(service=instance)
        with pytest.raises(ToolError, match="ambiguous"):
            await server.call_tool("get_document_outline", {"file_path": "clean_doc.md"})
    finally:
        instance.close()


async def test_bad_directory_is_an_anticipated_error(server: MCPServer[None]) -> None:
    with pytest.raises(ToolError, match="does not exist") as raised:
        await server.call_tool("index_directory", {"directory": "/definitely/not/here"})
    assert not isinstance(raised.value, UnexpectedToolError)


async def test_invalid_arguments_are_rejected_by_the_schema(server: MCPServer[None]) -> None:
    with pytest.raises(ToolError):
        await server.call_tool("search_docs", {"limit": 3})  # query missing
    with pytest.raises(ToolError):
        await server.call_tool("search_docs", {"query": "x", "limit": "many"})


async def test_search_docs_returns_sections_and_breadcrumbs(server: MCPServer[None]) -> None:
    await call(server, "index_directory")
    answer = await call(server, "search_docs", query="ORBIT_UPSTREAM_TIMEOUT_MS", limit=3)
    assert set(answer) == {"results", "index_status"}
    # A clean index says so quietly: a caveat on every answer is a caveat nobody reads.
    assert answer["index_status"] == {"coverage": "verified", "failures": [], "message": None}
    results = answer["results"]
    assert 1 <= len(results) <= 3
    top = results[0]
    assert top["heading_path"] == "Orbit Gateway > Configuration > Environment Variables"
    assert top["fts_rank"] == 1
    assert "ORBIT_UPSTREAM_TIMEOUT_MS" in top["content"]
    assert set(top) - {"matched_passage"} == {
        "file_path", "document_title", "heading_path", "heading_title", "lines",
        "score", "fts_rank", "vec_rank", "tokens", "content",
    }  # fmt: skip
    assert [r["score"] for r in results] == sorted((r["score"] for r in results), reverse=True)
    assert (await call(server, "search_docs", query="   "))["results"] == []


async def test_a_search_over_a_damaged_index_says_so_in_its_answer(
    server: MCPServer[None], docs_dir: Path
) -> None:
    """The caveat has to ride on the answer, not wait in a tool nobody calls.

    An agent searches and acts on what comes back. If the tree it searched is missing
    files, the moment to say so is in that reply - by the time anyone runs the indexer
    again the wrong conclusion has already been drawn.
    """
    await call(server, "index_directory")
    broken = docs_dir / "unreadable.md"
    broken.write_text("# Unreadable\n\nbody\n")
    broken.chmod(0o000)
    try:
        await call(server, "index_directory")
        answer = await call(server, "search_docs", query="gateway", limit=3)
        status = answer["index_status"]
        assert status["coverage"] == "unknown"
        assert [f["file_path"] for f in status["failures"]] == [str(broken)]
        assert status["message"] is not None and "could not be indexed" in status["message"]
        assert answer["results"], "the answer still comes, with the caveat attached"

        listed = await call(server, "list_documents")
        assert listed["index_status"]["coverage"] == "unknown"
    finally:
        broken.chmod(0o644)


async def test_search_reflects_reindexing(server: MCPServer[None], docs_dir: Path) -> None:
    await call(server, "index_directory")
    (docs_dir / "new.md").write_text("# Fresh\n\n## Zanzibar\n\nquokka habitat notes\n")
    (docs_dir / "messy_doc.md").unlink()
    summary = await call(server, "index_directory")
    assert "1 (re)indexed" in summary and "1 purged" in summary
    top = (await call(server, "search_docs", query="quokka"))["results"][0]
    assert top["heading_path"] == "Fresh > Zanzibar"
    stale = await call(server, "search_docs", query="MESSY_FLAG", limit=20)
    assert all("messy_doc.md" not in r["file_path"] for r in stale["results"])


async def test_tools_never_write_to_stdout(
    server: MCPServer[None], capfd: pytest.CaptureFixture[str]
) -> None:
    await call(server, "index_directory")
    await call(server, "list_documents")
    await call(server, "get_document_outline", file_path="clean_doc.md")
    await call(server, "read_section", file_path="clean_doc.md", heading_path="License")
    await call(server, "search_docs", query="graceful shutdown")
    assert capfd.readouterr().out == ""


def test_logging_is_routed_to_stderr(capfd: pytest.CaptureFixture[str]) -> None:
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    try:
        configure_logging("DEBUG")
        logging.getLogger("markdown_memory.test").info("to-stderr-only")
        streams = [getattr(handler, "stream", None) for handler in root.handlers]
        assert streams == [sys.stderr]
        captured = capfd.readouterr()
        assert captured.out == ""
        assert "to-stderr-only" in captured.err
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def test_source_tree_contains_no_print_calls() -> None:
    import ast

    package = Path(__file__).parent.parent / "src" / "markdown_memory"
    for source in package.glob("*.py"):
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "print", f"print() in {source.name}:{node.lineno}"
            if isinstance(node, ast.Attribute) and node.attr == "stdout":
                raise AssertionError(f"stdout referenced in {source.name}:{node.lineno}")


def test_config_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MARKDOWN_MEMORY_DB", str(tmp_path / "custom.db"))
    monkeypatch.setenv("MARKDOWN_MEMORY_DOCS_DIR", str(tmp_path))
    monkeypatch.setenv("MARKDOWN_MEMORY_MODEL_CACHE", str(tmp_path / "models"))
    config = ServerConfig.from_env()
    assert config.db_path == tmp_path / "custom.db"
    assert config.docs_dir == tmp_path
    assert config.model_cache_dir == tmp_path / "models"
    assert config.embedder == "embeddinggemma"
    monkeypatch.setenv("MARKDOWN_MEMORY_EMBEDDER", "bge-small")
    assert ServerConfig.from_env().embedder == "bge-small"
    monkeypatch.delenv("MARKDOWN_MEMORY_EMBEDDER")
    for name in ("MARKDOWN_MEMORY_DB", "MARKDOWN_MEMORY_DOCS_DIR", "MARKDOWN_MEMORY_MODEL_CACHE"):
        monkeypatch.delenv(name)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)
    defaults = ServerConfig.from_env()
    assert defaults.db_path == tmp_path / "data" / "markdown-memory" / "index.db"
    assert defaults.docs_dir == tmp_path


# ---------------------------------------------------------------------- pure helpers


def section(path: str, level: int, line: int, part_index: int = 0, content: str = "x") -> Section:
    title = path.split(" > ")[-1].split(" (Part")[0]
    return Section(
        id=line, doc_id=1, heading_title=title, heading_level=level, heading_path=path,
        content=content, start_line=line, end_line=line, part_index=part_index,
    )  # fmt: skip


class TestSelectSections:
    SECTIONS = (
        section("A", 1, 1),
        section("A > Setup", 2, 3),
        section("A > Setup > Linux", 3, 5),
        section("A > Big (Part 1)", 2, 7, part_index=1),
        section("A > Big (Part 2)", 2, 9, part_index=2),
        section("A > Other", 2, 11),
        section("A > Other > Linux", 3, 13),
    )

    def test_full_path(self) -> None:
        assert [s.id for s in select_sections(self.SECTIONS, "A > Setup")] == [3]

    def test_base_path_selects_every_part(self) -> None:
        assert [s.id for s in select_sections(self.SECTIONS, "A > Big")] == [7, 9]

    def test_part_path_selects_one_part(self) -> None:
        assert [s.id for s in select_sections(self.SECTIONS, "A > Big (Part 2)")] == [9]

    def test_subsections(self) -> None:
        chosen = select_sections(self.SECTIONS, "A > Setup", include_subsections=True)
        assert [s.id for s in chosen] == [3, 5]
        everything = select_sections(self.SECTIONS, "A", include_subsections=True)
        assert len(everything) == len(self.SECTIONS)

    def test_prefix_of_a_sibling_title_is_not_a_child(self) -> None:
        sections = (section("A > Set", 2, 1), section("A > Setup", 2, 3))
        assert [s.id for s in select_sections(sections, "A > Set", include_subsections=True)] == [1]

    def test_ambiguous_short_path(self) -> None:
        with pytest.raises(SectionNotFoundError, match="ambiguous") as raised:
            select_sections(self.SECTIONS, "Linux")
        assert "A > Setup > Linux | A > Other > Linux" in str(raised.value)

    def test_short_path_must_align_with_a_breadcrumb_segment(self) -> None:
        with pytest.raises(SectionNotFoundError, match="No section"):
            select_sections(self.SECTIONS, "etup")

    def test_empty_inputs(self) -> None:
        with pytest.raises(SectionNotFoundError, match="must not be empty"):
            select_sections(self.SECTIONS, "  ")
        with pytest.raises(SectionNotFoundError, match="no sections"):
            select_sections((), "A")

    def test_normalize_heading_path(self) -> None:
        assert normalize_heading_path("  A>B  >   C ") == "A > B > C"


def test_build_outline_nests_by_level_and_handles_skipped_levels() -> None:
    outline = build_outline(
        [
            section(PREAMBLE_TITLE, 0, 1),
            section("A", 1, 3),
            section("A > Deep", 4, 5),
            section("A > B (Part 1)", 2, 7, part_index=1, content="x" * 40),
            section("A > B (Part 2)", 2, 9, part_index=2, content="y" * 8),
            section("A > B > C", 3, 11),
            section("Z", 1, 13),
        ]
    )
    assert [node.heading_title for node in outline] == [PREAMBLE_TITLE, "A", "Z"]
    a = outline[1]
    assert [child.heading_path for child in a.children] == ["A > Deep", "A > B"]
    b = a.children[1]
    assert (b.part_count, b.token_estimate, b.start_line, b.end_line) == (2, 12, 7, 9)
    assert [child.heading_title for child in b.children] == ["C"]
    assert b.to_dict()["parts"] == 2
    assert "parts" not in a.to_dict()
