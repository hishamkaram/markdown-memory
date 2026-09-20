# markdown-memory

A local [Model Context Protocol](https://modelcontextprotocol.io) server that indexes
Markdown documentation into SQLite (FTS5 + [sqlite-vec](https://github.com/asg017/sqlite-vec))
so coding agents can read an outline and fetch one heading's text instead of ingesting
whole files.

Everything runs locally: parsing with `markdown-it-py`, embeddings with `fastembed`
(`BAAI/bge-small-en-v1.5`, ONNX, 384 dimensions), storage in a single SQLite file.

## Tools

| Tool | Purpose |
| --- | --- |
| `index_directory(directory=None)` | Scan a tree, (re)index new/changed `.md` files (SHA-256), purge deleted ones |
| `list_documents(directory="")` | Indexed paths, titles and section counts |
| `get_document_outline(file_path)` | Hierarchical TOC with line ranges and token estimates |
| `read_section(file_path, heading_path, include_subsections=False)` | Verbatim text of one section |
| `search_docs(query, limit=5)` | BM25 + cosine vector search fused with Reciprocal Rank Fusion (k = 60) |

Sections are addressed by breadcrumb: `Root > Child > Subchild`. Oversized sections
(> ~800 tokens) are stored as `Root > Child (Part 1)`, `(Part 2)`, ...; reading the base
path reassembles them byte-for-byte. `file_path` may be absolute, relative to the docs
root, or any unique path suffix. `heading_path` is matched exactly first, then ignoring
spacing around `>`, then ignoring case, then as a trailing fragment (`Child > Subchild`
or just the title); an ambiguous request lists the exact candidates.

## Install and run

```bash
uv sync
uv run markdown-memory            # serves MCP over stdio
```

Claude Code / Cursor (`.mcp.json` or `~/.cursor/mcp.json`) - `--project` keeps the
client's working directory, which becomes the default documentation root:

```json
{
  "mcpServers": {
    "markdown-memory": {
      "command": "uv",
      "args": ["run", "--project", "/absolute/path/to/markdown-memory", "markdown-memory"]
    }
  }
}
```

Claude Desktop has no meaningful working directory, so set the root explicitly:

```json
{
  "mcpServers": {
    "markdown-memory": {
      "command": "uv",
      "args": ["run", "--project", "/absolute/path/to/markdown-memory", "markdown-memory"],
      "env": { "MARKDOWN_MEMORY_DOCS_DIR": "/absolute/path/to/your/docs" }
    }
  }
}
```

## Configuration

| Environment variable | CLI flag | Default |
| --- | --- | --- |
| `MARKDOWN_MEMORY_DOCS_DIR` | `--docs-dir` | current working directory |
| `MARKDOWN_MEMORY_DB` | `--db` | `$XDG_DATA_HOME/markdown-memory/index.db` (`~/.local/share/...`) |
| `MARKDOWN_MEMORY_MODEL_CACHE` | - | `$XDG_CACHE_HOME/markdown-memory/models` (`~/.cache/...`) |
| `MARKDOWN_MEMORY_LOG_LEVEL` | `--log-level` | `INFO` |

Documents are stored under their absolute path, so one database can hold several
projects. The embedding model (~65 MB on disk) is downloaded once on first use. Switching
to a different model discards the whole index (vectors from two models cannot be
compared); `index_directory` reports that and every root must be indexed again.

All logging goes to **stderr**. stdout carries JSON-RPC frames only.

## How malformed Markdown is handled

- **Skipped heading levels** (`#` then `####`): a heading stack pops every level `>= L`
  before pushing, so breadcrumbs stay well formed.
- **Preamble**: badges/summary before the first heading become `[Overview / Preamble]`.
- **YAML front matter**: kept out of the AST (CommonMark would read it as a setext
  heading) and used as a title fallback. A leading `---` rule followed by prose is not
  mistaken for it.
- **Unclosed code fences**: CommonMark runs them to EOF - or to the closing marker of a
  *later* fence - swallowing the sections in between. The fence is closed before the next
  blank-line-preceded ATX heading instead. A level-1 `# ...` line is treated as a comment
  unless the fence language cannot have `#` comments (JSON, Go, ...). This is a heuristic:
  a fence that merely *looks* closed is only cut on strong evidence (it contains another
  opening fence with an info string, or the document ends inside a bare fence and the cut
  makes the rest well formed).
- **No headings / walls of text**: split on paragraph boundaries, then on lines, then on
  whitespace. A fenced block is only cut when it exceeds the limit by itself.
- **Colliding breadcrumbs** get a ` [2]`, ` [3]` suffix (also against generated
  `(Part n)` paths), so every stored path addresses exactly one section.
- **Headings like `Option<T>`** keep their type parameter; only lower-case formatting
  tags (`<b>`, `<sub>`, `<a>`, ...) are stripped from titles.

## Indexing rules

- `.md` / `.markdown`, regular files only, at most 10 MB; symlinked directories are not
  followed. `.git`, `node_modules`, virtualenvs and tool caches are pruned - index such a
  tree by passing a directory *inside* it, and it is then left alone when an ancestor is
  re-indexed.
- A document is purged only when the walk could have found it and did not. Files under a
  directory that cannot be listed are kept and the directory is reported as an error.

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy --strict src/
uv run pytest -v                      # unit + integration (real ONNX model for semantic tests)
uv run python scripts/live_test.py    # spawns the server, drives it over stdio JSON-RPC
```

Built on the MCP Python SDK 2.x, where `FastMCP` was renamed `MCPServer`.
