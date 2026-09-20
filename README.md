# markdown-memory

A local [Model Context Protocol](https://modelcontextprotocol.io) server that indexes
Markdown documentation into SQLite (FTS5 + [sqlite-vec](https://github.com/asg017/sqlite-vec))
so coding agents can read an outline and fetch one heading's text instead of ingesting
whole files.

Everything runs locally: parsing with `markdown-it-py`, embeddings with
[EmbeddingGemma-300m](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX)
(quantized ONNX on CPU, 768 dimensions), storage in a single SQLite file.

## Tools

| Tool | Purpose |
| --- | --- |
| `index_directory(directory=None)` | Scan a tree, (re)index new/changed `.md` files (SHA-256), purge deleted ones |
| `list_documents(directory="")` | Indexed paths, titles and section counts |
| `get_document_outline(file_path)` | Hierarchical TOC with line ranges and token estimates |
| `read_section(file_path, heading_path, include_subsections=False)` | Verbatim text of one section |
| `search_docs(query, limit=5)` | BM25 + passage-level vector search fused with Reciprocal Rank Fusion (k = 60); each hit reports the `matched_passage` |

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
| `MARKDOWN_MEMORY_EMBEDDER` | `--embedder` | `embeddinggemma` (or `bge-small`) |

Documents are stored under their absolute path, so one database can hold several
projects. Switching to a different model discards the whole index (vectors from two
models cannot be compared); `index_directory` reports that and every root must be
indexed again.

### Embedding models

| Preset | Download | Peak RAM | Query (median) | Indexing | Held-out Top-1 / Top-3 / Top-5 |
| --- | --- | --- | --- | --- | --- |
| `embeddinggemma` (default) | ~330 MB | ~1.6 GB | ~370 ms | ~5 vectors/s | 88% / 97% / 97% |
| `bge-small` | ~65 MB | ~1.1 GB | ~26 ms | ~12 vectors/s | 68% / 82% / 88% |

Measured by `scripts/eval_retrieval.py` on a laptop CPU: a frozen 54-section corpus and 34
held-out paraphrase queries written before any tuning (a second, deliberately adversarial
set scores 71% / 88% / 91% with EmbeddingGemma and 53% / 68% / 79% with bge-small). Exact
identifiers - flags, environment variables, error strings - are 100% Top-1 with either
preset, because they are answered by FTS5. Indexing is incremental, so the slow first
pass is paid once per file version. EmbeddingGemma is distributed under the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms); the revision is pinned.

## How search ranks

1. **Keywords (FTS5, BM25).** Stopwords are dropped, identifiers are kept verbatim. A hit
   only counts if it covers at least half of the query's IDF mass or matches an
   identifier-like term - a stray match on "data" or "deploy" no longer outvotes the
   vector index.
2. **Vectors.** Every paragraph, list item, table row (rendered as `Header: cell; ...`) and
   code block - also inside block quotes and list items - is embedded separately, alongside
   one vector for the section. A table split across `(Part n)` sections keeps its header
   for every part. A section is
   ranked by its closest vector, so one relevant table row is enough. Heading-only
   sections have no vectors and are never returned ahead of their children.
3. **Reciprocal Rank Fusion** of the two rankings.

Cross-encoder rerankers (MiniLM, bge-reranker-base, jina, ColBERT) were benchmarked and
rejected: every one lowered accuracy on technical documentation and cost 2-12 s a query.

All logging goes to **stderr**. stdout carries JSON-RPC frames only.

## How malformed Markdown is handled

- **Skipped heading levels** (`#` then `####`): a heading stack pops every level `>= L`
  before pushing, so breadcrumbs stay well formed.
- **Preamble**: badges/summary before the first heading become `[Overview / Preamble]`.
- **YAML front matter**: kept out of the AST (CommonMark would read it as a setext
  heading) and used as a title fallback. A leading `---` rule followed by prose is not
  mistaken for it. One case is inherently ambiguous - a single `key: value` line between
  two `---` lines - and is read as front matter, as static-site generators do, unless the
  key is an admonition word (`Note:`, `Warning:`, `TODO:` ...).
- **Unclosed code fences**: CommonMark runs them to EOF - or to the closing marker of a
  *later* fence - swallowing the sections in between. The fence is closed before the next
  blank-line-preceded ATX heading instead. A level-1 `# ...` line is treated as a comment
  unless the fence language cannot have `#` comments (JSON, Go, ...). This is a heuristic:
  a fence that merely *looks* closed is only cut on strong evidence (it contains another
  opening fence with an info string, or the document ends inside a bare fence and the cut
  makes the rest well formed), and never when the repair would lose a heading that was
  already found. Repair work is capped per document, so a pathological file costs a
  bounded number of extra parses rather than one per fence.
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
uv run python scripts/eval_retrieval.py --show-misses   # retrieval accuracy; fails on regression
uv run python scripts/reindex_docs.py DIR --force       # forced re-index + integrity verification
scripts/check.sh                                        # the whole pre-commit gate, fail-fast
```

### For AI coding agents

`CLAUDE.md` is the developer guide (commands, layout, architecture rules, the retrieval
regression policy). `AGENTS.md` and `.cursorrules` carry the short form for Codex and
Cursor. All three share one block of rules for *using* the MCP tools - search first, then
outline, then read a single section; never dump whole Markdown files - and a test keeps the
three copies identical and in step with the code. Project skills for Claude Code live in
`.claude/skills/`: `run-eval`, `reindex-docs`, `test-regression`.

Built on the MCP Python SDK 2.x, where `FastMCP` was renamed `MCPServer`.
