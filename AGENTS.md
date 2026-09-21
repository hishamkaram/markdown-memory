# markdown-memory - agent instructions

Local MCP server for AST-aware Markdown indexing: SQLite with FTS5 (BM25), `sqlite-vec`
section and passage vectors, EmbeddingGemma-300m (default) or bge-small embeddings, fused
with Reciprocal Rank Fusion. **`CLAUDE.md` is the full developer guide** (layout,
architecture rules, regression policy); this file is the short form for agents that do not
read it.

## Commands

```bash
uv sync                                     # set up
uv run src/markdown_memory/server.py        # run the MCP server (stdio)
uv run ruff check . && uv run ruff format . # lint, format
uv run mypy --strict src/                   # type check (zero errors required)
uv run pytest                               # tests
uv run python scripts/eval_retrieval.py     # retrieval gate: Top-1 >= 80%, Top-5 >= 90%, identifiers 100%
uv run python scripts/live_test.py          # end-to-end over stdio
scripts/check.sh                            # all of the above except the eval, fail-fast
```

## Non-negotiable rules

- Python 3.11+, `mypy --strict` clean, no untyped defs, no unconstrained `Any`.
- Domain models are `@dataclass(slots=True, frozen=True)`.
- SQLite: WAL mode, writes only inside `with db.transaction():`.
- Never write to stdout in `src/` (it is the JSON-RPC channel); log to `sys.stderr`.
- Any change to ranking, chunking or embedding must pass `scripts/eval_retrieval.py`.
  Never tune on the `held_out` queries.

<!-- markdown-memory:navigation-rules:start -->
## Reading Markdown documentation (when the `markdown-memory` MCP server is mounted)

**Never dump whole files.** Do not `Read`/`read_file`/`cat` a Markdown document longer than
~150 lines into context unless the user explicitly asks for the whole file. Fetch the one
section you need instead; it is typically 10-50x cheaper.

Work in this order:

1. **`search_docs(query, limit=5)`** - start here. Returns
   `{"results": [...], "index_status": {...}}`; each hit carries its breadcrumb
   (`heading_path`), line range, token estimate, full section `content`, and the
   `matched_passage` that matched best. Use exact identifiers verbatim (`--dry-run`,
   `HELIOS_BATCH`, `ENOSPC`): they are matched by keyword at 100% Top-1. Plain-language
   questions work too; read all returned hits, not just the first (Top-5 is ~97% reliable,
   Top-1 ~85%). When `index_status.coverage` is `"unknown"`, the documentation you just
   searched is missing files or was never indexed end to end - say so rather than
   concluding the docs do not cover it.
2. **`get_document_outline(file_path)`** - only when you need the structure of a document:
   the heading tree with line spans and per-section token estimates, at a few hundred
   tokens. Use it to choose a section, or to find sibling sections of a search hit.
3. **`read_section(file_path, heading_path)`** - fetch exactly one section, verbatim.
   `heading_path` is the breadcrumb `Root > Child > Subchild` as shown by the outline or
   by `search_docs`; a bare heading title works when it is unambiguous. It returns the
   section's own text only - pass `include_subsections=true` for a parent heading whose
   content lives in its children. Oversized sections appear as `Path (Part n)`; reading
   the base path returns all parts reassembled.

Supporting tools: `list_documents(directory="")` returns
`{"documents": [...], "index_status": {...}}` - what is indexed, and whether anything
vouches for it; `index_directory(directory=None)` (re)indexes - run it once per session if
searches come back empty or stale, or if `index_status.coverage` is `"unknown"`. It is
incremental (SHA-256 per file), so re-running is cheap.
`file_path` may be absolute, relative to the docs root, or any unique path suffix. An
error from `read_section` lists the valid heading paths - retry with one of them rather
than falling back to reading the file.
<!-- markdown-memory:navigation-rules:end -->
