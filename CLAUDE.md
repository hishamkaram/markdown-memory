# markdown-memory

Local MCP server that indexes Markdown documentation so coding agents fetch one heading's
text instead of whole files. Markdown is parsed into an AST (`markdown-it-py`), cut into
heading-delimited sections, and stored in a single SQLite file with three indexes: FTS5
(BM25 keywords), `sqlite-vec` section vectors, and `sqlite-vec` passage vectors (one per
paragraph, list item, table row, code block). Embeddings are local ONNX on CPU:
EmbeddingGemma-300m by default (768 dims), `bge-small-en-v1.5` via fastembed as the light
preset (384 dims). Search fuses keyword and vector rankings with Reciprocal Rank Fusion.
Built on the MCP Python SDK **2.x**, where `FastMCP` is named `MCPServer`.

## Commands

```bash
uv sync                                        # set up the environment
uv run src/markdown_memory/server.py           # run the MCP server over stdio (= uv run markdown-memory)
uv run ruff check .                            # lint            (add --fix to apply fixes)
uv run ruff format .                           # format          (--check in CI)
uv run mypy --strict src/                      # type check: must report zero errors
uv run pytest                                  # unit + integration tests (-m "not embedding" skips real-model tests)
uv run python scripts/eval_retrieval.py        # retrieval accuracy gate (see policy below)
uv run python scripts/live_test.py             # end-to-end: spawns the server, drives it over stdio JSON-RPC
uv run python scripts/reindex_docs.py DIR --force   # forced re-index + integrity verification
scripts/check.sh                               # ruff + format + mypy + pytest + live test, fail-fast
git config core.hooksPath .githooks             # once per clone: run that gate on every push
```

`.githooks/pre-push` runs `scripts/check.sh` before anything leaves the machine; it is
this project's CI, since the gate needs the local ONNX model. `git push --no-verify`
skips it for a work-in-progress branch.

The first run downloads the embedding model (~330 MB) into
`$XDG_CACHE_HOME/markdown-memory/models`. Tests that need the real model are marked
`embedding` and skip (not fail) when it cannot be loaded.

## Layout

| Path | Responsibility |
| --- | --- |
| `src/markdown_memory/models.py` | Frozen dataclasses: `SectionDraft` (+ `units`), `SectionVectors`, `Section`, `Document`, `OutlineNode`, `SearchResult`, `IndexReport` |
| `src/markdown_memory/exceptions.py` | `MarkdownMemoryError` hierarchy (`DatabaseError`, `ASTParseError`, `IndexingError` > `EmbeddingError` > `ModelLoadError`, `SearchError`, `DocumentNotFoundError`, `SectionNotFoundError`) |
| `src/markdown_memory/parser.py` | AST sectioniser: heading stack, preamble, front matter, unclosed-fence repair, oversized-section parts, `extract_units` |
| `src/markdown_memory/db.py` | `Database`: per-thread connections, WAL, migrations (schema v2), repository methods, `integrity_problems()` |
| `src/markdown_memory/indexer.py` | `Embedder` protocol, `EmbeddingGemmaEmbedder`, `FastEmbedEmbedder`, `create_embedder`, incremental `Indexer` |
| `src/markdown_memory/search.py` | `HybridSearcher`: FTS5 query building, IDF keyword gate, passage max-sim, RRF |
| `src/markdown_memory/server.py` | `ServerConfig`, `MarkdownMemoryService`, heading-path resolution, outline, MCP tool wiring, `main()` |
| `tests/` | `test_<area>.py` covers the module of that name; `test_<area>_regressions.py` pins every bug review found there. `fakes.py` holds `FakeEmbedder` (offline, deterministic), `helpers.py` the shared builders |
| `scripts/eval_data/` | Frozen eval corpus, labelled queries, `baseline.json` |

Storage: `documents` -> `sections` (ON DELETE CASCADE) -> `units` (ON DELETE CASCADE).
`sections_fts` is an FTS5 external-content table; `sections_vec` and `units_vec` are `vec0`
tables. Triggers on `sections` and `units` keep all three in sync, including rows removed
by cascade - application code writes only to `documents`, `sections`, `units` and the two
vec tables inside `Database.replace_document`.

## Code style and architecture rules

- **Python 3.11+, fully typed.** `mypy --strict` must pass with zero errors and zero
  untyped defs. No unconstrained `Any`: use `JsonValue`, Protocols or TypeAliases. `numpy`
  and `onnxruntime` stay inside `indexer.py`; vectors cross module boundaries as
  `list[float]`.
- **Immutable models.** Every domain entity is `@dataclass(slots=True, frozen=True)`. To
  change one, build a new one (`dataclasses.replace`).
- **SQLite discipline.** Connections are per-thread and opened with WAL,
  `synchronous=NORMAL`, `foreign_keys=ON`. All writes go through
  `with db.transaction():` (BEGIN IMMEDIATE, commit on success, rollback on any
  exception). Never write to `sections_fts` directly; never use `:memory:` databases.
  Driver errors are translated to `DatabaseError`.
- **No stdout pollution.** stdout carries JSON-RPC frames only. Never `print()` in
  `src/`; log with `logging` (routed to `sys.stderr` by `configure_logging`). A test scans
  the package AST for `print` calls and `stdout` references. Scripts under `scripts/` are
  clients and may print.
- **Errors the agent should see** must be a `MarkdownMemoryError`: tool handlers convert
  those to the SDK's `ToolError`. The SDK hides every other exception behind a generic
  "Error executing tool".
- **Ruff**: line length 100, rules `E,F,I,N,UP,B,A,C4,SIM,TID`. Match the surrounding
  comment density; comments explain why, not what.
- **Every bug fix gets a regression test** that fails when the fix is reverted.

## Retrieval regression policy

Any change to ranking, chunking, unit extraction, FTS query building, the embedder or its
prompts **must pass** `uv run python scripts/eval_retrieval.py` on the held-out set:

| Gate | Floor | Frozen baseline (EmbeddingGemma) |
| --- | --- | --- |
| Paraphrase Top-1 | >= 80% | 88% |
| Paraphrase Top-5 | >= 90% | 97% |
| Identifier Top-1 (dev and held-out) | = 100% | 100% |

The script exits non-zero below a floor and prints the delta against
`scripts/eval_data/baseline.json`. Tune on the `dev` queries only; the `held_out` queries
were written before any tuning and must never be used to choose a parameter. After an
accepted change, record the new numbers with `--update-baseline` and update the table in
`README.md`. Rerankers (MiniLM, bge-reranker-base, jina, ColBERT) were benchmarked and
rejected: all lowered accuracy and cost 2-12 s per query.

Changing the embedding model or its dimension invalidates every stored vector: the index
is discarded and rebuilt (`IndexReport.notes` says so).

<!-- markdown-memory:navigation-rules:start -->
## Reading Markdown documentation (when the `markdown-memory` MCP server is mounted)

**Never dump whole files.** Do not `Read`/`read_file`/`cat` a Markdown document longer than
~150 lines into context unless the user explicitly asks for the whole file. Fetch the one
section you need instead; it is typically 10-50x cheaper.

Work in this order:

1. **`search_docs(query, limit=5)`** - start here. Returns the best sections with their
   breadcrumb (`heading_path`), line range, token estimate, full section `content`, and the
   `matched_passage` that matched best. Use exact identifiers verbatim (`--dry-run`,
   `HELIOS_BATCH`, `ENOSPC`): they are matched by keyword at 100% Top-1. Plain-language
   questions work too; read all returned hits, not just the first (Top-5 is ~97% reliable,
   Top-1 ~88%).
2. **`get_document_outline(file_path)`** - only when you need the structure of a document:
   the heading tree with line spans and per-section token estimates, at a few hundred
   tokens. Use it to choose a section, or to find sibling sections of a search hit.
3. **`read_section(file_path, heading_path)`** - fetch exactly one section, verbatim.
   `heading_path` is the breadcrumb `Root > Child > Subchild` as shown by the outline or
   by `search_docs`; a bare heading title works when it is unambiguous. It returns the
   section's own text only - pass `include_subsections=true` for a parent heading whose
   content lives in its children. Oversized sections appear as `Path (Part n)`; reading
   the base path returns all parts reassembled.

Supporting tools: `list_documents(directory="")` shows what is indexed;
`index_directory(directory=None)` (re)indexes - run it once per session if searches come
back empty or stale. It is incremental (SHA-256 per file), so re-running is cheap.
`file_path` may be absolute, relative to the docs root, or any unique path suffix. An
error from `read_section` lists the valid heading paths - retry with one of them rather
than falling back to reading the file.
<!-- markdown-memory:navigation-rules:end -->

## Skills

Project skills live in `.claude/skills/`:

- **`run-eval`** - run the retrieval benchmark, check the gates, report deltas vs the baseline.
- **`reindex-docs`** - forced re-index of a directory, then SQLite and vector integrity checks.
- **`test-regression`** - the pre-commit gate: ruff, mypy, pytest, live end-to-end test.
