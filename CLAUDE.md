# markdown-memory

Local MCP server that indexes Markdown documentation so coding agents fetch one heading's
text instead of whole files. Markdown is parsed into an AST (`markdown-it-py`), cut into
heading-delimited sections, and stored in a single SQLite file with three indexes: FTS5
(BM25 keywords), `sqlite-vec` section vectors (the mean of a section's passage vectors, since the
embedder truncates at 512 tokens), and `sqlite-vec` passage vectors (one per
paragraph, list item, table row, code block). Embeddings are local ONNX on CPU:
EmbeddingGemma-300m by default (768 dims), `bge-small-en-v1.5` via fastembed as the light
preset (384 dims). Gemma runs the published 4-bit graph (`onnx/model_q4.onnx`), which
quantizes the 262144x768 vocabulary table with `GatherBlockQuantized` and the projections
with `MatMulNBits`, so the table is never materialised in float32. Search fuses keyword and
vector rankings with Reciprocal Rank Fusion.
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
scripts/check.sh                               # licences + ruff + format + mypy + pytest + live test
uv run python scripts/mutation_check.py        # delete guarded behaviours; a test must notice
git config core.hooksPath .githooks             # once per clone: run that gate on every push
```

`.githooks/pre-push` runs `scripts/check.sh` before anything leaves the machine, and
`.github/workflows/gate.yml` runs the same steps on every pull request and on pushes to
`main`,
across Python 3.11 to 3.14 on x86-64 Linux, plus one leg on arm64 Linux and - on `main`
and `workflow_dispatch` only, at roughly ten times the per-minute rate - one on Apple
Silicon: the 4-bit graph asks for the int8 `MatMulNBits` kernel and gets it only where the
CPU has one, so architecture is a thing the gate has to cover rather than assume. The ONNX model is restored from a cache
keyed on the pinned revision *and* the graph file, since one revision publishes several.
The hook is the one to satisfy - it is what you can run - but `git push
--no-verify` skips it, which is why CI also exists. `tests/test_agent_docs.py` holds the
two step lists in the same order. The retrieval gate stays out of CI: it holds an exclusive
lock and asserts on latency, which a shared runner cannot hold still.

Releasing: bump the version in a PR (`uv version --bump patch`), merge it through the gate,
then `git tag vX.Y.Z && git push origin vX.Y.Z`. `.github/workflows/release.yml` refuses a
tag that is not on `main` or does not match `pyproject.toml`, builds, checks the PyPI render
and the installed wheel's handshake (`scripts/handshake_check.py`), publishes through PyPI
Trusted Publishing (environment `pypi`, no stored token), then creates the GitHub release.
The version is written once, in `pyproject.toml`; `__version__`, `--version` and the MCP
`serverInfo` read it back from the installed metadata.

The first run downloads the embedding model (~218 MB) into
`$XDG_CACHE_HOME/markdown-memory/models`. Tests that need the real model are marked
`embedding` and skip (not fail) when it cannot be loaded.

## Layout

| Path | Responsibility |
| --- | --- |
| `src/markdown_memory/models.py` | Frozen dataclasses: `ParsedDocument`, `SectionDraft` (+ `units`), `SectionVectors`, `Section`, `Document`, `DocumentSummary`, `OutlineNode`, `SearchResult` (+ `Excerpt`), `FileFailure`, `IndexStatus`, `IndexReport` |
| `src/markdown_memory/exceptions.py` | `MarkdownMemoryError` hierarchy (`ConfigurationError`, `DatabaseError`, `ASTParseError`, `IndexingError` > `EmbeddingError` > `ModelLoadError`, `ForeignWeightsError`, `IndexBusyError`, `SearchError`, `DocumentNotFoundError`, `SectionNotFoundError`, `WorkTreeError`) |
| `src/markdown_memory/parser.py` | AST sectioniser: heading stack, preamble, front matter, unclosed-fence repair, oversized-section parts, `passages` (each passage with the lines it came from) and its projection `extract_units` (+ `_windows`: a passage over `MAX_UNIT_CHARS` is split, never truncated), `cuts_a_block` |
| `src/markdown_memory/db.py` | `Database`: per-thread connections, WAL, migrations (schema v7; a database of another vector size is refused before any of them), repository methods, `integrity_problems()` |
| `src/markdown_memory/discovery.py` | The walk: `iter_markdown_files`, `Scope` (exclusions, `git_ignored` -> `GitIgnore`, linked worktrees), `without_aliases` (symlinks to a file indexed anyway), symlink and unreadable-name handling, `read_regular_file` (`O_NONBLOCK` + `fstat`), `hash_bytes`, `MAX_FILE_BYTES` |
| `src/markdown_memory/model_cache.py` | The versioned/verified model cache: `gemma_model_dir`, `GEMMA_MANIFEST`, the pin, the graph file, the `.verified` stamp, `flock`, atomic writes |
| `src/markdown_memory/embedders.py` | `Embedder` protocol, `EmbeddingGemmaEmbedder`, `FastEmbedEmbedder`, `create_embedder`. `numpy` and `onnxruntime` live here and nowhere else; cache names are read as `model_cache.X` so one patch point holds. `weights_revision` names the revision *and* the graph, because one revision publishes several |
| `src/markdown_memory/indexer.py` | Incremental `Indexer`: the scan, SHA-256 change detection, the bounded window (workers embed, the driver writes), section/passage vectors |
| `src/markdown_memory/search.py` | `HybridSearcher`: FTS5 query building, IDF keyword gate, passage max-sim, RRF, the top hit's excerpt (`select_anchor`, `excerpt_lines`) |
| `src/markdown_memory/config.py` | `ServerConfig`, `resolve_config` (one precedence for every entry point), the default database per docs root and preset, the `MARKDOWN_MEMORY_*` names, `parse_exclusions` |
| `src/markdown_memory/freshness.py` | `FreshnessSweep`: how many indexed documents moved on, its single-entry cache, its TTL and the lock that makes a sweep one step |
| `src/markdown_memory/autoindex.py` | `AutoIndexer`: the stdio server's background re-indexing - one run at a time, started by a search's `index_status` - the first after the server starts, then whenever one is due (changed files above the post-run baseline, a new weights mismatch, or the walk interval); stopped between documents before the service closes |
| `src/markdown_memory/trees.py` | Which work tree a path is in (`work_tree`: one `git rev-parse`, never a parse of `.git`), where the docs root sits in another tree of the same repository (`counterpart`), `MAX_TREES` |
| `src/markdown_memory/headings.py` | Breadcrumb resolution (`Root > Child`), section selection, the outline tree |
| `src/markdown_memory/server.py` | `MarkdownMemoryService`, MCP tool wiring, `_ServiceProvider` (one service per work tree a call names by path or `cwd`, sharing the embedder and the run lock; another tree's index reuses the configured root's vectors for identical passage text), `main()` |
| `tests/` | `test_<area>.py` covers the module of that name; `test_<area>_regressions.py` pins every bug review found there. `fakes.py` holds `FakeEmbedder` (offline, deterministic), `helpers.py` the shared builders |
| `scripts/eval_data/` | Frozen eval corpus, labelled queries, `baseline.json` |

Storage: `documents` -> `sections` (ON DELETE CASCADE) -> `units` (ON DELETE CASCADE).
`sections_fts` is an FTS5 external-content table; `sections_vec` and `units_vec` are `vec0`
tables. Triggers on `sections` and `units` keep all three in sync, including rows removed
by cascade - the content tables are written only through `Database.replace_document`, and
never `sections_fts` directly. Two tables sit outside that path and carry the index's own
account of itself: `index_failures` (one row per file that could not be read) and
`index_coverage` (whether a full run of a root finished), written by `record_failures` and
the scan bookkeeping in `db.py`.

## Code style and architecture rules

- **Python 3.11+, fully typed.** `mypy --strict` must pass with zero errors and zero
  untyped defs. No unconstrained `Any`: use `JsonValue`, Protocols or TypeAliases. `numpy`
  and `onnxruntime` stay inside `embedders.py`; vectors cross module boundaries as
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
- **Ruff**: line length 100, rules `E,F,I,N,UP,B,A,C4,SIM,TID,ARG,ERA,RUF100` (`ARG` is
  ignored under `tests/`, where a fake takes the arguments of what it stands in for).
  Match the surrounding comment density; comments explain why, not what.
- **The shape of the package is a test, not a description.**
  `tests/test_architecture.py` fails when a module goes over its budget of logical lines,
  when a heavy dependency is imported outside the module that owns it, when the layout
  table above does not name a module, when a top-level symbol is dead, or when a star
  import hides what is used. Growing a module means raising its budget in a diff, on
  purpose. Dead-symbol detection resolves scope through `symtable`, so it cannot see a
  name reached only dynamically (`getattr`); `ALLOWED_UNREFERENCED` records any such
  claim, and is empty.
- **Every bug fix gets a regression test** that fails when the fix is reverted. Prove
  it: `scripts/mutation_check.py` deletes each guarded behaviour in a throwaway copy of
  the package and fails if the suite stays green. Add an entry when a fix protects
  something a test could pass without.

## Retrieval regression policy

Any change to ranking, chunking, unit extraction, FTS query building, the embedder or its
prompts **must pass** `uv run python scripts/eval_retrieval.py` on the held-out set. CI does
not run it - it holds an exclusive lock and asserts on latency - so this one is on you to
run before proposing the change; nothing will stop a regression at review time:

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

`--corpus v2` scores the same way over `scripts/eval_data/corpus_v2` - 97 real upstream
docs (cargo, compose-spec, gh, prometheus, ripgrep) with `queries_v2.json` - and is
report-only: it has no floors and exits zero, and its baseline is `<preset>@v2`. Run it
alongside the gate for any ranking change, and once the change is accepted record it too
(`--corpus v2 --update-baseline`). The same rule holds there: tune on its `dev` queries
only, never on `held_out`.

A passage vector is reused across work trees on the exact embedded text, the weights stamp
and `VECTOR_FORMAT` - the prompts are not part of the stamp, so **changing an embedding prompt
bumps `VECTOR_FORMAT`**, which makes every document re-embed itself.

Changing the embedding dimension never discards an index: a database holding vectors of
another size is refused on open, before anything is written, and each preset has a default
database of its own (`index.db`, `index-bge-small.db`), so switching preset does not reach
another's file (#82). Changing the weights at the same dimension does not discard either:
each document is stamped with the weights that embedded it (`documents.weights_revision`),
the index-wide revision is revoked while stamps disagree, and a run re-embeds the stale
documents in place, restoring the revision once no vector-bearing row in the database is
stale. Only a renamed model that cannot name its weights still discards the index.

The gate keeps its index in `$XDG_CACHE_HOME/markdown-memory/eval/`, keyed on the corpus
content, the chunking constants, the source of the modules that decide what is indexed
(`parser.py`, `indexer.py`, `embedders.py`, `model_cache.py`, `discovery.py`, `config.py`,
`db.py`, `models.py`),
the embedder's revision, prompts and dimension, the size and mtime of the model files
actually on disk, and `MARKDOWN_MEMORY_THREADS` (`scripts/eval_cache.py`). A cached index
is never trusted on its key alone: before it is scored, its parse fingerprint - every
section path, section content and passage text, verbatim and in order - is recomputed, the
recorded key is compared field by field, `integrity_problems()` runs, and a passage of the
corpus is re-embedded and must come back as its own nearest neighbour (cosine distance
below 1e-3; measured -6e-8 for a match against 0.384 for the next passage). A query cannot
do that last job: search fuses keyword and vector rankings, so an identifier query returns
the right section even when every vector came from a different model. Any check failing
discards the index and rebuilds it. `--rebuild` forces that by hand. Only one evaluation may run at a time
(`flock` on `eval.lock`): the script reports median and p95 latency, and a second job on the
same CPU moves those numbers further than the changes being measured.

<!-- markdown-memory:navigation-rules:start -->
## Reading Markdown documentation (when the `markdown-memory` MCP server is mounted)

**Never dump whole files.** Do not `Read`/`read_file`/`cat` a Markdown document longer than
~150 lines into context unless the user explicitly asks for the whole file. Fetch the one
section you need instead; it is typically 10-50x cheaper.

**Pass `cwd` - your working directory, as an absolute path - on every call.** In another git
worktree of the repository the server was started for, answers then come from that worktree's
own copy of the docs, indexed into its own database the first time it is named; anywhere else
it changes nothing. `index_status.root` names the documentation root that answered: when it is
not the tree you are working in, you are reading another checkout's docs. Every `file_path` in
an answer is relative to that root (absolute when another worktree answered): pass it back
verbatim, with the same `cwd`.

Work in this order:

1. **`search_docs(query, limit=5)`** - start here. Returns
   `{"results": [...], "keyword_match": ..., "index_status": {...}}`, plus a
   `keyword_message` whenever `keyword_match` is not `"matched"`. The first hit carries the
   full section `content` with its breadcrumb (`heading_path`), line range and token
   estimate - or, marked `excerpt: true`, only the passage that matched and its neighbours,
   verbatim, with `lines` naming them and `tokens` still what the whole section costs: when
   the excerpt is not enough, `read_section(file_path, heading_path)` returns the full
   section. The rest are pointers - `file_path`, `heading_path`, `lines`, `tokens` and,
   when a passage won the vector ranking, `matched_passage`: how that passage begins. Use exact
   identifiers verbatim (`--dry-run`, `HELIOS_BATCH`, `ENOSPC`): they are matched by
   keyword at 100% Top-1. Plain-language questions work too, but the
   first hit is not always the answer (Top-1 ~88%; Top-5 is ~97% reliable): when it is not,
   read the pointers' `matched_passage` and fetch the one that fits with `read_section`,
   passing its `heading_path` verbatim - for a `(Part n)` the base path returns every part.
   A `(Part n)` pointer also carries `part_preview`, how that part begins, so parts of one
   split section can be told apart before reading any of them.
   `keyword_match` says whether keyword search found the query's terms; when it is
   `"no_match"`, no indexed section contains them: a query of identifiers alone then returns
   no results, and any other query's hits are only semantic neighbours. Report such an
   identifier as not in the indexed documentation rather than answering from neighbours -
   unless `index_status.indexing` is true and coverage `"unknown"`: then it may only not be
   indexed yet, so search again once indexing is false.
   `index_status.gitignore` is what git said when a run of the root last finished: `applied`,
   `off`, `no_repository`, `unavailable` (git could not be asked, so ignored files may be in
   the index; the message says how to see why) or `unknown` - no run has recorded it yet, which
   is not a failure and needs nothing from you.
   When `index_status.coverage` is `"unknown"`, the documentation you just
   searched is missing files or was never indexed end to end - say so rather than
   concluding the docs do not cover it. When `index_status.changed_files` is non-zero,
   that many indexed documents could not be confirmed to be what was indexed - edited,
   unreadable or deleted - so a hit may quote text that is no longer there; re-run
   `index_directory` before relying on it. The count is best-effort: files created since
   the last run are not in it, and an edit that puts a file's modification time back is
   not seen, so a zero is "nothing detected", not "everything verified".
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
vouches for it; `index_directory(directory=None)` (re)indexes. The stdio server keeps its
own docs root indexed in the background (from the first search, and whenever a search sees
change), so call it only when `index_status.message` asks for it - never while
`index_status.indexing` is true. It is incremental (SHA-256 per file), so re-running is cheap.
`file_path` may be absolute, relative to the docs root, or any unique path suffix. An
error from `read_section` lists the valid heading paths - retry with one of them rather
than falling back to reading the file.
<!-- markdown-memory:navigation-rules:end -->

## Skills

Project skills live in `.claude/skills/`:

- **`run-eval`** - run the retrieval benchmark, check the gates, report deltas vs the baseline.
- **`reindex-docs`** - forced re-index of a directory, then SQLite and vector integrity checks.
- **`test-regression`** - the pre-commit gate: ruff, mypy, pytest, live end-to-end test.
