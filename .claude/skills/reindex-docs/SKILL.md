---
name: reindex-docs
description: Force a re-index of a documentation directory with markdown-memory, then verify SQLite integrity (PRAGMA integrity_check, foreign keys, FTS5 index) and vector-dimension consistency. Use when search results look stale or wrong, after switching the embedding model, after a crash during indexing, or when asked to "reindex", "rebuild the index", or "check the database".
---

# reindex-docs

`index_directory` is incremental: a file whose SHA-256 is unchanged is skipped, so it
cannot repair a damaged or stale index on its own. This skill drops the directory's
documents first, re-embeds everything, and then checks the store.

## Run

```bash
uv run python scripts/reindex_docs.py <directory> --force
```

- `<directory>` defaults to the configured docs root: `MARKDOWN_MEMORY_DOCS_DIR`, else
  `CLAUDE_PROJECT_DIR`, else the current directory.
- `--db PATH` targets a specific database. Without it the script resolves the same way the
  server does: `MARKDOWN_MEMORY_DB` if it is set, else the index keyed on the directory
  being re-indexed (`$XDG_DATA_HOME/markdown-memory/projects/<name>-<digest>/index.db`).
  That keying is what makes this verify the index the server actually searches; it used to
  take the database from the environment's root and silently re-index one project into
  another's.
- `--embedder {embeddinggemma,bge-small}` must match the server's. **Opening a database
  with the other embedder is destructive**: the vector size differs, so the whole index -
  every directory, not just this one - is discarded on open and must be rebuilt. The
  summary then carries a `NOTE ... discarded all N previously indexed documents`; repeat
  that to the user and re-index the other roots. Confirm the embedder before running
  against a database you did not create.
- Without `--force` it is a normal incremental run followed by the same checks.

Only documents under `<directory>` are dropped; other indexed directories are untouched.
Expect 2.4-3.8 vectors per second with EmbeddingGemma on CPU. A vector per paragraph, list
item, table row and code block; a section costs none of its own, since its vector is pooled
from its passages.

## What is verified

The script calls `Database.integrity_problems()` and exits non-zero if it returns anything:

- `PRAGMA integrity_check` is `ok` and `PRAGMA foreign_key_check` finds no orphans.
- FTS5 `integrity-check`: the keyword index matches the `sections` table.
- One FTS row per section; one vector per passage (`units` = `units_vec`); a section vector
  exactly for the sections that have passages (heading-only sections have none).
- Every stored vector has the configured dimension (768 EmbeddingGemma, 384 bge-small) and
  `meta.embedding_dim` agrees; schema is at the current version.

## Report

1. The `Indexed ...` summary line (scanned / re-indexed / purged, sections and passages).
2. Row counts and the vector dimension line.
3. `integrity: ok`, or every `INTEGRITY PROBLEM:` line verbatim.
4. Any `ERROR <file>:` lines - those files were not indexed.
5. The `INCOMPLETE: N file(s) could not be indexed` line, if it appears. It means the root
   is not certified whole, so every answer drawn from it reports
   `index_status.coverage: "unknown"` until a clean run finishes. Say which files.

## If it fails

- `ERROR` lines for individual files: fix or exclude the file, then re-run. A model that
  cannot load aborts the whole run instead (one error, not one per file).
- `could not verify the FTS5 index: the database is locked` is **not damage**: another
  process (usually the MCP server, mid-index) holds the write lock. Wait and re-run. Never
  delete a database because of this message.
- Any other integrity problem after a forced re-index means the database file itself is damaged:
  stop the MCP server, delete the database file (it is only a cache of the Markdown
  files) plus its `-wal` and `-shm` companions, and run the script again. Ask before
  deleting if the path is not the default one.
