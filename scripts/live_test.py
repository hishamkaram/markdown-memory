"""End-to-end live validation of the markdown-memory MCP server.

Spawns the real server as a subprocess and talks to it over stdio with the MCP client,
exactly as Claude Code / Cursor / Claude Desktop would. Nothing is mocked: documents are
parsed, embedded with the local ONNX model, stored in SQLite (FTS5 + sqlite-vec) and
queried through JSON-RPC tool calls.

    uv run python scripts/live_test.py

Exits non-zero on the first failed check. This script is a *client*, so it may print;
the server process it launches must keep its stdout pure JSON-RPC, which is verified.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters, stdio_client
from mcp.types import CallToolResult, TextContent

from markdown_memory.config import ServerConfig
from markdown_memory.db import SCHEMA_VERSION, Database
from markdown_memory.embedders import DEFAULT_EMBEDDER, GEMMA_DIMENSION
from markdown_memory.models import estimate_tokens

# How long [9] leaves the server idle before measuring what that idleness costs and what
# the query after it costs. Long enough that any onnxruntime spin window has closed.
_IDLE_SECONDS = 2.0

CONFIGURATION_MD = """\
---
title: Helios Configuration Reference
owner: platform-team
---

Every option can be set in `helios.toml`, by environment variable, or by flag.
Flags win over environment variables, which win over the file.

# Helios Configuration Reference

## Environment Variables

### Ingest

| Variable | Type | Default | Description |
| --- | --- | --- | --- |
| `HELIOS_INGEST_BATCH_SIZE` | int | `500` | Records pulled from the broker per poll |
| `HELIOS_INGEST_WORKERS` | int | `4` | Parallel decoder goroutines |
| `HELIOS_INGEST_MAX_LAG_MS` | int | `15000` | Lag that trips the backpressure circuit |

### Storage

| Variable | Type | Default | Description |
| --- | --- | --- | --- |
| `HELIOS_STORE_DSN` | string | *(required)* | PostgreSQL connection string |
| `HELIOS_STORE_POOL_MAX` | int | `32` | Upper bound of pooled connections |
| `HELIOS_WAL_SEGMENT_MB` | int | `64` | Size at which a write-ahead segment rolls |

## Command Line Flags

```text
helios serve [flags]

  --config string               path to helios.toml (default "/etc/helios/helios.toml")
  --replay-from-offset int      re-consume the topic starting at this offset
  --dry-run                     decode and validate without writing to storage
  --metrics-addr string         Prometheus listener (default ":9102")
```

`--replay-from-offset` is destructive when combined with `--truncate`; take a snapshot first.

## File Format

```toml
[ingest]
batch_size = 500
workers = 4

[store]
dsn = "postgres://helios@db.internal/helios"
pool_max = 32
```
"""

ARCHITECTURE_MD = """\
# Ingest Pipeline Architecture

Status: **accepted** - revision 7.

## Overview

Helios consumes change events from the broker, normalises them, and writes them to the
columnar store. The pipeline is a chain of bounded stages connected by channels.

## Stages

### Decoder

The decoder turns Avro payloads into the internal `Record` struct. Schema lookups are
cached for ten minutes; a cache miss costs one round trip to the schema registry.

```go
type Record struct {
    Key       []byte
    Timestamp time.Time
    Fields    map[string]Value
}
```

### Deduplicator

Records are deduplicated on `(key, timestamp)` with a sliding window of two minutes
backed by a cuckoo filter. False positives drop at most 0.01% of legitimate records.

### Writer

The writer groups records into row groups of 64 MiB and flushes them with a two-phase
commit against the metadata catalog.

## Backpressure

When the writer falls behind, channel buffers fill up and the decoder blocks. Once lag
exceeds the configured ceiling the consumer stops polling the broker entirely, which lets
the broker retain the data instead of the process exhausting its memory. Polling resumes
after lag falls under half of the ceiling.

## Failure Modes

| Failure | Detection | Automatic response |
| --- | --- | --- |
| Schema registry down | lookup timeout | serve from cache, then pause the partition |
| Catalog commit conflict | optimistic lock error | retry with jitter, max 5 attempts |
| Disk full | `ENOSPC` on flush | stop ingest, page on-call |

## Capacity Planning

{capacity}
"""

DEPLOYMENT_MD = """\
# Deployment Guide

## Kubernetes

Apply the manifests with kustomize:

```bash
kubectl apply -k deploy/overlays/production
kubectl rollout status deployment/helios
```

```yaml
resources:
  requests:
    cpu: "2"
    memory: 4Gi
```

## Secrets Management

API tokens and database passwords are injected from Vault by the agent sidecar. Rotate
them every thirty days; the process re-reads the mounted files on `SIGHUP`, so rotation
needs no restart.

## Rolling Back

Run `helios migrate down --steps 1`, then redeploy the previous image tag. Migrations are
backwards compatible for exactly one release.

## Observability

Dashboards live in Grafana under *Helios / Ingest*. Alerts fire on consumer lag, flush
latency and error-budget burn rate.
"""

TROUBLESHOOTING_MD = """\
![runbook](https://img.shields.io/badge/runbook-live-blue.svg)
Pasted from the on-call wiki; formatting is rough.

# Troubleshooting

#### Consumer stuck at the same offset

Check for a poison message. Skip it with the replay flag after capturing the payload.

## Out Of Memory Kills

The container is OOM-killed when batch size times record size exceeds the memory limit.

```bash
kubectl top pod -l app=helios

# list the largest recent batches
helios debug batches --top 10

## Slow Queries After Upgrade

Statistics are stale after a major version bump. Run `ANALYZE` on the fact tables.

### Verifying The Fix

Compare p99 latency before and after in the dashboard.
"""

CHANGELOG_MD = """\
# Changelog

## 2.1.0

### Added

- Backpressure circuit with hysteresis.

### Fixed

- Catalog commit retry storm.

## 2.0.0

### Added

- Two-phase commit writer.

### Fixed

- Decoder panic on empty payloads.
"""


def build_corpus(root: Path) -> dict[str, str]:
    capacity = "\n\n".join(
        f"**Scenario {number}.** "
        + "A cluster sized for this scenario needs headroom for replays, compaction and "
        "traffic spikes, so provision for twice the steady-state throughput and verify it "
        "with a synthetic load test before onboarding new tenants. " * 3
        for number in range(1, 11)
    )
    files = {
        "reference/configuration.md": CONFIGURATION_MD,
        "architecture/ingest-pipeline.md": ARCHITECTURE_MD.replace("{capacity}", capacity),
        "guides/deployment.md": DEPLOYMENT_MD,
        "guides/troubleshooting.md": TROUBLESHOOTING_MD,
        "CHANGELOG.md": CHANGELOG_MD,
        "notes/empty.md": "",
    }
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (root / "node_modules" / "left-pad").mkdir(parents=True)
    (root / "node_modules" / "left-pad" / "README.md").write_text("# vendored, must be skipped\n")
    return files


def server_memory_mb() -> tuple[float, float] | None:
    """``(current RSS, peak RSS)`` in MB of the spawned server process (Linux only)."""
    me = str(os.getpid())
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            if entry.joinpath("stat").read_text().rsplit(")", 1)[1].split()[1] != me:
                continue
            if b"markdown_memory.server" not in entry.joinpath("cmdline").read_bytes():
                continue
            fields = dict(
                line.split(":", 1) for line in entry.joinpath("status").read_text().splitlines()
            )
            return (
                int(fields["VmRSS"].split()[0]) / 1024,
                int(fields["VmHWM"].split()[0]) / 1024,
            )
        except (OSError, KeyError, IndexError, ValueError):
            continue
    return None


def server_cpu_seconds() -> float | None:
    """CPU the spawned server has used, in seconds (Linux only).

    Fields 14 and 15 of /proc/<pid>/stat are the process's user and system ticks, summed
    over every thread it owns - which is the point, because the cost being measured is
    onnxruntime's thread pool, not the thread that happens to answer the request.
    """
    me = str(os.getpid())
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            after_name = entry.joinpath("stat").read_text().rsplit(")", 1)[1].split()
            if after_name[1] != me:
                continue
            if b"markdown_memory.server" not in entry.joinpath("cmdline").read_bytes():
                continue
            # after_name[0] is field 3, so fields 14 and 15 are at offsets 11 and 12.
            ticks = int(after_name[11]) + int(after_name[12])
            return ticks / os.sysconf("SC_CLK_TCK")
        except (OSError, IndexError, ValueError):
            continue
    return None


def check_schema_integrity(test: LiveTest, db_path: Path) -> None:
    """Open the database the server just closed and verify every index agrees."""
    # Read the stored size with plain sqlite3 first: opening a Database with a different
    # embedding size REBUILDS it, and an inspection must never be able to do that.
    plain = sqlite3.connect(db_path)
    try:
        stored = plain.execute("SELECT value FROM meta WHERE key = 'embedding_dim'").fetchone()
    finally:
        plain.close()
    test.check(
        stored is not None and int(stored[0]) == GEMMA_DIMENSION,
        f"index was built with {GEMMA_DIMENSION}-dimensional vectors (meta says {stored})",
    )
    with Database(db_path, embedding_dim=GEMMA_DIMENSION) as database:
        conn = database.connection()
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        test.check(version == SCHEMA_VERSION, f"schema is at version {SCHEMA_VERSION}")
        test.check(database.pragma("journal_mode") == "wal", "journal mode is WAL")
        test.check(
            [row[0] for row in conn.execute("PRAGMA integrity_check")] == ["ok"],
            "PRAGMA integrity_check: ok",
        )
        test.check(
            conn.execute("PRAGMA foreign_key_check").fetchall() == [],
            "PRAGMA foreign_key_check: no orphans",
        )
        conn.execute("INSERT INTO sections_fts(sections_fts, rank) VALUES ('integrity-check', 1)")
        test.check(True, "FTS5 integrity-check passed (index matches the sections table)")
        counts = {
            name: database.count_rows(name)
            for name in (
                "documents",
                "sections",
                "sections_fts",
                "sections_vec",
                "units",
                "units_vec",
            )
        }
        print(f"        rows: {counts}")
        test.check(
            counts["documents"] > 0 and counts["sections"] > 0 and counts["units"] > 0,
            "the re-opened index still holds the documents (an empty one proves nothing)",
        )
        test.check(
            counts["sections"] == counts["sections_fts"], "every section is in the FTS index"
        )
        test.check(counts["units"] == counts["units_vec"], "every passage has exactly one vector")
        with_body = int(conn.execute("SELECT COUNT(DISTINCT section_id) FROM units").fetchone()[0])
        test.check(
            counts["sections_vec"] == with_body < counts["sections"],
            f"{with_body} sections with a body have a vector; heading-only sections have none",
        )
        dims = {len(row[0]) // 4 for row in conn.execute("SELECT embedding FROM units_vec LIMIT 5")}
        test.check(dims == {GEMMA_DIMENSION}, f"stored vectors are {GEMMA_DIMENSION}-dimensional")
        test.check(database.get_meta("embedding_dim") == str(GEMMA_DIMENSION), "meta records 768")
        problems = database.integrity_problems()
        test.check(problems == [], f"Database.integrity_problems() is empty {problems or ''}")
        model = database.get_meta("embedding_model") or ""
        test.check("embeddinggemma-300m-ONNX@" in model, f"meta records the pinned model: {model}")


class CheckFailedError(Exception):
    """A live-test expectation did not hold."""


class LiveTest:
    def __init__(self, client: Client, docs: Path, files: dict[str, str]) -> None:
        self.client = client
        self.docs = docs
        self.files = files
        self.checks = 0
        self.metrics: dict[str, float] = {}

    # ------------------------------------------------------------------ plumbing

    def check(self, condition: bool, description: str) -> None:
        if not condition:
            print(f"  FAIL  {description}")
            raise CheckFailedError(description)
        self.checks += 1
        print(f"  ok    {description}")

    async def call(self, tool: str, **arguments: Any) -> tuple[Any, str, float]:
        """Returns (structured result, raw text exactly as the model would read it, ms)."""
        started = time.perf_counter()
        outcome = await self.client.call_tool(tool, arguments)
        elapsed_ms = (time.perf_counter() - started) * 1000
        assert isinstance(outcome, CallToolResult)
        text = "\n".join(block.text for block in outcome.content if isinstance(block, TextContent))
        if outcome.is_error:
            return None, text, elapsed_ms
        structured = outcome.structured_content or {}
        # A tool returning an object is its own structured content; only a bare list or
        # scalar arrives wrapped in "result".
        payload = structured["result"] if set(structured) == {"result"} else structured
        return payload, text, elapsed_ms

    async def median_latency(self, tool: str, runs: int = 7, **arguments: Any) -> float:
        samples = [(await self.call(tool, **arguments))[2] for _ in range(runs)]
        return statistics.median(samples)

    # ------------------------------------------------------------------ scenarios

    async def run(self) -> None:
        await self.handshake()
        await self.indexing()
        await self.outline()
        await self.read_section()
        await self.keyword_search()
        await self.semantic_search()
        await self.error_reporting()
        await self.incremental()

    async def handshake(self) -> None:
        print("\n[1] MCP handshake and tool discovery (stdio JSON-RPC)")
        listing = await self.client.list_tools()
        names = sorted(tool.name for tool in listing.tools)
        self.check(
            names
            == [
                "get_document_outline",
                "index_directory",
                "list_documents",
                "read_section",
                "search_docs",
            ],
            f"server exposes exactly the five tools: {', '.join(names)}",
        )

    async def indexing(self) -> None:
        print("\n[2] index_directory: parse -> embed (local ONNX) -> SQLite")
        summary, _, elapsed = await self.call("index_directory")
        self.metrics["index_cold_ms"] = elapsed
        print(f"        {summary}")
        self.check("6 scanned, 6 (re)indexed" in summary, "all 6 Markdown files indexed")
        self.check("ERROR" not in summary, "no per-file errors")
        answer, _, _ = await self.call("list_documents")
        self.check(
            "list_documents returns an envelope", set(answer) == {"documents", "index_status"}
        )
        self.check(
            "a freshly indexed root vouches for itself",
            answer["index_status"]
            == {"coverage": "verified", "failures": [], "changed_files": 0, "message": None},
        )
        documents = answer["documents"]
        titles = {Path(d["file_path"]).name: d["title"] for d in documents}
        self.check(len(documents) == 6, "list_documents reports 6 documents")
        self.check(
            not any("node_modules" in d["file_path"] for d in documents),
            "vendored node_modules tree was skipped",
        )
        self.check(
            titles["configuration.md"] == "Helios Configuration Reference"
            and titles["troubleshooting.md"] == "Troubleshooting"
            and titles["empty.md"] == "empty",
            "titles come from H1 / front matter / file name",
        )
        sections = sum(d["section_count"] for d in documents)
        self.metrics["sections"] = sections
        self.check(sections > 30, f"{sections} sections stored")
        self.check("passages)" in summary, "per-passage vectors were embedded too")
        scoped_answer, _, _ = await self.call("list_documents", directory="guides")
        scoped = scoped_answer["documents"]
        self.check(len(scoped) == 2, "list_documents(directory='guides') filters to 2 documents")

    async def outline(self) -> None:
        print("\n[3] get_document_outline: hierarchical TOC at a fraction of the file's tokens")
        relative = "architecture/ingest-pipeline.md"
        outline, raw, elapsed = await self.call("get_document_outline", file_path=relative)
        self.metrics["outline_ms"] = elapsed
        document_tokens = len(self.files_text(relative)) // 4
        outline_tokens = len(raw) // 4
        self.metrics["outline_tokens"] = outline_tokens
        self.metrics["document_tokens"] = document_tokens
        print(f"        document ~{document_tokens} tokens, outline ~{outline_tokens} tokens")
        self.check(outline_tokens * 4 < document_tokens, "outline costs < 25% of the document")
        self.check("content" not in raw, "outline carries no section bodies")
        root = outline[0]
        self.check(
            root["title"] == "Ingest Pipeline Architecture" and root["level"] == 1,
            "single H1 root node",
        )
        children = [child["title"] for child in root["children"]]
        self.check(
            children
            == ["Overview", "Stages", "Backpressure", "Failure Modes", "Capacity Planning"],
            f"H2 children in order: {children}",
        )
        stages = root["children"][1]
        self.check(
            [c["heading_path"] for c in stages["children"]]
            == [
                "Ingest Pipeline Architecture > Stages > Decoder",
                "Ingest Pipeline Architecture > Stages > Deduplicator",
                "Ingest Pipeline Architecture > Stages > Writer",
            ],
            "H3 breadcrumbs are 'Root > Child > Subchild'",
        )
        capacity = root["children"][4]
        self.check(
            capacity.get("parts", 1) >= 2 and capacity["tokens"] > 800,
            f"oversized section reported as {capacity.get('parts')} parts "
            f"(~{capacity['tokens']} tokens)",
        )
        source = self.files_text(relative).split("\n")
        start, end = (int(n) for n in stages["children"][0]["lines"].split("-"))
        self.check(
            source[start - 1] == "### Decoder" and source[end - 1] == "```",
            f"line range {start}-{end} points at the Decoder section",
        )

        messy, _, _ = await self.call("get_document_outline", file_path="guides/troubleshooting.md")
        top = [node["title"] for node in messy]
        self.check(top == ["[Overview / Preamble]", "Troubleshooting"], "preamble captured")
        nested = {c["title"]: c for c in messy[1]["children"]}
        self.check(
            nested["Consumer stuck at the same offset"]["level"] == 4,
            "skipped level (# -> ####) nests under the H1",
        )
        self.check(
            "Slow Queries After Upgrade" in nested
            and [c["title"] for c in nested["Slow Queries After Upgrade"]["children"]]
            == ["Verifying The Fix"],
            "sections after an unclosed code fence were recovered",
        )
        self.check(
            not any("largest recent batches" in title for title in nested),
            "shell comment inside the unclosed fence did not become a heading",
        )

    async def read_section(self) -> None:
        print("\n[4] read_section: exact boundary extraction")
        relative = "reference/configuration.md"
        heading = "Helios Configuration Reference > Environment Variables > Storage"
        text, _, elapsed = await self.call("read_section", file_path=relative, heading_path=heading)
        self.metrics["read_section_ms"] = elapsed
        source = self.files_text(relative).split("\n")
        expected = "\n".join(
            source[source.index("### Storage") : source.index("## Command Line Flags")]
        ).rstrip("\n")
        self.check(text == expected, "returned text is byte-identical to the source slice")
        self.check(text.startswith("### Storage"), "starts at the heading line")
        self.check("HELIOS_INGEST_BATCH_SIZE" not in text, "previous sibling not included")
        self.check("--replay-from-offset" not in text, "next section not included")
        table_rows = [line for line in text.split("\n") if line.startswith("|")]
        self.check(
            len(table_rows) == 5 and all(row.count("|") == 5 for row in table_rows),
            "Markdown table intact: header, separator and 3 data rows of 4 columns",
        )
        self.check(
            "| `HELIOS_WAL_SEGMENT_MB` | int | `64` | Size at which a write-ahead segment rolls |"
            in table_rows,
            "table row preserved verbatim",
        )
        whole_tokens = len(self.files_text(relative)) // 4
        print(f"        section ~{len(text) // 4} tokens vs whole file ~{whole_tokens} tokens")

        absolute = str(self.docs / relative)
        flags, _, _ = await self.call(
            "read_section", file_path=absolute, heading_path="command line flags"
        )
        self.check(
            flags.startswith("## Command Line Flags")
            and flags.rstrip().endswith("snapshot first."),
            "absolute path + case-insensitive short heading path resolve",
        )
        self.check("helios serve [flags]" in flags, "fenced code block preserved verbatim")

        relative = "architecture/ingest-pipeline.md"
        base = "Ingest Pipeline Architecture > Capacity Planning"
        joined, _, _ = await self.call("read_section", file_path=relative, heading_path=base)
        source_text = self.files_text(relative)
        expected = source_text[source_text.index("## Capacity Planning") :].rstrip("\n")
        self.check(joined == expected, "multi-part section reassembles byte-for-byte")
        part, _, _ = await self.call(
            "read_section", file_path=relative, heading_path=f"{base} (Part 2)"
        )
        self.check(
            part.startswith("**Scenario") and len(part) <= 3200 and part in joined,
            f"'(Part 2)' addresses one ~{len(part) // 4}-token chunk",
        )

        parent, _, _ = await self.call(
            "read_section", file_path=relative, heading_path="Ingest Pipeline Architecture > Stages"
        )
        self.check(parent == "## Stages", "parent heading returns only its own text")
        subtree, _, _ = await self.call(
            "read_section",
            file_path=relative,
            heading_path="Ingest Pipeline Architecture > Stages",
            include_subsections=True,
        )
        lines = source_text.split("\n")
        expected = "\n".join(lines[lines.index("## Stages") : lines.index("## Backpressure")])
        self.check(
            subtree == expected.rstrip("\n"), "include_subsections returns the exact subtree"
        )

        first, _, _ = await self.call(
            "read_section", file_path="CHANGELOG.md", heading_path="Changelog > 2.1.0 > Fixed"
        )
        second, _, _ = await self.call(
            "read_section", file_path="CHANGELOG.md", heading_path="Changelog > 2.0.0 > Fixed"
        )
        self.check(
            "retry storm" in first and "empty payloads" in second,
            "repeated headings stay distinct through their breadcrumbs",
        )

    async def keyword_search(self) -> None:
        print("\n[5] search_docs: exact keyword queries are answered by FTS5 (BM25)")
        for query, expected_path in (
            (
                "HELIOS_WAL_SEGMENT_MB",
                "Helios Configuration Reference > Environment Variables > Storage",
            ),
            ("--replay-from-offset", "Helios Configuration Reference > Command Line Flags"),
            ("ENOSPC", "Ingest Pipeline Architecture > Failure Modes"),
        ):
            answer, _, elapsed = await self.call("search_docs", query=query, limit=5)
            results = answer["results"]
            top = results[0]
            print(
                f"        {query!r} -> {top['heading_path']} "
                f"(fts_rank={top['fts_rank']}, vec_rank={top['vec_rank']}, {elapsed:.1f} ms)"
            )
            self.check(top["heading_path"] == expected_path, f"top hit for {query!r} is correct")
            self.check(top["fts_rank"] == 1, "FTS5 ranked it first")
            self.check(query in top["content"], "the literal identifier is in the returned section")
            self.check(answer["keyword_match"] == "matched", "keyword_match says it matched")
        absent, _, _ = await self.call("search_docs", query="maxItemErrors", limit=5)
        self.check(
            absent["keyword_match"] == "no_match" and "keyword_message" in absent,
            "an identifier no section contains is no_match, and says so",
        )
        self.check(absent["results"] == [], "an identifier no section contains gets no neighbours")
        results = (await self.call("search_docs", query="HELIOS_WAL_SEGMENT_MB", limit=5))[0][
            "results"
        ]
        top, pointers = results[0], results[1:]
        self.check(
            abs(top["score"] - (1 / (60 + top["fts_rank"]) + 1 / (60 + top["vec_rank"]))) < 1e-6,
            "score equals 1/(60+fts_rank) + 1/(60+vec_rank)",
        )
        self.check(
            bool(pointers)
            and all(
                {"file_path", "heading_path", "lines", "tokens"} <= set(p)
                and not {"content", "score", "fts_rank", "vec_rank"} & set(p)
                for p in pointers
            ),
            "every hit after the first is a pointer: where, how big, why - no content, no ranks",
        )
        followed, _, _ = await self.call(
            "read_section",
            file_path=pointers[0]["file_path"],
            heading_path=pointers[0]["heading_path"],
        )
        self.check(
            isinstance(followed, str) and estimate_tokens(followed) == pointers[0]["tokens"],
            "a pointer followed verbatim with read_section costs what it said",
        )
        for hostile in ('"unbalanced', "NEAR(", "a AND", "*", "col:umn", "--", "'; DROP TABLE x;"):
            outcome = await self.client.call_tool("search_docs", {"query": hostile})
            assert isinstance(outcome, CallToolResult)
            if outcome.is_error:
                self.check(False, f"FTS5 syntax in user input must not error: {hostile!r}")
        self.check(True, "FTS5 operator characters in queries are neutralised")

    async def semantic_search(self) -> None:
        print("\n[6] search_docs: conceptual queries are answered by the vector index")
        # (query, target section, worst acceptable position). None of the queries' content
        # words occur in their target, so FTS5 cannot retrieve it: fts_rank must be None.
        cases = (
            (
                "what happens when the system is overloaded",
                "Ingest Pipeline Architecture > Backpressure",
                1,
            ),
            ("credential expiry policy", "Deployment Guide > Secrets Management", 1),
            # "deploy" also matches the Kubernetes section lexically, so keywords and vectors
            # disagree here; fusion must still surface the semantically right section.
            ("how do I undo a bad deploy", "Deployment Guide > Rolling Back", 3),
        )
        for query, expected_path, worst_position in cases:
            answer, _, elapsed = await self.call("search_docs", query=query, limit=5)
            results = answer["results"]
            hit = next((r for r in results if r["heading_path"] == expected_path), None)
            position = results.index(hit) + 1 if hit else None
            print(
                f"        {query!r} -> #{position} {expected_path} "
                f"(fts_rank={hit and hit.get('fts_rank')}, vec_rank={hit and hit.get('vec_rank')}, "
                f"{elapsed:.1f} ms)"
            )
            self.check(
                hit is not None and position is not None and position <= worst_position,
                f"target section ranked within the top {worst_position}",
            )
            assert hit is not None
            if position == 1:  # only the first hit carries its ranks; a pointer has none
                self.check(hit["fts_rank"] is None, "FTS5 did not match it: keywords alone miss")
                self.check(hit["vec_rank"] == 1, "vector index ranked it first")
            else:
                self.check("content" not in hit and "tokens" in hit, "found as a pointer")

    async def error_reporting(self) -> None:
        print("\n[7] errors are actionable for the agent")
        outcome = await self.client.call_tool(
            "read_section", {"file_path": "CHANGELOG.md", "heading_path": "Changelog > 9.9.9"}
        )
        assert isinstance(outcome, CallToolResult)
        text = "\n".join(b.text for b in outcome.content if isinstance(b, TextContent))
        self.check(outcome.is_error is True, "unknown heading -> is_error result, server stays up")
        self.check("Changelog > 2.1.0 > Added" in text, "error lists the available heading paths")
        outcome = await self.client.call_tool("get_document_outline", {"file_path": "missing.md"})
        assert isinstance(outcome, CallToolResult)
        self.check(outcome.is_error is True, "unknown file -> is_error result")

    async def incremental(self) -> None:
        print("\n[8] incremental re-indexing (SHA-256 cache, purge of deleted files)")
        summary, _, elapsed = await self.call("index_directory")
        self.metrics["index_noop_ms"] = elapsed
        self.check("0 (re)indexed, 6 unchanged, 0 purged" in summary, "unchanged files are skipped")
        changelog = self.docs / "CHANGELOG.md"
        changelog.write_text(
            changelog.read_text() + "\n## 1.9.0\n\n- Introduced the zstd codec.\n", encoding="utf-8"
        )
        (self.docs / "guides" / "deployment.md").unlink()
        summary, _, elapsed = await self.call("index_directory")
        self.metrics["index_delta_ms"] = elapsed
        print(f"        {summary}")
        self.check("1 (re)indexed, 4 unchanged, 1 purged" in summary, "1 changed, 1 deleted")
        results = (await self.call("search_docs", query="zstd codec"))[0]["results"]
        self.check(results[0]["heading_path"] == "Changelog > 1.9.0", "new section is searchable")
        results = (await self.call("search_docs", query="Vault sidecar SIGHUP", limit=20))[0][
            "results"
        ]
        self.check(
            not any("deployment.md" in r["file_path"] for r in results),
            "purged file left no sections, FTS rows or vectors behind",
        )

    async def latency(self) -> None:
        print("\n[9] warm query latency (median of 7, end-to-end over stdio JSON-RPC)")
        # Sampled around this one series and nothing else, so the number divides by a
        # known count of queries. Reported, never asserted: a CPU threshold on a shared
        # runner measures the runner.
        before_cpu = server_cpu_seconds()
        self.metrics["search_keyword_ms"] = await self.median_latency(
            "search_docs", query="HELIOS_INGEST_BATCH_SIZE"
        )
        after_cpu = server_cpu_seconds()
        if before_cpu is not None and after_cpu is not None:
            self.metrics["search_cpu_ms"] = (after_cpu - before_cpu) * 1000 / 7
        self.metrics["search_semantic_ms"] = await self.median_latency(
            "search_docs", query="how do I stop the process from running out of memory"
        )
        self.metrics["outline_warm_ms"] = await self.median_latency(
            "get_document_outline", file_path="architecture/ingest-pipeline.md"
        )
        self.metrics["read_section_warm_ms"] = await self.median_latency(
            "read_section",
            file_path="architecture/ingest-pipeline.md",
            heading_path="Ingest Pipeline Architecture > Backpressure",
        )

        # What an idle server costs. onnxruntime's threads spin-wait between operators by
        # default, which kept burning ~0.5 core-seconds per second after a query returned;
        # with spinning off this reads as ~0. Measured with the connection open and no
        # request in flight, which is what a server beside an editor does almost always.
        idle_start = server_cpu_seconds()
        if idle_start is not None:
            await asyncio.sleep(_IDLE_SECONDS)
            idle_end = server_cpu_seconds()
            if idle_end is not None:
                self.metrics["idle_cpu_ms"] = (idle_end - idle_start) * 1000
        # And what the first query after that idle costs. Named for what it is: after two
        # seconds any spin window has long closed, so this is not the wake-up trade-off,
        # it is the latency a client actually sees, since queries arrive in gaps.
        payload, _, after_idle_ms = await self.call("search_docs", query="ENOSPC")
        # `call` returns None for a tool error, and a latency recorded for a query that
        # failed is a number describing nothing.
        self.check(payload is not None, "the query after a long idle still answers")
        self.metrics["query_after_idle_ms"] = after_idle_ms

    def files_text(self, relative: str) -> str:
        return (self.docs / relative).read_text(encoding="utf-8")


async def _search_finds(probe: LiveTest, token: str, timeout: float) -> bool:
    """Search for ``token`` until a hit quotes it, or give up after ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        payload, _, _ = await probe.call("search_docs", query=token)
        hits = payload["results"] if payload else []
        # Only the first hit carries `content`; a pointer shows why it matched.
        if any(token in (hit.get("content") or hit.get("matched_passage") or "") for hit in hits):
            return True
        await asyncio.sleep(1)
    return False


async def auto_index_check(test: LiveTest, root: Path, model_cache: str) -> None:
    docs = root / "auto-docs"
    docs.mkdir()
    guide = docs / "guide.md"
    guide.write_text("# Guide\n\nThe batcher reads HELIOS_BATCH at start.\n", encoding="utf-8")
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "markdown_memory.server"],
        env={
            **os.environ,
            "MARKDOWN_MEMORY_DB": str(root / "auto.db"),
            "MARKDOWN_MEMORY_DOCS_DIR": str(docs),
            "MARKDOWN_MEMORY_MODEL_CACHE": model_cache,
            "MARKDOWN_MEMORY_EMBEDDER": DEFAULT_EMBEDDER,
            "MARKDOWN_MEMORY_AUTO_INDEX": "1",
        },
    )
    with (root / "auto.stderr.log").open("w", encoding="utf-8") as errlog:
        async with Client(stdio_client(parameters, errlog=errlog)) as client:
            probe = LiveTest(client, docs, {})
            test.check(
                await _search_finds(probe, "HELIOS_BATCH", timeout=180),
                "the first search started the catch-up run without being asked",
            )
            guide.write_text(
                "# Guide\n\nThe batcher now reads ZEPHYR_QUOTA instead.\n", encoding="utf-8"
            )
            await asyncio.sleep(11)  # past the gap after the last run
            test.check(
                await _search_finds(probe, "ZEPHYR_QUOTA", timeout=180),
                "an edited file became searchable once a search noticed it",
            )


async def main() -> int:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="markdown-memory-live-") as workspace:
        root = Path(workspace)
        docs = root / "docs"
        files = build_corpus(docs)
        server_log = root / "server.stderr.log"
        # Resolved exactly as the server does (MARKDOWN_MEMORY_MODEL_CACHE / XDG_CACHE_HOME)
        # so the model is downloaded once and shared, never into the throw-away workspace.
        model_cache = str(ServerConfig.from_env().model_cache_dir)
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "markdown_memory.server"],
            env={
                **os.environ,
                "MARKDOWN_MEMORY_DB": str(root / "index.db"),
                "MARKDOWN_MEMORY_DOCS_DIR": str(docs),
                "MARKDOWN_MEMORY_MODEL_CACHE": model_cache,
                "MARKDOWN_MEMORY_LOG_LEVEL": "INFO",
                # Pinned: an exported MARKDOWN_MEMORY_EMBEDDER must not change what is tested.
                "MARKDOWN_MEMORY_EMBEDDER": DEFAULT_EMBEDDER,
                # Off here: these scenarios call index_directory themselves and time it, and
                # a background run holding the lock would make them busy. [13] turns it on.
                "MARKDOWN_MEMORY_AUTO_INDEX": "0",
            },
        )
        print(f"workspace : {root}")
        print(f"server    : {sys.executable} -m markdown_memory.server  (stdio)")
        transport_errors: list[Exception] = []
        failed: str | None = None

        async def on_message(message: object) -> None:
            if isinstance(message, Exception):  # e.g. a non-JSON line on the server's stdout
                transport_errors.append(message)

        with server_log.open("w", encoding="utf-8") as errlog:
            async with Client(
                stdio_client(parameters, errlog=errlog), message_handler=on_message
            ) as client:
                test = LiveTest(client, docs, files)
                memory: tuple[float, float] | None = None
                try:  # handled here so the stdio transport still shuts down cleanly
                    await test.run()
                    await test.latency()
                    memory = server_memory_mb()
                except CheckFailedError as failure:
                    failed = str(failure)

        log_text = server_log.read_text(encoding="utf-8")
        if failed is not None:
            print("\n--- server stderr ---")
            print(log_text[-3000:])
            print(f"\nLIVE TEST FAILED: {failed}")
            return 1

        print("\n[10] stdio protocol hygiene")
        test.check(not transport_errors, "client saw no malformed frame on the server's stdout")
        test.check("markdown-memory serving" in log_text, "server logs went to stderr")
        test.check("Indexed " in log_text, "indexing reports were logged to stderr, not stdout")
        # search_docs degrades to the surviving index when one fails, so "no tool error"
        # alone would not prove the hostile queries above were valid FTS5 expressions.
        test.check(
            "search failed" not in log_text,
            "no query made FTS5 (or the vector index) fail behind the fallback",
        )
        test.check((root / "index.db").exists(), "SQLite database persisted on disk")

        print("\n[11] schema integrity (database re-opened after the server exited)")
        check_schema_integrity(test, root / "index.db")

        print("\n[12] server memory")
        if memory is None:
            print("        skipped: /proc is not available on this platform")
        else:
            print(f"        RSS now {memory[0]:.0f} MB, peak {memory[1]:.0f} MB")
            # A loose sentinel, not the memory gate: the real one is the VmHWM test in
            # tests/test_embedders.py, which measures one run in its own process. This
            # covers a whole server - index, search, several sessions - so it sits above
            # the ~450 MB one run of the 4-bit graph costs, with room for the index and
            # the interpreter, and still far below the ~1.45 GB an unquantized vocabulary
            # table alone would take.
            test.check(memory[1] < 800, "peak resident memory under 800 MB")
        print("        --- last server stderr lines ---")
        for line in log_text.strip().splitlines()[-4:]:
            print(f"        {line[:150]}")

        print("\n[13] automatic indexing (a second server, nobody calls index_directory)")
        await auto_index_check(test, root, model_cache)

        metrics = test.metrics
        print("\n================ timing metrics ================")
        rows = (
            ("cold index: 6 files, parse + embed + store", "index_cold_ms"),
            ("no-op re-index (all hashes match)", "index_noop_ms"),
            ("delta re-index (1 changed, 1 purged)", "index_delta_ms"),
            ("search_docs keyword   (warm median)", "search_keyword_ms"),
            ("search_docs semantic  (warm median)", "search_semantic_ms"),
            ("get_document_outline  (warm median)", "outline_warm_ms"),
            ("read_section          (warm median)", "read_section_warm_ms"),
            (f"search_docs after {_IDLE_SECONDS:.0f}s idle", "query_after_idle_ms"),
        )
        for label, key in rows:
            print(f"  {label:<46} {metrics[key]:>10.1f} ms")
        # Reported, not asserted: both depend on how many cores the runner gave us.
        for label, key in (
            ("server CPU per warm search_docs", "search_cpu_ms"),
            (f"server CPU while idle for {_IDLE_SECONDS:.0f}s", "idle_cpu_ms"),
        ):
            if key in metrics:
                print(f"  {label:<46} {metrics[key]:>10.1f} ms CPU")
        print(f"  {'sections indexed':<46} {int(metrics['sections']):>10}")
        print(
            f"  {'outline vs document tokens':<46} "
            f"{int(metrics['outline_tokens']):>5} / {int(metrics['document_tokens'])}"
        )
        test.check(metrics["index_cold_ms"] < 120_000, "cold indexing finished within 120 s")
        test.check(metrics["index_noop_ms"] < 1_000, "no-op re-index under 1 s")
        test.check(metrics["search_keyword_ms"] < 1500, "keyword query median under 1.5 s")
        test.check(metrics["search_semantic_ms"] < 1500, "semantic query median under 1.5 s")
        test.check(metrics["outline_warm_ms"] < 100, "outline median under 100 ms")
        test.check(metrics["read_section_warm_ms"] < 100, "read_section median under 100 ms")
        print(json.dumps({key: round(value, 2) for key, value in metrics.items()}))
        print(f"\nLIVE TEST PASSED: {test.checks} checks in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
