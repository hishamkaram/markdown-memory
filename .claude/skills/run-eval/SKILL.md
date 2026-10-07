---
name: run-eval
description: Run the markdown-memory retrieval benchmark, check the accuracy gates (held-out Top-1 >= 80%, Top-5 >= 90%, identifiers 100%) and report latency and the delta against the frozen baseline. Use after any change to ranking, chunking, unit extraction, FTS query building, the embedder or its prompts, or when asked to "run the eval", "check retrieval accuracy", or "did search get worse".
---

# run-eval

Scores the **shipped** `search_docs` pipeline against the frozen corpus
(`scripts/eval_data/corpus/`, 54 sections) and the labelled queries in
`scripts/eval_data/queries.json`: 34 paraphrase queries in each split, plus 10 dev and
8 held-out identifier queries. A separate no-answer stratum - 4 queries per split the corpus
cannot answer - is reported by how often the default call abstains; it never gates, and at
that size a change of one query is 25pp.

## Run

```bash
uv run python scripts/eval_retrieval.py --show-misses
```

Takes about a minute with the default embedder (EmbeddingGemma-300m; the first run also
downloads ~218 MB). The script always scores the default embedder, whatever
`MARKDOWN_MEMORY_EMBEDDER` is set to, so the gate cannot be switched off by the
environment. `--embedder bge-small` scores the light preset instead: that run is
informational, prints `GATES NOT CHECKED`, and must never be reported as "gates passed".

## Read the result

The script prints one row per query set, then the delta against
`scripts/eval_data/baseline.json`, then exactly one verdict line: `OK: ...`,
`REGRESSION: ...` (exit code non-zero) or `GATES NOT CHECKED: ...`. Only an `OK` line
means the gates passed - an exit code of 0 alone does not.

| Gate (held-out set) | Floor | Frozen baseline |
| --- | --- | --- |
| Paraphrase Top-1 | >= 80% | 88% |
| Paraphrase Top-5 | >= 90% | 97% |
| Identifier Top-1 (dev and held-out) | = 100% | 100% |

A single query changing rank moves a paraphrase row ~3pp (34 queries), a dev identifier
row 10pp and a held-out identifier row 12.5pp - so read a delta as queries, not as a
trend, and look at `--show-misses` first. The identifier gate is `= 100%`: one lost query
fails it. Latency deltas are informational: they depend on the machine and on what else
is running.

## Report

State, in this order:

1. Verdict: gates passed or which gate failed (quote the `REGRESSION:` line).
2. The held-out paraphrase row (Top-1 / Top-3 / Top-5) and both identifier rows.
3. Delta vs baseline in percentage points for each set, and the median latency delta.
4. Any query that newly misses Top-1 compared with the baseline run (from `--show-misses`).

## Tune on dev, then compare query by query

While choosing a parameter, score the dev queries alone - nothing from held-out is
validated, searched, scored or printed, and the run checks no floor:

```bash
uv run python scripts/eval_retrieval.py --split dev --record candidate.json
```

To prove a change makes **no query** worse - totals can hide one loss among several gains -
record the base revision from the same checkout, with its code on `PYTHONPATH`, then compare:

```bash
git worktree add ../base main
PYTHONPATH=../base/src uv run python scripts/eval_retrieval.py --split dev --record base.json
uv run python scripts/eval_compare.py base.json candidate.json
```

`eval_compare.py` exits 0 when no query got worse, 1 naming each one that did (rank down,
any-valid@1 lost, nDCG@5 down by more than 1e-9, a default page that came back empty, a
changed no-answer page), and 2 when the records are malformed or measure different things.
When the change moves section boundaries, score the base with its own labels
(`--queries ../base/scripts/eval_data/queries.json`) and pass `--allow-label-changes`; every
migrated label is listed. A PASS covers its own corpus, preset and split only: every shipped
preset on every evaluated corpus must pass unless excluded, with its reason, before any result
is seen; compare held-out once, at the end.

## Rules

- **Never tune on `held_out`.** Choose parameters using the `dev` rows only; the held-out
  queries were written before any tuning and exist to catch overfitting.
- Do not edit `queries.json` or the corpus to make a change pass. If a label is genuinely
  wrong, say so and let the user decide.
- Only after the user accepts a change: `uv run python scripts/eval_retrieval.py
  --update-baseline`, then update the accuracy tables in `README.md` and `CLAUDE.md`.
  Record the report-only corpus as well (`--corpus v2 --update-baseline`): its baseline
  is `<preset>@v2`, and no document quotes its numbers.
- A failed gate blocks the change. Report it; do not lower the floors in
  `scripts/eval_retrieval.py`.
