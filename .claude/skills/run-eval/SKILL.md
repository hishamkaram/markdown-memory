---
name: run-eval
description: Run the markdown-memory retrieval benchmark, check the accuracy gates (held-out Top-1 >= 80%, Top-5 >= 90%, identifiers 100%) and report latency and the delta against the frozen baseline. Use after any change to ranking, chunking, unit extraction, FTS query building, the embedder or its prompts, or when asked to "run the eval", "check retrieval accuracy", or "did search get worse".
---

# run-eval

Scores the **shipped** `search_docs` pipeline against the frozen corpus
(`scripts/eval_data/corpus/`, 54 sections) and the labelled queries
(`scripts/eval_data/queries.json`: 34 paraphrase + identifier queries per split).

## Run

```bash
uv run python scripts/eval_retrieval.py --show-misses
```

Takes about a minute with the default embedder (EmbeddingGemma-300m; the first run also
downloads ~330 MB). Add `--embedder bge-small` to score the light preset (informational
only - the gates are calibrated for the default embedder).

## Read the result

The script prints one row per query set, then the delta against
`scripts/eval_data/baseline.json`, then `OK` or `REGRESSION`. **Its exit code is the
verdict**: non-zero means a gate failed.

| Gate (held-out set) | Floor | Frozen baseline |
| --- | --- | --- |
| Paraphrase Top-1 | >= 80% | 88% |
| Paraphrase Top-5 | >= 90% | 97% |
| Identifier Top-1 (dev and held-out) | = 100% | 100% |

One query is ~3 percentage points (34 queries per set), so a 3pp move is a single query
changing rank - look at `--show-misses` before calling it a trend. Latency deltas are
informational: they depend on the machine and on what else is running.

## Report

State, in this order:

1. Verdict: gates passed or which gate failed (quote the `REGRESSION:` line).
2. The held-out paraphrase row (Top-1 / Top-3 / Top-5) and both identifier rows.
3. Delta vs baseline in percentage points for each set, and the median latency delta.
4. Any query that newly misses Top-1 compared with the baseline run (from `--show-misses`).

## Rules

- **Never tune on `held_out`.** Choose parameters using the `dev` rows only; the held-out
  queries were written before any tuning and exist to catch overfitting.
- Do not edit `queries.json` or the corpus to make a change pass. If a label is genuinely
  wrong, say so and let the user decide.
- Only after the user accepts a change: `uv run python scripts/eval_retrieval.py
  --update-baseline`, then update the accuracy tables in `README.md` and `CLAUDE.md`.
- A failed gate blocks the change. Report it; do not lower the floors in
  `scripts/eval_retrieval.py`.
