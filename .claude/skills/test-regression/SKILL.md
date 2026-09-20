---
name: test-regression
description: Pre-commit verification gate for markdown-memory - runs ruff (lint and format check), mypy --strict, pytest and the live end-to-end test in sequence, stopping at the first failure. Use before committing or opening a PR, after finishing a change, or when asked to "run all checks", "verify everything passes", or "is this ready to commit".
---

# test-regression

## Run

```bash
scripts/check.sh            # ruff check, ruff format --check, mypy --strict, pytest, live_test.py
scripts/check.sh --fast     # same without the live end-to-end test
```

The script is fail-fast (`set -euo pipefail`): the first failing step stops the run and its
exit code is non-zero. A full run takes about two minutes; most of that is the real
embedding model loading in pytest and in the live test.

## The steps

| Step | Command | Passing looks like |
| --- | --- | --- |
| Lint | `uv run ruff check .` | `All checks passed!` |
| Format | `uv run ruff format --check .` | `N files already formatted` |
| Types | `uv run mypy --strict src/` | `Success: no issues found` |
| Tests | `uv run pytest -q` | all passed (tests marked `embedding` skip only if the model cannot load) |
| End-to-end | `uv run python scripts/live_test.py` | `LIVE TEST PASSED: N checks` |

The live test spawns the real server as a subprocess and drives it over stdio JSON-RPC:
tool discovery, indexing, outline, exact section boundaries, keyword and semantic search,
error reporting, incremental re-indexing, stdout hygiene, schema integrity, server memory
(< 3 GB peak) and warm latency.

## On failure

1. Report the failing step and its output verbatim. Do not summarise an error away.
2. Fix the cause, not the check: no `# type: ignore`, `# noqa`, skipped tests, loosened
   assertions or raised thresholds to get to green. If a check is genuinely wrong, say so
   and let the user decide.
3. `uv run ruff check . --fix` and `uv run ruff format .` may be applied freely; re-run the
   gate afterwards.
4. Re-run the **whole** gate after a fix, not just the step that failed.

## Not covered here

Changes to ranking, chunking, unit extraction or embedding also need the retrieval gate -
use the `run-eval` skill (`uv run python scripts/eval_retrieval.py`). It is separate because
it measures search quality rather than correctness.

Committing is a separate, explicit step: a green gate is not permission to commit.

`.githooks/pre-push` runs this same gate on every push once the clone has been pointed at
it (`git config core.hooksPath .githooks`), so a red gate cannot reach the remote. Running
it by hand first is still faster than finding out at push time.
