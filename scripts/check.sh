#!/usr/bin/env bash
# Pre-commit verification gate: vendored licences, lint, format, types, tests,
# live end-to-end test.
# Stops at the first failing step and exits non-zero.
#
#   scripts/check.sh            # everything
#   scripts/check.sh --fast     # skip the live end-to-end test (no model load over stdio)
#
# Changes to ranking, chunking or embedding must ALSO pass:
#   uv run python scripts/eval_retrieval.py
set -euo pipefail
cd "$(dirname "$0")/.."

fast=0
[[ "${1:-}" == "--fast" ]] && fast=1

step() {
    printf '\n==> %s\n' "$*"
    "$@"
}

# First, and cheapest: the vendored evaluation corpus carries upstream licence and
# NOTICE files that Apache-2.0 requires to travel with the copy. Nothing else in this
# gate would notice one going missing.
step uv run python scripts/fetch_eval_corpus.py --check
step uv run ruff check .
step uv run ruff format --check .
step uv run mypy --strict src/
step uv run pytest -q
if [[ $fast -eq 0 ]]; then
    step uv run python scripts/live_test.py
fi

printf '\nAll checks passed.\n'
