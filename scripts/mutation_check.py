"""Break each guarded behaviour on purpose and check that a test notices.

    uv run python scripts/mutation_check.py            # every mutation
    uv run python scripts/mutation_check.py --only gate gemma

A passing test suite only proves the tests do not fail. Twice now a behaviour this
project depends on - ranking a section by its closest passage, ordering keyword hits by
BM25 - could be deleted with every test still green. Each entry below removes one
behaviour and names the tests that must fail; a mutation nothing catches is a hole in the
suite, not a bug in the code.

The package is copied to a temporary directory and mutated there, never in the working
tree, and the tests run against the copy through PYTHONPATH.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Mutation:
    """One behaviour, deleted. ``tests`` is a ``pytest -k`` expression."""

    name: str
    module: str
    old: str
    new: str
    tests: str


MUTATIONS = (
    Mutation(
        name="max-sim: rank a section by its own vector, never by its best passage",
        module="search.py",
        old=(
            "            if distance < best.get(section_id, math.inf):\n"
            "                best[section_id] = distance"
        ),
        new="            pass",
        tests="TestRankingIsActuallyTested",
    ),
    Mutation(
        name="BM25: return keyword hits in row order instead of by relevance",
        module="db.py",
        old="ORDER BY rank LIMIT ?",
        new="ORDER BY rowid LIMIT ?",
        tests="TestRankingIsActuallyTested",
    ),
    Mutation(
        name="BM25: weight a heading match the same as body text",
        module="db.py",
        old="'rank', 'bm25(5.0, 3.0, 1.0)'",
        new="'rank', 'bm25(1.0, 1.0, 1.0)'",
        tests="TestRankingIsActuallyTested",
    ),
    Mutation(
        name="gate: let any identifier-shaped term bypass the IDF gate, however common",
        module="search.py",
        old="if _is_identifier(term) and frequencies[term] <= rare:",
        new="if _is_identifier(term):",
        tests="TestIdentifierGateBypassNeedsRarity",
    ),
    Mutation(
        name="search: hand back a short page when a re-index removes ranked sections",
        module="search.py",
        old="""        for _ in range(_STALE_RETRIES):
            results, stale = self._search_once(query, limit)
            if not stale:
                return results
            logger.info("Sections changed during the search; ranking again")
        return self._search_once(query, limit)[0]""",
        new="        return self._search_once(query, limit)[0]",
        tests="TestSearchDuringReindex",
    ),
    Mutation(
        name="storage: let a deleted section lend its id to the next one",
        module="db.py",
        old="        self._seed_section_ids()\n",
        new="",
        tests="TestSectionIdsOnAnUpgradedDatabase or TestMigratingARealOldDatabase",
    ),
    Mutation(
        name="indexer: consume the index-discarded notice before the report carries it",
        module="indexer.py",
        old="            self._db.dismiss_notices(notices)\n",
        new="",
        tests="TestNoticesSurviveAnAbortedRun",
    ),
    Mutation(
        name="parser: cut a fenced block that would fit in a part of its own",
        module="parser.py",
        old="""    for fence_start, fence_end in _fenced_runs(content, begin, end):
        pieces.extend(_line_pieces(content, position, fence_start, max_chars))
        if fence_end - fence_start <= max_chars:
            pieces.append((fence_start, fence_end))
        else:
            pieces.extend(_line_pieces(content, fence_start, fence_end, max_chars))
        position = fence_end
    pieces.extend(_line_pieces(content, position, end, max_chars))""",
        new="    pieces.extend(_line_pieces(content, position, end, max_chars))",
        tests="TestFencesInsideOversizedBlocks",
    ),
    Mutation(
        name="parser: stop rescuing headings hidden by an unclosed fence",
        module="parser.py",
        old="    token.content = state.getLines(",
        new="    return True\n    token.content = state.getLines(",
        tests="TestMalformedMarkdown or TestUnclosedFenceBeforeLaterFences",
    ),
    Mutation(
        name="parser: let an HTML comment leak its text into the passage index",
        module="parser.py",
        old='_HTML_TAG.sub(" ", _HTML_COMMENT.sub(" ", token.content))',
        new='_HTML_TAG.sub(" ", token.content)',
        tests="TestUnitsRound4",
    ),
)


def apply(mutation: Mutation, package: Path) -> bool:
    path = package / "markdown_memory" / mutation.module
    text = path.read_text(encoding="utf-8")
    if text.count(mutation.old) != 1:
        print(
            f"  SETUP ERROR: {mutation.module} has {text.count(mutation.old)} matches", flush=True
        )
        return False
    path.write_text(text.replace(mutation.old, mutation.new), encoding="utf-8")
    return True


def run_tests(package: Path, selector: str) -> bool:
    """True when the selected tests fail, which is what a mutation should cause."""
    environment = dict(os.environ, PYTHONPATH=str(package))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-m", "not embedding",
         "-k", selector, "--no-header", "-x"],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=1800,
    )  # fmt: skip
    if "no tests ran" in result.stdout:
        print(f"  SETUP ERROR: no test matches {selector!r}", flush=True)
        return False
    return result.returncode != 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--only", nargs="*", default=[], help="substrings of mutation names")
    arguments = parser.parse_args()
    chosen = [
        mutation
        for mutation in MUTATIONS
        if not arguments.only or any(word in mutation.name for word in arguments.only)
    ]
    if not chosen:
        print("no mutation matched")
        return 1

    missed: list[str] = []
    for mutation in chosen:
        with tempfile.TemporaryDirectory(prefix="mdmem-mutation-") as workspace:
            package = Path(workspace) / "src"
            shutil.copytree(ROOT / "src", package)
            if not apply(mutation, package):
                missed.append(mutation.name)
                continue
            caught = run_tests(package, mutation.tests)
        print(f"{'caught ' if caught else 'MISSED '} {mutation.name}", flush=True)
        if not caught:
            missed.append(mutation.name)

    print()
    if missed:
        print(f"{len(missed)} of {len(chosen)} mutation(s) survived - the suite has a hole:")
        for name in missed:
            print(f"  {name}")
        return 1
    print(f"all {len(chosen)} mutations were caught")
    return 0


if __name__ == "__main__":
    sys.exit(main())
