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
    # ``src`` mutates the package; ``scripts`` mutates the developer scripts, which the
    # tests import by name through pytest's ``pythonpath``.
    area: str = "src"


MUTATIONS = (
    Mutation(
        name="max-sim: rank a section by its own vector, never by its best passage",
        module="search.py",
        old=(
            "                if distance < best.get(section_id, math.inf):\n"
            "                    best[section_id] = distance"
        ),
        new="                pass",
        tests="TestRankingIsActuallyTested",
    ),
    Mutation(
        name="BM25: return keyword hits in row order instead of by relevance",
        module="db.py",
        old=(
            '"WHERE sections_fts MATCH ? AND substr(d.file_path, 1, length(?)) = ? "\n'
            '                    "ORDER BY rank LIMIT ?"'
        ),
        new=(
            '"WHERE sections_fts MATCH ? AND substr(d.file_path, 1, length(?)) = ? "\n'
            '                    "ORDER BY rowid LIMIT ?"'
        ),
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
    Mutation(
        name="cache: score a cached index built from a different corpus",
        module="eval_cache.py",
        area="scripts",
        old="    if stored != expected_fingerprint:\n        raise StaleCacheError(",
        new="    if False:\n        raise StaleCacheError(",
        tests="test_a_cache_built_from_another_corpus_is_refused",
    ),
    Mutation(
        name="cache: fingerprint the passages but not the section around them",
        module="eval_cache.py",
        area="scripts",
        old='            digest.update(hashlib.sha256(section.content.encode("utf-8")).digest())\n',
        new="",
        tests="test_text_no_passage_carries_still_changes_the_fingerprint",
    ),
    Mutation(
        name="cache: fingerprint the section but not how it was cut into passages",
        module="eval_cache.py",
        area="scripts",
        old=(
            "            for ordinal, unit in enumerate(section.unit_texts):\n"
            '                digest.update(b"\\x1f")\n'
            '                digest.update(str(ordinal).encode("ascii"))\n'
            '                digest.update(hashlib.sha256(unit.encode("utf-8")).digest())'
        ),
        new="            pass",
        tests="test_a_change_in_how_passages_are_cut_changes_the_fingerprint",
    ),
    Mutation(
        name="cache: key on the requested model revision, not the files on disk",
        module="eval_cache.py",
        area="scripts",
        old='        "artifacts": _model_identity(model_cache_dir, embedder),',
        new='        "artifacts": "",',
        tests="test_a_reinstalled_model_invalidates_the_cache",
    ),
    Mutation(
        name="cache: trust the directory name instead of the recorded key",
        module="eval_cache.py",
        area="scripts",
        old=(
            "        if meta.get(field) != getattr(key, field):\n"
            "            raise StaleCacheError("
            'f"cached index was built with a different {field}")'
        ),
        new="        pass",
        tests="test_a_cache_directory_copied_from_another_key_is_refused",
    ),
    Mutation(
        name="exclusions: anchor every pattern at the docs root",
        module="indexer.py",
        old=(
            "        elif any(fnmatchcase(part, pattern) for part in parts):\n"
            "            return True"
        ),
        new="        elif False:\n            return True",
        tests="test_a_bare_name_excludes_that_directory_at_any_depth",
    ),
    Mutation(
        name="exclusions: split a configured pattern on colons as well as commas",
        module="indexer.py",
        old='    for part in value.split(","):',
        new='    for part in __import__("re").split(r"[,:]", value):',
        tests="test_a_pattern_containing_a_colon_is_one_pattern",
    ),
    Mutation(
        name="cache: let two evaluations measure latency on the same CPU",
        module="eval_cache.py",
        area="scripts",
        old="fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)",
        new="fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)",
        tests="test_a_second_evaluation_refuses_to_start",
    ),
    Mutation(
        name="cache: leave the write-ahead log behind when discarding an index",
        module="eval_cache.py",
        area="scripts",
        old='    for suffix in ("", "-wal", "-shm"):',
        new='    for suffix in ("",):',
        tests="test_discard_removes_the_write_ahead_log_too",
    ),
    Mutation(
        name="cache: trust the cache key instead of reading the stored vectors",
        module="eval_cache.py",
        area="scripts",
        old="    if distance > VECTOR_PROBE_TOLERANCE or passage != probe.text:",
        new="    if False:",
        tests="test_an_index_whose_vectors_came_from_another_model_is_refused",
    ),
    Mutation(
        name="cache: accept any passage as the probe's nearest neighbour",
        module="eval_cache.py",
        area="scripts",
        old="    if distance > VECTOR_PROBE_TOLERANCE or passage != probe.text:",
        new="    if distance > VECTOR_PROBE_TOLERANCE:",
        tests="test_a_probe_matched_to_the_wrong_passage_is_refused",
    ),
    Mutation(
        name="cache: cache a build over a corpus that moved while it was indexed",
        module="eval_cache.py",
        area="scripts",
        old="    if parse_fingerprint(corpus, exclude) != fingerprint:",
        new="    if False:",
        tests="test_a_corpus_that_moved_during_indexing_is_not_cached",
    ),
    Mutation(
        name="cache: probe one passage and call the whole index verified",
        module="eval_cache.py",
        area="scripts",
        old="    spread = found[::step][:count]",
        new="    spread = found[:1]",
        tests="test_the_probes_are_spread_through_the_corpus",
    ),
    Mutation(
        name="cache: read a named pipe in the corpus like any other file",
        module="eval_cache.py",
        area="scripts",
        old=(
            "    if not stat.S_ISREG(info.st_mode):\n"
            '        return f"not-a-regular-file:{stat.S_IFMT(info.st_mode)}"'
        ),
        new="    pass",
        tests="test_a_named_pipe_in_the_corpus_does_not_hang_the_cache",
    ),
    Mutation(
        name="eval: score and cache an index that failed on some files",
        module="eval_retrieval.py",
        area="scripts",
        old="    if report.errors:",
        new="    if False:",
        tests="test_a_partial_build_is_neither_scored_nor_cached",
    ),
    Mutation(
        name="exclusions: strip every leading dot and slash from a pattern",
        module="indexer.py",
        old='        cleaned = part.strip().removeprefix("./").rstrip("/")',
        new='        cleaned = part.strip().lstrip("./").rstrip("/")',
        tests="test_a_dot_prefixed_name_is_not_mistaken_for_a_relative_path",
    ),
    Mutation(
        name="exclusions: fold case when matching a pattern",
        module="indexer.py",
        old="        elif any(fnmatchcase(part, pattern) for part in parts):",
        new="        elif any(fnmatchcase(part.lower(), pattern.lower()) for part in parts):",
        tests="test_matching_does_not_depend_on_the_platform_case_rules",
    ),
    Mutation(
        name="config: refuse a directory whose real name contains a dollar-brace",
        module="server.py",
        old='    if "${" in value and not Path(value).expanduser().exists():',
        new='    if "${" in value:',
        tests="test_a_directory_really_named_like_a_variable_is_allowed",
    ),
    Mutation(
        name="scope: search every documentation root the database happens to hold",
        module="search.py",
        old="                allowed = self._db.sections_under(list(best), self._scope)",
        new="                allowed = set(best)",
        tests="test_one_project_never_answers_with_another_project_s_documentation",
    ),
    Mutation(
        name="scope: stop widening the vector search when the page is not full",
        module="search.py",
        old="            if self._scope is None or len(best) >= limit or fetch >= ceiling:",
        new="            if True:",
        tests="test_the_vector_side_keeps_looking_past_a_crowded_neighbour",
    ),
    Mutation(
        name="scope: list every project's documents, not this project's",
        module="server.py",
        old=(
            "        return self._db.list_documents("
            "str(self._resolve_directory(directory or None)))"
        ),
        new="        return self._db.list_documents()",
        tests="test_listing_documents_shows_this_project_only",
    ),
    Mutation(
        name="usage: miss the calls a subagent made",
        module="usage_from_transcripts.py",
        area="scripts",
        old='        yield from sorted(directory.rglob("*.jsonl"))',
        new='        yield from sorted(directory.glob("*.jsonl"))',
        tests="test_a_subagents_calls_count_too",
    ),
    Mutation(
        name="usage: count another MCP server's tools as ours",
        module="usage_from_transcripts.py",
        area="scripts",
        old='SERVER_TOOL = re.compile(r"^mcp__[^_]*markdown[^_]*__(?P<tool>\\w+)$", re.IGNORECASE)',
        new='SERVER_TOOL = re.compile(r"^mcp__.*__(?P<tool>\\w+)$", re.IGNORECASE)',
        tests="test_another_servers_tools_are_not_counted_as_ours",
    ),
    Mutation(
        name="usage: stop noticing that the agent gave up and read the file",
        module="usage_from_transcripts.py",
        area="scripts",
        old=(
            "            if any(_touches_markdown(later) for later in window):\n"
            "                report.searches_followed_by_file_access += 1"
        ),
        new=("            if False:\n                report.searches_followed_by_file_access += 1"),
        tests="test_a_search_abandoned_for_the_file_system or test_a_search_abandoned_for_grep",
    ),
    Mutation(
        name="usage: blame a search for a file read that came much later",
        module="usage_from_transcripts.py",
        area="scripts",
        old="            window = calls[position + 1 : position + 1 + FALLBACK_WINDOW]",
        new="            window = calls[position + 1 :]",
        tests="test_a_file_read_long_after_a_search_is_not_attributed_to_it",
    ),
    Mutation(
        name="usage: lose a whole session to one half-written line",
        module="usage_from_transcripts.py",
        area="scripts",
        old=(
            "        except ValueError:\n"
            "            continue  # a transcript being written to can end mid-line"
        ),
        new="        except ValueError:\n            return []",
        tests="test_a_half_written_line_does_not_lose_the_session",
    ),
    Mutation(
        name="units: throw away everything past the character limit again",
        module="parser.py",
        old="            for window in _windows(unit):",
        new="            for window in [unit[:MAX_UNIT_CHARS]]:",
        tests="test_no_line_of_a_long_block_is_dropped or test_the_tail_reaches_the_embedder",
    ),
    Mutation(
        name="units: let a window run past the limit instead of splitting",
        module="parser.py",
        old="    if len(text) <= MAX_UNIT_CHARS:\n        return [text]",
        new="    if True:\n        return [text]",
        tests="test_every_window_still_respects_the_limit",
    ),
    Mutation(
        name="units: drop the remainder after the last full window",
        module="parser.py",
        old="    if remaining:\n        pieces.append(remaining)",
        new="    if False:\n        pieces.append(remaining)",
        tests="test_the_windows_reassemble_into_exactly_what_arrived",
    ),
    Mutation(
        name="units: cut a window mid-word instead of at a boundary",
        module="parser.py",
        old='        cut = max(head.rfind("\\n"), head.rfind(". "), _last_space(head))',
        new="        cut = MAX_UNIT_CHARS",
        tests="test_a_window_does_not_end_mid_word",
    ),
    Mutation(
        name="preflight: count requests only in assistant turns, shifting every task",
        module="preflight.py",
        area="scripts",
        old="        if not isinstance(content, list):\n            continue",
        new=(
            '        if not isinstance(content, list) or message.get("role") != "assistant":\n'
            "            continue"
        ),
        tests="test_a_tool_call_outside_an_assistant_turn_does_not_shift_attribution",
    ),
    Mutation(
        name="units: let a long final block overrun the passage cap",
        module="parser.py",
        old=(
            "                if len(passages) >= MAX_UNITS_PER_SECTION:\n"
            "                    return tuple(passages)"
        ),
        new="                if False:\n                    return tuple(passages)",
        tests="test_windows_never_push_a_section_past_the_passage_cap",
    ),
    Mutation(
        name="units: break only on a literal space, never on a tab",
        module="parser.py",
        old="        if text[index].isspace():",
        new='        if text[index] == " ":',
        tests="test_a_run_broken_only_by_tabs_breaks_there",
    ),
    Mutation(
        name="vectors: embed the whole section again, truncation and all",
        module="indexer.py",
        old="            vectors.append(SectionVectors(section=_mean_vector(units), units=units))",
        new=(
            "            vectors.append("
            "SectionVectors(section=units[0] if units else None, units=units))"
        ),
        tests="test_the_section_vector_sits_among_its_passages",
    ),
    Mutation(
        name="vectors: give a body-less section a vector pointing nowhere",
        module="indexer.py",
        old="    if not vectors:\n        return None",
        new="    if not vectors:\n        return [0.0]",
        tests="test_a_section_with_no_body_still_has_no_vector",
    ),
    Mutation(
        name="index: let a partial run read as a clean one",
        module="models.py",
        old=(
            "            lines.append(\n"
            '                f"INCOMPLETE: {len(self.errors)} file(s) could not be indexed; "'
        ),
        new=(
            "        if False:\n            lines.append(\n"
            '                f"INCOMPLETE: {len(self.errors)} file(s) could not be indexed; "'
        ),
        tests="test_the_run_that_hits_it_calls_the_index_incomplete",
    ),
    Mutation(
        name="index: forget by the next run that the tree was incomplete",
        module="indexer.py",
        old="            self._db.record_notice(",
        new="            _ = lambda *a: None; _(",
        tests="test_the_next_run_is_told_too",
    ),
)


def _ignore_caches(directory: str, names: list[str]) -> set[str]:
    # eval_data holds the frozen corpus; copying it for a mutation run is pure cost.
    return {name for name in names if name in {"__pycache__", "eval_data"}}


def apply(mutation: Mutation, workspace: Path) -> bool:
    path = (
        workspace / "src" / "markdown_memory" / mutation.module
        if mutation.area == "src"
        else workspace / "scripts" / mutation.module
    )
    text = path.read_text(encoding="utf-8")
    if text.count(mutation.old) != 1:
        print(
            f"  SETUP ERROR: {mutation.module} has {text.count(mutation.old)} matches", flush=True
        )
        return False
    path.write_text(text.replace(mutation.old, mutation.new), encoding="utf-8")
    return True


def run_tests(workspace: Path, mutation: Mutation) -> bool:
    """True when the selected tests fail, which is what a mutation should cause."""
    environment = dict(os.environ, PYTHONPATH=str(workspace / "src"))
    # The scripts are imported by name, and pytest's own ``pythonpath`` setting wins over
    # PYTHONPATH - so point that setting at the mutated copy instead.
    override = (
        ["-o", f"pythonpath=tests {workspace / 'scripts'}"] if mutation.area == "scripts" else []
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-m", "not embedding",
         "-k", mutation.tests, "--no-header", "-x", *override],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=1800,
    )  # fmt: skip
    if "no tests ran" in result.stdout:
        print(f"  SETUP ERROR: no test matches {mutation.tests!r}", flush=True)
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
        with tempfile.TemporaryDirectory(prefix="mdmem-mutation-") as directory:
            workspace = Path(directory)
            shutil.copytree(ROOT / "src", workspace / "src")
            if mutation.area == "scripts":
                shutil.copytree(ROOT / "scripts", workspace / "scripts", ignore=_ignore_caches)
            if not apply(mutation, workspace):
                missed.append(mutation.name)
                continue
            caught = run_tests(workspace, mutation)
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
