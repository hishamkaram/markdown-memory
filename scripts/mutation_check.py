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
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Outcome:
    """What running one mutation proved.

    ``problem`` is the harness failing, not the suite: an anchor that no longer matches,
    a selector naming no test, a mutant that crashed. Reported apart from a survivor,
    because a survivor means the tests are too weak while a problem means they never ran.
    """

    caught: bool = False
    problem: str = ""


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
    # How the test is expected to notice. The default is an assertion, because a mutation
    # that makes the code raise proves only that the code was reached: the assertions that
    # describe the behaviour never run, and the mutation is scored green for crashing.
    # A mutation whose honest outcome really is an exception names that exception here.
    fails_with: str = "assertion"


MUTATIONS = (
    Mutation(
        name="scope: resolve the configured docs root again on every scan",
        module="server.py",
        old=(
            "        if directory is None or not directory.strip():\n"
            "            return Path(self._root)"
        ),
        new=(
            "        if directory is None or not directory.strip():\n"
            "            return _absolute(self._config.docs_dir, IndexingError)"
        ),
        tests="test_a_scan_after_a_retarget_stays_with_the_root_it_serves",
    ),
    Mutation(
        name="config: keep the launcher's database when another project's root is named",
        module="server.py",
        old=(
            "            db.expanduser() if db else configured_db "
            "if configured_db else _project_database(root)"
        ),
        new=(
            "            db.expanduser() if db else configured_db "
            "if configured_db else base.db_path"
        ),
        tests="test_two_docs_dir_flags_do_not_share_the_launcher_s_database "
        "or test_naming_a_directory_rekeys_the_database "
        "or test_the_script_resolves_its_configuration_the_same_way",
    ),
    Mutation(
        name="config: put every project's index back in one shared database",
        module="server.py",
        old=(
            "            db_path=(db_path if db_path else "
            "_project_database(docs_dir if docs_dir else root)),"
        ),
        new=(
            "            db_path=(db_path if db_path else "
            '_xdg_dir("XDG_DATA_HOME", ".local/share") / "markdown-memory" / "index.db"),'
        ),
        tests="test_one_working_directory_two_projects_two_databases",
    ),
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
            '                    "ORDER BY f.rowid LIMIT ?"'
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
        name="diagram: print a token count the files stopped matching",
        module="make_diagram.py",
        area="scripts",
        old='    ("README.md", 5424),',
        new='    ("README.md", 5062),',
        tests="test_every_file_on_the_diagram_still_costs_what_it_says "
        "or test_the_totals_the_readme_prints_are_the_sum_of_those_files",
    ),
    Mutation(
        name="diagram: let a machine with no browser report a half-redrawn picture as done",
        module="make_diagram.py",
        area="scripts",
        old="    if missing:",
        new="    if False:",
        tests="test_a_half_redrawn_diagram_is_a_failure_and_not_a_warning",
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
        old="        if distance > VECTOR_PROBE_TOLERANCE or passage != probe.text:",
        new="        if False:",
        tests="test_an_index_whose_vectors_came_from_another_model_is_refused",
    ),
    Mutation(
        name="cache: accept any passage as the probe's nearest neighbour",
        module="eval_cache.py",
        area="scripts",
        old="        if distance > VECTOR_PROBE_TOLERANCE or passage != probe.text:",
        new="        if distance > VECTOR_PROBE_TOLERANCE:",
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
        fails_with="markdown_memory.exceptions.ConfigurationError",
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
            "        return self._db.list_documents(str(self._within_root(scope, IndexingError)))"
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
        old=(
            "            vectors.append("
            "SectionVectors(section=_section_vector(units), units=units))"
        ),
        new=(
            "            vectors.append("
            "SectionVectors(section=units[0] if units else None, units=units))"
        ),
        tests="test_the_section_vector_is_the_centroid_of_its_passages",
    ),
    Mutation(
        name="vectors: give a body-less section a vector pointing nowhere",
        module="indexer.py",
        old="    if not units:\n        return None",
        new="    if not units:\n        return [0.0]",
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
            "            _ = (\n"
            '                f"INCOMPLETE: {len(self.errors)} file(s) could not be indexed; "'
        ),
        tests="test_the_run_that_hits_it_calls_the_index_incomplete",
    ),
    Mutation(
        name="index: forget that a root was left incomplete",
        module="indexer.py",
        old="            self._db.record_failures(",
        new="            _ = lambda *a, **k: None; _(",
        tests="test_every_later_session_is_told_while_it_is_still_broken",
    ),
    Mutation(
        name="vectors: normalise rounding noise into a direction",
        module="indexer.py",
        old="    if norm < _MIN_POOLED_NORM:",
        new="    if norm == 0.0:",
        tests="test_passages_that_cancel_produce_no_vector_rather_than_noise",
    ),
    Mutation(
        name="index: let every tree answer with every other tree's failures",
        module="db.py",
        old=(
            '                    "SELECT file_path, message FROM index_failures "\n'
            '                    "WHERE file_path = ? OR substr(file_path, 1, length(?)) = ? "'
        ),
        new=(
            '                    "SELECT file_path, message FROM index_failures "\n'
            '                    "WHERE ? IS NOT NULL AND ? IS NOT NULL AND ? IS NOT NULL "'
        ),
        tests="test_one_root_never_wears_another_root_s_failure",
    ),
    Mutation(
        name="index: hide a failure recorded under a subdirectory",
        module="db.py",
        old=(
            '                    "SELECT file_path, message FROM index_failures "\n'
            '                    "WHERE file_path = ? OR substr(file_path, 1, length(?)) = ? "'
        ),
        new=(
            '                    "SELECT file_path, message FROM index_failures "\n'
            '                    "WHERE file_path = ? AND ? IS NOT NULL AND ? IS NOT NULL "'
        ),
        tests="test_a_failure_under_a_subdirectory_is_visible_from_the_root",
    ),
    Mutation(
        name="index: leave a failure standing for a file that is gone",
        module="db.py",
        old=(
            '                "DELETE FROM index_failures WHERE file_path = ?", '
            "[(path,) for path in clear]"
        ),
        new=('                "DELETE FROM index_failures WHERE file_path = ? AND ? IS NULL", []'),
        tests="test_a_file_that_failed_then_vanished_stops_being_reported",
    ),
    Mutation(
        name="index: let one root's clean run clear every root's failures",
        module="db.py",
        old=(
            "            conn.executemany(\n"
            '                "DELETE FROM index_failures WHERE file_path = ?"'
        ),
        new=(
            '            conn.execute("DELETE FROM index_failures")\n'
            "            conn.executemany(\n"
            '                "DELETE FROM index_failures WHERE file_path = ?"'
        ),
        tests="test_one_root_never_wears_another_root_s_failure",
    ),
    Mutation(
        name="coverage: vouch for a tree that a run left half-written",
        module="indexer.py",
        old="                    self._db.mark_scan_started(str(root))",
        new="                    pass",
        tests="test_a_run_that_dies_partway_leaves_the_tree_unvouched_for",
    ),
    Mutation(
        name="coverage: let a root nested inside the scan vouch for itself",
        module="db.py",
        old='                "OR substr(root, 1, length(?)) = ?",',
        new='                "OR ? IS NULL AND ? IS NULL",',
        tests="test_a_scan_retracts_the_roots_nested_inside_it_too",
    ),
    Mutation(
        name="coverage: catch a similarly named neighbour in the retraction",
        module="db.py",
        old=(
            "        prefix = _directory_prefix(root)\n"
            "        with self.transaction() as conn:\n"
            "            conn.execute(\n"
            '                "UPDATE index_coverage SET verified = 0 "'
        ),
        new=(
            "        prefix = root\n"
            "        with self.transaction() as conn:\n"
            "            conn.execute(\n"
            '                "UPDATE index_coverage SET verified = 0 "'
        ),
        tests="test_a_scan_leaves_a_similarly_named_neighbour_alone",
    ),
    Mutation(
        name="coverage: let a subdirectory scan leave the root vouching for itself",
        module="db.py",
        old='                "OR substr(?, 1, length(root) + 1) = root || ? "',
        new='                "OR ? IS NULL AND ? IS NULL "',
        tests="test_a_scan_of_a_subdirectory_retracts_the_root_it_sits_in",
    ),
    Mutation(
        name="coverage: let a run rewrite the tree without retracting anything",
        module="indexer.py",
        old="        about_to_write()\n        self._db.replace_document(",
        new="        self._db.replace_document(",
        tests="test_a_run_that_dies_partway_leaves_the_tree_unvouched_for",
    ),
    Mutation(
        name="coverage: retract before knowing whether anything will be written",
        module="indexer.py",
        old="        parsed = self._parser.parse(",
        new="        about_to_write()\n        parsed = self._parser.parse(",
        tests="test_a_run_that_committed_nothing_leaves_the_certificate_alone",
    ),
    Mutation(
        name="index: let two runs index one database at once",
        module="indexer.py",
        old="                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)",
        new="                fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)",
        tests="test_the_scan_lock_is_exclusive_not_merely_held",
    ),
    Mutation(
        name="index: make a second tool call wait out a 25-minute scan",
        module="indexer.py",
        old="        if not self._run_lock.acquire(blocking=False):",
        new="        if not self._run_lock.acquire(blocking=True):",
        tests="test_a_second_thread_is_refused_the_same_way",
    ),
    Mutation(
        name="vectors: let a cancelling section take its whole file out of the index",
        module="indexer.py",
        old="    return pooled if pooled is not None else list(units[0])",
        new="    return pooled",
        tests="test_cancelling_passages_do_not_cost_the_file_its_place",
    ),
    Mutation(
        name="index: treat one root's two spellings as two roots",
        module="indexer.py",
        old="            root = directory.expanduser().resolve(strict=True)",
        new="            root = directory.expanduser()",
        tests="test_the_same_root_spelled_two_ways_is_one_root",
    ),
    Mutation(
        name="scope: filter search by the docs root as spelled, not as resolved",
        module="server.py",
        old="        self._root = str(_absolute(self._config.docs_dir, SearchError))",
        new="        self._root = str(self._config.docs_dir)",
        tests="test_a_symlinked_docs_root_still_answers",
    ),
    Mutation(
        name="vectors: keep the old truncated section vectors across the format change",
        module="indexer.py",
        old="        if known is not None and known == (content_hash, VECTOR_FORMAT):",
        new="        if known is not None and known[0] == content_hash:",
        tests="test_a_v2_database_keeps_everything_and_rebuilds_its_vectors_in_place",
    ),
    Mutation(
        name="index: lock whichever spelling of the database reached us",
        module="indexer.py",
        old='            lock_path = str(Path(self._db.path).resolve()) + ".lock"',
        new='            lock_path = str(self._db.path) + ".lock"',
        tests="test_two_spellings_of_one_database_take_one_lock",
    ),
    Mutation(
        name="index: speak for a pruned tree this walk never entered",
        module="indexer.py",
        old=(
            "            reachable = self._reachable("
            "root, self._db.failure_paths(str(root)), unreadable)"
        ),
        new="            reachable = self._db.failure_paths(str(root))",
        tests="test_a_failure_inside_a_pruned_directory_outlives_a_parent_scan",
    ),
    Mutation(
        name="coverage: vouch for an index discarded while the scan walked",
        module="db.py",
        old="            if current != generation:",
        new="            if False:",
        tests="test_a_scan_does_not_certify_an_index_discarded_under_it",
    ),
    Mutation(
        name="answers: hide vectors left behind by an older format",
        module="db.py",
        old="                whole = rows == [] and stale_vectors == 0",
        new="                whole = rows == []",
        tests="test_a_document_left_on_an_older_vector_format_is_not_called_whole",
    ),
    Mutation(
        name="coverage: count a run's own model-change wipe against it",
        module="indexer.py",
        old="            generation = self._db.generation()\n            known_hashes =",
        new="            generation = 0\n            known_hashes =",
        tests="test_a_clean_run_after_a_model_change_vouches_for_the_tree",
    ),
    Mutation(
        name="index: ask whether a pruned directory's parent was walkable",
        module="indexer.py",
        old="        if not _is_walkable(os.path.relpath(path, root).split(os.sep)):",
        new="        if not _is_walkable(os.path.relpath(path, root).split(os.sep)[:-1]):",
        tests="test_a_pruned_directory_that_could_not_be_listed_keeps_its_own_failure",
    ),
    Mutation(
        name="index: speak for an excluded tree this walk never entered",
        module="indexer.py",
        old="        return not (self._exclude and _is_excluded(Path(path), root, self._exclude))",
        new="        return True",
        tests="test_a_failure_inside_an_excluded_directory_outlives_a_parent_scan",
    ),
    Mutation(
        name="answers: report the root's stale vectors as a subdirectory's own",
        module="db.py",
        old="                        (named, prefix, prefix, VECTOR_FORMAT),",
        new=(
            "                        (root, _directory_prefix(root), "
            "_directory_prefix(root), VECTOR_FORMAT),"
        ),
        tests="test_a_subdirectory_is_not_blamed_for_the_root_s_stale_vectors",
    ),
    Mutation(
        name="scope: resolve the docs root again on every question",
        module="server.py",
        old="            return self._db.index_status(self._root)",
        new=(
            "            return self._db.index_status("
            "str(_absolute(self._config.docs_dir, SearchError)))"
        ),
        tests="test_status_and_search_always_describe_the_same_tree",
    ),
    Mutation(
        name="index: purge what a walk could not see behind a symlink",
        module="indexer.py",
        old="            if _behind_symlink(root, file_path):\n                continue",
        new="            if False:\n                continue",
        tests="test_a_directory_that_became_a_symlink_costs_nothing",
    ),
    Mutation(
        name="index: clear failures behind a symlink this walk never followed",
        module="indexer.py",
        old="        if _behind_symlink(root, path) or _is_shadowing_symlink(path):",
        new="        if _is_shadowing_symlink(path):",
        tests="test_a_failure_behind_a_symlinked_directory_outlives_a_parent_scan",
    ),
    Mutation(
        name="answers: call a tree whole from one read while naming faults from another",
        module="db.py",
        old="        verified = certificate is not None and bool(certificate[0]) and whole",
        new="        verified = certificate is not None and bool(certificate[0])",
        tests="test_a_narrowed_status_is_never_internally_contradictory",
    ),
    Mutation(
        name="answers: never mention that the index is missing files",
        module="server.py",
        old="            return self._db.index_status(self._root)",
        new="            return IndexStatus(verified=True)",
        tests="test_a_search_says_the_index_is_missing_files",
    ),
    Mutation(
        name="index: keep a failure standing for a file the walk could not see and that is gone",
        module="indexer.py",
        old=(
            "            if self._walk_would_visit(root, path, blocked) "
            "or _certainly_gone(root, path):"
        ),
        new="            if self._walk_would_visit(root, path, blocked):",
        tests="test_a_failure_out_of_the_walk_s_reach_goes_when_the_file_does",
    ),
    Mutation(
        name="index: keep answering from a deleted file the walk could not see",
        module="indexer.py",
        old="            if _certainly_gone(root, file_path):",
        new="            if False:",
        tests="test_a_deleted_document_inside_a_pruned_tree_stops_answering",
    ),
    Mutation(
        name="index: read a directory it may not list as an empty one",
        module="indexer.py",
        old=(
            "    except OSError:  # no permission, symlink loop, "
            "unreachable mount: no evidence either way\n        return False"
        ),
        new=(
            "    except OSError:  # no permission, symlink loop, "
            "unreachable mount: no evidence either way\n        return True"
        ),
        tests="test_a_file_under_an_unreadable_directory_is_not_taken_for_deleted",
    ),
    Mutation(
        name="index: read a broken symlink's target as a deletion",
        module="indexer.py",
        old="    if _behind_symlink(root, path) or os.path.islink(path):\n        return False",
        new="    if False:\n        return False",
        tests="test_a_directory_replaced_by_a_broken_symlink_keeps_its_documents",
    ),
    Mutation(
        name="index: clear a failure on a directory the walk only saw the name of",
        module="indexer.py",
        old="        if _behind_symlink(root, path) or _is_shadowing_symlink(path):",
        new="        if _behind_symlink(root, path):",
        tests="test_a_failure_on_a_directory_that_became_a_symlink_outlives_the_swap",
    ),
    Mutation(
        name="index: take a name it could not decode for a file that is gone",
        module="indexer.py",
        old="    if _UNDECODABLE in path:\n        return False",
        new="    if False:\n        return False",
        tests="test_a_failure_whose_name_could_not_be_decoded_is_not_taken_for_deleted",
    ),
    Mutation(
        name="scope: answer about a directory outside the root this server serves",
        module="server.py",
        old="        if resolved != root and root not in resolved.parents:",
        new="        if False:",
        tests="test_every_spelling_of_outside_is_refused",
    ),
    Mutation(
        name="answers: call a root nobody indexed complete",
        module="db.py",
        old="        verified = certificate is not None and bool(certificate[0]) and whole",
        new="        verified = (certificate is None or bool(certificate[0])) and whole",
        tests="test_a_root_nobody_indexed_does_not_claim_to_be_whole",
    ),
    Mutation(
        name="config: overrule a database that was configured on purpose",
        module="server.py",
        old=(
            "            db.expanduser() if db else configured_db "
            "if configured_db else _project_database(root)"
        ),
        new=("            configured_db if configured_db else _project_database(root)"),
        tests="test_an_explicitly_configured_database_still_wins",
    ),
    Mutation(
        name="config: index the trees the operator excluded",
        module="server.py",
        old="        exclude=tuple(exclude) or base.exclude,",
        new="        exclude=tuple(exclude),",
        tests="test_exclusions_are_inherited_rather_than_dropped "
        "or test_the_script_resolves_its_configuration_the_same_way",
    ),
)


def _ignore_caches(directory: str, names: list[str]) -> set[str]:
    # eval_data holds the frozen corpus; copying it for a mutation run is pure cost.
    return {name for name in names if name in {"__pycache__", "eval_data"}}


def _occurrences(text: str, needle: str) -> list[int]:
    found, index = [], text.find(needle)
    while index != -1:
        found.append(index)
        index = text.find(needle, index + 1)
    return found


def apply(mutation: Mutation, workspace: Path) -> str:
    """Mutate the copy. Returns the setup problem that stopped it, or "" on success."""
    path = (
        workspace / "src" / "markdown_memory" / mutation.module
        if mutation.area == "src"
        else workspace / "scripts" / mutation.module
    )
    text = path.read_text(encoding="utf-8")
    # An anchor that starts mid-line matches a more deeply indented copy of itself, and
    # replacing it leaves the leftover indentation glued to the next line - a syntax error
    # that reads exactly like a caught mutation. Anchors address whole lines.
    if mutation.old.startswith(" ") and not all(
        index == 0 or text[index - 1] == "\n" for index in _occurrences(text, mutation.old)
    ):
        return f"{mutation.module} matches {mutation.old[:40]!r} mid-line"
    if text.count(mutation.old) != 1:
        return f"{mutation.module} has {text.count(mutation.old)} matches for the anchor"
    path.write_text(text.replace(mutation.old, mutation.new), encoding="utf-8")
    return ""


def run_tests(workspace: Path, mutation: Mutation) -> Outcome:
    """Caught when the selected tests fail, which is what a mutation should cause."""
    environment = dict(os.environ, PYTHONPATH=str(workspace / "src"))
    # The scripts are imported by name, and pytest's own ``pythonpath`` setting wins over
    # PYTHONPATH - so point that setting at the mutated copy instead.
    override = (
        ["-o", f"pythonpath=tests {workspace / 'scripts'}"] if mutation.area == "scripts" else []
    )
    # A selector naming a test that no longer exists makes pytest exit 5, which reads as
    # "the mutation was caught" - so every mutation whose test was renamed would pass
    # while proving nothing. Count the matches first.
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-m", "not embedding",
         "-k", mutation.tests, "--collect-only", "--no-header", *override],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=600,
    )  # fmt: skip
    if collected.returncode != 0 or " test" not in collected.stdout.rsplit("\n", 3)[-2]:
        return Outcome(problem=f"no test matches {mutation.tests!r}")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-m", "not embedding",
         "-k", mutation.tests, "--no-header", "-x", "--tb=line", *override],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=1800,
    )  # fmt: skip
    if "no tests ran" in result.stdout:
        return Outcome(problem=f"no test matches {mutation.tests!r}")
    if result.returncode == 0:
        return Outcome(caught=False)
    return _failed_for_the_right_reason(result.stdout, mutation)


# ``--tb=line`` reports each failure as "<file>:<line>: <detail>", where the detail is the
# exception and its message - except for a bare ``assert x in y``, whose detail is the
# rewritten assertion itself, with no exception name anywhere in it.
_RAISED = re.compile(r"^.+:\d+: (\S.*)$")


def _failed_for_the_right_reason(output: str, mutation: Mutation) -> Outcome:
    """Did the test fail on its assertions, rather than crash on the way to them?

    A mutation that raises makes the suite red while proving nothing: the code blew up
    before the assertions describing the behaviour could run, so the test would look just
    as red if it asserted nothing at all. One mutation here used to replace a hash call
    with ``len(...)``, and the ``AttributeError`` that followed was scored as a catch.
    """
    raised = [
        _what_went_wrong(match.group(1))
        for line in output.splitlines()
        if (match := _RAISED.match(line))
    ]
    if not raised:
        return Outcome(problem="the test failed without naming what went wrong")
    wanted = "assertion" if mutation.fails_with == "assertion" else mutation.fails_with
    wrong = [name for name in raised if name != wanted]
    if wrong:
        return Outcome(problem=f"expected {wanted}, the test hit {wrong[0]}")
    return Outcome(caught=True)


def _what_went_wrong(detail: str) -> str:
    """Name what the test hit: a failed assertion, or the exception raised instead.

    ``Failed`` is pytest's own: a ``pytest.raises`` block that did not raise, or a
    ``pytest.fail`` call. Both are the test deciding it failed, which is exactly what a
    caught mutation looks like - the assertion just happens to be spelled as a context
    manager rather than an ``assert``.
    """
    if detail.startswith(("assert", "AssertionError", "Failed")):
        return "assertion"
    return detail.split(":", 1)[0].strip()


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
    broken: list[str] = []
    for mutation in chosen:
        with tempfile.TemporaryDirectory(prefix="mdmem-mutation-") as directory:
            workspace = Path(directory)
            shutil.copytree(ROOT / "src", workspace / "src")
            if mutation.area == "scripts":
                shutil.copytree(ROOT / "scripts", workspace / "scripts", ignore=_ignore_caches)
            outcome = Outcome(problem=apply(mutation, workspace))
            if not outcome.problem:
                outcome = run_tests(workspace, mutation)
        if outcome.problem:
            print(f"SETUP   {mutation.name}\n        {outcome.problem}", flush=True)
            broken.append(mutation.name)
            continue
        print(f"{'caught ' if outcome.caught else 'MISSED '} {mutation.name}", flush=True)
        if not outcome.caught:
            missed.append(mutation.name)

    print()
    if broken:
        print(f"{len(broken)} mutation(s) never ran - the harness is stale, not the suite:")
        for name in broken:
            print(f"  {name}")
    if missed:
        print(f"{len(missed)} of {len(chosen)} mutation(s) survived - the suite has a hole:")
        for name in missed:
            print(f"  {name}")
    if broken or missed:
        return 1
    print(f"all {len(chosen)} mutations were caught")
    return 0


if __name__ == "__main__":
    sys.exit(main())
