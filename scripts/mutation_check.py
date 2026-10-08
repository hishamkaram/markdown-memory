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
        name="preset: rebuild an index of another vector size on open again",
        module="db.py",
        old="        if version >= 1:\n            self._refuse_another_dimension(conn)",
        new="        if False:\n            self._refuse_another_dimension(conn)",
        tests="test_a_database_of_another_dimension_is_refused_and_left_unchanged "
        "or test_another_dimension_is_refused_and_every_root_keeps_its_index "
        "or test_a_mismatched_database_stops_the_server_with_the_reason",
    ),
    Mutation(
        name="preset: trust a file another preset built while this one waited",
        module="db.py",
        old=(
            "                if current >= 1:\n"
            "                    # And it may have been another preset"
        ),
        new="                if False:\n                    # And it may have been another preset",
        tests="test_another_preset_that_built_the_file_first_is_still_refused",
    ),
    Mutation(
        name="preset: leave out which model built the refused index",
        module="db.py",
        old=(
            '        built_by = f"built by {model[0]}" if model else "built by an unrecorded model"'
        ),
        new='        built_by = "built by an unrecorded model"',
        tests="test_the_refusal_names_the_path_both_sizes_and_the_model",
    ),
    Mutation(
        name="preset: hold on to the file a refused open refused",
        module="db.py",
        old="            self.close()  # a refused open must not hold the file it refused",
        new="            pass",
        tests="test_a_refused_open_lets_go_of_the_file",
    ),
    Mutation(
        name="preset: one default database for every preset again",
        module="config.py",
        old=(
            '        / ("index.db" if embedder == DEFAULT_EMBEDDER '
            'else f"index-{_safe(embedder)}.db")'
        ),
        new='        / "index.db"',
        tests="test_each_preset_has_its_own_default_database "
        "or test_two_presets_on_one_root_keep_both_indexes",
    ),
    Mutation(
        name="preset: name a preset's database with whatever the environment says",
        module="config.py",
        old=(
            '        / ("index.db" if embedder == DEFAULT_EMBEDDER '
            'else f"index-{_safe(embedder)}.db")'
        ),
        new='        / ("index.db" if embedder == DEFAULT_EMBEDDER else f"index-{embedder}.db")',
        tests="test_each_preset_has_its_own_default_database",
    ),
    Mutation(
        name="preset: derive the environment's database before its preset is known",
        module="config.py",
        old=(
            "                db_path if db_path else "
            "_project_database(docs_dir if docs_dir else root, embedder)"
        ),
        new=(
            "                db_path if db_path else "
            "_project_database(docs_dir if docs_dir else root, DEFAULT_EMBEDDER)"
        ),
        tests="test_each_preset_has_its_own_default_database",
    ),
    Mutation(
        name="preset: key the command line's database on the environment's preset",
        module="config.py",
        old="            else _project_database(root, preset)",
        new="            else _project_database(root, base.embedder)",
        tests="test_each_preset_has_its_own_default_database",
    ),
    Mutation(
        name="preset: put every preset's work-tree index in one file",
        module="config.py",
        old="    default = _project_database(docs_dir, config.embedder)",
        new="    default = _project_database(docs_dir, DEFAULT_EMBEDDER)",
        tests="test_a_trees_database_is_beside_a_chosen_one_and_derived_otherwise",
    ),
    Mutation(
        name="scope: resolve the configured docs root again on every scan",
        module="server.py",
        old=(
            "        if directory is None or not directory.strip():\n"
            "            return Path(self._root)"
        ),
        new=(
            "        if directory is None or not directory.strip():\n"
            "            return headings._absolute(self._config.docs_dir, IndexingError)"
        ),
        tests="test_a_scan_after_a_retarget_stays_with_the_root_it_serves",
    ),
    Mutation(
        name="config: keep the launcher's database when another project's root is named",
        module="config.py",
        old="            else _project_database(root, preset)",
        new="            else base.db_path",
        tests="test_two_docs_dir_flags_do_not_share_the_launcher_s_database "
        "or test_naming_a_directory_rekeys_the_database "
        "or test_the_script_resolves_its_configuration_the_same_way",
    ),
    Mutation(
        name="config: put every project's index back in one shared database",
        module="config.py",
        old=(
            "                db_path if db_path else "
            "_project_database(docs_dir if docs_dir else root, embedder)"
        ),
        new=(
            "                db_path if db_path else "
            '_xdg_dir("XDG_DATA_HOME", ".local/share") / "markdown-memory" / "index.db"'
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
            page, stale = self._search_once(query, limit)
            if not stale:
                return page
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
        name="cache: trust size, mtime and inode, and drop ctime from the stamp",
        module="model_cache.py",
        old='        "ctime_ns": info.st_ctime_ns,\n',
        new="",
        tests="test_a_file_rewritten_with_its_old_mtime_restored_is_still_caught",
    ),
    Mutation(
        name="cache: trust a model file that is a symlink out of the cache",
        module="model_cache.py",
        old="""    if not stat.S_ISREG(info.st_mode):
        return None  # a symlink into a blob store is not a file this cache vouches for
""",
        new="",
        tests="test_a_model_file_replaced_by_a_symlink_is_not_trusted",
    ),
    Mutation(
        name="cache: take what was downloaded on trust",
        module="embedders.py",
        old="            still_wrong = model_cache._unverified(self._model_dir)\n",
        new="            still_wrong: list[str] = []\n",
        tests="test_a_download_that_does_not_match_the_manifest_fails_without_downloading_again",
    ),
    Mutation(
        name="cache: treat any load failure as corruption and re-download",
        module="embedders.py",
        old="                        self._session, self._tokenizer = self._open()\n",
        new="""                        try:
                            self._session, self._tokenizer = self._open()
                        except ModelLoadError:
                            self._fetch()
                            raise
""",
        tests="test_a_load_failure_on_verified_files_is_not_treated_as_corruption",
    ),
    Mutation(
        name="cache: repair under the shared lock, beside whoever is reading",
        module="embedders.py",
        old="                    with model_cache._model_cache_lock("
        "self._cache_dir, exclusive=True):\n",
        new="                    with model_cache._model_cache_lock("
        "self._cache_dir, exclusive=False):\n",
        tests="test_two_processes_starting_at_once_download_once_between_them",
    ),
    Mutation(
        name="cache: leave a directory sitting where the stamp goes",
        module="model_cache.py",
        old="    if stamped.is_dir() and not stamped.is_symlink():\n",
        new="    if False:  # a directory there is somebody else's problem\n",
        tests="test_a_directory_where_the_stamp_belongs_is_repaired",
        fails_with="markdown_memory.exceptions.ModelLoadError",
    ),
    Mutation(
        name="cache: try to unlink a directory where a model file belongs",
        module="embedders.py",
        old="            model_cache._remove(path)  # only what is proven wrong\n",
        new="            path.unlink(missing_ok=True)\n",
        tests="test_a_directory_where_a_model_file_belongs_is_repaired",
        # The honest outcome of the bug is the exception the guard exists to prevent,
        # which the repair wrapper now reports as the domain error carrying its text.
        fails_with="markdown_memory.exceptions.ModelLoadError",
    ),
    Mutation(
        name="cache: record the weights over an index this run did not build",
        module="indexer.py",
        old=(
            "        self._db.forget_weights_revision()\n"
            "        self._db.record_weights_mismatch(None)\n"
        ),
        new="        self._db.record_weights_mismatch(None)\n",
        tests="test_the_weights_are_recorded_only_for_vectors_every_one_of_which_they_embedded",
    ),
    Mutation(
        name="storage: keep the weights revision after discarding every document",
        module="db.py",
        old='    if conn.execute("SELECT 1 FROM units_vec LIMIT 1").fetchone() is not None:\n',
        new="    if True:  # the revision outlives the vectors it described\n",
        tests="test_discarding_every_document_discards_the_revision_that_described_them",
    ),
    Mutation(
        name="cache: follow a symlinked directory out of the model cache",
        module="model_cache.py",
        old="    current = model_dir\n",
        new="    return model_dir / name\n    current = model_dir\n",
        tests="test_a_symlinked_directory_on_the_way_is_never_written_through",
    ),
    Mutation(
        name="cache: assume a removal that did nothing worked",
        module="model_cache.py",
        old="    if path.exists() or path.is_symlink():\n",
        new="    if False:\n",
        tests="test_a_file_that_cannot_be_cleared_says_so_where_the_path_is_known",
    ),
    Mutation(
        name="weights: vouch for the index while another root holds other weights' vectors",
        module="db.py",
        old="            if not stale:\n",
        new="            if True:\n",
        tests="test_vectors_from_other_weights_left_in_another_root_keep_the_index_revoked",
    ),
    Mutation(
        name="weights: treat a model that cannot say which weights it is as the right one",
        module="indexer.py",
        old="            if recorded in {None, WEIGHTS_REVOKED}:\n",
        new="            if True:\n",
        tests="test_weights_that_cannot_be_identified_are_not_assumed_to_be_the_right_ones",
    ),
    Mutation(
        name="cache: let a repair failure escape as whatever the Hub raised",
        module="embedders.py",
        old="                        except Exception as exc:\n",
        new="                        except MarkdownMemoryError as exc:\n",
        tests="test_a_cache_that_cannot_be_repaired_fails_as_a_domain_error",
        fails_with="OSError",
    ),
    Mutation(
        name="cache: follow a symlink planted at a temporary path",
        module="model_cache.py",
        old=(
            "    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT "
            "| os.O_EXCL | os.O_NOFOLLOW, 0o600)\n"
        ),
        new="    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)\n",
        tests="test_the_helper_that_writes_those_files_refuses_a_symlink_outright",
    ),
    Mutation(
        name="weights: let other weights write without revoking the index first",
        module="indexer.py",
        old="                    self._settle_weights()\n",
        new="                    pass  # whichever model is loaded writes its vectors\n",
        tests="test_vectors_from_other_weights_left_in_another_root_keep_the_index_revoked",
    ),
    Mutation(
        name="weights: skip a document stamped by other weights",
        module="indexer.py",
        old="            and (identity is None or known[3] == identity)\n",
        new="            and (identity is None or True)\n",
        tests="test_a_model_whose_weights_changed_re_embeds_the_index_in_place",
    ),
    Mutation(
        name="weights: store a document without the weights that embedded it",
        module="indexer.py",
        old="                    weights_revision=self._embedder.weights_revision,\n",
        new="",
        tests="test_a_repair_killed_partway_resumes_where_it_stopped",
    ),
    Mutation(
        name="weights: never vouch for the index again once it is repaired",
        module="indexer.py",
        old="            self._db.settle_weights(self._embedder.weights_revision)\n",
        new="",
        tests="test_a_model_whose_weights_changed_re_embeds_the_index_in_place",
    ),
    Mutation(
        name="weights: refuse a renamed model that names its weights",
        module="indexer.py",
        old=(
            "                if self._embedder.weights_revision is None:\n"
            "                    raise ForeignWeightsError(\n"
        ),
        new="                if True:\n                    raise ForeignWeightsError(\n",
        tests="test_a_renamed_model_that_names_its_weights_once_loaded_repairs_too",
        fails_with="markdown_memory.exceptions.ForeignWeightsError",
    ),
    Mutation(
        name="weights: refuse a renamed lazy model before it can name its weights",
        module="indexer.py",
        old=(
            "                self._embedder.warm_up()\n"
            "                if self._embedder.weights_revision is None:\n"
        ),
        new="                if self._embedder.weights_revision is None:\n",
        tests="test_a_renamed_model_that_names_its_weights_once_loaded_repairs_too",
        fails_with="markdown_memory.exceptions.ForeignWeightsError",
    ),
    Mutation(
        name="weights: let a renamed model that cannot name its weights write anyway",
        module="indexer.py",
        old="                    raise ForeignWeightsError(\n",
        new="                    _ = (\n",
        tests="test_a_renamed_model_that_cannot_name_its_weights_is_refused_not_discarded",
    ),
    Mutation(
        name="weights: discard every root before refusing a renamed unnamed model",
        module="indexer.py",
        old=(
            "                if self._embedder.weights_revision is None:\n"
            "                    raise ForeignWeightsError(\n"
        ),
        new=(
            "                if self._embedder.weights_revision is None:\n"
            "                    self._db.clear()\n"
            "                    raise ForeignWeightsError(\n"
        ),
        tests="test_a_renamed_model_that_cannot_name_its_weights_is_refused_not_discarded",
    ),
    Mutation(
        name="weights: discard every root on a rename over documents that hold no vector",
        module="indexer.py",
        old="            self._db.set_meta(MODEL_META_KEY, self._embedder.model_name)\n",
        new=(
            "            if previous_model not in {None, self._embedder.model_name}:\n"
            "                self._db.clear()\n"
            "            self._db.set_meta(MODEL_META_KEY, self._embedder.model_name)\n"
        ),
        tests="test_a_renamed_model_over_an_index_without_vectors_takes_it_over",
    ),
    Mutation(
        name="weights: record the refusal where no run of the old model can clear it",
        module="indexer.py",
        old=(
            "                if self._embedder.weights_revision is None:\n"
            "                    raise ForeignWeightsError(\n"
        ),
        new=(
            "                if self._embedder.weights_revision is None:\n"
            "                    self._db.record_weights_mismatch('refused')\n"
            "                    raise ForeignWeightsError(\n"
        ),
        tests=(
            "test_a_renamed_model_that_cannot_name_its_weights_is_refused_not_discarded"
            " or test_a_renamed_model_that_cannot_name_its_weights_is_refused_once_not_per_search"
        ),
    ),
    Mutation(
        name="weights: carry on past a renamed model that will not load",
        module="indexer.py",
        old=(
            "                self._embedder.warm_up()\n"
            "                if self._embedder.weights_revision is None:\n"
        ),
        new=(
            "                with contextlib.suppress(ModelLoadError):\n"
            "                    self._embedder.warm_up()\n"
            "                if self._embedder.weights_revision is None:\n"
        ),
        tests="test_a_renamed_model_that_will_not_load_leaves_the_index_it_found",
        fails_with="markdown_memory.exceptions.ForeignWeightsError",
    ),
    Mutation(
        name="weights: refuse a rename over documents that hold no vector",
        module="indexer.py",
        old="                and self._db.has_vectors()\n",
        new='                and self._db.count_rows("documents") > 0\n',
        tests="test_a_renamed_model_over_an_index_without_vectors_takes_it_over",
        fails_with="markdown_memory.exceptions.ForeignWeightsError",
    ),
    Mutation(
        name="weights: leave a pending repair to a model that has not loaded",
        module="indexer.py",
        old="            try:\n                self._embedder.warm_up()\n",
        new="            try:\n                pass\n",
        tests="test_a_pending_repair_loads_the_model_at_the_start_of_the_run",
    ),
    Mutation(
        name="weights: leave vectors no revision vouches for to whichever search notices",
        module="indexer.py",
        old='            or (recorded is None and self._db.count_rows("units_vec") > 0)\n',
        new="            or False\n",
        tests="test_vectors_no_revision_vouches_for_are_a_pending_repair",
    ),
    Mutation(
        name="search: rank vectors stored unvouched while the lookup ran",
        module="search.py",
        old="                best and recorded != self._embedder.weights_revision\n",
        new="                False\n",
        tests="test_vectors_written_unvouched_during_the_lookup_are_not_ranked",
    ),
    Mutation(
        name="storage: keep a revision claimed for vectors that never arrived",
        module="db.py",
        old=(
            "        if (recorded, mismatch) != (None, None) "
            'and self.count_rows("units_vec") == 0:\n'
        ),
        new="        if False:\n",
        tests="test_a_revision_claimed_for_vectors_that_never_arrived_is_dropped_by_the_next_run",
    ),
    Mutation(
        name="search: warn about weights recorded over an index with no vector",
        module="search.py",
        old=(
            "        if recorded is not None and weights != recorded "
            'and self._db.count_rows("units_vec") == 0:\n'
        ),
        new="        if False:\n",
        tests="test_a_revision_over_no_vectors_is_not_reported_as_a_mismatch",
    ),
    Mutation(
        name="storage: let a search overwrite the indexer's account of a mismatch",
        module="db.py",
        old=(
            "                    + "
            '("DO UPDATE SET value = excluded.value" if replace else "DO NOTHING"),\n'
        ),
        new='                    + "DO UPDATE SET value = excluded.value",\n',
        tests="test_a_search_does_not_replace_the_indexers_account_of_a_mismatch",
    ),
    Mutation(
        name="auto: start a run for every save instead of one per burst",
        module="autoindex.py",
        old=(
            "                (status.changed_files > self._baseline "
            "and since >= self._change_gap)\n"
        ),
        new="                (status.changed_files > self._baseline)\n",
        tests="test_an_edit_a_search_sees_starts_a_run_once_the_gap_has_passed",
    ),
    Mutation(
        name="auto: walk the tree again on every search over a file that cannot be indexed",
        module="autoindex.py",
        old=(
            "                (status.changed_files > self._baseline "
            "and since >= self._change_gap)\n"
        ),
        new="                (status.changed_files > 0 and since >= self._change_gap)\n",
        tests="test_a_file_the_last_run_could_not_index_does_not_start_one_per_search",
    ),
    Mutation(
        name="auto: let a clean run adopt an edit made while it ran as the norm",
        module="autoindex.py",
        old="            baseline = self._changed() if report.errors else 0\n",
        new="            baseline = self._changed()\n",
        tests="test_an_edit_made_while_a_clean_run_ran_starts_the_next_one",
    ),
    Mutation(
        name="auto: let a stopped run absorb the edits it never saw",
        module="autoindex.py",
        old='            logger.info("Automatic index run stopped")\n',
        new=(
            '            logger.info("Automatic index run stopped")\n'
            "            baseline = self._changed()\n"
        ),
        tests="test_a_stopped_run_keeps_the_baseline",
    ),
    Mutation(
        name="auto: retry a run that failed whole for the same edit every few seconds",
        module="autoindex.py",
        old=(
            '            logger.exception("Automatic index run failed")\n'
            "            baseline = self._changed()\n"
        ),
        new='            logger.exception("Automatic index run failed")\n',
        tests="test_a_run_that_fails_whole_is_not_retried_for_the_same_edit",
    ),
    Mutation(
        name="auto: mark a mismatch a search recorded mid-run as already handled",
        module="autoindex.py",
        old="            if after is None or after == before:\n",
        new="            if True:\n",
        tests="test_a_mismatch_recorded_while_a_run_was_busy_is_still_acted_on",
    ),
    Mutation(
        name="auto: wait a whole walk interval after a run another process blocked",
        module="autoindex.py",
        old="                self._last_finished -= self._walk_gap - self._change_gap\n",
        new="                pass\n",
        tests="test_a_run_another_process_blocked_is_retried_once_per_gap",
    ),
    Mutation(
        name="auto: retry a blocked run on every search while the lock is held",
        module="autoindex.py",
        old="                self._retry_after = self._last_finished + self._walk_gap\n",
        new="                pass\n",
        tests="test_a_run_another_process_blocked_is_retried_once_per_gap",
    ),
    Mutation(
        name="auto: never walk the tree for files nobody indexed yet",
        module="autoindex.py",
        old="                or since >= self._walk_gap\n",
        new="                or False\n",
        tests="test_the_walk_that_finds_new_files_is_due_after_the_interval",
    ),
    Mutation(
        name="auto: leave a weights mismatch a search found for the next walk",
        module="autoindex.py",
        old="                    status.weights_mismatch is not None\n",
        new="                    False\n",
        tests="test_a_weights_mismatch_starts_one_run_not_one_per_search",
    ),
    Mutation(
        name="auto: start a run per search over a mismatch no run can repair",
        module="autoindex.py",
        old="                    and status.weights_mismatch != self._seen_mismatch\n",
        new="                    and True\n",
        tests="test_a_weights_mismatch_starts_one_run_not_one_per_search",
    ),
    Mutation(
        name="auto: start a second run beside a running one, or after stop",
        module="autoindex.py",
        old=(
            "    def _start(self) -> bool:\n"
            "        if self._stopping or self._thread is not None:\n"
        ),
        new="    def _start(self) -> bool:\n        if False:\n",
        tests="test_nothing_starts_while_a_run_is_running_or_after_stop",
    ),
    Mutation(
        name="auto: return from stop while the run still holds the database",
        module="autoindex.py",
        old="            thread.join()\n",
        new="            pass\n",
        tests="test_stop_asks_the_run_to_stop_and_waits_for_it",
    ),
    Mutation(
        name="auto: close the database under a running background run",
        module="server.py",
        old="            self._auto.stop()\n",
        new="            pass\n",
        tests="test_close_stops_the_run_before_the_database_closes",
    ),
    Mutation(
        name="auto: never look at what a search measured",
        module="server.py",
        old="                self._auto.consider(status)\n",
        new="                pass\n",
        tests="test_a_search_after_an_edit_brings_the_index_up_to_date",
    ),
    Mutation(
        name="auto: start the background run whatever the configuration says",
        module="server.py",
        old="    if config.auto_index:\n",
        new="    if True:\n",
        tests="test_only_the_stdio_server_arms_it_and_only_when_asked_to",
    ),
    Mutation(
        name="install: start the catch-up run before the client's handshake is answered",
        module="server.py",
        old="        service.start_auto_index(request=False)\n",
        new="        service.start_auto_index()\n",
        tests="test_only_the_stdio_server_arms_it_and_only_when_asked_to",
    ),
    Mutation(
        name="install: arm the runner and let no first search ever start it",
        module="autoindex.py",
        old=(
            '                float("inf") if self._last_finished is None '
            "else self._clock() - self._last_finished\n"
        ),
        new=(
            "                0.0 if self._last_finished is None "
            "else self._clock() - self._last_finished\n"
        ),
        tests="test_an_armed_runner_catches_up_at_the_first_search",
    ),
    Mutation(
        name="install: answer the handshake with no version",
        module="server.py",
        old=(
            '        "markdown-memory", version=__version__, instructions=SERVER_INSTRUCTIONS, '
            "lifespan=lifespan\n"
        ),
        new='        "markdown-memory", instructions=SERVER_INSTRUCTIONS, lifespan=lifespan\n',
        tests="test_the_handshake_names_the_version",
    ),
    Mutation(
        name="install: drop the --version flag an install is checked with",
        module="server.py",
        old=(
            '    parser.add_argument("--version", action="version", '
            'version=f"%(prog)s {__version__}")\n'
        ),
        new="",
        tests="test_version_is_the_installed_one_and_builds_nothing",
    ),
    Mutation(
        name="install: build the whole service to download a model",
        module="server.py",
        old="    if arguments.download_model:\n",
        new="    if False:\n",
        tests="test_download_model_loads_the_configured_embedder_and_nothing_else",
    ),
    Mutation(
        name="install: report a failed model download as success",
        module="server.py",
        old=(
            '        logger.exception("Cannot download the embedding model")\n'
            "        raise SystemExit(1) from None\n"
        ),
        new='        logger.exception("Cannot download the embedding model")\n        return\n',
        tests="test_a_download_that_fails_exits_non_zero",
    ),
    Mutation(
        name="install: call a fresh download's own bookkeeping leftover weights",
        module="embedders.py",
        old="                ours = entry.is_relative_to(bookkeeping)\n",
        new="                ours = False\n",
        tests="test_the_graph_an_upgrade_left_behind_is_reported_not_hidden",
    ),
    Mutation(
        name="scope: walk into a linked worktree below the root",
        module="discovery.py",
        old="        if _is_linked_worktree(current):\n",
        new="        if False:\n",
        tests="test_a_linked_worktree_is_left_out_and_its_copies_purged",
    ),
    Mutation(
        name="scope: take any gitdir for a worktree's, submodules and dead ones included",
        module="discovery.py",
        old='    return os.path.isfile(directory / named / "commondir")',
        new="    return os.path.exists(directory / named)",
        tests="test_a_checkout_not_known_to_be_a_copy_is_indexed",
    ),
    Mutation(
        name="scope: resolve a worktree's gitdir only one way",
        module="discovery.py",
        old='    return os.path.isfile(directory / named / "commondir")',
        new='    return os.path.isfile(directory / named.lstrip("/") / "commondir")',
        tests="test_a_linked_worktree_is_left_out_and_its_copies_purged",
    ),
    Mutation(
        name="scope: take a bare path in a `.git` file for git's gitfile format",
        module="discovery.py",
        old="    if named == first or not named:\n",
        new="    if False:\n",
        tests="test_a_checkout_not_known_to_be_a_copy_is_indexed",
    ),
    Mutation(
        name="scope: take `gitdir: ` with no path for the directory's own gitdir",
        module="discovery.py",
        old="    if named == first or not named:\n",
        new="    if named == first:\n",
        tests="test_a_checkout_not_known_to_be_a_copy_is_indexed",
    ),
    Mutation(
        name="scope: trim a `.git` file before asking whether it is git's format",
        module="discovery.py",
        old='    first = os.fsdecode(head).splitlines()[0] if head else ""\n',
        new='    first = os.fsdecode(head).splitlines()[0].strip() if head else ""\n',
        tests="test_a_checkout_not_known_to_be_a_copy_is_indexed",
    ),
    Mutation(
        name="git: call a folder without git a repository git could not be asked about",
        module="discovery.py",
        old="        if not _in_repository(root):\n",
        new="        if False:\n",
        tests="test_a_failed_git_is_no_answer_and_says_so_only_for_a_repository",
    ),
    Mutation(
        name="git: call a folder git says is no repository one it could not be asked about",
        module="discovery.py",
        old='        if exc.returncode == 128 and "not a git repository (or any" in reason:\n',
        new="        if False:\n",
        tests="test_a_failed_git_is_no_answer_and_says_so_only_for_a_repository",
    ),
    Mutation(
        name="git: take a `.git` that names nothing usable for no repository at all",
        module="discovery.py",
        old='        if exc.returncode == 128 and "not a git repository (or any" in reason:\n',
        new='        if exc.returncode == 128 and "not a git repository" in reason:\n',
        tests="test_a_failed_git_is_no_answer_and_says_so_only_for_a_repository",
    ),
    Mutation(
        name="git: let a git hook's environment choose the repository",
        module="discovery.py",
        old="    env = {name: value for name, value in os.environ.items() if name not in _GIT_LOCATION_VARIABLES}\n",  # noqa: E501
        new="    env = dict(os.environ)\n",
        tests="test_a_git_hook_s_environment_does_not_choose_the_repository",
    ),
    Mutation(
        name="git: keep quiet to the agent when git could not be asked",
        module="models.py",
        old='            if self.gitignore == "unavailable":\n',
        new="            if False:\n",
        tests="test_only_a_whole_tree_mentions_git "
        "or test_a_failed_git_is_no_answer_and_says_so_only_for_a_repository",
    ),
    Mutation(
        name="git: keep quiet to a manual run when git could not be asked",
        module="indexer.py",
        old='    if git.state != "unavailable":\n',
        new="    if True:\n",
        tests="test_a_failed_git_is_no_answer_and_says_so_only_for_a_repository",
    ),
    Mutation(
        name="git: forget what git said once the walk is done",
        module="db.py",
        old="                (root, gitignore),\n",
        new="                (root, None),\n",
        tests="test_the_state_is_the_last_finished_walks "
        "or test_a_failed_git_is_no_answer_and_says_so_only_for_a_repository",
    ),
    Mutation(
        name="scope: refuse a checkout even when it is the root asked for",
        module="discovery.py",
        old="    while current != root and current.parent != current:\n",
        new="    while current.parent != current:\n",
        tests="test_a_checkout_pointed_at_directly_is_still_indexed",
    ),
    Mutation(
        name="scope: let a git that hangs fail the whole run",
        module="discovery.py",
        old="    except (OSError, subprocess.SubprocessError) as exc:\n",
        new="    except OSError as exc:\n",
        tests="test_a_failed_git_is_no_answer_and_says_so_only_for_a_repository",
    ),
    Mutation(
        name="scope: index whatever git ignores",
        module="discovery.py",
        old="        if self.ignored and any(\n",
        new="        if False and any(\n",
        tests="test_what_git_ignores_is_left_out_and_purged_but_tracked_files_stay",
    ),
    Mutation(
        name="scope: look only at the path, not at the ignored directory above it",
        module="discovery.py",
        old=(
            '            _printable("/".join(parts[:end])) in self.ignored for end in'
            " range(1, len(parts) + 1)\n"
        ),
        new=(
            '            _printable("/".join(parts[:end])) in self.ignored for end in'
            " [len(parts)]\n"
        ),
        tests="test_a_directory_that_became_an_ignored_symlink_takes_its_rows_with_it",
    ),
    Mutation(
        name="scope: keep rows behind a symlink that the root has disowned",
        module="indexer.py",
        old="            if scope.excludes(file_path):\n",
        new="            if False:\n",
        tests="test_a_directory_that_became_an_ignored_symlink_takes_its_rows_with_it",
    ),
    Mutation(
        name="scope: ask git even when told not to",
        module="indexer.py",
        old=(
            "            git = discovery.git_ignored(root) if self._gitignore"
            ' else discovery.GitIgnore("off")\n'
        ),
        new="            git = discovery.git_ignored(root)\n",
        tests="test_switched_off_git_is_never_asked "
        "or test_what_git_ignores_is_left_out_and_purged_but_tracked_files_stay",
    ),
    Mutation(
        name="scope: ignore MARKDOWN_MEMORY_GITIGNORE",
        module="config.py",
        old="            gitignore=_switched_on(ENV_GITIGNORE),\n",
        new="            gitignore=True,\n",
        tests="test_it_can_be_switched_off",
    ),
    Mutation(
        name="scope: never hand the gitignore switch to the indexer",
        module="server.py",
        old="            gitignore=config.gitignore,\n",
        new="            gitignore=True,\n",
        tests="test_the_server_passes_the_switch_to_its_indexer",
    ),
    Mutation(
        name="aliases: index a symlinked file a second time under the link",
        module="discovery.py",
        old="            if (target.st_dev, target.st_ino) in originals:\n",
        new="            if False:\n",
        tests="test_a_link_to_a_file_already_indexed_is_not_indexed_again",
    ),
    Mutation(
        name="aliases: drop the only link to a file that cannot be indexed itself",
        module="discovery.py",
        old="            if stat.S_ISREG(info.st_mode) and _encodable(str(path)):\n",
        new="            if stat.S_ISREG(info.st_mode):\n",
        tests="test_a_link_to_a_file_that_cannot_be_indexed_stands_in_for_it",
    ),
    Mutation(
        name="walk: visit dot-directories before the project's own docs",
        module="discovery.py",
        old='    return (name.startswith("."), name)\n',
        new="    return (False, name)\n",
        tests="test_the_projects_own_docs_come_before_dot_directories",
    ),
    Mutation(
        name="walk: ignore a stop until the whole tree is walked",
        module="discovery.py",
        old=(
            "            if should_stop is not None and should_stop():\n"
            '                raise IndexCancelled("index run stopped by its owner")\n'
            "            path = here / filename\n"
        ),
        new="            path = here / filename\n",
        tests="test_a_stop_is_honoured_during_the_walk",
    ),
    Mutation(
        name="failures: fail a whole run on a failure row recorded twice",
        module="db.py",
        old=(
            '                    "INSERT INTO index_failures(file_path, message) VALUES (?,'
            ' ?) "\n'
            '                    "ON CONFLICT(file_path) DO UPDATE SET message ='
            ' excluded.message",\n'
        ),
        new='                    "INSERT INTO index_failures(file_path, message) VALUES (?, ?)",\n',
        tests="test_recording_a_failure_twice_restates_it",
    ),
    Mutation(
        name="auto: keep telling the agent to run what is already running",
        module="models.py",
        old="        if self.indexing:\n",
        new="        if False:\n",
        tests="test_the_status_says_a_run_is_in_progress_instead_of_asking_for_one",
    ),
    Mutation(
        name="auto: never report the background run in the status",
        module="server.py",
        old="        active = self._auto is not None and self._auto.active\n",
        new="        active = False\n",
        tests="test_the_status_says_a_run_is_in_progress_instead_of_asking_for_one",
    ),
    Mutation(
        name="index: never ask whether the run should stop",
        module="indexer.py",
        old="                        if should_stop is not None and should_stop():\n",
        new="                        if False:\n",
        tests="test_a_stop_asked_once_the_walk_is_done_embeds_nothing",
    ),
    Mutation(
        name="index: write the file embedded while a stop was asked",
        module="indexer.py",
        old="                            elif should_stop is not None and should_stop():\n",
        new="                            elif False:\n",
        tests="test_the_file_being_embedded_is_not_written_and_not_called_a_failure",
    ),
    Mutation(
        name="index: let the per-file handler swallow a stop as one file's failure",
        module="exceptions.py",
        old="class IndexCancelled(Exception):",
        new="class IndexCancelled(MarkdownMemoryError):",
        tests="test_the_file_being_embedded_is_not_written_and_not_called_a_failure",
    ),
    Mutation(
        name="weights: fail a run over a pending repair whose model will not load",
        module="indexer.py",
        old="            except ModelLoadError as exc:\n",
        new="            except ZeroDivisionError as exc:\n",
        tests="test_a_pending_repair_whose_model_will_not_load_does_not_fail_the_run",
        fails_with="markdown_memory.exceptions.ModelLoadError",
    ),
    Mutation(
        name="weights: let named weights write beside vectors nobody vouched for",
        module="indexer.py",
        old="        if weights == recorded:\n",
        new="        if weights == recorded or recorded is None:\n",
        tests="test_vectors_nobody_vouched_for_are_not_ranked_beside_named_ones",
    ),
    Mutation(
        name="weights: count a document of headings alone as holding other weights' vectors",
        module="db.py",
        old='                    "AND EXISTS (SELECT 1 FROM sections AS s JOIN units AS u "\n',
        new='                    "AND EXISTS (SELECT 1 FROM sections AS s LEFT JOIN units AS u "\n',
        tests="test_a_document_of_headings_alone_never_holds_up_the_certificate",
    ),
    Mutation(
        name="weights: vouch for an index that holds no vector",
        module="db.py",
        old='            if conn.execute("SELECT 1 FROM units_vec LIMIT 1").fetchone() is None:\n',
        new="            if False:\n",
        tests="test_a_document_that_embeds_nothing_records_no_provenance",
    ),
    Mutation(
        name="storage: leave every document unstamped on upgrade",
        module="db.py",
        old=(
            "                        tx.execute("
            '"UPDATE documents SET weights_revision = ?", (recorded[0],))\n'
        ),
        new="                        pass\n",
        tests="test_an_upgrade_stamps_every_document_with_the_recorded_weights",
    ),
    Mutation(
        name="storage: rank vectors no record vouches for after an upgrade",
        module="db.py",
        old=(
            "                    elif tx.execute("
            '"SELECT 1 FROM units_vec LIMIT 1").fetchone() is not None:\n'
        ),
        new="                    elif False:\n",
        tests="test_an_upgrade_quarantines_vectors_no_record_vouches_for",
    ),
    Mutation(
        name="storage: revoke the weights and give the reason in two transactions",
        module="db.py",
        old="            _revoke_weights(conn, message)\n",
        new=(
            "            pass\n"
            "        self.set_meta(WEIGHTS_META_KEY, WEIGHTS_REVOKED)\n"
            "        self.record_weights_mismatch(message)\n"
        ),
        tests="test_revoking_writes_the_revision_and_its_reason_together",
    ),
    Mutation(
        name="storage: restore the weights and clear the reason in two transactions",
        module="db.py",
        old=(
            '                conn.execute("DELETE FROM meta WHERE key = ?", '
            "(WEIGHTS_MISMATCH_KEY,))\n"
            "                return\n            # Named by directory"
        ),
        new=(
            '                conn.execute("COMMIT")\n'
            '                conn.execute("BEGIN IMMEDIATE")\n'
            '                conn.execute("DELETE FROM meta WHERE key = ?", '
            "(WEIGHTS_MISMATCH_KEY,))\n"
            "                return\n            # Named by directory"
        ),
        tests="test_certifying_writes_the_revision_and_clears_the_reason_together",
    ),
    Mutation(
        name="search: describe an index being re-embedded as one built by other weights",
        module="search.py",
        old="        if recorded == WEIGHTS_REVOKED:\n",
        new="        if False:\n",
        tests="test_a_revoked_index_keeps_the_reason_the_indexer_gave",
    ),
    Mutation(
        name="weights: refuse a database whose recorded revision describes nothing",
        module="indexer.py",
        old='        if self._db.count_rows("units_vec") == 0:\n',
        new="        if False:  # a revision outliving its vectors still speaks for them\n",
        tests="test_a_revision_left_over_a_vectorless_index_does_not_refuse_unnamed_weights",
        fails_with="markdown_memory.exceptions.ForeignWeightsError",
    ),
    Mutation(
        name="weights: ask an unloaded model which weights it is",
        module="indexer.py",
        old=(
            "        self._embedder.warm_up()\n"
            "        weights = self._embedder.weights_revision\n"
            "        if weights is None:\n"
        ),
        new=("        weights = self._embedder.weights_revision\n        if weights is None:\n"),
        tests="test_the_model_is_loaded_before_it_is_asked_which_weights_it_is",
        # Unloaded, it answers None, which is refused as weights that cannot be named.
        fails_with="markdown_memory.exceptions.ForeignWeightsError",
    ),
    Mutation(
        name="search: rank this model's query against another model's vectors",
        module="search.py",
        old="        recorded = self._refuse_foreign_vectors()\n",
        new="        recorded = self._db.get_meta(WEIGHTS_META_KEY)\n",
        tests="test_a_cache_that_changes_while_no_document_does_still_stops_semantic_ranking",
    ),
    Mutation(
        name="search: read the weights before the query has loaded the model",
        module="search.py",
        old="        embedding = self._embedder.embed_query(query)\n",
        new=(
            "        self._refuse_foreign_vectors()  # before the model has loaded\n"
            "        embedding = self._embedder.embed_query(query)\n"
        ),
        tests="test_the_weights_are_read_after_the_query_is_embedded_not_before",
    ),
    Mutation(
        name="weights: put a revision on an index that holds no vector",
        module="db.py",
        old="            _forget_weights_without_vectors(conn)\n        return Document(\n",
        new="        return Document(\n",
        tests="test_a_document_whose_prose_becomes_headings_leaves_no_provenance_behind",
    ),
    Mutation(
        name="storage: keep a revision after purging the last document that had one",
        module="db.py",
        old="            _forget_weights_without_vectors(conn)\n        return deleted\n",
        new="        return deleted\n",
        tests="test_purging_the_last_document_forgets_what_its_vectors_came_from",
    ),
    Mutation(
        name="search: trust a revision checked before the rows were read",
        module="search.py",
        old="            self._db.get_meta(WEIGHTS_META_KEY) != recorded\n",
        new="            False  # the check speaks for rows read after it\n",
        tests="test_an_index_rebuilt_by_another_model_mid_search_is_not_ranked_on",
    ),
    Mutation(
        name="search: trust a model name checked before the rows were read",
        module="search.py",
        old="            or self._db.get_meta(MODEL_META_KEY) != model\n",
        new="            or False\n",
        tests="test_a_model_renamed_mid_search_is_not_ranked_on",
    ),
    Mutation(
        name="search: rank another model's vectors when neither names its weights",
        module="search.py",
        old=(
            "                if renamed is not None:\n"
            "                    raise SearchError(renamed)\n"
        ),
        new="                if False:\n                    raise SearchError(renamed)\n",
        tests="test_vectors_another_model_built_are_not_ranked_when_neither_names_its_weights",
    ),
    Mutation(
        name="cache: look only at other revisions, and miss the graph this one replaced",
        module="embedders.py",
        old="            wanted = {self._model_dir / name for name in model_cache.GEMMA_FILES}\n",
        new="            wanted = {entry for entry in self._model_dir.rglob('*')}\n",
        tests="test_the_graph_an_upgrade_left_behind_is_reported_not_hidden",
    ),
    Mutation(
        name="weights: shorten two graphs' identities to the same twelve characters",
        module="embedders.py",
        old='    return f"{revision[:12]}/{graph}" if separator else revision[:12]\n',
        new="    return revision[:12]\n",
        tests="test_a_mismatch_message_distinguishes_two_graphs_at_one_revision",
    ),
    Mutation(
        name="cache: trust a stamp that vouches for a different graph's files",
        module="model_cache.py",
        old="    if not isinstance(recorded, dict) or set(recorded) != set(GEMMA_FILES):\n",
        new="    if not isinstance(recorded, dict):\n",
        tests="test_a_cache_holding_another_graph_is_refetched_rather_than_trusted",
    ),
    Mutation(
        name="weights: name the revision but not which of its graphs answered",
        module="embedders.py",
        old='        return f"{model_cache.GEMMA_REVISION}/{model_cache.GEMMA_MODEL_FILE}"\n',
        new="        return model_cache.GEMMA_REVISION\n",
        tests="test_two_graphs_at_one_revision_report_different_weights",
    ),
    Mutation(
        name="weights: report one model's revision for another model's weights",
        module="embedders.py",
        old="        if self._model_name != model_cache.BGE_SMALL_MODEL_NAME:\n",
        new="        if False:  # every model is assumed to live in that one folder\n",
        tests="test_only_the_model_whose_cache_it_can_find_reports_a_revision",
    ),
    Mutation(
        name="weights: claim the index only once its vectors are already written",
        module="indexer.py",
        old="            self._claim_empty_index()\n",
        new="            return\n",
        tests="test_the_revision_is_written_before_the_vectors_it_describes",
    ),
    Mutation(
        name="search: keep the mismatch to itself while the status says all is well",
        module="search.py",
        old="        self._db.record_weights_mismatch(message, replace=False)\n",
        new="        pass  # the status goes on calling the index healthy\n",
        tests="test_a_search_that_finds_the_weights_changed_says_so_in_the_index_status",
    ),
    Mutation(
        name="weights: demand a working model for a file that embeds nothing",
        module="indexer.py",
        old="                if prepared.has_vectors and not weights_settled:\n",
        new="                if not weights_settled:\n",
        tests="test_a_file_that_embeds_nothing_does_not_need_a_model_that_loads",
        fails_with="markdown_memory.exceptions.ModelLoadError",
    ),
    Mutation(
        name="search: withdraw a pending repair because these weights agree",
        module="search.py",
        old="            # repair is pending. That run withdraws it once the whole index agrees.\n",
        new=(
            "            # repair is pending. That run withdraws it once the whole index agrees.\n"
            "            self._db.record_weights_mismatch(None)\n"
        ),
        tests="test_weights_that_come_back_rank_again_and_leave_the_mismatch_to_a_run",
    ),
    Mutation(
        name="search: rank named weights against vectors no revision vouches for",
        module="search.py",
        old="            if not self._db.has_vectors():\n",
        new="            if True:\n",
        tests="test_vectors_no_revision_vouches_for_are_not_ranked_by_named_weights",
    ),
    Mutation(
        name="status: report a mismatched index as verified anyway",
        module="db.py",
        old="            verified=verified and weights_mismatch is None,\n",
        new="            verified=verified,\n",
        tests="TestAnIndexAnsweringFromAnotherModelsVectors",
    ),
    Mutation(
        name="indexer: let onnxruntime's intra-op pool spin-wait again",
        module="embedders.py",
        old="            options.add_session_config_entry(*_SPIN_CONFIG)\n",
        new="",
        tests="test_gemma_session_disables_intra_op_spinning",
    ),
    Mutation(
        name="indexer: document MARKDOWN_MEMORY_THREADS for bge-small but drop it again",
        module="embedders.py",
        old="                        threads=_inference_threads() or None,\n",
        new="",
        tests="test_fastembed_passes_the_thread_override",
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
        name="excerpt harness: count a split verdict as a yes",
        module="eval_excerpts.py",
        area="scripts",
        old='    return 2 * sum(judge[item_id] == "yes" for judge in judges) > len(judges)',
        new='    return 2 * sum(judge[item_id] == "yes" for judge in judges) >= len(judges)',
        tests="test_a_verdict_is_a_strict_majority_and_a_split_is_not_a_yes",
    ),
    Mutation(
        name="excerpt harness: pass a small sample on its point estimate alone",
        module="eval_excerpts.py",
        area="scripts",
        old="            passed &= retention >= FLOOR_RETENTION and bound >= FLOOR_RETENTION_BOUND",
        new="            passed &= retention >= FLOOR_RETENTION",
        tests="test_the_gates_count_only_frozen_items_and_hold_only_on_v2",
    ),
    Mutation(
        name="excerpt harness: score items outside the frozen denominator",
        module="eval_excerpts.py",
        area="scripts",
        old='        if key["id"] in frozen:',
        new="        if True:",
        tests="test_the_gates_count_only_frozen_items_and_hold_only_on_v2",
    ),
    Mutation(
        name="excerpt harness: let a lost excerpt cost nothing more",
        module="eval_excerpts.py",
        area="scripts",
        old='    cost = key["payload_tokens"] + (0 if kept else key["read_tokens"])',
        new='    cost = key["payload_tokens"]',
        tests="test_a_lost_excerpt_costs_the_read_and_a_whole_section_is_kept_as_sent",
    ),
    Mutation(
        name="excerpt harness: score a top hit that moved to another section",
        module="eval_excerpts.py",
        area="scripts",
        old=(
            '        if key["id"] in set(frozen["eligible"]) and clusters.get(key["id"])'
            ' != key["cluster"]'
        ),
        new='        if key["id"] in set(frozen["eligible"]) and False',
        tests="test_a_build_that_drifted_from_the_frozen_denominator_is_not_scored",
    ),
    Mutation(
        name="excerpt harness: price a baseline that names another top hit",
        module="eval_excerpts.py",
        area="scripts",
        old=(
            '        raise RuntimeError(f"{item_id}: the whole-section response'
            ' names another top hit")'
        ),
        new="        pass",
        tests="test_the_baseline_is_the_whole_section_response_for_the_same_top_hit",
    ),
    Mutation(
        name="excerpt harness: gate v1, which has no sealed set",
        module="eval_excerpts.py",
        area="scripts",
        old='GATED = ("v2",)',
        new='GATED = ("v1", "v2")',
        tests="test_the_gates_count_only_frozen_items_and_hold_only_on_v2",
    ),
    Mutation(
        name="excerpt harness: gate the hoped-for cost instead of a material saving",
        module="eval_excerpts.py",
        area="scripts",
        old="CEILING_COST_RATIO = 0.90",
        new="CEILING_COST_RATIO = 0.80",
        tests="test_the_gates_count_only_frozen_items_and_hold_only_on_v2 "
        "or test_the_cost_target_is_reported_but_does_not_gate",
    ),
    Mutation(
        name="excerpt: stop the window one block after the anchor again",
        module="search.py",
        old="BLOCKS_AFTER = 3",
        new="BLOCKS_AFTER = 1",
        tests="test_the_anchor_one_block_before_and_three_after",
    ),
    Mutation(
        name="excerpt: leave out the block before the anchor",
        module="search.py",
        old="BLOCKS_BEFORE = 1",
        new="BLOCKS_BEFORE = 0",
        tests="test_the_anchor_one_block_before_and_three_after",
    ),
    Mutation(
        name="excerpt harness: show the judges a text without the hit it came from",
        module="eval_excerpts.py",
        area="scripts",
        old='                    "hit": {"file_path": file_path, "heading_path": heading_path},\n',
        new="",
        tests="test_a_judge_sees_one_text_the_section_or_the_excerpt",
    ),
    Mutation(
        name="diagram: print a token count the files stopped matching",
        module="make_diagram.py",
        area="scripts",
        old='    ("README.md", 11360),',
        new='    ("README.md", 5654),',
        tests="test_every_file_on_the_diagram_still_costs_what_it_says "
        "or test_the_totals_the_readme_prints_are_the_sum_of_those_files",
    ),
    Mutation(
        name="diagram: claim an excerpt the anchored passage does not give",
        module="make_diagram.py",
        area="scripts",
        old="EXCERPT_ANCHOR = 0",
        new="EXCERPT_ANCHOR = 1",
        tests="test_the_excerpt_the_worked_example_shows_is_the_size_it_claims",
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
        name="diagram: price the call at a figure the README does not print",
        module="make_diagram.py",
        area="scripts",
        old="CALL_TOKENS = 608\n",
        new="CALL_TOKENS = 609\n",
        tests="test_the_worked_example_says_what_comes_back "
        "or test_the_call_the_headline_prices_is_the_call_search_docs_sends",
    ),
    Mutation(
        name="diagram: leave the call out of what the picture says to a screen reader",
        module="make_diagram.py",
        area="scripts",
        old="f'~{CALL_TOKENS}-token call with {spell(POINTER_COUNT)} pointers to the rest.\">'",
        new="f'{spell(POINTER_COUNT)} pointers to the rest.\">'",
        tests="test_the_committed_drawing_is_the_one_the_generator_draws",
    ),
    Mutation(
        name="diagram: stop drawing the token count beside each returned section",
        module="make_diagram.py",
        area="scripts",
        old='        o.append(text(776, y + 10, f"{tokens}", fill=c["muted"], size=11, anchor="end", font=MONO))\n',  # noqa: E501
        new="",
        tests="test_the_committed_drawing_is_the_one_the_generator_draws",
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
        module="discovery.py",
        old=(
            "        elif any(fnmatchcase(part, pattern) for part in parts):\n"
            "            return True"
        ),
        new="        elif False:\n            return True",
        tests="test_a_bare_name_excludes_that_directory_at_any_depth",
    ),
    Mutation(
        name="exclusions: split a configured pattern on colons as well as commas",
        module="config.py",
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
        name="cost: measure a call other than the default one",
        module="eval_retrieval.py",
        area="scripts",
        old='server.call_tool("search_docs", {"query": query})',
        new='server.call_tool("search_docs", {"query": query, "limit": 1})',
        tests="test_the_payload_is_the_text_block_the_default_call_sends",
    ),
    Mutation(
        name="cost: divide the answer by the payload",
        module="eval_retrieval.py",
        area="scripts",
        old="            payload / answer for payload, answer in",
        new="            answer / payload for payload, answer in",
        tests="test_the_payload_is_the_text_block_the_default_call_sends",
    ),
    Mutation(
        name="cost: ignore the file a qualified label names",
        module="eval_retrieval.py",
        area="scripts",
        old="                    if not qualified or name in _names(file_path, service.root)",
        new="                    if True",
        tests="test_a_qualified_label_resolves_in_the_file_it_names",
    ),
    Mutation(
        name="v2: let a label name a file by its name alone, never its path",
        module="eval_retrieval.py",
        area="scripts",
        old="    return path.name, path.relative_to(root).as_posix()",
        new="    return path.name, path.name",
        tests="test_a_hit_answers_to_its_file_name_and_its_whole_path_only",
    ),
    Mutation(
        name="v2: let any trailing part of a path name a file",
        module="eval_retrieval.py",
        area="scripts",
        old="    return path.name, path.relative_to(root).as_posix()",
        new="    return path.name, *('/'.join(path.parts[i:]) for i in range(len(path.parts)))",
        tests="test_a_label_names_a_file_by_its_whole_path_from_the_corpus_root",
    ),
    Mutation(
        name="v2: share the gate's cached index",
        module="eval_retrieval.py",
        area="scripts",
        old='        return root if self.name == "v1" else root.with_name(f"eval-{self.name}")',
        new="        return root",
        tests="test_v2_keeps_its_own_index_and_baseline",
    ),
    Mutation(
        name="v2: rank the churn report without the corpus root its labels need",
        module="cross_host_churn.py",
        area="scripts",
        old="    return [_labels(hit, service.root) for hit in service.search_docs(query, TOP_N)]",
        new="    return [_labels(hit) for hit in service.search_docs(query, TOP_N)]",
        tests="test_the_churn_report_ranks_by_the_same_labels_as_the_gate",
        fails_with="TypeError",
    ),
    Mutation(
        name="cost: look only at top-level headings for a label",
        module="eval_retrieval.py",
        area="scripts",
        old="        yield from _outline_paths(node.children)",
        new="        yield from ()",
        tests="test_a_nested_heading_resolves",
    ),
    Mutation(
        name="cost: take an ambiguous label for the first section it names",
        module="eval_retrieval.py",
        area="scripts",
        old="                if len(found) != 1:",
        new="                if not found:",
        tests="test_a_label_naming_no_section_or_two_is_a_broken_fixture",
    ),
    Mutation(
        name="cost: search before the fixture is known to be sound",
        module="eval_retrieval.py",
        area="scripts",
        old="        answers = resolve_answers(service, queries, splits)",
        new="        answers = {}",
        tests="test_a_broken_fixture_stops_the_run_before_any_search",
    ),
    Mutation(
        name="split: score held-out on a dev run",
        module="eval_retrieval.py",
        area="scripts",
        old="        scores: dict[str, Scores] = {}\n        for split in splits:",
        new="        scores: dict[str, Scores] = {}\n        for split in SPLITS:",
        tests="test_a_dev_run_never_reaches_held_out",
    ),
    Mutation(
        name="split: cost held-out on a dev run",
        module="eval_retrieval.py",
        area="scripts",
        old="    costs: dict[str, Cost] = {}\n    for split in splits:",
        new="    costs: dict[str, Cost] = {}\n    for split in SPLITS:",
        tests="test_a_dev_run_never_reaches_held_out",
    ),
    Mutation(
        name="split: search held-out no-answer cases on a dev run",
        module="eval_retrieval.py",
        area="scripts",
        old="    measured: dict[str, NoAnswer] = {}\n    for split in splits:",
        new="    measured: dict[str, NoAnswer] = {}\n    for split in SPLITS:",
        tests="test_a_dev_run_never_reaches_held_out",
    ),
    Mutation(
        name="split: resolve held-out labels on a dev run",
        module="eval_retrieval.py",
        area="scripts",
        old="    for split in splits:  # only the splits scored",
        new="    for split in SPLITS:  # only the splits scored",
        tests="test_a_dev_run_never_reaches_held_out",
    ),
    Mutation(
        name="split: check held-out no-answer cases on a dev run",
        module="eval_retrieval.py",
        area="scripts",
        old='    cases = [case for split in splits for case in queries[split]["no_answer"]]',
        new='    cases = [case for split in SPLITS for case in queries[split]["no_answer"]]',
        tests="test_a_dev_run_never_reaches_held_out",
    ),
    Mutation(
        name="split: let a report-only run write the baseline",
        module="eval_retrieval.py",
        area="scripts",
        old="    if report_only and arguments.update_baseline:",
        new="    if False:",
        tests="test_a_report_only_run_cannot_write_the_baseline",
    ),
    Mutation(
        name="split: check the gate on another label file",
        module="eval_retrieval.py",
        area="scripts",
        old="    if report_only:\n",
        new="    if False:\n",
        tests="test_another_label_file_checks_no_gate",
    ),
    Mutation(
        name="record: let two cases share one key",
        module="eval_retrieval.py",
        area="scripts",
        old="        if key in cases:\n",
        new="        if False:\n",
        tests="test_a_case_named_twice_is_not_recorded",
    ),
    Mutation(
        name="record: keep a non-finite score",
        module="eval_retrieval.py",
        area="scripts",
        old="                if not math.isfinite(outcome.ndcg5):",
        new="                if False:",
        tests="test_a_non_finite_score_is_not_recorded",
    ),
    Mutation(
        name="compare: let a rank drop",
        module="eval_compare.py",
        area="scripts",
        old="    if _rank(after) > _rank(before):",
        new="    if False:",
        tests="test_one_query_worse_rejects",
    ),
    Mutation(
        name="compare: let the first answer stop being one",
        module="eval_compare.py",
        area="scripts",
        old='    if before["any_valid"] and not after["any_valid"]:',
        new="    if False:",
        tests="test_one_query_worse_rejects",
    ),
    Mutation(
        name="compare: let nDCG@5 fall",
        module="eval_compare.py",
        area="scripts",
        old="    if _ndcg(after) < _ndcg(before) - NDCG_TOLERANCE:",
        new="    if False:",
        tests="test_one_query_worse_rejects or test_a_drop_within_the_tolerance_is_not_a_loss",
    ),
    Mutation(
        name="compare: let a page come back empty",
        module="eval_compare.py",
        area="scripts",
        old='    if before["hits"] and not after["hits"]:',
        new="    if False:",
        tests="test_one_query_worse_rejects",
    ),
    Mutation(
        name="compare: let a no-answer page change",
        module="eval_compare.py",
        area="scripts",
        old='        if page_before != (after["hits"], after["keyword_match"]):',
        new="        if False:",
        tests="test_a_no_answer_page_must_not_change",
    ),
    Mutation(
        name="compare: compare relabelled cases unreviewed",
        module="eval_compare.py",
        area="scripts",
        old="    if changed and not allow_label_changes:",
        new="    if False:",
        tests="test_a_label_change_needs_an_explicit_review",
    ),
    Mutation(
        name="compare: read a duplicated JSON key",
        module="eval_compare.py",
        area="scripts",
        old="    if len(keys) != len(set(keys)):",
        new="    if False:",
        tests="test_a_malformed_record_is_not_compared",
    ),
    Mutation(
        name="compare: accept a non-finite nDCG",
        module="eval_compare.py",
        area="scripts",
        old="    if not (_is_int(ndcg) or isinstance(ndcg, float)) or not math.isfinite(ndcg):",
        new="    if not (_is_int(ndcg) or isinstance(ndcg, float)):",
        tests="test_a_malformed_record_is_not_compared",
    ),
    Mutation(
        name="compare: accept a rank outside 1..20",
        module="eval_compare.py",
        area="scripts",
        old="    if rank is not None and (not _is_int(rank) or not 1 <= rank <= MAX_RANK):",
        new="    if False:",
        tests="test_a_malformed_record_is_not_compared",
    ),
    Mutation(
        name="compare: read a record that is not an object",
        module="eval_compare.py",
        area="scripts",
        old="    if not isinstance(record, dict):\n",
        new="    if False:\n",
        tests="test_a_malformed_record_is_not_compared",
    ),
    Mutation(
        name="compare: read a case that is missing a field",
        module="eval_compare.py",
        area="scripts",
        old="    if absent:",
        new="    if False:",
        tests="test_a_malformed_record_is_not_compared",
    ),
    Mutation(
        name="compare: let two empty records agree",
        module="eval_compare.py",
        area="scripts",
        old="        if stratum.endswith(ANSWERABLE) and not counts[stratum]:",
        new="        if False:",
        tests="test_a_malformed_record_is_not_compared",
    ),
    Mutation(
        name="compare: compare records of different instruments",
        module="eval_compare.py",
        area="scripts",
        old='    for field in ("corpus", "preset", "splits", "corpus_sha256", "evaluator"):',
        new='    for field in ("corpus", "splits"):',
        tests="test_records_of_different_things_are_not_compared",
    ),
    Mutation(
        name="cache: key on the source beside the script",
        module="eval_cache.py",
        area="scripts",
        old="SOURCE = Path(markdown_memory.__file__).resolve().parent",
        new='SOURCE = Path(__file__).parent.parent / "src" / "markdown_memory"',
        tests="test_another_revision_on_pythonpath_keys_its_own_index",
    ),
    Mutation(
        name="cache: one key for the same corpus in two worktrees",
        module="eval_cache.py",
        area="scripts",
        old='        "root": str(corpus.resolve()),\n',
        new="",
        tests="test_the_same_corpus_at_two_roots_is_two_indexes",
    ),
    Mutation(
        name="cache: keep the identity that misattributed entries",
        module="eval_cache.py",
        area="scripts",
        old="CACHE_VERSION = 2\n",
        new="CACHE_VERSION = 1\n",
        tests="test_an_entry_from_the_old_identity_is_not_reused",
    ),
    Mutation(
        name="cache: trust metadata of another cache version",
        module="eval_cache.py",
        area="scripts",
        old='    if meta.get("version") != CACHE_VERSION:',
        new="    if False:",
        tests="test_an_entry_from_the_old_identity_is_not_reused",
    ),
    Mutation(
        name="lru: keep one index per corpus again",
        module="eval_cache.py",
        area="scripts",
        old="KEEP_INDEXES = 4\n",
        new="KEEP_INDEXES = 1\n",
        tests="TestTheCacheKeepsRecentIndexes or TestEvaluationsReuseWhatTheyBuilt",
    ),
    Mutation(
        name="lru: never refresh an index's recency",
        module="eval_cache.py",
        area="scripts",
        old="    os.utime(root / keep, follow_symlinks=False)\n",
        new="",
        tests="TestTheCacheKeepsRecentIndexes or TestEvaluationsReuseWhatTheyBuilt",
    ),
    Mutation(
        name="lru: evict the most recent instead",
        module="eval_cache.py",
        area="scripts",
        old="        reverse=True,\n",
        new="        reverse=False,\n",
        tests="TestTheCacheKeepsRecentIndexes",
    ),
    Mutation(
        name="lru: let the current index be evicted",
        module="eval_cache.py",
        area="scripts",
        old="        if entry.name == keep or not _DIGEST.fullmatch(entry.name):",
        new="        if not _DIGEST.fullmatch(entry.name):",
        tests="test_the_current_index_stays_whatever_its_date",
    ),
    Mutation(
        name="lru: delete directories the cache did not make",
        module="eval_cache.py",
        area="scripts",
        old="        if entry.name == keep or not _DIGEST.fullmatch(entry.name):",
        new="        if entry.name == keep:",
        tests="test_only_cache_entries_are_considered",
    ),
    Mutation(
        name="lru: follow a symlinked entry",
        module="eval_cache.py",
        area="scripts",
        old="        if not stat.S_ISDIR(status.st_mode):\n            continue\n",
        new="",
        tests="test_a_symlink_is_neither_counted_followed_nor_removed",
    ),
    Mutation(
        name="lru: keep unfinished builds",
        module="eval_cache.py",
        area="scripts",
        old="        if not complete:",
        new="        if False:",
        tests="test_an_unfinished_build_is_swept or test_failed_builds_never_pile_up",
    ),
    Mutation(
        name="lru: let reuse leave recency alone",
        module="eval_retrieval.py",
        area="scripts",
        old="            eval_cache.prune(root, key.digest)  # only on success",
        new="            pass  # only on success",
        tests="TestEvaluationsReuseWhatTheyBuilt",
    ),
    Mutation(
        name="lru: let a build evict nothing",
        module="eval_retrieval.py",
        area="scripts",
        old="    eval_cache.prune(root, key.digest)\n    return service, True",
        new="    return service, True",
        tests="TestEvaluationsReuseWhatTheyBuilt",
    ),
    Mutation(
        name="lru: evict before knowing the run succeeds",
        module="eval_retrieval.py",
        area="scripts",
        old="    eval_cache.drop_incomplete(root, key.digest)\n",
        new="    eval_cache.prune(root, key.digest)\n",
        tests="test_failed_builds_never_pile_up",
    ),
    Mutation(
        name="lru: let failed builds pile up",
        module="eval_retrieval.py",
        area="scripts",
        old="    eval_cache.drop_incomplete(root, key.digest)\n",
        new="",
        tests="test_failed_builds_never_pile_up",
    ),
    Mutation(
        name="mixed: score no mixed stratum",
        module="eval_retrieval.py",
        area="scripts",
        old='ANSWERABLE = ("paraphrase", "identifier", "mixed")',
        new='ANSWERABLE = ("paraphrase", "identifier")',
        tests="test_a_mixed_stratum_is_scored_and_recorded",
    ),
    Mutation(
        name="mixed: require a mixed stratum of every query file",
        module="eval_retrieval.py",
        area="scripts",
        old="if k not in OPTIONAL_KINDS or k in queries[split])",
        new="if True)",
        tests="test_a_fixture_without_a_mixed_stratum_is_scored_as_before",
    ),
    Mutation(
        name="mixed: refuse every record without a mixed stratum",
        module="eval_compare.py",
        area="scripts",
        old='OPTIONAL = ("mixed",)',
        new="OPTIONAL = ()",
        tests="test_records_without_a_mixed_stratum_still_compare",
    ),
    Mutation(
        name="mixed: accept an empty mixed stratum",
        module="eval_compare.py",
        area="scripts",
        old='ANSWERABLE = ("paraphrase", "identifier", "mixed")',
        new='ANSWERABLE = ("paraphrase", "identifier")',
        tests="test_a_declared_mixed_stratum_must_hold_queries",
    ),
    Mutation(
        name="mixed: refuse mixed cases as an unknown kind",
        module="eval_compare.py",
        area="scripts",
        old='KINDS = ("paraphrase", "identifier", "mixed", "no_answer")',
        new='KINDS = ("paraphrase", "identifier", "no_answer")',
        tests="test_a_mixed_query_worse_rejects",
    ),
    Mutation(
        name="keyed: let a server.py change reuse the index",
        module="eval_cache.py",
        area="scripts",
        old='    "server.py",\n',
        new="",
        tests="test_a_change_to_what_builds_the_index_changes_the_key",
    ),
    Mutation(
        name="keyed: let a headings.py change reuse the index",
        module="eval_cache.py",
        area="scripts",
        old='    "headings.py",\n',
        new="",
        tests="test_a_change_to_what_builds_the_index_changes_the_key",
    ),
    Mutation(
        name="keyed: let an exceptions.py change reuse the index",
        module="eval_cache.py",
        area="scripts",
        old='    "exceptions.py",\n)',
        new=")",
        tests="test_a_change_to_what_builds_the_index_changes_the_key",
    ),
    Mutation(
        name="keyed: rebuild the index for a ranking change",
        module="eval_cache.py",
        area="scripts",
        old='    "exceptions.py",\n)',
        new='    "exceptions.py",\n    "search.py",\n)',
        tests="test_a_ranking_change_keeps_the_key",
    ),
    Mutation(
        name="keyed: excuse a module without a reason",
        module="eval_cache.py",
        area="scripts",
        old='"never started by an evaluation: only main() starts background indexing"',
        new='" "',
        tests="test_every_package_module_is_classified",
    ),
    Mutation(
        name="cost: let the informational table decide how the run ends",
        module="eval_retrieval.py",
        area="scripts",
        old="        except Exception as exc:  # informational: it must",
        new="        except ZeroDivisionError as exc:  # informational: it must",
        tests="test_a_cost_pass_that_fails_does_not_decide_the_exit_code",
    ),
    Mutation(
        name="cost: accept --show-costs and list nothing",
        module="eval_retrieval.py",
        area="scripts",
        old="    if per_query:",
        new="    if False:",
        tests="test_show_costs_lists_every_query",
    ),
    Mutation(
        name="eval: take the slowest query for the p95",
        module="eval_retrieval.py",
        area="scripts",
        old="    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]",
        new="    return ordered[-1]",
        tests="test_p95_is_the_same_rule_for_latency_and_cost",
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
        module="config.py",
        old='        cleaned = part.strip().removeprefix("./").rstrip("/")',
        new='        cleaned = part.strip().lstrip("./").rstrip("/")',
        tests="test_a_dot_prefixed_name_is_not_mistaken_for_a_relative_path",
    ),
    Mutation(
        name="exclusions: fold case when matching a pattern",
        module="discovery.py",
        old="        elif any(fnmatchcase(part, pattern) for part in parts):",
        new="        elif any(fnmatchcase(part.lower(), pattern.lower()) for part in parts):",
        tests="test_matching_does_not_depend_on_the_platform_case_rules",
    ),
    Mutation(
        name="config: refuse a directory whose real name contains a dollar-brace",
        module="config.py",
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
        old="        elif any(_touches_markdown(later) for later in window):",
        new="        elif False:",
        tests=(
            "test_a_read_of_another_file_is_not_a_same_file_fallback"
            " or test_a_search_abandoned_for_grep"
        ),
    ),
    Mutation(
        name="usage: blame a search for a file read that came much later",
        module="usage_from_transcripts.py",
        area="scripts",
        old="        window = calls[position + 1 : position + 1 + FALLBACK_WINDOW]",
        new="        window = calls[position + 1 :]",
        tests="test_a_file_read_long_after_a_search_is_not_attributed_to_it",
    ),
    Mutation(
        name="usage: pair a reused id with the newest call, not the oldest",
        module="usage_from_transcripts.py",
        area="scripts",
        old="                    position = queue.popleft()",
        new="                    position = queue.pop()",
        tests="test_a_reused_id_pairs_first_in_first_out",
    ),
    Mutation(
        name="usage: count a subagent's transcript as a session of its own",
        module="usage_from_transcripts.py",
        area="scripts",
        old='        session = session or str(record.get("sessionId") or "")',
        new='        session = ""',
        tests="test_a_subagent_belongs_to_its_parents_session",
    ),
    Mutation(
        name="usage: drop the sessions that never called a tool",
        module="usage_from_transcripts.py",
        area="scripts",
        old="    for transcript in transcripts:",
        new="    for transcript in (t for t in transcripts if t.calls):",
        tests="test_a_session_with_no_calls_is_still_a_session",
    ),
    Mutation(
        name="usage: hide the report when nobody called the server",
        module="usage_from_transcripts.py",
        area="scripts",
        old="    print_report(report)",
        new="    if report.sessions_using_server:\n        print_report(report)",
        tests="test_zero_adoption_still_reports_who_needed_docs",
    ),
    Mutation(
        name="usage: only count a session that called the server as needing docs",
        module="usage_from_transcripts.py",
        area="scripts",
        old="        if retrieval or by_hand:",
        new="        if retrieval:",
        tests="test_a_session_that_read_docs_by_hand_is_silence",
    ),
    Mutation(
        name="usage: call any Markdown read a same-file fallback",
        module="usage_from_transcripts.py",
        area="scripts",
        old="        if any(_read_path(later) in hits for later in reads):",
        new="        if reads:",
        tests="test_a_read_of_another_file_is_not_a_same_file_fallback",
    ),
    Mutation(
        name="usage: read a Grep's search pattern as a file it searched",
        module="usage_from_transcripts.py",
        area="scripts",
        old='_markdown_path(arguments.get(key), call.project) for key in ("path", "glob")',
        new='_markdown_path(json.dumps(arguments), call.project) for key in ("path", "glob")',
        tests="test_only_fields_that_name_files_count",
    ),
    Mutation(
        name="usage: count an agent's own notes as documentation",
        module="usage_from_transcripts.py",
        area="scripts",
        old="    return any(path == root or path.startswith(root + os.sep) for root in SCRATCH)",
        new="    return False",
        tests="test_an_agents_own_notes_are_not_documentation",
    ),
    Mutation(
        name="usage: count reading the file you edit as looking something up",
        module="usage_from_transcripts.py",
        area="scripts",
        old="edited = {_read_path(call) for call in calls if call.name in EDIT_TOOLS}",
        new="edited = {_read_path(call) for call in calls if False}",
        tests=(
            "test_reading_the_file_you_edit_is_maintenance"
            " or test_an_edit_in_a_subagent_counts_for_the_session"
        ),
    ),
    Mutation(
        name="usage: count indexing as looking something up",
        module="usage_from_transcripts.py",
        area="scripts",
        old="        retrieval = [call for call in calls if call.tool in RETRIEVAL_TOOLS]",
        new="        retrieval = [call for call in calls if call.tool]",
        tests="test_indexing_is_not_looking_anything_up",
    ),
    Mutation(
        name="usage: count listing Markdown file names as reading them",
        module="usage_from_transcripts.py",
        area="scripts",
        old="if _reads_markdown(call) and _read_path(call) not in edited",
        new="if _touches_markdown(call) and _read_path(call) not in edited",
        tests="test_listing_markdown_files_reads_none",
    ),
    Mutation(
        name="usage: call every result that says is_error a failure",
        module="usage_from_transcripts.py",
        area="scripts",
        old='                        failed=part.get("is_error") is True,',
        new='                        failed="is_error" in part,',
        tests="test_the_value_of_is_error_decides_a_failure",
    ),
    Mutation(
        name="usage: compare paths as they were typed",
        module="usage_from_transcripts.py",
        area="scripts",
        old="    return os.path.normpath(path)",
        new="    return path",
        tests="test_a_relative_read_matches_the_absolute_hit",
    ),
    Mutation(
        name="usage: forget the manual read after a failed search",
        module="usage_from_transcripts.py",
        area="scripts",
        old="        elif any(_touches_markdown(later) for later in window):",
        new="        elif page is not None and any(_touches_markdown(later) for later in window):",
        tests="test_a_manual_read_after_a_failed_search_still_counts",
    ),
    Mutation(
        name="usage: miss a reformulation chain of two",
        module="usage_from_transcripts.py",
        area="scripts",
        old="            if chain >= 2:",
        new="            if chain >= 3:",
        tests="test_reformulation_chains_are_counted_apart",
    ),
    Mutation(
        name="usage: leave the failures out of the JSON report",
        module="usage_from_transcripts.py",
        area="scripts",
        old='            "failures": self.failures,',
        new='            "failures": 0,',
        tests="test_the_json_report_carries_the_estimator_and_failures",
    ),
    Mutation(
        name="usage: ignore the name the server was registered under",
        module="usage_from_transcripts.py",
        area="scripts",
        old="    if arguments.server:",
        new="    if False:",
        tests="test_a_server_registered_under_another_name",
    ),
    Mutation(
        name="preflight: call an outline after a search a retreat to the file system",
        module="preflight.py",
        area="scripts",
        old='if any(step in miner.FILE_TOOLS or step == "Bash" for step in',
        new='if any(step not in ("search_docs", "read_section") for step in',
        tests="test_an_outline_after_a_search_is_not_a_retreat",
    ),
    Mutation(
        name="usage: count a search rooted in the agent's notes as documentation",
        module="usage_from_transcripts.py",
        area="scripts",
        old="    return isinstance(path, str) and _scratch(_norm(path, call.project))",
        new="    return False",
        tests="test_an_agents_own_notes_are_not_documentation",
    ),
    Mutation(
        name="usage: call reading back a scratch hit a fallback",
        module="usage_from_transcripts.py",
        area="scripts",
        old="if later.name in READ_TOOLS and _reads_markdown(later)]",
        new="if later.name in READ_TOOLS]",
        tests="test_reading_back_a_hit_in_scratch_is_not_a_fallback",
    ),
    Mutation(
        name="usage: print no failures whatever failed",
        module="usage_from_transcripts.py",
        area="scripts",
        old='    print(f"failed retrieval calls           {report.failures}")',
        new='    print(f"failed retrieval calls           {0}")',
        tests="test_the_json_report_carries_the_estimator_and_failures",
    ),
    Mutation(
        name="usage: lose a whole session to one half-written line",
        module="usage_from_transcripts.py",
        area="scripts",
        old=(
            "        except ValueError:\n"
            "            continue  # a transcript being written to can end mid-line"
        ),
        new="        except ValueError:\n            return Transcript(path.stem, [])",
        tests="test_a_half_written_line_does_not_lose_the_session",
    ),
    Mutation(
        name="units: throw away everything past the character limit again",
        module="parser.py",
        old="            for window in _windows(unit.text):",
        new="            for window in [unit.text[:MAX_UNIT_CHARS]]:",
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
        old="                about_to_write()\n                self._db.replace_document(\n",
        new=(
            "                pass  # rewrite the tree without saying the tree changed\n"
            "                self._db.replace_document(\n"
        ),
        tests="test_a_run_that_dies_partway_leaves_the_tree_unvouched_for",
    ),
    Mutation(
        name="coverage: retract before knowing whether anything will be written",
        module="indexer.py",
        old="            files = iter(\n",
        new="            about_to_write()\n            files = iter(\n",
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
        old="        self._root = str(headings._absolute(self._config.docs_dir, SearchError))",
        new="        self._root = str(self._config.docs_dir)",
        tests="test_a_symlinked_docs_root_still_answers",
    ),
    Mutation(
        name="vectors: keep the old truncated section vectors across the format change",
        module="indexer.py",
        old="            and known[:2] == (content_hash, VECTOR_FORMAT)\n",
        new="            and known[0] == content_hash\n",
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
            "            elif not discovery._is_walkable("
            "os.path.relpath(path, root).split(os.sep)):\n"
        ),
        new="            elif False:\n",
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
        name="coverage: count a revocation from before the run against it",
        module="indexer.py",
        old="            generation = self._db.generation()\n            identity =",
        new="            generation = 0\n            identity =",
        tests="test_a_clean_run_after_an_earlier_revocation_vouches_for_the_tree",
    ),
    Mutation(
        name="index: ask whether a pruned directory's parent was walkable",
        module="indexer.py",
        old=(
            "            elif not discovery._is_walkable("
            "os.path.relpath(path, root).split(os.sep)):\n"
        ),
        new=(
            "            elif not discovery._is_walkable("
            "os.path.relpath(path, root).split(os.sep)[:-1]):\n"
        ),
        tests="test_a_pruned_directory_that_could_not_be_listed_keeps_its_own_failure",
    ),
    Mutation(
        name="scope: keep a disowned failure holding the root's coverage back for good",
        module="indexer.py",
        old="            elif scope.excludes(path):\n",
        new="            elif False:\n",
        tests="test_a_failure_inside_an_excluded_directory_is_disowned_by_a_parent_scan",
    ),
    Mutation(
        name="scope: let an inner root vouch for itself after losing its recorded fault",
        module="indexer.py",
        old="            if disowned:\n",
        new="            if False:\n",
        tests="test_a_failure_inside_an_excluded_directory_is_disowned_by_a_parent_scan",
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
        old=("        return self._with_freshness(self._db.index_status(self._root), self._root)"),
        new=(
            "        return self._with_freshness(self._db.index_status("
            "str(headings._absolute(self._config.docs_dir, SearchError))), self._root)"
        ),
        tests="test_status_and_search_always_describe_the_same_tree",
    ),
    Mutation(
        name="index: purge what a walk could not see behind a symlink",
        module="indexer.py",
        old="            if discovery._behind_symlink(root, file_path):\n                continue",
        new="            if False:\n                continue",
        tests="test_a_directory_that_became_a_symlink_costs_nothing",
    ),
    Mutation(
        name="index: clear failures behind a symlink this walk never followed",
        module="indexer.py",
        old=(
            "        return not (discovery._behind_symlink(root, path) "
            "or discovery._is_shadowing_symlink(path))\n"
        ),
        new="        return not discovery._is_shadowing_symlink(path)\n",
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
        old=("        return self._with_freshness(self._db.index_status(self._root), self._root)"),
        new="        return IndexStatus(verified=True)",
        tests="test_a_search_says_the_index_is_missing_files",
    ),
    Mutation(
        name="index: keep a failure standing for a file the walk could not see and that is gone",
        module="indexer.py",
        old="            if discovery._certainly_gone(root, path):\n",
        new="            if False:\n",
        tests="test_a_failure_out_of_the_walk_s_reach_goes_when_the_file_does",
    ),
    Mutation(
        name="index: keep answering from a deleted file the walk could not see",
        module="indexer.py",
        old="            if discovery._certainly_gone(root, file_path):",
        new="            if False:",
        tests="test_a_deleted_document_inside_a_pruned_tree_stops_answering",
    ),
    Mutation(
        name="index: read a directory it may not list as an empty one",
        module="discovery.py",
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
        module="discovery.py",
        old="    if _behind_symlink(root, path) or os.path.islink(path):\n        return False",
        new="    if False:\n        return False",
        tests="test_a_directory_replaced_by_a_broken_symlink_keeps_its_documents",
    ),
    Mutation(
        name="index: clear a failure on a directory the walk only saw the name of",
        module="indexer.py",
        old=(
            "        return not (discovery._behind_symlink(root, path) "
            "or discovery._is_shadowing_symlink(path))\n"
        ),
        new="        return not discovery._behind_symlink(root, path)\n",
        tests="test_a_failure_on_a_directory_that_became_a_symlink_outlives_the_swap",
    ),
    Mutation(
        name="index: take a name it could not decode for a file that is gone",
        module="discovery.py",
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
        module="config.py",
        old="            db.expanduser()\n            if db\n            else configured_db",
        new="            configured_db",
        tests="test_an_explicitly_configured_database_still_wins",
    ),
    Mutation(
        name="config: index the trees the operator excluded",
        module="config.py",
        old="        exclude=tuple(exclude) or base.exclude,",
        new="        exclude=tuple(exclude),",
        tests="test_exclusions_are_inherited_rather_than_dropped "
        "or test_the_script_resolves_its_configuration_the_same_way",
    ),
    Mutation(
        name="parallel: write files in whatever order the workers finish",
        module="indexer.py",
        old="                        file_path, future = pending.popleft()",
        new="                        file_path, future = pending.pop()",
        tests="test_several_workers_build_exactly_the_index_one_worker_builds",
    ),
    Mutation(
        name="parallel: read the whole tree ahead of the writes",
        module="indexer.py",
        old="            window = max(2 * self._workers, 2)",
        new="            window = 1_000_000",
        tests="test_the_driver_reads_ahead_by_a_bounded_window_not_by_the_whole_tree",
    ),
    Mutation(
        name="weights: settle the provenance again for every file",
        module="indexer.py",
        old="                    weights_settled = True",
        new="                    pass  # ask again, once per file, while the run is writing",
        tests="test_the_weights_of_a_run_are_settled_once_however_many_workers_embed",
    ),
    Mutation(
        name="status: say nothing of a renamed model that cannot name its weights",
        module="server.py",
        old="            renamed = unnamed_rename(self._db, self._embedder)\n",
        new="            renamed = None\n",
        tests="test_a_renamed_model_that_cannot_name_its_weights_is_reported_while_it_lasts",
    ),
    Mutation(
        name="status: derive the rename for the root only, not for a directory",
        module="server.py",
        old=(
            "            self._with_freshness("
            "self._db.index_status(self._root, str(scope)), str(scope))\n"
        ),
        new=(
            "            dataclasses.replace(self._db.index_status(self._root, str(scope)), "
            "changed_files=self._freshness.changed_files(str(scope)))\n"
        ),
        tests="test_a_renamed_model_that_cannot_name_its_weights_is_reported_while_it_lasts",
    ),
    Mutation(
        name="freshness: never look at the disk",
        module="server.py",
        old=(
            "        status = dataclasses.replace("
            "status, changed_files=self._freshness.changed_files(scope))"
        ),
        new="        status = status",
        tests="test_an_edited_document_is_reported_without_unverifying_the_walk",
    ),
    Mutation(
        name="freshness: call a file changed because its timestamp moved",
        module="freshness.py",
        old="    return _Verdict.SAME_BYTES_NEW_TIME, info.st_mtime_ns",
        new="    return _Verdict.CHANGED, info.st_mtime_ns  # a touch is a change",
        tests="test_a_touch_that_changes_no_byte_is_not_a_change",
    ),
    Mutation(
        name="freshness: read a missing modification time as the epoch",
        module="freshness.py",
        old="    if mtime_ns is not None and info.st_mtime_ns == mtime_ns:",
        new="    if info.st_mtime_ns == (mtime_ns or 0):  # nothing recorded is the epoch",
        tests="test_a_row_from_before_nanoseconds_were_recorded_is_answered_by_its_bytes",
    ),
    Mutation(
        name="freshness: sweep the filesystem on every single query",
        module="freshness.py",
        old="                cached is not None",
        new="                False  # measure it again, several times per conversational turn",
        tests="test_the_sweep_speaks_for_a_few_seconds_rather_than_per_query",
    ),
    Mutation(
        name="freshness: store no modification time to compare against",
        module="indexer.py",
        old="                    mtime_ns=prepared.mtime_ns,",
        new="                    mtime_ns=0,",
        tests="test_indexing_records_the_modification_time_it_read",
    ),
    Mutation(
        name="no-answer: resolve a label for a case that has none",
        module="eval_retrieval.py",
        area="scripts",
        old="        for kind in _kinds(queries, split):  # no-answer cases",
        new="        for kind in queries[split]:  # no-answer cases",
        tests="test_resolving_labels_skips_cases_that_have_none",
        fails_with="KeyError",
    ),
    Mutation(
        name="no-answer: score a fixture without checking its unanswerable queries",
        module="eval_retrieval.py",
        area="scripts",
        old="        check_no_answer(queries, corpus.root, splits)",
        new="        pass",
        tests="test_a_no_answer_query_the_corpus_contains_stops_the_run_before_any_search",
    ),
    Mutation(
        name="no-answer: let a query the corpus contains pass for unanswerable",
        module="eval_retrieval.py",
        area="scripts",
        old="            if any(words[i : i + width] == phrase",
        new="            if False and any(words[i : i + width] == phrase",
        tests="test_the_fixture_guard_finds_a_query_the_corpus_contains",
    ),
    Mutation(
        name="no-answer: accept a malformed no-answer case",
        module="eval_retrieval.py",
        area="scripts",
        old='        if "expected" in case or case.get("shape")',
        new="        if False and case.get('shape')",
        tests="test_the_fixture_guard_rejects_a_malformed_case",
    ),
    Mutation(
        name="no-answer: read the corpus when there is nothing to check",
        module="eval_retrieval.py",
        area="scripts",
        old="    if not cases:\n        return\n    documents",
        new="    documents",
        tests="test_the_fixture_guard_reads_no_corpus_without_cases",
        fails_with="FileNotFoundError",
    ),
    Mutation(
        name="no-answer: count no_match from the wrong state",
        module="eval_retrieval.py",
        area="scripts",
        old='                no_match=tuple(page.keyword_match == "no_match" for page in pages),',
        new='                no_match=tuple(page.keyword_match == "matched" for page in pages),',
        tests="test_the_default_call_is_measured_per_shape",
    ),
    Mutation(
        name="no-answer: count a page with hits as an abstention",
        module="eval_retrieval.py",
        area="scripts",
        old="            abstained = sum(hits == 0 for hits in result.hits) / n",
        new="            abstained = sum(hits > 0 for hits in result.hits) / n",
        tests="test_the_report_counts_an_empty_page_as_abstaining",
    ),
    Mutation(
        name="no-answer: print a rate over no cases",
        module="eval_retrieval.py",
        area="scripts",
        old="            if not result.queries:\n                continue",
        new="            if False:\n                continue",
        tests="test_the_report_counts_an_empty_page_as_abstaining",
        fails_with="ZeroDivisionError",
    ),
    Mutation(
        name="no-answer: list an abstention as a miss",
        module="eval_retrieval.py",
        area="scripts",
        old="                if hits:\n",
        new="                if True:\n",
        tests="test_the_report_counts_an_empty_page_as_abstaining",
    ),
    Mutation(
        name="no-answer: let a failed no-answer pass decide the exit code",
        module="eval_retrieval.py",
        area="scripts",
        old="        except Exception as exc:  # informational, like the cost pass below",
        new="        except ZeroDivisionError as exc:  # informational, like the cost pass below",
        tests="test_a_no_answer_pass_that_fails_does_not_decide_the_exit_code",
        fails_with="RuntimeError",
    ),
    Mutation(
        name="baseline: print a delta against a baseline scored on other queries",
        module="eval_retrieval.py",
        area="scripts",
        old="        if scored_on != cases_sha256(queries, name):",
        new="        if False:",
        tests="test_a_delta_is_printed_only_against_the_same_cases "
        "or test_another_label_file_gets_no_delta_for_a_set_it_changes",
    ),
    Mutation(
        name="baseline: take a baseline that names no queries for one scored on these",
        module="eval_retrieval.py",
        area="scripts",
        old='        scored_on = before.get("cases_sha256")',
        new='        scored_on = before.get("cases_sha256", cases_sha256(queries, name))',
        tests="test_a_delta_is_printed_only_against_the_same_cases",
    ),
    Mutation(
        name="baseline: record numbers without the queries they were scored on",
        module="eval_retrieval.py",
        area="scripts",
        old='            "cases_sha256": cases_sha256(queries, name),\n',
        new="",
        tests="test_a_recorded_baseline_holds_no_cost",
    ),
    Mutation(
        name="baseline: let key order change a set's fingerprint",
        module="eval_retrieval.py",
        area="scripts",
        old="json.dumps(queries[split][kind], sort_keys=True)",
        new="json.dumps(queries[split][kind])",
        tests="test_the_fingerprint_follows_every_field_of_a_case_and_no_key_order",
    ),
    Mutation(
        name="abstain: miss a camelCase identifier",
        module="search.py",
        old="        or any(lower.islower() and upper.isupper()",
        new="        or any(False and upper.isupper()",
        tests="test_an_absent_identifier_is_no_match_and_abstains",
    ),
    Mutation(
        name="abstain: read a backticked flag as a word",
        module="search.py",
        old="""    term = quoted_term.strip("\\"'`")""",
        new="""    term = quoted_term.strip('"')""",
        tests="test_an_identifier_lookup_nothing_contains_abstains",
    ),
    Mutation(
        name="abstain: hand out neighbours for an identifier nothing contains",
        module="search.py",
        old='        if keyword_match == "no_match" and _is_identifier_lookup(fts_terms(query)):',
        new="        if False:",
        tests="test_an_identifier_lookup_nothing_contains_abstains",
    ),
    Mutation(
        name="abstain: on any keyword state, not only no_match",
        module="search.py",
        old='        if keyword_match == "no_match" and _is_identifier_lookup(fts_terms(query)):',
        new="        if _is_identifier_lookup(fts_terms(query)):",
        tests=(
            "test_a_failed_keyword_index_never_abstains"
            " or test_candidates_the_gate_refuses_never_abstain"
        ),
    ),
    Mutation(
        name="abstain: when one term merely looks like an identifier",
        module="search.py",
        old="< _MAX_QUERY_TERMS and all(_is_identifier(term)",
        new="< _MAX_QUERY_TERMS and any(_is_identifier(term)",
        tests="test_anything_else_nothing_contains_keeps_its_neighbours",
    ),
    Mutation(
        name="abstain: speak for terms a cut query never searched",
        module="search.py",
        old="    return 0 < len(terms) < _MAX_QUERY_TERMS and",
        new="    return 0 < len(terms) <= _MAX_QUERY_TERMS and",
        tests="test_anything_else_nothing_contains_keeps_its_neighbours",
    ),
    Mutation(
        name="abstain: explain an empty page as neighbours",
        module="models.py",
        old='        if self.keyword_match == "no_match" and not self.results:',
        new="        if False:",
        tests="test_an_identifier_lookup_nothing_contains_abstains",
    ),
    Mutation(
        name="part_preview: look up no first passage",
        module="search.py",
        old="        firsts = self._db.first_passages(",
        new="        firsts = {} if True else self._db.first_passages(",
        tests="test_every_part_carries_its_first_passage_shortened",
    ),
    Mutation(
        name="part_preview: preview an unsplit section too",
        module="search.py",
        old="if hydrated[sid][0].part_index > 0])",
        new="if True])",
        tests="test_a_section_stored_whole_has_none",
    ),
    Mutation(
        name="part_preview: preview from the second passage",
        module="db.py",
        old='f"WHERE section_id IN ({placeholders}) AND ordinal = 0",',
        new='f"WHERE section_id IN ({placeholders}) AND ordinal = 1",',
        tests="test_every_part_carries_its_first_passage_shortened",
    ),
    Mutation(
        name="part_preview: leave it off the pointer",
        module="models.py",
        old='            pointer["part_preview"] = self.part_preview',
        new="            pass",
        tests="test_a_part_with_a_matched_passage_carries_both",
    ),
    Mutation(
        name="part_preview: pay for it beside rank 1's full text",
        module="models.py",
        old='                "content": self.content,',
        new='                "content": self.content, "part_preview": self.part_preview,',
        tests="test_every_part_carries_its_first_passage_shortened",
    ),
    Mutation(
        name="part_preview: cut mid-word",
        module="models.py",
        old="    cut = next((i for i in range(limit, 0, -1) if text[i].isspace()), 0)",
        new="    cut = 0",
        tests="test_longer_text_is_cut_between_words",
    ),
    Mutation(
        name="part_preview: leave nothing when there is nowhere to cut",
        module="models.py",
        old='    return text[: cut or limit].rstrip() + "…"',
        new='    return text[:cut].rstrip() + "…"',
        tests="test_text_with_nowhere_to_cut_is_cut_at_the_limit",
    ),
    Mutation(
        name="part_preview: shorten text that already fits",
        module="models.py",
        old="    if len(text) <= limit:\n        return text",
        new="    if len(text) < limit:\n        return text",
        tests="test_text_that_fits_is_unchanged",
        fails_with="IndexError",
    ),
    Mutation(
        name="keyword_match: call a query with no searchable terms a miss",
        module="search.py",
        old='            return [], "no_terms"',
        new='            return [], "no_match"',
        tests="test_a_query_with_no_searchable_terms",
    ),
    Mutation(
        name="keyword_match: call a blank query a miss",
        module="search.py",
        old='            return SearchPage((), "no_terms")',
        new='            return SearchPage((), "no_match")',
        tests="test_a_query_with_no_searchable_terms",
    ),
    Mutation(
        name="keyword_match: hide that no section contains the terms",
        module="search.py",
        old='            return [], "no_match"',
        new='            return [], "filtered"',
        tests="test_an_absent_identifier_is_no_match_and_abstains",
    ),
    Mutation(
        name="keyword_match: say nothing contains terms the gate merely refused",
        module="search.py",
        old='            return [], "filtered"',
        new='            return [], "no_match"',
        tests="test_candidates_the_gate_refuses_are_filtered_not_no_match",
    ),
    Mutation(
        name="keyword_match: report a matched keyword search as a miss",
        module="search.py",
        old='        return [hit for hit in hits if hit in with_body] or hits, "matched"',
        new='        return [hit for hit in hits if hit in with_body] or hits, "no_match"',
        tests="test_a_present_identifier_matched",
    ),
    Mutation(
        name="keyword_match: pass off a failed keyword index as a match",
        module="search.py",
        old='_settle(fts_future, _Keyword([], "unavailable"))',
        new='_settle(fts_future, _Keyword([], "matched"))',
        tests="test_a_failed_keyword_index_is_unavailable_while_vectors_answer",
    ),
    Mutation(
        name="identifiers: check only the first twenty keyword hits for the identifier",
        module="search.py",
        old="LITERAL_CANDIDATES = 200",
        new="LITERAL_CANDIDATES = 20",
        tests="test_a_literal_beyond_the_first_twenty_keyword_hits_is_found",
    ),
    Mutation(
        name="identifiers: find --pre inside --pre-glob",
        module="search.py",
        old='_IDENTIFIER_EDGE = "A-Za-z0-9_-"',
        new='_IDENTIFIER_EDGE = "A-Za-z0-9_"',
        tests="test_an_identifier_is_found_as_itself_and_not_inside_another",
    ),
    Mutation(
        name="identifiers: find v1.2 inside v1.2.3",
        module="search.py",
        old=(
            '        tail = r"(?=\\s*\\()" if call else '
            'rf"(?![{_IDENTIFIER_EDGE}])(?!\\.[A-Za-z0-9_])"'
        ),
        new='        tail = r"(?=\\s*\\()" if call else rf"(?![{_IDENTIFIER_EDGE}])"',
        tests="test_an_identifier_is_found_as_itself_and_not_inside_another",
    ),
    Mutation(
        name="identifiers: read name() as the literal text, parentheses and all",
        module="search.py",
        old='        call = len(term) > 2 and term.endswith("()")',
        new="        call = False",
        tests=(
            "test_an_identifier_is_found_as_itself_and_not_inside_another"
            " or test_plain_call_matches_arguments_at_identifier_boundaries"
        ),
    ),
    Mutation(
        name="calls: omit plain-call identifier recognition",
        module="search.py",
        old="        or _is_plain_call(term)\n",
        new="",
        tests="test_plain_call_definition_wins",
    ),
    Mutation(
        name="calls: search a call written with arguments as the whole expression",
        module="search.py",
        old="        _called(term)\n",
        new="        term\n",
        tests="test_a_call_written_with_arguments_is_answered_by_its_definition",
    ),
    Mutation(
        name="calls: read any parenthesised word as a call",
        module="search.py",
        old="        or not any(\n",
        new="        or False and not any(\n",
        tests="test_plain_call_syntax_does_not_expand_other_queries",
    ),
    Mutation(
        name="calls: miss a call written in backticks or ending a question",
        module="search.py",
        old="    call = _CALL_WITH_ARGUMENTS.fullmatch(_unwrapped(term))\n",
        new="    call = _CALL_WITH_ARGUMENTS.fullmatch(term)\n",
        tests="test_a_call_written_with_arguments_is_answered_by_its_definition",
    ),
    Mutation(
        name="calls: miss a call whose closing parenthesis is in another piece",
        module="search.py",
        old='_CALL_WITH_ARGUMENTS = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\\((.+)")',
        new='_CALL_WITH_ARGUMENTS = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\\((.+)\\)")',
        tests="test_a_call_written_with_arguments_is_searched_as_the_call",
    ),
    Mutation(
        name="calls: rank an argument call against the call it is an argument of",
        module="search.py",
        old='        or call[2].count(")") > call[2].count("(") + 1\n',
        new="",
        tests=(
            "test_a_call_written_with_arguments_is_searched_as_the_call"
            " or test_a_call_written_with_arguments_is_answered_by_its_definition"
        ),
    ),
    Mutation(
        name="calls: discard stopword-named calls",
        module="search.py",
        old="if _is_plain_call(term) or not word.isalpha()",
        new="if not word.isalpha()",
        tests="test_stopword_call_survives_mixed_query",
    ),
    Mutation(
        name="identifiers: miss gh_repo when GH_REPO is asked for",
        module="search.py",
        old="        self._loose = re.compile(pattern, re.IGNORECASE)",
        new="        self._loose = re.compile(pattern)",
        tests="test_exact_case_is_told_apart_from_a_case_folded_match",
    ),
    Mutation(
        name="identifiers: forget the section headed by the identifier",
        module="search.py",
        old="                not heads,",
        new="                False,",
        tests="test_the_section_headed_by_the_identifier_comes_first_and_its_first_part_first",
    ),
    Mutation(
        name="identifiers: rank the parts of the headed section by BM25 alone",
        module="search.py",
        old="                section.part_index if heads else 0,",
        new="                0,",
        tests="test_the_section_headed_by_the_identifier_comes_first_and_its_first_part_first",
    ),
    Mutation(
        name="identifiers: ignore the spelling asked for",
        module="search.py",
        old="                not exact,",
        new="                False,",
        tests="test_the_spelling_asked_for_ranks_first_among_sections_naming_it",
    ),
    Mutation(
        name="ties: break an exact fused tie by section id again",
        module="search.py",
        old=(
            "        ordered = self._in_order("
            "{section_id: -score for section_id, score in scores.items()})\n"
        ),
        new="        ordered = sorted(scores, key=lambda sid: (-scores[sid], sid))\n",
        tests="test_a_tie_between_documents_survives_an_edit",
    ),
    Mutation(
        name="ties: break an exact vector-distance tie by section id again",
        module="search.py",
        old="        ranking = self._in_order(best)[:limit]\n",
        new="        ranking = sorted(best, key=lambda sid: (best[sid], sid))[:limit]\n",
        tests="test_a_vector_distance_tie_goes_to_the_walk_up_to_the_cut",
    ),
    Mutation(
        name="ties: forget where in its document a tied section is",
        module="search.py",
        old="            return values[sid], (0, walk_order(file_path), start_line, part_index)\n",
        new="            return values[sid], (0, walk_order(file_path))\n",
        tests="test_a_tie_inside_one_document_goes_to_the_earlier_section",
    ),
    Mutation(
        name="ties: forget which part of a cut line a tied section is",
        module="search.py",
        old="            return values[sid], (0, walk_order(file_path), start_line, part_index)\n",
        new="            return values[sid], (0, walk_order(file_path), start_line)\n",
        tests="test_a_tie_between_parts_of_one_line_goes_to_the_earlier_part",
    ),
    Mutation(
        name="ties: let a section gone since the ranking win its tie on its id",
        module="search.py",
        old="                return values[sid], (1,)\n",
        new="                return values[sid], (0, (), sid, 0)\n",
        tests="test_a_section_gone_since_the_ranking_goes_after_its_tie",
    ),
    Mutation(
        name="ties: read every candidate's position, tied or not",
        module="search.py",
        old="        positions = self._db.section_positions(tied) if tied else {}\n",
        new="        positions = self._db.section_positions(list(values))\n",
        tests="test_positions_are_read_only_when_something_ties",
    ),
    Mutation(
        name="ties: put a directory's sub-directories before its own files",
        module="discovery.py",
        old="(0, False, name))\n",
        new="(2, False, name))\n",
        tests=(
            "test_walk_order_is_the_order_of_the_walk"
            " or test_a_fresh_build_numbers_sections_in_walk_order"
        ),
    ),
    Mutation(
        name="identifiers: let a vector-only neighbour win the tie on its place in the walk",
        module="search.py",
        old="        if keyword.literal:",
        new="        if False:",
        tests="test_the_section_naming_the_identifier_beats_prose_holding_its_words",
    ),
    Mutation(
        name="identifiers: let mentions with vector support push the headed section down",
        module="search.py",
        old=(
            "                *keyword.headings,\n"
            "                *(sid for sid in ordered "
            "if sid in keyword.literal - set(keyword.headings)),"
        ),
        new="                *(sid for sid in ordered if sid in keyword.literal),",
        tests="test_the_section_headed_by_the_identifier_comes_first_and_its_first_part_first",
    ),
    Mutation(
        name="identifiers: treat a term found everywhere as an identifier",
        module="search.py",
        old="            if found[literal] and len(found[literal]) * spread(term) <= rare",
        new="            if found[literal]",
        tests="test_an_identifier_named_everywhere_is_vocabulary",
    ),
    Mutation(
        name="identifiers: judge rarity by the candidates checked alone",
        module="search.py",
        old="            if found[literal] and len(found[literal]) * spread(term) <= rare",
        new="            if found[literal] and len(found[literal]) <= rare",
        tests="test_a_term_common_beyond_the_candidates_checked_is_vocabulary",
    ),
    Mutation(
        name="identifiers: judge rarity by the keyword index's count alone",
        module="search.py",
        old="            if found[literal] and len(found[literal]) * spread(term) <= rare",
        new="            if found[literal] and frequencies[term] <= rare",
        tests="test_an_identifier_whose_words_are_common_is_still_rare",
    ),
    Mutation(
        name="identifiers: let one vocabulary term turn off the lookup of the others",
        module="search.py",
        old="        if not wanted:",
        new="        if len(wanted) < len(literals):",
        tests="test_rarity_is_judged_per_term",
    ),
    Mutation(
        name="identifiers: hide a candidate that vanished while it was checked",
        module="search.py",
        old="        stale = len(texts) < len(ranking)",
        new="        stale = False",
        tests="test_a_section_vanishing_while_checked_asks_for_a_second_pass",
    ),
    Mutation(
        name="identifiers: forget a vanished candidate when nothing left names the identifier",
        module="search.py",
        old=(
            "                stale = literal.stale  "
            "# a vanished candidate may have been the one naming it"
        ),
        new="                pass",
        tests="test_losing_the_only_section_naming_it_still_asks_for_a_second_pass",
    ),
    Mutation(
        name="identifiers: sample a term over the whole pool, not the candidates matching it",
        module="search.py",
        old="            checked = max(1, len(self._db.fts_matching(term, list(texts))))",
        new="            checked = len(texts)",
        tests="test_a_term_is_sampled_among_the_candidates_matching_it_not_the_whole_pool",
    ),
    Mutation(
        name="identifiers: count another root's sections towards this one's rarity",
        module="search.py",
        old="            return max(1.0, frequencies[term] / checked)",
        new="            return max(1.0, self._db.term_counts([term], None)[1][term] / checked)",
        tests="test_rarity_in_one_root_is_not_judged_by_another_roots_words",
    ),
    Mutation(
        name="rarity: size the lookup's limit from every root in the database",
        module="search.py",
        old="        rare = _rare(total)\n        literals = [_Literal(term) for term in terms]",
        new=(
            '        rare = _rare(self._db.count_rows("sections"))\n'
            "        literals = [_Literal(term) for term in terms]"
        ),
        tests="test_a_large_neighbour_does_not_make_a_common_identifier_one",
    ),
    Mutation(
        name="rarity: size the gate's limit from every root in the database",
        module="search.py",
        old="        rare = _rare(total)\n        exact: set[int] = set()",
        new=(
            '        rare = _rare(self._db.count_rows("sections"))\n        exact: set[int] = set()'
        ),
        tests="test_a_large_neighbour_does_not_make_a_common_term_rare",
    ),
    Mutation(
        name="rarity: weigh a query's terms by every root's words",
        module="search.py",
        old="term_counts(terms, self._scope)\n        weights",
        new="term_counts(terms, None)\n        weights",
        tests="test_another_roots_words_do_not_reweigh_this_roots_query",
    ),
    Mutation(
        name="rarity: count by the index alone whatever else the database holds",
        module="db.py",
        old="                outside = scope is not None and bool(",
        new="                outside = False and bool(",
        tests="test_term_counts_are_a_roots_own_whatever_else_the_database_holds",
    ),
    Mutation(
        name="rarity: count a scope's matches in every root",
        module="db.py",
        old='"WHERE sections_fts MATCH ? AND substr(d.file_path, 1, length(?)) = ?",',
        new='"WHERE sections_fts MATCH ? AND ? = ?",',
        tests="test_term_counts_are_a_roots_own_whatever_else_the_database_holds",
    ),
    Mutation(
        name="rarity: count a scope's sections in every root",
        module="db.py",
        old='                        "WHERE substr(d.file_path, 1, length(?)) = ?",',
        new='                        "WHERE ? = ?",',
        tests="test_term_counts_are_a_roots_own_whatever_else_the_database_holds",
    ),
    Mutation(
        name="identifiers: keep a backtick a sentence's full stop left behind",
        module="search.py",
        old='        term = term.rstrip("?!,;:").removesuffix(".").strip("\'`\\"")',
        new='        term = term.rstrip("?!,;:").removesuffix(".")',
        tests="test_an_identifier_is_found_as_itself_and_not_inside_another",
    ),
    Mutation(
        name="identifiers: miss a heading that quotes only the name of a call",
        module="search.py",
        old='    return text.replace("`", "").strip().removesuffix("()").strip().lower()',
        new='    return text.strip().strip("`").strip().removesuffix("()").lower()',
        tests="test_a_heading_naming_the_call_heads_it_however_it_is_quoted",
    ),
    Mutation(
        name="identifiers: return an empty page from a pass that lost sections as final",
        module="search.py",
        old="            return SearchPage((), keyword_match), keyword.stale",
        new="            return SearchPage((), keyword_match), False",
        tests="test_an_empty_page_from_a_pass_that_lost_sections_is_ranked_again",
    ),
    Mutation(
        name="identifiers: keep the backtick of a call quoted as `name`()",
        module="search.py",
        old=(
            '        self.name = term[:-2].strip("\'`\\"") if call else term  '
            "# `name`() quotes the name only"
        ),
        new="        self.name = term[:-2] if call else term",
        tests="test_an_identifier_is_found_as_itself_and_not_inside_another",
    ),
    Mutation(
        name="keyword_match: let a refused candidate claim nothing contains the terms",
        module="models.py",
        old='    "filtered": "The top keyword',
        new='    "filtered": "No section contains the terms. The top keyword',
        tests="test_only_no_match_says_nothing_contains_the_terms",
    ),
    Mutation(
        name="keyword_match: leave the state out of the response",
        module="server.py",
        old='            "keyword_match": page.keyword_match,',
        new="",
        tests="test_search_docs_returns_sections_and_breadcrumbs",
    ),
    Mutation(
        name="keyword_match: leave the explanation out of the response",
        module="server.py",
        old='            payload["keyword_message"] = message',
        new="            pass",
        tests="test_an_absent_identifier_abstains_and_says_why",
        fails_with="KeyError",
    ),
    Mutation(
        name="pointers: pay for the matched passage twice in the full result",
        module="models.py",
        old='                "content": self.content,',
        new='                "content": self.content, "matched_passage": self.matched_passage,',
        tests="test_best_passage_decides_and_is_reported",
    ),
    Mutation(
        name="pointers: send the section's text with every pointer",
        module="models.py",
        old="        return pointer",
        new='        pointer["content"] = self.content\n        return pointer',
        tests="test_hits_after_the_first_are_pointers_that_read_section_follows",
    ),
    Mutation(
        name="pointers: send the ranks with every pointer",
        module="models.py",
        old="        return pointer",
        new='        pointer["vec_rank"] = self.vec_rank\n        return pointer',
        tests="test_hits_after_the_first_are_pointers_that_read_section_follows",
    ),
    Mutation(
        name="pointers: drop why a pointer matched",
        module="models.py",
        old="        if self.matched_passage is not None:\n            # How the passage",
        new="        if False:\n            # How the passage",
        tests="test_best_passage_decides_and_is_reported",
        fails_with="KeyError",
    ),
    Mutation(
        name="pointers: report a passage for a hit no passage won",
        module="models.py",
        old="        if self.matched_passage is not None:\n            # How the passage",
        new="        if True:\n            # How the passage",
        tests="test_result_without_a_winning_passage_omits_the_field",
        fails_with="TypeError",  # preview(None): the field is built for a hit that has none
    ),
    Mutation(
        name="pointers: return every hit in full",
        module="server.py",
        old="                _relative(hit.to_dict() if rank == 0 else hit.to_pointer(), root)",
        new="                _relative(hit.to_dict(), root)",
        tests="test_hits_after_the_first_are_pointers_that_read_section_follows",
    ),
    Mutation(
        name="pointers: return even the best hit as a pointer",
        module="server.py",
        old="                _relative(hit.to_dict() if rank == 0 else hit.to_pointer(), root)",
        new="                _relative(hit.to_pointer(), root)",
        tests="test_the_tool_keeps_the_order_and_the_limit_the_service_gives_it",
    ),
    Mutation(
        name="freshness: keep quiet about documents that moved on",
        module="models.py",
        old="            if self.changed_files:",
        new="            if False:  # a verified walk has nothing left to say",
        tests="test_an_edited_document_is_reported_without_unverifying_the_walk",
    ),
    Mutation(
        name="freshness: leave the count out of the envelope",
        module="models.py",
        old='            "changed_files": self.changed_files,',
        new="",
        tests="test_a_clean_index_of_an_untouched_tree_stays_quiet",
        fails_with="KeyError",
    ),
    Mutation(
        name="parallel: run the workers one at a time",
        module="indexer.py",
        old="                max_workers=self._workers,",
        new="                max_workers=1,",
        tests="test_the_workers_really_do_embed_at_the_same_time",
        fails_with="threading.BrokenBarrierError",
    ),
    Mutation(
        name="freshness: keep reporting what the last sweep found after a re-index",
        module="server.py",
        old="            self._freshness.invalidate()",
        new="            pass  # the sweep still speaks for the tree it measured",
        tests="test_indexing_forgets_what_the_last_sweep_found",
    ),
    Mutation(
        name="freshness: answer one scope's question with another scope's sweep",
        module="freshness.py",
        old="                and cached[0] == scope",
        new="                and True  # whatever it swept, it answers for",
        tests="test_a_narrowed_status_sweeps_the_directory_it_was_asked_about",
    ),
    Mutation(
        name="parallel: let one rejected document end the whole run",
        module="indexer.py",
        old="                                store(prepared)",
        new="                                pass  # the run dies with the document",
        tests="test_a_document_the_storage_layer_rejects_fails_alone",
    ),
    Mutation(
        name="freshness: publish a sweep of a tree that changed under it",
        module="freshness.py",
        old="        self._lock = threading.Lock()",
        new="        self._lock = threading.Semaphore(8)  # not mutual exclusion",
        tests="test_a_sweep_and_an_invalidation_cannot_overlap",
    ),
    Mutation(
        name="freshness: leave a skipped file with the timestamp it was stored with",
        module="indexer.py",
        old="            if known[2] == info.st_mtime_ns:",
        new="            if True:  # the time it was stored with is time enough",
        tests="test_a_file_whose_bytes_did_not_change_still_has_its_timestamp_brought_up_to_date",
    ),
    Mutation(
        name="freshness: hash a touched file again on every sweep",
        module="freshness.py",
        old=(
            "                    self._db.record_modification_time("
            "file_path, content_hash, mtime_ns, seen_ns)"
        ),
        new="                    pass  # hash it again next window, and the one after",
        tests="test_a_touch_is_hashed_once_and_then_written_down",
    ),
    Mutation(
        name="freshness: stamp a verified time onto whatever the row holds now",
        module="db.py",
        old='                "WHERE file_path = ? AND content_hash = ? AND mtime_ns IS ?",',
        new='                "WHERE file_path = ? AND ? IS NOT NULL AND ? IS NOT NULL",',
        tests="test_the_write_back_refuses_to_stamp_a_time_onto_somebody_else_s_content",
    ),
    Mutation(
        name="migration: invent a modification time for rows that never had one",
        module="db.py",
        old='                    tx.execute("ALTER TABLE documents ADD COLUMN mtime_ns INTEGER")',
        new=(
            '                    tx.execute("ALTER TABLE documents ADD COLUMN '
            'mtime_ns INTEGER NOT NULL DEFAULT 0")'
        ),
        tests="test_a_v4_database_keeps_its_documents_and_learns_to_time_them",
    ),
    Mutation(
        name="trees: let cwd decide over the path a call names",
        module="server.py",
        old="        probe = _probe(cwd, path)",
        new="        probe = _probe(cwd, None)",
        tests="test_a_worktree_path_routes_there_without_cwd",
    ),
    Mutation(
        name="trees: answer another repository from its own copy of the docs path",
        module="server.py",
        old="other.common_dir != home.common_dir",
        new="other.common_dir is None",
        tests="test_cwd_in_the_primary_or_anywhere_else_answers_from_the_configured_root",
    ),
    Mutation(
        name="trees: give the configured root's own checkout a second index",
        module="server.py",
        old="        if other.top == home.top:",
        new="        if False:",
        tests="test_cwd_in_the_primary_or_anywhere_else_answers_from_the_configured_root",
    ),
    Mutation(
        name="trees: let every tree run an index at the same time",
        module="server.py",
        old="                run_lock=configured.run_lock,",
        new="                run_lock=None,",
        tests="test_concurrent_first_calls_for_one_tree_share_one_service",
    ),
    Mutation(
        name="trees: embed another tree from scratch",
        module="server.py",
        old="                if donor is None",
        new="                if True",
        tests="test_unchanged_passages_are_not_embedded_again",
    ),
    Mutation(
        name="trees: leave other trees' services open after the session",
        module="server.py",
        old="        for service in closing:\n            service.close()",
        new="        for service in closing:\n            pass",
        tests="test_the_two_indexes_never_hold_each_others_paths",
    ),
    Mutation(
        name="trees: answer for any number of trees",
        module="server.py",
        old="            if len(self._trees) >= trees.MAX_TREES:",
        new="            if False:",
        tests="test_one_server_answers_for_a_bounded_number_of_trees",
    ),
    Mutation(
        name="trees: leave out which root answered",
        module="server.py",
        old="dataclasses.replace(status, indexing=active, root=self._root)",
        new="dataclasses.replace(status, indexing=active)",
        tests="test_cwd_in_a_worktree_answers_from_that_worktree",
    ),
    Mutation(
        name="trees: resolve a relative path without the agent's cwd",
        module="server.py",
        old="[Path(self._root) / path, Path(cwd.strip()).expanduser() / path]",
        new="[Path(self._root) / path]",
        tests="test_a_relative_path_is_resolved_where_the_agent_is",
    ),
    Mutation(
        name="trees: ask git with a hook's GIT_DIR still set",
        module="discovery.py",
        old="        env=git_environment(),",
        new="        env=None,",
        tests="test_a_git_dir_inherited_from_a_hook_does_not_misroute",
    ),
    Mutation(
        name="trees: call a bare repository an error rather than no work tree",
        module="trees.py",
        old='"must be run in a work tree"',
        new='"no such message"',
        tests="test_a_bare_repository_has_no_work_tree",
        fails_with="markdown_memory.exceptions.WorkTreeError",
    ),
    Mutation(
        name="trees: serve a tree that has no copy of the docs root",
        module="trees.py",
        old="    if not docs.is_dir():",
        new="    if False:",
        tests="test_a_tree_without_the_docs_directory_is_refused_with_the_reason",
    ),
    Mutation(
        name="trees: reuse vectors other weights produced",
        module="db.py",
        old="d.weights_revision = ? AND d.vector_format = ?",
        new="? IS NOT NULL AND d.vector_format = ?",
        tests="test_only_identical_input_under_identical_weights_and_format_is_reused",
    ),
    Mutation(
        name="trees: reuse vectors an older pooling scheme wrote",
        module="db.py",
        old="d.weights_revision = ? AND d.vector_format = ?",
        new="d.weights_revision = ? AND ? IS NOT NULL",
        tests="test_only_identical_input_under_identical_weights_and_format_is_reused",
    ),
    Mutation(
        name="trees: reuse a vector that does not decode to a usable one",
        module="db.py",
        old="                    if _is_usable_vector(vector):",
        new="                    if True:",
        tests="test_a_vector_that_does_not_decode_to_a_usable_one_is_embedded_instead",
    ),
    Mutation(
        name="trees: match vectors against weights nobody named",
        module="indexer.py",
        old="if self._reuse is not None and identity is not None and texts",
        new="if self._reuse is not None and texts",
        tests="test_weights_nobody_can_name_are_never_matched",
    ),
    Mutation(
        name="trees: put another tree's index wherever the default would",
        module="config.py",
        old="    if not config.db_explicit:",
        new="    if True:",
        tests="test_a_trees_database_is_beside_a_chosen_one_and_derived_otherwise",
    ),
    Mutation(
        name="trees: count a database the environment derived as chosen",
        module="config.py",
        old="db_explicit=db_path is not None,",
        new="db_explicit=True,",
        tests="test_only_a_database_the_environment_derived_counts_as_derived",
    ),
    Mutation(
        name="trees: count a database resolve_config derived as chosen",
        module="config.py",
        old="db_explicit=bool(db or configured_db),",
        new="db_explicit=True,",
        tests="test_only_a_database_the_environment_derived_counts_as_derived",
    ),
    Mutation(
        name="trees: read a relative directory against the docs root despite cwd",
        module="server.py",
        old="        directory = _anchored(directory, cwd)\n        return services",
        new="        return services",
        tests="test_a_relative_directory_is_the_agents_when_it_exists_there",
    ),
    Mutation(
        name="trees: list a relative directory against the docs root despite cwd",
        module="server.py",
        old='        directory = _anchored(directory, cwd) or ""',
        new="        directory = directory",
        tests="test_a_relative_directory_is_the_agents_when_it_exists_there",
    ),
    Mutation(
        name="trees: start a tree's background run inside the call that made it",
        module="server.py",
        old="                tree.start_auto_index(request=False)",
        new="                tree.start_auto_index(request=True)",
        tests="test_a_trees_background_run_is_armed_not_started_by_the_call_that_made_it",
    ),
    Mutation(
        name="trees: index a tree before learning whose weights could be reused",
        module="indexer.py",
        old="            self._reuse is not None\n            or recorded",
        new="            False\n            or recorded",
        tests="test_a_model_that_names_its_weights_once_loaded_still_reuses",
    ),
    Mutation(
        name="trees: call a term undocumented while a fresh tree is still indexing",
        module="models.py",
        old="status and status.indexing and not status.verified:",
        new="status and False:",
        tests="test_a_miss_while_an_unvouched_tree_is_indexing_says_so",
    ),
    Mutation(
        name="trees: say it while a tree walked whole is only being refreshed",
        module="models.py",
        old="status and status.indexing and not status.verified:",
        new="status and status.indexing:",
        tests="test_a_miss_while_an_unvouched_tree_is_indexing_says_so",
    ),
    Mutation(
        name="trees: build the search message without the status beside it",
        module="server.py",
        old="        message = page.keyword_message(status)",
        new="        message = page.keyword_message()",
        tests="test_search_docs_tells_the_agent_to_wait_rather_than_conclude",
    ),
    Mutation(
        name="git: miss the submodule's way of saying its repository is bare",
        module="discovery.py",
        old='_BARE_CHECKOUT = ("must be run in a work tree", "unable to set up work tree',
        new='_BARE_CHECKOUT = ("must be run in a work tree", "nothing like this',
        tests="test_a_submodule_configured_bare_is_read_from_its_own_directory",
    ),
    Mutation(
        name="git: give up on a checkout whose repository says it is bare",
        module="discovery.py",
        old="        checkout = _bare_checkout(directory)",
        new="        checkout = None",
        tests="TestABareConfiguredCheckout",
    ),
    Mutation(
        name="git: read a git directory as a checkout",
        module="discovery.py",
        old='    if inside != b"false":',
        new="    if False:",
        tests="test_inside_the_git_directory_is_in_no_work_tree",
    ),
    Mutation(
        name="git: look for the checkout along a symlink's own path",
        module="discovery.py",
        old="    start = directory.resolve()",
        new="    start = directory",
        tests="test_a_symlink_into_it_finds_the_same_checkout",
    ),
    Mutation(
        name="git: warn about the same checkout on every call",
        module="discovery.py",
        old="            first = str(checkout) not in _warned",
        new="            first = True",
        tests="test_it_is_logged_once_with_the_fix_nobody_ran",
    ),
    Mutation(
        name="unindexed: call a missing file not indexed",
        module="discovery.py",
        old="    if not path.exists():",
        new="    if False:",
        tests="test_for_a_path_the_agent_spelled",
    ),
    Mutation(
        name="unindexed: ask for an index of a file that is not Markdown",
        module="discovery.py",
        old="    if path.is_dir() or path.suffix.lower() not in MARKDOWN_SUFFIXES:",
        new="    if path.is_dir():",
        tests="test_for_a_path_the_agent_spelled",
    ),
    Mutation(
        name="unindexed: treat a path outside the root as one under it",
        module="discovery.py",
        old="    if real != root and root not in real.parents:",
        new="    if False:",
        tests="test_outside_the_root_says_so_and_points_at_cwd",
    ),
    Mutation(
        name="unindexed: call a skipped directory's file excluded",
        module="discovery.py",
        old="    if skipped is not None:",
        new="    if False:",
        tests="test_for_a_path_the_agent_spelled",
    ),
    Mutation(
        name="unindexed: give a linked worktree's file the generic advice",
        module="discovery.py",
        old="    if worktree is not None:",
        new="    if False:",
        tests="test_a_linked_worktree_named_relative_to_the_root_is_pointed_at_cwd",
    ),
    Mutation(
        name="unindexed: forget the exclusion patterns",
        module="discovery.py",
        old="    if _is_excluded(real, root, exclude):",
        new="    if False:",
        tests="test_for_a_path_the_agent_spelled",
    ),
    Mutation(
        name="unindexed: ask git about ignores the server does not apply",
        module="discovery.py",
        old="    elif gitignore and _git_ignores(real, root):",
        new="    elif _git_ignores(real, root):",
        tests="test_ignored_output_is_not_called_ignored_when_git_is_not_asked",
    ),
    Mutation(
        name="unindexed: let a file name read as a git option",
        module="discovery.py",
        old='            "--", relative.parts[0],',
        new="            relative.parts[0],",
        tests="test_for_a_path_the_agent_spelled",
    ),
    Mutation(
        name="unindexed: ask git about the whole path below an ignored directory",
        module="discovery.py",
        old='            "--", relative.parts[0],',
        new='            "--", str(relative),',
        tests="test_for_a_path_the_agent_spelled",
    ),
    Mutation(
        name="unindexed: judge a bare name as a path",
        module="server.py",
        old='        if not (path.is_absolute() or "/" in requested or requested.startswith(".")):',
        new="        if False:",
        tests="test_a_bare_name_is_a_suffix_nothing_matched",
    ),
    Mutation(
        name="unindexed: judge a relative path as a bare name",
        module="server.py",
        old='        if not (path.is_absolute() or "/" in requested or requested.startswith(".")):',
        new='        if not (path.is_absolute() or requested.startswith(".")):',
        tests="test_a_relative_path_is_judged_where_it_points",
    ),
    Mutation(
        name="unindexed: judge the first candidate even when another exists",
        module="server.py",
        old="        target = next((c for c in candidates if os.path.lexists(c)), candidates[0])",
        new="        target = candidates[0]",
        tests="test_the_candidate_that_exists_is_the_one_judged",
    ),
    Mutation(
        name="unindexed: judge a link by where it points",
        module="discovery.py",
        old="    real = path.parent.resolve() / path.name",
        new="    real = path.resolve()",
        tests="test_a_link_is_judged_where_it_is_not_where_it_points",
    ),
    Mutation(
        name="unindexed: read a file name as pathspec magic",
        module="discovery.py",
        old='"--literal-pathspecs",',
        new="",
        tests="test_for_a_path_the_agent_spelled",
    ),
    Mutation(
        name="unindexed: miss a file whose directory git ignores",
        module="discovery.py",
        old='    return any("/".join(parts[:end]) in entries for end in range(1, len(parts) + 1))',
        new='    return "/".join(parts) in entries',
        tests="test_for_a_path_the_agent_spelled",
    ),
    Mutation(
        name="unindexed: ask for a run while one is indexing",
        module="server.py",
        old="        if self._auto is not None and self._auto.active:\n            return (",
        new="        if False:\n            return (",
        tests="test_a_run_in_progress_is_named_instead_of_asking_for_one",
    ),
    Mutation(
        name="gitignore: point an agent at a log it cannot read",
        module="models.py",
        old='["git", "-C", self.root, "status"] if self.root else ["git", "status"]',
        new='["git", "status"]',
        tests="test_unavailable_names_a_command_the_agent_can_run",
    ),
    Mutation(
        name="transport: send search results twice, as text and as structuredContent",
        module="server.py",
        old=(
            "    @server.tool(structured_output=False)\n"
            "    @anticipated_errors\n"
            "    def search_docs("
        ),
        new="    @server.tool()\n    @anticipated_errors\n    def search_docs(",
        tests="test_the_five_tools_are_registered_with_typed_schemas",
    ),
    Mutation(
        name="transport: let index_directory declare an output schema",
        module="server.py",
        old=(
            "    @server.tool(structured_output=False)\n"
            "    @anticipated_errors\n"
            "    def index_directory("
        ),
        new="    @server.tool()\n    @anticipated_errors\n    def index_directory(",
        tests="test_the_five_tools_are_registered_with_typed_schemas",
    ),
    Mutation(
        name="transport: let list_documents declare an output schema",
        module="server.py",
        old=(
            "    @server.tool(structured_output=False)\n"
            "    @anticipated_errors\n"
            "    def list_documents("
        ),
        new="    @server.tool()\n    @anticipated_errors\n    def list_documents(",
        tests="test_the_five_tools_are_registered_with_typed_schemas",
    ),
    Mutation(
        name="transport: let get_document_outline declare an output schema",
        module="server.py",
        old=(
            "    @server.tool(structured_output=False)\n"
            "    @anticipated_errors\n"
            "    def get_document_outline("
        ),
        new="    @server.tool()\n    @anticipated_errors\n    def get_document_outline(",
        tests="test_the_five_tools_are_registered_with_typed_schemas",
    ),
    Mutation(
        name="transport: let read_section declare an output schema",
        module="server.py",
        old=(
            "    @server.tool(structured_output=False)\n"
            "    @anticipated_errors\n"
            "    def read_section("
        ),
        new="    @server.tool()\n    @anticipated_errors\n    def read_section(",
        tests="test_the_five_tools_are_registered_with_typed_schemas",
    ),
    Mutation(
        name="transport: escape every non-ASCII character",
        module="server.py",
        old='    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)',
        new='    return json.dumps(value, separators=(",", ":"))',
        tests="test_non_ascii_text_is_sent_as_itself_not_escaped",
    ),
    Mutation(
        name="transport: indent the JSON every reader pays for",
        module="server.py",
        old='    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)',
        new="    return json.dumps(value, indent=2, ensure_ascii=False)",
        tests="test_each_result_is_one_text_block_of_compact_json_or_raw_text",
    ),
    Mutation(
        name="transport: JSON-encode a section's raw Markdown",
        module="server.py",
        old=(
            "        return services.get(cwd, file_path).read_section(\n"
            "            file_path, heading_path, include_subsections=include_subsections, "
            "cwd=cwd\n"
            "        )"
        ),
        new=(
            "        return _json(services.get(cwd, file_path).read_section(\n"
            "            file_path, heading_path, include_subsections=include_subsections, "
            "cwd=cwd\n"
            "        ))"
        ),
        tests="test_each_result_is_one_text_block_of_compact_json_or_raw_text",
    ),
    Mutation(
        name="pointers: quote the whole passage instead of how it begins",
        module="models.py",
        old='            pointer["matched_passage"] = preview(self.matched_passage)',
        new='            pointer["matched_passage"] = self.matched_passage',
        tests="test_a_pointer_shows_how_a_long_passage_begins_not_all_of_it",
    ),
    Mutation(
        name="top hit: send the ranks an agent has no use for",
        module="models.py",
        old='                "content": self.content,',
        new='                "content": self.content, "fts_rank": self.fts_rank,',
        tests="test_search_docs_returns_sections_and_breadcrumbs",
    ),
    Mutation(
        name="live test: a status check that cannot fail",
        module="live_test.py",
        area="scripts",
        old='            answer["index_status"]\n            == {',
        new='            True or answer["index_status"]\n            == {',
        tests="test_a_root_that_does_not_vouch_for_itself_fails_the_run",
    ),
    Mutation(
        name="excerpt: let a table row lose the line its table starts on",
        module="parser.py",
        old='            rows.append(Passage("; ".join(labelled), row, table))',
        new='            rows.append(Passage("; ".join(labelled), row, None))',
        tests="test_a_table_row_brings_its_header_and_delimiter",
    ),
    Mutation(
        name="excerpt: give a list item no lines",
        module="parser.py",
        old="            fragments, item = [], _span(token)",
        new="            fragments, item = [], None",
        tests="test_every_kind_of_block_names_the_lines_it_was_cut_from",
    ),
    Mutation(
        name="excerpt: call no part boundary a cut through a block",
        module="parser.py",
        old="        return any(start < cut < end for cut in cuts for start, end in blocks)",
        new="        return False",
        tests=(
            "test_a_cut_inside_a_fence_a_table_or_a_list_is_not"
            " or test_a_part_cut_through_a_fence_is_sent_whole"
            " or test_a_part_cut_inside_one_long_line_is_sent_whole"
        ),
    ),
    Mutation(
        name="excerpt: end a block on the next line's first character",
        module="parser.py",
        old="            (starts[token.map[0]], starts[token.map[1]] - 1)",
        new="            (starts[token.map[0]], starts[token.map[1]])",
        tests="test_a_cut_between_blocks_or_among_blank_lines_is_clean",
    ),
    Mutation(
        name="excerpt: never see a fence left open",
        module="parser.py",
        old="    return open_fence is not None",
        new="    return False",
        tests=(
            "test_an_unclosed_fence_is_seen or test_a_window_that_leaves_a_fence_open_is_sent_whole"
        ),
    ),
    Mutation(
        name="excerpt: anchor a query of stopwords",
        module="search.py",
        old='    if not terms or all(_is_stopword(term.replace("`", "")) for term in terms):',
        new="    if not terms:",
        tests="test_a_query_of_stopwords_has_no_anchor_even_with_a_vector_passage",
    ),
    Mutation(
        name="excerpt: let backticks hide a stopword",
        module="search.py",
        old='    if not terms or all(_is_stopword(term.replace("`", "")) for term in terms):',
        new="    if not terms or all(_is_stopword(term) for term in terms):",
        tests="test_a_query_of_stopwords_has_no_anchor_even_with_a_vector_passage",
    ),
    Mutation(
        name="excerpt: anchor an identifier lookup on the vector passage",
        module="search.py",
        old="    if _is_identifier_lookup(terms):\n        return next(",
        new="    if _is_identifier_lookup(terms) and ordinal is None:\n        return next(",
        tests="test_an_identifier_lookup_is_anchored_where_the_identifier_is "
        "or test_an_identifier_lookup_is_anchored_where_it_is_first_named",
    ),
    Mutation(
        name="excerpt: fall through to the vector passage when only the heading names it",
        module="search.py",
        old="        return next((index for index, count in enumerate(found) if count), None)",
        new="        return next((index for index, count in enumerate(found) if count), ordinal)",
        tests="test_an_identifier_only_the_heading_names_gets_the_whole_section",
    ),
    Mutation(
        name="excerpt: ignore the passage that won the vector ranking",
        module="search.py",
        old="    if ordinal is not None and 0 <= ordinal < len(passages):\n        return ordinal",
        new="    if False:\n        return ordinal",
        tests="test_any_other_query_is_anchored_on_the_vector_passage",
    ),
    Mutation(
        name="excerpt: anchor on a passage that holds no term",
        module="search.py",
        old="    return found.index(best) if best else None",
        new="    return found.index(best) if found else None",
        tests="test_without_one_the_passage_holding_the_most_terms_lowest_first",
    ),
    Mutation(
        name="excerpt: cut by a passage that has no lines",
        module="search.py",
        old="    if len(passages) <= 3 or any(span is None for span in spans):",
        new="    if len(passages) <= 3:",
        tests="test_a_passage_without_lines_sends_the_section_whole",
    ),
    Mutation(
        name="excerpt: cut a section of three passages",
        module="search.py",
        old="    if len(passages) <= 3 or any(span is None for span in spans):",
        new="    if any(span is None for span in spans):",
        tests="test_three_passages_or_fewer_are_sent_whole",
    ),
    Mutation(
        name="excerpt: show a table row without its header",
        module="search.py",
        old=(
            "    first = min([first, *(passage.table for passage in window"
            " if passage.table is not None)])"
        ),
        new="    first = min([first])",
        tests="test_a_table_row_brings_its_header_and_delimiter",
    ),
    Mutation(
        name="excerpt: keep the blank line markdown-it gives a list item",
        module="search.py",
        old="        end -= 1  # markdown-it counts the blank line after a list item as part of it",
        new="        break",
        tests="test_a_list_item_does_not_bring_the_blank_line_after_it",
    ),
    Mutation(
        name="excerpt: cut a window that covers every passage",
        module="search.py",
        old=(
            "    if all(p.lines is not None and first <= p.lines[0] and p.lines[1] <= end"
            " for p in passages):\n        return None"
        ),
        new=(
            "    if False and all(p.lines is not None and first <= p.lines[0]"
            " and p.lines[1] <= end for p in passages):\n        return None"
        ),
        tests="test_a_window_that_grows_over_every_passage_is_sent_whole",
    ),
    Mutation(
        name="excerpt: send a window no smaller than its section",
        module="search.py",
        old="    if ends_inside_fence(text) or estimate_tokens(text) >= estimate_tokens(content):",
        new="    if ends_inside_fence(text):",
        tests="test_a_window_no_smaller_than_the_section_is_sent_whole",
    ),
    Mutation(
        name="excerpt: trust passages that differ from the stored ones",
        module="search.py",
        old="                if [passage.text for passage in cut] == stored:",
        new="                if True:",
        tests="test_passages_that_differ_from_the_stored_ones_send_the_section",
    ),
    Mutation(
        name="excerpt: cut a section at the passage cap",
        module="search.py",
        old="            if not stored or len(stored) >= MAX_UNITS_PER_SECTION:",
        new="            if not stored:",
        tests="test_a_section_at_the_passage_cap_is_sent_whole",
    ),
    Mutation(
        name="excerpt: cut a part without looking at the section it came from",
        module="search.py",
        old=(
            "            if passages is None or (section.part_index"
            " and self._cut_through(section, parser)):"
        ),
        new="            if passages is None:",
        tests=(
            "test_a_part_cut_through_a_fence_is_sent_whole"
            " or test_a_part_cut_inside_one_long_line_is_sent_whole"
        ),
    ),
    Mutation(
        name="excerpt: let a failure while cutting fail the search",
        module="search.py",
        old=(
            "        except Exception:  # an excerpt is an optimisation:"
            " whatever fails sends the section"
        ),
        new="        except ValueError:  # narrowed",
        tests="test_a_failure_while_cutting_sends_the_section",
        fails_with="RuntimeError",
    ),
    Mutation(
        name="excerpt: lose which passage of its section won the vector ranking",
        module="search.py",
        old="                passages.setdefault(section_id, (ordinal, passage))",
        new="                passages.setdefault(section_id, (0, passage))",
        tests="test_a_section_is_ranked_by_its_closest_passage_not_its_average",
    ),
    Mutation(
        name="excerpt: send the whole section's line range with the excerpt",
        module="models.py",
        old='            "lines": f"{self.excerpt.start_line}-{self.excerpt.end_line}",',
        new='            "lines": f"{self.start_line}-{self.end_line}",',
        tests="test_the_excerpt_is_verbatim_lines_of_the_file",
    ),
    Mutation(
        name="excerpt: price the excerpt instead of what read_section costs",
        module="models.py",
        old=('            "tokens": estimate_tokens(self.content),\n            "excerpt": True,'),
        new=(
            '            "tokens": estimate_tokens(self.excerpt.text),\n'
            '            "excerpt": True,'
        ),
        tests="test_the_excerpt_is_verbatim_lines_of_the_file",
    ),
    Mutation(
        name="excerpt: count each list item as a block of its own",
        module="search.py",
        old="    keys = [passage.listing or passage.lines for passage in passages]",
        new="    keys = [passage.lines for passage in passages]",
        tests="test_a_list_is_one_block_with_the_sentence_that_introduces_it",
    ),
    Mutation(
        name="excerpt: let a list item forget which list it is in",
        module="parser.py",
        old="    listing = _span(block[0])",
        new="    listing = None",
        tests="test_a_list_is_one_block_with_the_sentence_that_introduces_it",
    ),
    Mutation(
        name="excerpt: cut a section whose heading names the whole query",
        module="search.py",
        old="    if all(literal.found(heading) for literal in literals):",
        new="    if False:",
        tests="test_a_section_whose_heading_names_the_whole_query_is_sent_whole",
    ),
    Mutation(
        name="excerpt: send whole every section whose heading shares a query term",
        module="search.py",
        old="    if all(literal.found(heading) for literal in literals):",
        new="    if any(literal.found(heading) for literal in literals):",
        tests="test_a_section_whose_heading_names_the_whole_query_is_sent_whole",
    ),
    Mutation(
        name="excerpt: anchor the top hit without telling it its heading",
        module="search.py",
        old="            anchor = select_anchor(terms, stored, ordinal, section.heading_title)",
        new="            anchor = select_anchor(terms, stored, ordinal)",
        tests="test_a_section_headed_by_the_identifier_looked_up_is_sent_whole",
    ),
    Mutation(
        name="paths: let the agent's cwd shadow the root's document again",
        module="server.py",
        old="[Path(self._root) / path, Path(cwd.strip()).expanduser() / path]",
        new="[Path(cwd.strip()).expanduser() / path, Path(self._root) / path]",
        tests="test_the_roots_document_wins_over_a_same_named_one_in_cwd",
    ),
    Mutation(
        name="paths: follow a symlink before looking the path up as spelled",
        module="server.py",
        old=(
            "            document = self._db.get_document(spelled)"
            " or self._db.get_document(resolved)"
        ),
        new="            document = self._db.get_document(resolved)",
        tests="test_an_indexed_symlink_is_found_under_its_own_name",
        # The honest outcome: the suffix lookup finds two `shared.md` and says so.
        fails_with="mcp.server.mcpserver.exceptions.ToolError",
    ),
    Mutation(
        name="paths: shorten only the pointers' paths",
        module="server.py",
        old="                _relative(hit.to_dict() if rank == 0 else hit.to_pointer(), root)",
        new="                hit.to_dict() if rank == 0 else _relative(hit.to_pointer(), root)",
        tests="test_the_tool_keeps_the_order_and_the_limit_the_service_gives_it",
    ),
    Mutation(
        name="paths: shorten only the top hit's path",
        module="server.py",
        old="                _relative(hit.to_dict() if rank == 0 else hit.to_pointer(), root)",
        new="                _relative(hit.to_dict(), root) if rank == 0 else hit.to_pointer()",
        tests="test_the_tool_keeps_the_order_and_the_limit_the_service_gives_it",
    ),
    Mutation(
        name="paths: list documents by their absolute paths",
        module="server.py",
        old="                    _relative(summary.to_dict(), root)",
        new="                    summary.to_dict()",
        tests=(
            "test_every_listed_path_reads_back_its_own_document_from_any_cwd"
            " or test_a_narrowed_listing_is_still_relative_to_the_root"
        ),
    ),
    Mutation(
        name="paths: leave a listing's failures absolute",
        module="server.py",
        old=(
            '                "index_status": _relative_failures('
            "service.index_status(scope).to_dict(), root),"
        ),
        new='                "index_status": service.index_status(scope).to_dict(),',
        tests="test_a_search_over_a_damaged_index_says_so_in_its_answer",
    ),
    Mutation(
        name="paths: leave a search's failures absolute",
        module="server.py",
        old='        payload["index_status"] = _relative_failures(status.to_dict(), root)',
        new='        payload["index_status"] = status.to_dict()',
        tests="test_a_search_over_a_damaged_index_says_so_in_its_answer",
    ),
    Mutation(
        name="paths: shorten index_status.root too",
        module="server.py",
        old='    return {**status, "failures": shown}',
        new='    return {**status, "failures": shown, "root": "."}',
        tests="test_the_tool_keeps_the_order_and_the_limit_the_service_gives_it",
    ),
    Mutation(
        name="paths: shorten another work tree's paths",
        module="server.py",
        old="        return service.root if service is self._configured() else None",
        new="        return service.root",
        tests="test_a_worktree_answer_keeps_absolute_paths_that_route_without_cwd",
    ),
    Mutation(
        name="paths: shorten a path that is not under the root",
        module="server.py",
        old=(
            "    if root is None or not isinstance(path, str)"
            " or not Path(path).is_relative_to(root):"
        ),
        new="    if root is None or not isinstance(path, str):",
        tests="test_a_failure_outside_the_root_stays_absolute",
        fails_with="ValueError",
    ),
    Mutation(
        name="usage: read a relative hit against the project instead of the root",
        module="usage_from_transcripts.py",
        area="scripts",
        old="    base = root if isinstance(root, str) and root else cwd",
        new="    base = cwd",
        tests="test_a_relative_hit_is_the_file_under_the_root_that_answered",
    ),
)


def _ignore_caches(_directory: str, names: list[str]) -> set[str]:
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
