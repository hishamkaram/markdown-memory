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
            "            return headings._absolute(self._config.docs_dir, IndexingError)"
        ),
        tests="test_a_scan_after_a_retarget_stays_with_the_root_it_serves",
    ),
    Mutation(
        name="config: keep the launcher's database when another project's root is named",
        module="config.py",
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
        module="config.py",
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
        name="weights: discard every document when a model with named weights is renamed",
        module="indexer.py",
        old=(
            "            if renamed and self._embedder.weights_revision is None:\n"
            "                # Vectors from"
        ),
        new="            if renamed:\n                # Vectors from",
        tests="test_a_renamed_model_with_named_weights_repairs_instead_of_discarding",
    ),
    Mutation(
        name="weights: discard the index before a renamed lazy model can name its weights",
        module="indexer.py",
        old=(
            "                with contextlib.suppress(ModelLoadError):\n"
            "                    self._embedder.warm_up()\n"
        ),
        new="                with contextlib.suppress(ModelLoadError):\n                    pass\n",
        tests="test_a_renamed_model_that_names_its_weights_once_loaded_repairs_too",
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
        old="            best and recorded != self._embedder.weights_revision\n",
        new="            False\n",
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
        old='            key=lambda name: (name.startswith("."), name),\n',
        new="            key=lambda name: name,\n",
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
        old="        if self._db.get_meta(WEIGHTS_META_KEY) != recorded or (\n",
        new="        if False or (  # the check speaks for rows read after it\n",
        tests="test_an_index_rebuilt_by_another_model_mid_search_is_not_ranked_on",
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
        old='            if weights is None or self._db.count_rows("units_vec") == 0:\n',
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
        name="diagram: print a token count the files stopped matching",
        module="make_diagram.py",
        area="scripts",
        old='    ("README.md", 8766),',
        new='    ("README.md", 5654),',
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
        old='{"query": str(case["query"])})',
        new='{"query": str(case["query"]), "limit": 1})',
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
        old="                    if not qualified or Path(file_path).name == name",
        new="                    if True",
        tests="test_a_qualified_label_resolves_in_the_file_it_names",
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
        old="        answers = resolve_answers(service, queries)",
        new="        answers = {}",
        tests="test_a_broken_fixture_stops_the_run_before_any_search",
    ),
    Mutation(
        name="cost: let the informational table decide how the run ends",
        module="eval_retrieval.py",
        area="scripts",
        old="        except Exception as exc:  # informational",
        new="        except ZeroDivisionError as exc:  # informational",
        tests="test_a_cost_pass_that_fails_does_not_decide_the_exit_code",
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
        name="coverage: count a run's own model-change wipe against it",
        module="indexer.py",
        old="            generation = self._db.generation()\n            identity =",
        new="            generation = 0\n            identity =",
        tests="test_a_clean_run_after_a_model_change_vouches_for_the_tree",
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
        old=(
            "            db.expanduser() if db else configured_db "
            "if configured_db else _project_database(root)"
        ),
        new=("            configured_db if configured_db else _project_database(root)"),
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
        name="freshness: never look at the disk",
        module="server.py",
        old=(
            "        return dataclasses.replace("
            "status, changed_files=self._freshness.changed_files(scope))"
        ),
        new="        return status",
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
