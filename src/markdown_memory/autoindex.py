"""Keeping the server's own documentation root indexed without being asked.

Nothing watches the filesystem. The server already looks at it on every search - the
freshness sweep counts indexed files that moved on - so that look is what decides when to
re-index, and one run at start catches whatever changed while no server was running. A
run is the ordinary incremental `index_directory`, in one background thread at a time.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from markdown_memory.exceptions import IndexBusyError, IndexCancelled
from markdown_memory.models import IndexReport, IndexStatus

logger = logging.getLogger(__name__)

#: How soon after a run finishes an edit found by a search may start the next one. Short
#: enough that an agent's edit is searchable within a turn or two; long enough that a
#: burst of saves is one run rather than one each.
CHANGE_GAP_SECONDS = 10.0
#: How long a run speaks for when nothing edited was seen. Only a walk finds a file
#: nobody indexed yet, or an edit that put its modification time back.
WALK_GAP_SECONDS = 300.0


class AutoIndexer:
    """One background index run at a time, started by what the server sees on use.

    Every field below is read and written under one lock, so a request can never slip
    between a run finishing and the runner forgetting it, and nothing starts once `stop`
    has been called.
    """

    def __init__(
        self,
        run: Callable[[Callable[[], bool]], IndexReport],
        measure: Callable[[], IndexStatus],
        *,
        clock: Callable[[], float] = time.monotonic,
        change_gap: float = CHANGE_GAP_SECONDS,
        walk_gap: float = WALK_GAP_SECONDS,
    ) -> None:
        self._run = run
        self._measure = measure
        self._clock = clock
        self._change_gap = change_gap
        self._walk_gap = walk_gap
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stopping = False
        self._last_finished: float | None = None
        #: Changed files the last finished run could not clear - files it failed on. A
        #: search that still sees exactly those is not a reason to walk again; one more
        #: is. Reset to zero by a run that failed nothing, so an edit made while it ran
        #: is picked up by the next search.
        self._baseline = 0
        #: The weights mismatch as the last run left it: a new one is repaired at once,
        #: one this runner could not repair does not start a run per search.
        self._seen_mismatch: str | None = None

    @property
    def active(self) -> bool:
        with self._lock:
            return self._thread is not None

    def request(self) -> bool:
        """Start a run now unless one is running or the runner is stopping."""
        with self._lock:
            return self._start()

    def consider(self, status: IndexStatus) -> bool:
        """Start a run if what a search just measured says one is due."""
        with self._lock:
            if self._stopping or self._thread is not None:
                return False
            since = (
                float("inf") if self._last_finished is None else self._clock() - self._last_finished
            )
            due = (
                (status.changed_files > self._baseline and since >= self._change_gap)
                or (
                    status.weights_mismatch is not None
                    and status.weights_mismatch != self._seen_mismatch
                )
                or since >= self._walk_gap
            )
            return self._start() if due else False

    def stop(self) -> None:
        """Stop the running run between two documents, and wait for it to let go."""
        with self._lock:
            self._stopping = True
            thread = self._thread
        if thread is not None:
            thread.join()

    def _start(self) -> bool:
        if self._stopping or self._thread is not None:
            return False
        # Not a daemon: the interpreter would kill it mid-write at exit. `stop` is what
        # ends it, and the service calls that before it closes the database.
        self._thread = threading.Thread(target=self._work, name="mdmem-autoindex")
        self._thread.start()
        return True

    def _is_stopping(self) -> bool:
        with self._lock:
            return self._stopping

    def _work(self) -> None:
        baseline: int | None = None
        busy = False
        before = self._mismatch()
        try:
            report = self._run(self._is_stopping)
            baseline = self._changed() if report.errors else 0
        except IndexCancelled:
            logger.info("Automatic index run stopped")
        except IndexBusyError as exc:
            # Someone else holds the lock - maybe on another root of a shared database -
            # so nothing is learnt about this tree; the next search asks again.
            logger.info("Automatic index run skipped: %s", exc)
            busy = True
        except Exception:
            # The whole run failed - a model that will not load, say. Retrying it for the
            # same edit every few seconds would load the model every few seconds: what
            # it left behind becomes the baseline, and only a further change is news.
            logger.exception("Automatic index run failed")
            baseline = self._changed()
        after = self._mismatch()
        with self._lock:
            self._last_finished = self._clock()
            if busy:
                # Nothing ran, so nothing is known: ask again once the change gap has
                # passed rather than a whole walk interval later - a root nobody has
                # indexed has no changed files to prompt the next attempt.
                self._last_finished -= self._walk_gap - self._change_gap
                self._thread = None
                return
            if baseline is not None:
                self._baseline = baseline
            # Seen only if this run met it and could not clear it. One a search recorded
            # while the run was busy is news the run never acted on.
            if after is None or after == before:
                self._seen_mismatch = after
            self._thread = None

    def _changed(self) -> int | None:
        try:
            return self._measure().changed_files
        except Exception:
            logger.exception("Cannot read the index status around an automatic run")
            return None

    def _mismatch(self) -> str | None:
        try:
            return self._measure().weights_mismatch
        except Exception:
            logger.exception("Cannot read the index status around an automatic run")
            with self._lock:
                return self._seen_mismatch
