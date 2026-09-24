"""Whether the indexed documents are still what is on disk.

The database's own status is one SQLite snapshot and says nothing about the filesystem,
so this is composed beside it rather than inside it. ``FreshnessSweep`` owns the whole
of that state: the single-entry cache, its TTL, and the lock that makes reading the
cache, walking the tree and publishing the answer one step.
"""

from __future__ import annotations

import stat
import threading
import time
from enum import Enum, auto
from pathlib import Path

from markdown_memory.db import Database
from markdown_memory.discovery import MAX_FILE_BYTES, hash_bytes, read_regular_file

# How long one filesystem sweep speaks for. An agent fires several searches in a single
# turn, and every one of them asks for the status: on a local ext4 tree 100 stats cost
# ~0.1 ms, but across a WSL2 or network boundary they cost 100-300 ms, which would double
# the latency of a query to re-answer a question whose answer cannot have changed much.
# Short enough that an edit is reported by the next search but one.
_FRESHNESS_TTL_SECONDS = 3.0


class _Verdict(Enum):
    """What one file's freshness probe found."""

    UNCHANGED = auto()
    CHANGED = auto()
    #: Same bytes, a time that has moved: nothing to report, but worth writing down.
    SAME_BYTES_NEW_TIME = auto()


def _compare(path: Path, content_hash: str, mtime_ns: int | None) -> tuple[_Verdict, int]:
    """Whether the file behind an indexed document differs from what was indexed.

    The modification time is the cheap question and the bytes are the expensive one, so
    the hash is only computed where the time has moved: a `touch`, a checkout that
    rewrites a file with its own contents, or a copy that preserves nothing but the text
    must not be reported as a change an agent should act on. A file that has vanished, no
    longer resolves to a regular file, or cannot be read counts as changed - not because
    its bytes are known to differ, but because they cannot be checked at all. That applies
    where the bytes had to be read: a file whose recorded time still matches is answered
    from the time alone, so losing permission to read it - without touching it - is not
    reported here. The next index run cannot read it either, and records a failure, which
    is what takes `coverage` to `"unknown"`.

    A stored `None` means no modification time was recorded - a row written before the
    column existed - rather than a time of zero, so no real timestamp can be mistaken for
    it, the epoch included. Those files are answered by their bytes until an index run
    writes a time for them.

    The one edit this cannot see is a file rewritten with its modification time put back
    to what it was: no timestamp moved, so no hash is taken. Indexing itself is not fooled
    - it hashes every file it walks - so `index_directory` still rebuilds that document;
    what is missed is only the hint that it is worth running. Seeing it here would mean
    hashing every indexed file on every query, or storing a second timestamp to compare
    against, and this signal is not worth either.
    """
    try:
        info = path.stat()
    except OSError:
        return _Verdict.CHANGED, 0
    if not stat.S_ISREG(info.st_mode):
        return _Verdict.CHANGED, 0
    if mtime_ns is not None and info.st_mtime_ns == mtime_ns:
        return _Verdict.UNCHANGED, info.st_mtime_ns
    try:
        data = read_regular_file(path)
    except OSError:
        return _Verdict.CHANGED, 0
    if data is None or len(data) > MAX_FILE_BYTES or hash_bytes(data) != content_hash:
        return _Verdict.CHANGED, 0
    # The time from the stat that came *before* the read, never a fresher one: a file
    # rewritten after these bytes were hashed must not be recorded as verified at the
    # moment of its rewrite, or the next sweep would trust a time that belongs to content
    # nobody checked.
    return _Verdict.SAME_BYTES_NEW_TIME, info.st_mtime_ns


class FreshnessSweep:
    """How many indexed documents are no longer what was indexed, cheaply and often.

    One owner for three things that only make sense together: the count, the moment it
    was taken, and the lock that keeps a sweep whole. Before this they were three
    attributes on the service, which made it possible to reset one and not the others.
    """

    def __init__(self, db: Database, ttl: float = _FRESHNESS_TTL_SECONDS) -> None:
        self._db = db
        self._ttl = ttl
        #: One entry, not a map keyed on the caller's path: an agent fires several
        #: searches per turn against the same scope, and a map would grow for the life of
        #: the server, one entry per spelling anybody ever asked about.
        self._cache: tuple[str, float, int] | None = None
        #: Held for the whole of a sweep, so that reading the cache, walking the
        #: filesystem and storing the answer are one step. Without it a sweep that
        #: indexing overtook would publish a count of a tree that no longer exists - the
        #: one moment an agent is most likely to ask - and two sweeps racing could leave
        #: the older one's answer behind. It also means a second caller arriving mid-sweep
        #: waits and is served the result rather than walking the tree again.
        self._lock = threading.Lock()

    def invalidate(self) -> None:
        """Forget the cached count: indexing has changed what the answer would be."""
        with self._lock:
            self._cache = None

    def changed_files(self, scope: str) -> int:
        with self._lock:
            cached = self._cache
            if (
                cached is not None
                and cached[0] == scope
                and time.monotonic() - cached[1] < self._ttl
            ):
                return cached[2]
            fingerprints = self._db.document_fingerprints(scope)
            changed = 0
            for file_path, (content_hash, mtime_ns) in fingerprints.items():
                verdict, seen_ns = _compare(Path(file_path), content_hash, mtime_ns)
                if verdict is _Verdict.CHANGED:
                    changed += 1
                elif verdict is _Verdict.SAME_BYTES_NEW_TIME:
                    # The hash was computed to answer this, and the answer was "unchanged".
                    # Writing the time down means the next sweep reads it instead of the
                    # file - otherwise one `touch` costs a full hash every window until an
                    # index run happens to come past. A file that moved again in between
                    # is simply hashed again next time; nothing is lost by missing it.
                    self._db.record_modification_time(file_path, content_hash, mtime_ns, seen_ns)
            # Stamped when the answer was produced, not when the sweep began: the walk
            # itself takes time on a slow mount, and a window that starts before the
            # measurement is a window the measurement was never true for.
            self._cache = (scope, time.monotonic(), changed)
            return changed
