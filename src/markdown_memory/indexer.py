"""Incremental indexing: directory scan, SHA-256 change detection, embedding sync.

Workers read, parse and embed; one driver thread writes. The embedders themselves live in
``embedders.py``, the model cache in ``model_cache.py``, and the directory walk in
``discovery.py``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import logging
import math
import os
import stat
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from markdown_memory import discovery
from markdown_memory.db import (
    MODEL_META_KEY,
    VECTOR_FORMAT,
    WEIGHTS_META_KEY,
    WEIGHTS_MISMATCH_KEY,
    WEIGHTS_REVOKED,
    Database,
)
from markdown_memory.embedders import Embedder, short_weights
from markdown_memory.exceptions import (
    EmbeddingError,
    ForeignWeightsError,
    IndexBusyError,
    IndexCancelled,
    IndexingError,
    MarkdownMemoryError,
    ModelLoadError,
)
from markdown_memory.models import FileFailure, IndexReport, SectionDraft, SectionVectors
from markdown_memory.parser import MarkdownParser

logger = logging.getLogger(__name__)


#: Files embedded at the same time. One ONNX session is shared by all of them: the weights
#: are mmapped and counted once however many threads run against them, so a second worker
#: costs the ~150 MB of one in-flight forward pass and nothing more. Embedding alone, on 16
#: cores over 32 passages of 512 tokens: 1 worker 0.72 vectors/s at 725 MB peak, 2 workers
#: 1.63 at 883 MB, 4 workers 2.17 at 1,168 MB, 8 workers 2.85 at 1,784 MB. End to end over
#: 24 files of the eval corpus (879 passages) the gain is smaller, because parsing and the
#: writes are serial and a long file holds the head of the queue: 196.3 s at 1 worker,
#: 145.9 s at 2 (1.35x), 96.3 s at 4 (2.04x). Two is the default because it is the last
#: setting whose peak - 757-814 MB across live-test runs - is nowhere near the 1.2 GB this
#: tool budgets for itself while running beside an editor. A machine with cores to spare
#: sets the variable higher and is paid ~2x for four.
_INDEX_WORKERS_ENV = "MARKDOWN_MEMORY_INDEX_WORKERS"


DEFAULT_INDEX_WORKERS = 2


#: Vectors another index already holds for a file's passages: given the file, the texts it
#: is about to embed and the weights this run embeds with, those it may reuse, by text.
Reuse = Callable[[str, Sequence[str], str], Mapping[str, Sequence[float]]]


# Below this the pooled direction is rounding noise rather than a direction. Unit vectors
# that genuinely cancel land near 1e-16; a real centroid of normalised passages is >= 1/n
# of one passage, which for the 64-passage ceiling is ~0.015.
_MIN_POOLED_NORM = 1e-6


def _index_workers() -> int:
    """How many files are read, parsed and embedded at once. Never below one."""
    override = os.environ.get(_INDEX_WORKERS_ENV, "").strip()
    if override.isdigit() and int(override) > 0:
        return int(override)
    return DEFAULT_INDEX_WORKERS


def _section_vector(units: Sequence[Sequence[float]]) -> list[float] | None:
    """The vector stored for a section: pooled, or one of its passages if pooling fails.

    A section with passages must have a vector - the storage layer rejects the whole file
    otherwise - so passages that cancel each other out cannot be allowed to cost the file
    its place in the index. Falling back to the first passage keeps a direction that is
    at least the section's own text.
    """
    if not units:
        return None
    pooled = _mean_vector(units)
    return pooled if pooled is not None else list(units[0])


def _mean_vector(vectors: Sequence[Sequence[float]]) -> list[float] | None:
    """The centroid of ``vectors``, renormalised, or ``None`` for a section with no body.

    A section's own vector used to be a separate embedding of its whole text, which the
    model truncates at 512 tokens: 126 of 1,589 sections in the vendored corpus were
    longer than that, the largest half again over, and their tails were simply absent
    from the section-level signal. Averaging the passages covers the section entirely,
    and costs one embedding call fewer per section rather than one more.
    """
    if not vectors:
        return None
    totals = [math.fsum(values) for values in zip(*vectors, strict=True)]
    norm = math.sqrt(math.fsum(value * value for value in totals))
    # Not `== 0.0`: passages that point opposite ways cancel to float residue near 1e-16,
    # and dividing that by its own magnitude turns rounding noise into a full-length
    # vector aimed in an arbitrary direction, which then matches arbitrary queries.
    if norm < _MIN_POOLED_NORM:
        return None
    return [value / norm for value in totals]


@dataclasses.dataclass(slots=True, frozen=True)
class _Prepared:
    """One file, read and embedded, waiting to be written.

    Everything a worker produces and nothing it may do: the write is the driver's, so
    that one thread owns section ids, the certificate and the provenance metadata.
    """

    file_path: str
    title: str
    content_hash: str
    last_modified: int
    mtime_ns: int
    sections: tuple[SectionDraft, ...]
    vectors: tuple[SectionVectors, ...]
    #: The file is what the index already holds; only when it was last written has moved.
    unchanged: bool = False
    #: The time the row held when this was prepared, for the write-back to compare against.
    previous_mtime_ns: int | None = None

    @property
    def has_vectors(self) -> bool:
        return any(vector.units for vector in self.vectors)

    @property
    def counts(self) -> tuple[int, int]:
        return len(self.sections), sum(len(section.units) for section in self.sections)


class Indexer:
    """Keeps the database in sync with the Markdown files of a directory tree."""

    def __init__(
        self,
        db: Database,
        embedder: Embedder,
        workers: int | None = None,
        exclude: Sequence[str] = (),
        gitignore: bool = True,
        *,
        reuse: Reuse | None = None,
        run_lock: threading.Lock | None = None,
    ) -> None:
        if embedder.dimension != db.embedding_dim:
            raise IndexingError(
                f"Embedder produces {embedder.dimension}-dimensional vectors but the "
                f"database stores {db.embedding_dim}"
            )
        self._db = db
        self._embedder = embedder
        # One parser per thread. `MarkdownParser` holds a `MarkdownIt` with mutable
        # ruler and env state, so two files parsed through one instance at the same time
        # would read each other's tokens.
        self._parsers = threading.local()
        self._workers = max(1, workers if workers is not None else _index_workers())
        self._exclude = tuple(exclude)
        self._gitignore = gitignore
        self._reuse = reuse
        # Shared by every tree one server serves: two runs at once would hold two sets of
        # embedding buffers, which is what the memory budget is measured without.
        self._run_lock = run_lock or threading.Lock()

    @property
    def _parser(self) -> MarkdownParser:
        parser: MarkdownParser | None = getattr(self._parsers, "parser", None)
        if parser is None:
            parser = MarkdownParser()
            self._parsers.parser = parser
        return parser

    @contextlib.contextmanager
    def _scan_lock(self) -> Iterator[None]:
        """One scan at a time over this database, across threads and across processes.

        A thread lock cannot see another process, and every ordering rule this feature
        tried instead of a lock was wrong in one direction or the other. Both halves
        refuse rather than wait: a scan can run for 25 minutes, and a tool call that
        blocks that long is a client timeout, which reads to the agent as a broken
        server rather than a busy one.

        `flock` is released by the kernel when the process dies, so a killed run cannot
        strand it - the one guarantee a row in the database could not give.
        """
        if not self._run_lock.acquire(blocking=False):
            raise IndexBusyError(
                "Another index run is in progress in this process; try again shortly."
            )
        try:
            # Inside the try: opening the lock file can fail on its own (a read-only
            # directory, no file descriptors left), and a thread lock taken above and
            # never released would refuse every later run in this process for good.
            # Resolved: two spellings of one database - a symlink, a relative path -
            # would otherwise take two different locks and both scans would proceed.
            lock_path = str(Path(self._db.path).resolve()) + ".lock"
            handle = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        except OSError:
            self._run_lock.release()
            raise
        try:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise IndexBusyError(
                    f"Another process is indexing {self._db.path}; try again shortly."
                ) from error
            os.truncate(handle, 0)
            os.write(handle, f"{os.getpid()}\n".encode("ascii"))
            yield
        finally:
            os.close(handle)
            self._run_lock.release()

    def index_directory(
        self, directory: Path, should_stop: Callable[[], bool] | None = None
    ) -> IndexReport:
        """Index new/changed files, skip unchanged ones, purge files that disappeared.

        A failure in one file is recorded in the report and does not abort the run.
        ``should_stop`` is asked between documents; once it answers true the run raises
        `IndexCancelled` and leaves what a killed run leaves.
        """
        try:
            root = directory.expanduser().resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as exc:  # missing, ~unknown, NUL, loop
            raise IndexingError(f"Directory does not exist: {directory}") from exc
        if not root.is_dir():
            raise IndexingError(f"Not a directory: {root}")
        try:
            str(root).encode("utf-8")
        except UnicodeEncodeError:
            raise IndexingError(
                "Directory name is not valid UTF-8 and cannot be indexed: "
                f"{discovery._printable(str(root))}"
            ) from None

        with self._scan_lock():
            started = time.perf_counter()
            # Set at the first write of the run, not here: a run that changes nothing has
            # no business retracting a certificate that is still true. Once it does write,
            # the certificate stays retracted until a full pass finishes - so a run killed
            # partway leaves the tree honestly described as unvouched-for, with no marker
            # to clean up and nothing to go stale.
            retracted = False

            def about_to_write() -> None:
                nonlocal retracted
                if not retracted:
                    self._db.mark_scan_started(str(root))
                    retracted = True

            previous_model = self._db.get_meta(MODEL_META_KEY)
            if (
                previous_model not in {None, self._embedder.model_name}
                and self._embedder.weights_revision is None
                and self._db.has_vectors()
            ):
                # Vectors from different models are not comparable, and they share one
                # vector table. A model that names its weights - up front, or once loaded,
                # which is why it is loaded now - leaves this to the per-document stamps,
                # which re-embed each document in place while keyword search keeps
                # answering. One that cannot would leave nothing to tell old vectors from
                # new, and rebuilding every root in the file is minutes to hours of
                # embedding that is for the person to decide (#91): refused, as a vector
                # size is (#90), before anything is written. A model that will not load
                # stops here too, with the index it found.
                self._embedder.warm_up()
                if self._embedder.weights_revision is None:
                    raise ForeignWeightsError(
                        f"This index was built by {previous_model}, and the configured model "
                        f"{self._embedder.model_name} cannot say which weights it runs, so its "
                        "vectors could never be told apart from the stored ones. Nothing was "
                        f"changed in the index. Configure {previous_model} again, point --db / "
                        f"MARKDOWN_MEMORY_DB at another file, or delete {self._db.path} with "
                        "its -wal and -shm files while no markdown-memory process uses it; "
                        "the next run rebuilds it from the Markdown files."
                    )
            self._db.set_meta(MODEL_META_KEY, self._embedder.model_name)
            # Whatever emptied the index (an older vector format) left a notice. They are
            # dismissed only once the report carrying them exists: a run that aborts - the
            # model cannot be loaded - leaves them for the next one.
            notices = self._db.pending_notices()
            # Captured just before the run reads the hashes it will trust: a discard *after*
            # this point means the walk measured a database that no longer exists.
            generation = self._db.generation()
            identity = self._run_identity()
            known_hashes = self._db.document_hashes(str(root))
            weights_settled = False
            git = discovery.git_ignored(root) if self._gitignore else discovery.GitIgnore("off")
            scope = discovery.Scope(root, self._exclude, git.ignored)

            seen: set[str] = set()
            indexed = unchanged = sections_indexed = passages_indexed = 0
            failures: list[FileFailure] = []
            unreadable: list[str] = []

            def record_unreadable(error: OSError) -> None:
                location = str(error.filename or root)
                unreadable.append(location)
                failures.append(
                    FileFailure(
                        file_path=discovery._printable(location),
                        message=f"Cannot list directory: {error.strerror or error}",
                    )
                )

            def store(prepared: _Prepared) -> None:
                nonlocal weights_settled, indexed, unchanged, sections_indexed
                nonlocal passages_indexed
                if prepared.unchanged:
                    # Not an index: the document stands, and only the time it was last
                    # written is brought up to date. The certificate is not retracted for
                    # it either - nothing an answer is drawn from has changed.
                    self._db.record_modification_time(
                        prepared.file_path,
                        prepared.content_hash,
                        prepared.previous_mtime_ns,
                        prepared.mtime_ns,
                    )
                    unchanged += 1
                    return
                if prepared.has_vectors and not weights_settled:
                    # Once per run, on the one thread that writes, and only once a vector
                    # really exists to be written. A document of headings alone produces
                    # none, and asking would make a file that needs no model fail when no
                    # model can be loaded. The embedding is already spent by the time a
                    # refusal lands, but nothing is stored, which is what the guard is for.
                    self._settle_weights()
                    weights_settled = True
                # Here, and not in the worker: parsing and embedding can fail without
                # touching the index, and a run that changed nothing must leave a standing
                # certificate alone.
                about_to_write()
                self._db.replace_document(
                    file_path=prepared.file_path,
                    title=prepared.title,
                    content_hash=prepared.content_hash,
                    last_modified=prepared.last_modified,
                    mtime_ns=prepared.mtime_ns,
                    sections=prepared.sections,
                    vectors=prepared.vectors,
                    weights_revision=self._embedder.weights_revision,
                )
                indexed += 1
                sections_indexed += prepared.counts[0]
                passages_indexed += prepared.counts[1]

            # Workers embed; this thread writes. Embedding is ~17 s per file and a write
            # is under 5 ms, so nothing is gained by letting workers write and a great
            # deal is given up: SQLite takes one writer at a time anyway, and section ids
            # are allocated as documents are stored. Results are therefore drained in
            # submission order, so one tree builds one index whatever the worker count
            # (search breaks a scoring tie by where the walk reaches a section, not by id,
            # #103). The window bounds what is held in memory: a prepared file carries
            # every vector of every passage it has.
            # Walked to the end before anything is embedded, which costs nothing next to
            # embedding: a symlink is only recognised as an alias of a file once that file
            # has been seen, and it may come first. The order is the walk's all the same.
            files = iter(
                discovery.without_aliases(
                    list(
                        discovery.iter_markdown_files(
                            root, record_unreadable, scope=scope, should_stop=should_stop
                        )
                    )
                )
            )
            window = max(2 * self._workers, 2)
            pending: deque[tuple[str, Future[_Prepared | None]]] = deque()
            with ThreadPoolExecutor(
                max_workers=self._workers, thread_name_prefix="markdown-memory-index"
            ) as pool:
                try:
                    exhausted = False
                    while True:
                        if should_stop is not None and should_stop():
                            raise IndexCancelled("index run stopped by its owner")
                        while not exhausted and len(pending) < window:
                            path = next(files, None)
                            if path is None:
                                exhausted = True
                                break
                            file_path = str(path)
                            seen.add(file_path)
                            pending.append(
                                (
                                    file_path,
                                    pool.submit(
                                        self._prepare_file,
                                        path,
                                        known_hashes.get(file_path),
                                        identity,
                                    ),
                                )
                            )
                        if not pending:
                            break
                        file_path, future = pending.popleft()
                        try:
                            prepared = future.result()
                            # The write is inside the same guard as the read: storing one
                            # document can fail on its own - a vector the storage layer
                            # rejects, a row that will not go in - and that is this file's
                            # failure to carry, not the run's to die of.
                            if prepared is None:
                                unchanged += 1
                            elif should_stop is not None and should_stop():
                                # Checked again after the wait: embedding one file can take
                                # seconds, and a stop asked meanwhile writes nothing more.
                                raise IndexCancelled("index run stopped by its owner")
                            else:
                                store(prepared)
                        except (ModelLoadError, ForeignWeightsError):
                            raise  # not this file's fault: every other file fails the same
                        except (MarkdownMemoryError, OSError) as exc:
                            logger.warning(
                                "Failed to index %s: %s", discovery._printable(file_path), exc
                            )
                            failures.append(
                                FileFailure(
                                    file_path=discovery._printable(file_path), message=str(exc)
                                )
                            )
                            continue
                except BaseException:
                    # Whatever has not started will not start. What is already running is
                    # joined by the pool on the way out; there is nowhere to put its result.
                    for _, queued in pending:
                        queued.cancel()
                    raise

            vanished = self._vanished(scope, known_hashes, seen, unreadable)
            if vanished:
                about_to_write()  # deleting is changing it, even if no file was read
            purged = self._db.delete_documents(vanished)
            # This run's own failures are already in `errors`, and the search tools read
            # the recorded ones straight from the database, so nothing is added to
            # `notes`: a warning repeated in three places is how a warning becomes noise.
            # Only what this walk could have reached: a failure inside a pruned directory
            # or one that could not be listed is not this run's to forget, however far
            # under its root it sits.
            reachable, disowned = self._reachable(
                scope, self._db.failure_paths(str(root)), unreadable
            )
            if disowned:
                # An inner root indexed on its own may have been vouching for a tree whose
                # only fault was one of these rows; it may not go on doing so without it.
                about_to_write()
            self._db.record_failures(
                reachable,
                {
                    failure.file_path: (
                        f"{failure.message} (indexing {discovery._printable(str(root))})"
                    )
                    for failure in failures
                },
            )
            # Whole-database, so it cannot vouch for rows this walk never reached.
            self._db.settle_weights(self._embedder.weights_revision)
            # The walk finished, which is all this records; what it could not read is
            # recorded separately, and `index_status` refuses to call a tree whole while
            # anything under it is still listed there. Two facts, two places, one answer.
            self._db.mark_scan_complete(str(root), generation, git.state)
            report = IndexReport(
                directory=discovery._printable(str(root)),
                files_scanned=len(seen),
                files_indexed=indexed,
                files_unchanged=unchanged,
                files_purged=purged,
                sections_indexed=sections_indexed,
                passages_indexed=passages_indexed,
                elapsed_seconds=time.perf_counter() - started,
                errors=tuple(failures),
                notes=(*notices.values(), *_git_notes(git)),
            )
            self._db.dismiss_notices(notices)
        logger.info(report.summary())
        return report

    def _run_identity(self) -> str | None:
        """The weights this run embeds with, when they can be known before it embeds.

        A model that names its weights without loading (EmbeddingGemma) always answers.
        One that learns them by loading (bge-small) is loaded here only when a repair is
        pending - the index is being re-embedded, disagrees with some model, or holds
        vectors no revision vouches for - because such a run has embedding to do anyway;
        otherwise a run that changes nothing would pay for a model load. None means
        documents are compared on content and format alone, as before stamps existed; a
        stale one left behind that way keeps the certificate withheld at the end of the
        run, and the next run repairs it.
        """
        recorded = self._db.get_meta(WEIGHTS_META_KEY)
        if self._embedder.weights_revision is None and (
            # A tree that could reuse another's vectors needs to know whose they would be.
            self._reuse is not None
            or recorded == WEIGHTS_REVOKED
            or self._db.get_meta(WEIGHTS_MISMATCH_KEY) is not None
            or (recorded is None and self._db.count_rows("units_vec") > 0)
        ):
            try:
                self._embedder.warm_up()
            except ModelLoadError as exc:
                # Degrade rather than fail: a run whose files need no embedding can still
                # finish, and the recorded message keeps saying what is wrong.
                logger.warning("Model not loaded, stored weights not compared: %s", exc)
        return self._embedder.weights_revision

    def _settle_weights(self) -> None:
        """Make sure the index says it is being re-embedded before other weights write in.

        Called once per run, from the driver, at the first document that really has
        vectors: the earliest moment a lazily-loaded embedder can be asked what it is
        without making a run that needs no model load one, and on the only thread allowed
        to write what the answer implies. Weights that differ from the recorded ones - or
        any known weights joining vectors nobody vouched for - revoke the certificate and
        carry on: every search, whatever model it runs, then ranks on keywords alone until
        `settle_weights` finds every vector-bearing document stamped with one revision.
        Nothing is discarded first, so keyword search answers throughout, and a run killed
        partway resumes where it stopped, because each document's stamp is written with
        its vectors.

        Weights that cannot be named still refuse: vectors written now would be
        indistinguishable from the ones already stored, and no stamp could repair that.
        """
        if self._db.count_rows("units_vec") == 0:
            # No vector here for any of this to be about. Documents are the wrong
            # question: a file of nothing but headings is stored and embeds nothing, so an
            # index can hold documents and no vectors at all. Whatever this run is about
            # to write is therefore the whole of it, and it may say so - before the write
            # rather than after, so that another process reading these vectors a moment
            # from now finds them labelled. A run that dies in between leaves a revision
            # over no vectors, which the next one clears exactly here.
            self._claim_empty_index()
            return
        recorded = self._db.get_meta(WEIGHTS_META_KEY)
        # Already loaded: a worker embedded the document this is about to store. Kept
        # anyway, because `warm_up` is what makes `weights_revision` answerable and this
        # is called from tests and from runs whose first document came from a cache.
        self._embedder.warm_up()
        weights = self._embedder.weights_revision
        if weights is None:
            if recorded in {None, WEIGHTS_REVOKED}:
                # No provenance to contradict, or none left to protect: a revoked index is
                # already ranked by keyword alone, and only a run whose weights have a name
                # can stamp its way back out of that.
                return
            # The model loaded, so something answered - it just cannot say which weights
            # it is. That is not "nothing to compare": the vectors written now would be
            # unlabelled and indistinguishable from the ones already stored, which is the
            # state this guard exists to prevent.
            message = (
                f"Which weights {self._embedder.model_name} is running could not be read, so "
                "there is no way to tell whether they are the ones that built this index "
                f"({short_weights(recorded)}). Nothing has been discarded and no vector has been "
                "stored - a document of headings alone, which embeds nothing, may have "
                "been updated before this was reached; repair the model cache and run "
                "index_directory again."
            )
            self._db.record_weights_mismatch(message)
            self._db.revoke_coverage()
            raise ForeignWeightsError(message)
        if weights == recorded:
            self._db.record_weights_mismatch(None)
            return
        self._db.revoke_weights(
            f"{self._embedder.model_name} is re-embedding this index with other weights "
            f"({short_weights(weights)}). Only keyword ranking is used until every "
            "document has been re-embedded; semantic ranking resumes when index_directory "
            "finishes."
        )

    def _claim_empty_index(self) -> None:
        """Take ownership of an index that holds no vectors, and drop what described none.

        A model *name* is not enough for bge-small: fastembed pins no revision, so a
        re-download can bring different weights under the same name and nothing in the
        index would notice. This is the one moment a revision can honestly be written -
        the vectors that follow are all there will be, and there are none yet to
        contradict. A model that cannot say which weights it is records nothing, and the
        index carries no provenance rather than a provenance that might be wrong.
        """
        self._db.forget_weights_revision()
        self._db.record_weights_mismatch(None)
        self._embedder.warm_up()
        weights = self._embedder.weights_revision
        if weights is not None:
            self._db.set_meta(WEIGHTS_META_KEY, weights)

    def _reachable(
        self, scope: discovery.Scope, paths: Sequence[str], unreadable: Sequence[str]
    ) -> tuple[list[str], bool]:
        """The failure rows in ``paths`` this run may clear, and whether any left its scope.

        Pruned directories (`.venv`, `node_modules`) and directories that could not be
        listed are never entered, so this run saw nothing inside them and may not speak for
        what it did not see: their rows stay. Out-of-scope paths are the opposite - the run
        disowns them, and a failure recorded there must not keep this root's coverage
        "unknown" forever, since no run of it will ever look there again.

        Every component is tested, including the last. A recorded failure is usually a
        file, but an unreadable *directory* is recorded under its own path - and dropping
        the final component would ask whether `.venv`'s parent is walkable rather than
        whether `.venv` is, and then clear it.

        A path out of the walk's sight is still retired once it is observably gone
        (`discovery._certainly_gone`), or its row would outlive the file and no run could ever
        retire it.
        """
        root = scope.root
        blocked = tuple(location.rstrip(os.sep) + os.sep for location in unreadable)
        visitable: list[str] = []
        disowned = False
        for path in paths:
            if discovery._certainly_gone(root, path):
                visitable.append(path)
            elif not discovery._is_walkable(os.path.relpath(path, root).split(os.sep)):
                continue
            elif scope.excludes(path):
                visitable.append(path)
                disowned = True
            elif self._walk_would_visit(root, path, blocked):
                visitable.append(path)
        return visitable, disowned

    @staticmethod
    def _walk_would_visit(root: Path, path: str, blocked: tuple[str, ...]) -> bool:
        """Whether a walk of ``root`` reaches ``path``, given the directories it could not list."""
        if blocked and path.startswith(blocked):
            return False
        return not (discovery._behind_symlink(root, path) or discovery._is_shadowing_symlink(path))

    @staticmethod
    def _vanished(
        scope: discovery.Scope,
        known: Mapping[str, object],
        seen: set[str],
        unreadable: Sequence[str],
    ) -> list[str]:
        """Known documents that this walk *would* have found had they still existed.

        A document is only purged when its absence is evidence of deletion. It is kept
        when the walk could not have reached it: it lives under a directory that could
        not be listed, or under a pruned tree (``node_modules`` ...) that was indexed
        explicitly by pointing ``index_directory`` inside it.

        Unless the file is observably gone (`discovery._certainly_gone`). Not being visited is not
        evidence of deletion; `ENOENT` on that one name is exactly that evidence, and
        without it a deleted document under a pruned tree keeps answering searches with
        text that is not on disk any more, until someone re-indexes that tree by hand.

        Out of scope (`discovery.Scope`) is decided before the unreadable and symlink cases:
        it is a statement about the path, not about what the walk could see, so a worktree
        that has since become a symlink still has its copies purged.
        """
        root = scope.root
        blocked = tuple(location.rstrip(os.sep) + os.sep for location in unreadable)
        vanished: list[str] = []
        for file_path in sorted(set(known) - seen):
            if discovery._certainly_gone(root, file_path):
                vanished.append(file_path)
                continue
            relative = os.path.relpath(file_path, root)
            if not discovery._is_walkable(relative.split(os.sep)[:-1]):
                continue
            if scope.excludes(file_path):
                vanished.append(file_path)
                continue
            if blocked and file_path.startswith(blocked):
                continue
            if discovery._behind_symlink(root, file_path):
                continue
            vanished.append(file_path)
        return vanished

    def _prepare_file(
        self,
        path: Path,
        known: tuple[str, int, int | None, str | None] | None,
        identity: str | None,
    ) -> _Prepared | None:
        """Read, parse and embed one file. ``None`` when it is unchanged.

        Runs on a worker thread and writes nothing: every database write of a run belongs
        to the driver, so that section ids are handed out in walk order and the index's
        account of itself - certificate, provenance, failures - has a single author.
        """
        file_path = str(path)
        try:
            file_path.encode("utf-8")
        except UnicodeEncodeError:
            raise IndexingError("File name is not valid UTF-8; skipped") from None
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            # A FIFO or device named *.md would block or stream forever when read.
            raise IndexingError("Not a regular file; skipped")
        # Checked again on the descriptor: between that stat and this open the path can be
        # replaced by a FIFO, and a blocking open would then wait for a writer that may
        # never come - with a worker of the pool in its hand.
        data = discovery.read_regular_file(path)
        if data is None:
            raise IndexingError("Not a regular file; skipped")
        if len(data) > discovery.MAX_FILE_BYTES:
            raise IndexingError(f"File is larger than {discovery.MAX_FILE_BYTES} bytes; skipped")
        content_hash = discovery.hash_bytes(data)
        # The format counts as much as the content: a file whose bytes never changed still
        # has to be rebuilt if its vectors were pooled by an older scheme, or it would keep
        # them forever and the table would answer one query two different ways. So do the
        # weights, once this run knows its own: a document stamped by other ones is
        # re-embedded in place, which is how a changed model repairs the index file by file.
        if (
            known is not None
            and known[:2] == (content_hash, VECTOR_FORMAT)
            and (identity is None or known[3] == identity)
        ):
            if known[2] == info.st_mtime_ns:
                return None
            # Same bytes, a different timestamp: nothing to parse, embed or store, but the
            # time has to be written down or the freshness check hashes this file again on
            # every sweep from here on.
            return _Prepared(
                file_path=file_path,
                title="",
                content_hash=content_hash,
                last_modified=int(info.st_mtime),
                mtime_ns=info.st_mtime_ns,
                sections=(),
                vectors=(),
                unchanged=True,
                previous_mtime_ns=known[2],
            )
        parsed = self._parser.parse(
            data.decode("utf-8", errors="replace"), fallback_title=path.stem
        )
        # One embedding call per file, over the passages alone. The section vector is the
        # mean of its passages rather than a separate embedding of the whole section:
        # that text ran past the model's 512-token limit for 7.9% of the vendored corpus
        # and lost its tail, and embedding it cost one extra call per section.
        texts: list[str] = []
        for section in parsed.sections:
            texts.extend(section.unit_texts)
        # A vector is a function of the exact text and the weights alone - no path is part
        # of the input - so one another tree already holds for the same text is the one
        # this embedding would produce, and only the rest need the model.
        reused = (
            self._reuse(file_path, texts, identity)
            if self._reuse is not None and identity is not None and texts
            else {}
        )
        missing = [text for text in texts if text not in reused]
        fresh = self._embedder.embed_documents(missing)
        if len(fresh) != len(missing):
            raise EmbeddingError(f"Got {len(fresh)} vectors for {len(missing)} texts")
        computed = iter(fresh)
        embedded = iter([reused[text] if text in reused else next(computed) for text in texts])
        vectors = []
        for section in parsed.sections:
            units = tuple(next(embedded) for _ in section.units)
            vectors.append(SectionVectors(section=_section_vector(units), units=units))
        return _Prepared(
            file_path=file_path,
            title=parsed.title,
            content_hash=content_hash,
            last_modified=int(info.st_mtime),
            mtime_ns=info.st_mtime_ns,
            sections=tuple(parsed.sections),
            vectors=tuple(vectors),
        )


def _git_notes(git: discovery.GitIgnore) -> tuple[str, ...]:
    if git.state != "unavailable":
        return ()
    return (
        f"git could not list what it ignores here ({git.cause}), so files it ignores were "
        "indexed too.",
    )
