"""The on-disk model cache: where weights live, and how we know they are ours.

The embedder that uses this is in ``embedders.py``; the cache is its own module because
the pin, the manifest and the verification stamp are consulted from outside it too -
``scripts/eval_cache.py`` keys the retrieval gate on them, and the graph-patch tests read
the directory directly.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import stat
from collections.abc import Iterator, Mapping
from pathlib import Path

from markdown_memory.exceptions import ModelLoadError

BGE_SMALL_MODEL_NAME = "BAAI/bge-small-en-v1.5"
GEMMA_REPOSITORY = "onnx-community/embeddinggemma-300m-ONNX"


# Pinned so that an upstream re-export can never silently change stored vectors.
GEMMA_REVISION = "5090578d9565bb06545b4552f76e6bc2c93e4a66"


GEMMA_MODEL_FILE = "onnx/model_quantized.onnx"


# Size and sha256 of every file at GEMMA_REVISION, from the Hub's paths-info API. This is
# what stands between a damaged cache and onnxruntime: huggingface_hub checks only the
# size of what it downloads, and hands back a file that is already on disk without reading
# it at all.
GEMMA_MANIFEST: Mapping[str, tuple[int, str]] = {
    GEMMA_MODEL_FILE: (
        567_874,
        "172efde319fe1542dc41f31be6154910b05b78f7a861c265c4600eec906bd6d8",
    ),
    GEMMA_MODEL_FILE + "_data": (
        308_890_624,
        "705626e28e4c23c82ade34566b4197d97f534c12275fa406dfb71e9937d388c0",
    ),
    "tokenizer.json": (
        20_323_312,
        "4dda02faaf32bc91031dc8c88457ac272b00c1016cc679757d1c441b248b9c47",
    ),
}


GEMMA_FILES: tuple[str, ...] = tuple(GEMMA_MANIFEST)


# The graph the embedder actually runs is derived from the downloaded one on this machine:
# the same weights, with the vocabulary gathered before it is dequantized, which is worth
# about 1 GB per query (see graph_patch). It is never fetched, and it is regenerated - not
# re-downloaded - whenever it is missing or does not match. Its sha256 is also the
# rewrite's version: changing the rewriter changes this, and the tests say so.
DERIVED_GRAPH_FILE = "onnx/model_quantized.gather_first.onnx"


DERIVED_GRAPH_SHA256 = "ce47d05e0aa9abd97a474a7a951c2814060ddc2f9822dbb0ad30a407aa6e95ea"


# Versions before this one kept the files in `_GEMMA_DIR_PREFIX` itself, with no revision
# anywhere in the path: moving GEMMA_REVISION would have kept serving the old weights
# under a model name that claims to be the new ones. That folder is migrated, not
# re-downloaded, the first time this runs.
_GEMMA_DIR_PREFIX = "embeddinggemma-300m-onnx"


# One lock for every revision, so two versions starting at once still exclude each other.
_GEMMA_LOCK_NAME = f"{_GEMMA_DIR_PREFIX}.lock"


_VERIFIED_STAMP = ".verified"


def model_cache_root(cache_dir: Path | None = None) -> Path:
    """Where every model this package downloads is kept."""
    return cache_dir or Path.home() / ".cache" / "markdown-memory" / "models"


def gemma_model_dir(cache_dir: Path | None = None) -> Path:
    """The folder holding the pinned EmbeddingGemma revision.

    A *sibling* of the unversioned folder older versions used, never a child, so the
    migration can be one rename inside one directory - which no reader can catch
    half-done.
    """
    return model_cache_root(cache_dir) / f"{_GEMMA_DIR_PREFIX}-{GEMMA_REVISION[:12]}"


def fastembed_model_dir(cache_dir: Path | None) -> Path | None:
    """The folder fastembed keeps bge-small in, or None when it cannot be derived.

    Not guessable from the model name: fastembed downloads its own re-export of the
    model (`qdrant/bge-small-en-v1.5-onnx-q`), so the repository is read out of its
    registry rather than assumed.
    """
    if cache_dir is None:
        return None
    from fastembed import TextEmbedding

    for entry in TextEmbedding.list_supported_models():
        if not isinstance(entry, dict) or entry.get("model") != BGE_SMALL_MODEL_NAME:
            continue
        sources = entry.get("sources")
        repository = sources.get("hf") if isinstance(sources, dict) else None
        if isinstance(repository, str) and repository:
            return cache_dir / ("models--" + repository.replace("/", "--"))
    return None


def _cache_path(model_dir: Path, name: str) -> Path | None:
    """``model_dir/name``, or None when any directory on the way there is a symlink.

    `_file_identity` only ever looked at the last component, so an `onnx` that pointed
    somewhere else was trusted - and then repaired, which meant deleting and overwriting
    files outside the cache entirely.
    """
    current = model_dir
    for part in Path(name).parts:
        if current.is_symlink():
            return None
        current = current / part
    return current


def _file_identity(path: Path) -> dict[str, int] | None:
    """What a stamp remembers about one model file; None when it is not a plain file.

    `ctime_ns` earns its place: `cp -p`, `tar x` and `rsync --inplace` all rewrite a
    file's contents and then restore its old mtime, so size, mtime and inode can agree
    across different bytes. Nothing in user space can set ctime back.
    """
    try:
        info = path.lstat()
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None  # a symlink into a blob store is not a file this cache vouches for
    return {
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
        "inode": info.st_ino,
        "device": info.st_dev,
    }


def _stamped_files(model_dir: Path) -> list[str]:
    """Everything a stamp vouches for: the downloaded files, and the derived graph."""
    names = list(GEMMA_MANIFEST)
    if _identity_in(model_dir, DERIVED_GRAPH_FILE) is not None:
        names.append(DERIVED_GRAPH_FILE)
    return names


def _identity_in(model_dir: Path, name: str) -> dict[str, int] | None:
    """The identity of one cached file, refusing a path that leaves the cache."""
    path = _cache_path(model_dir, name)
    return None if path is None else _file_identity(path)


def _stamp_is_current(model_dir: Path) -> bool:
    """Whether every file still looks exactly as it did when it was last verified.

    Hashing 330 MB costs most of a second of one core, which is too much for every
    server start when an editor starts one per session. This is a handful of `stat`
    calls; anything that disagrees sends the files back to be hashed.
    """
    try:
        stamp = json.loads((model_dir / _VERIFIED_STAMP).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(stamp, dict) or stamp.get("revision") != GEMMA_REVISION:
        return False
    if stamp.get("derived") != DERIVED_GRAPH_SHA256:
        return False
    recorded = stamp.get("files")
    if not isinstance(recorded, dict) or set(recorded) != set(_stamped_files(model_dir)):
        return False  # a derived graph that has gone missing is regenerated, not ignored
    if model_dir.is_symlink():
        return False
    for name in recorded:
        identity = _identity_in(model_dir, name)
        # `None` means the path is not a regular file - a symlink, most likely. Nothing
        # should be able to stamp one (`_stamped_files` leaves it out, `_unverified`
        # calls it wrong and `_derive_graph` replaces it), and this is the line that
        # makes a stamp that somehow recorded `None` stop matching `None` for ever.
        if identity is None or recorded[name] != identity:
            return False
    return True


def _write_stamp(model_dir: Path) -> None:
    """Record what was just verified. Atomically: a half-written stamp is a false claim."""
    stamp = {
        "revision": GEMMA_REVISION,
        # The rewrite's version. Without it, a package update that moves the pin - or
        # fixes the rewriter - leaves the old derived graph in place, because the file
        # itself has not changed and every identity still matches.
        "derived": DERIVED_GRAPH_SHA256,
        "files": {name: _file_identity(model_dir / name) for name in _stamped_files(model_dir)},
    }
    temporary = model_dir / f"{_VERIFIED_STAMP}.{os.getpid()}"
    _remove(temporary)  # a stamp left behind by a crash under this same pid
    _write_new_file(temporary, json.dumps(stamp).encode("utf-8"))
    stamped = model_dir / _VERIFIED_STAMP
    if stamped.is_dir() and not stamped.is_symlink():
        # `os.replace` will not put a file where a directory is, and the stamp is read
        # back as simply invalid, so the repair it ends would run again on every start
        # and end the same way. Nothing of ours is ever a directory here.
        _remove(stamped)
    os.replace(temporary, stamped)


def _write_new_file(path: Path, payload: bytes) -> None:
    """Create ``path`` with its contents, refusing to follow a symlink or reuse a file.

    The temporary names these writes use are predictable (the pid), and a plain write
    follows a symlink planted at one of them: the caller would then overwrite whatever it
    points at, anywhere the user can write. `O_EXCL | O_NOFOLLOW` makes both refusals the
    kernel's.
    """
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
    except BaseException:
        with contextlib.suppress(OSError):
            path.unlink()
        raise


def _remove(path: Path) -> None:
    """Delete whatever sits at ``path``, file or directory.

    A directory where a model file belongs is not something `unlink` can clear, and a
    cache that cannot be repaired is a server that never starts again.
    """
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)
    if path.exists() or path.is_symlink():
        # Say so here, where the path is known, rather than failing three lines later on
        # a rename that cannot explain itself.
        raise ModelLoadError(f"Cannot clear {path} to repair the model cache")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _unverified(model_dir: Path) -> list[str]:
    """The model files that are missing, the wrong size, or the wrong bytes."""
    problems: list[str] = []
    for name, (size, checksum) in GEMMA_MANIFEST.items():
        path = _cache_path(model_dir, name)
        before = None if path is None else _file_identity(path)
        if path is None or before is None or before["size"] != size:
            problems.append(name)
        elif _hash_file(path) != checksum or _file_identity(path) != before:
            # Second identity: a file rewritten while it was being read was never hashed
            # as it now stands, so the answer that came back means nothing.
            problems.append(name)
    return problems


@contextlib.contextmanager
def _model_cache_lock(cache_dir: Path | None, *, exclusive: bool) -> Iterator[None]:
    """Serialise verification, repair and session construction across processes.

    Shared while a verified cache is being opened, exclusive while it is being changed,
    so nothing can repair files another process has verified but not yet handed to
    onnxruntime. The kernel drops a `flock` when the process dies, so a crash leaves
    nothing held. A model cache on NFS or SMB shared between machines is out of scope:
    `flock` can be local-only there - the boundary SQLite's WAL already has.
    """
    root = model_cache_root(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    with (root / _GEMMA_LOCK_NAME).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
