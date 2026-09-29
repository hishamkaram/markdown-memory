"""Domain-specific exception hierarchy for markdown-memory."""

from __future__ import annotations


class MarkdownMemoryError(Exception):
    """Base class for every error raised deliberately by this package."""


class ConfigurationError(MarkdownMemoryError):
    """The server was started with configuration it cannot act on."""


class DatabaseError(MarkdownMemoryError):
    """The SQLite store could not be opened, migrated, read, or written."""


class ASTParseError(MarkdownMemoryError):
    """A Markdown source could not be tokenised into an AST."""


class IndexingError(MarkdownMemoryError):
    """The indexing pipeline failed (bad directory, unreadable file, ...)."""


class ForeignWeightsError(IndexingError):
    """The model about to embed is not the one whose vectors the index already holds.

    Raised from the one place that can tell - just before a vector is produced, where a
    lazily-loaded embedder has had to load and can finally say what it is. It aborts the
    whole run rather than failing one file, because every other file would fail the same
    way and for the same reason.
    """


class IndexBusyError(IndexingError):
    """Another indexing run holds the lock on this database.

    Its own class because the answer is "try again shortly", not "this failed": the
    caller did nothing wrong and nothing is broken.
    """


class IndexCancelled(Exception):  # noqa: N818 - a request honoured, not an error
    """An index run stopped because its owner asked it to, between two documents.

    Deliberately outside `MarkdownMemoryError`: the indexer's per-file handler catches
    that hierarchy and carries on with the next file, and a stop that was swallowed as
    one file's failure would not be a stop. What it leaves is what a killed run leaves -
    the documents written so far, and coverage withdrawn until a run finishes.
    """


class EmbeddingError(IndexingError):
    """The embedding model failed to load or to produce usable vectors."""


class ModelLoadError(EmbeddingError):
    """The embedding model itself cannot be loaded: no file can be embedded at all."""


class SearchError(MarkdownMemoryError):
    """A search index failed in a way the storage layer did not anticipate."""


class DocumentNotFoundError(MarkdownMemoryError):
    """The requested file is not present in the index."""


class SectionNotFoundError(MarkdownMemoryError):
    """The requested heading path does not exist in the indexed document."""
