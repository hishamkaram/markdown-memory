"""Domain-specific exception hierarchy for markdown-memory."""

from __future__ import annotations


class MarkdownMemoryError(Exception):
    """Base class for every error raised deliberately by this package."""


class DatabaseError(MarkdownMemoryError):
    """The SQLite store could not be opened, migrated, read, or written."""


class ASTParseError(MarkdownMemoryError):
    """A Markdown source could not be tokenised into an AST."""


class IndexingError(MarkdownMemoryError):
    """The indexing pipeline failed (bad directory, unreadable file, ...)."""


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
