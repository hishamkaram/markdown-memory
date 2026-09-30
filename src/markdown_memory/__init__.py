"""markdown-memory: AST-aware Markdown indexing and hybrid retrieval over MCP."""

from importlib.metadata import PackageNotFoundError, version

from markdown_memory.exceptions import (
    ASTParseError,
    DatabaseError,
    DocumentNotFoundError,
    EmbeddingError,
    IndexingError,
    MarkdownMemoryError,
    SectionNotFoundError,
)
from markdown_memory.models import (
    Document,
    DocumentSummary,
    IndexReport,
    OutlineNode,
    SearchResult,
    Section,
    SectionDraft,
)

try:
    # pyproject.toml is the one place the version is written; this reads it back from the
    # installed metadata rather than repeating it.
    __version__ = version("markdown-memory")
except PackageNotFoundError:  # a bare source tree nobody installed
    __version__ = "0+unknown"

__all__ = [
    "ASTParseError",
    "DatabaseError",
    "Document",
    "DocumentNotFoundError",
    "DocumentSummary",
    "EmbeddingError",
    "IndexReport",
    "IndexingError",
    "MarkdownMemoryError",
    "OutlineNode",
    "SearchResult",
    "Section",
    "SectionDraft",
    "SectionNotFoundError",
    "__version__",
]
