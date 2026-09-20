"""markdown-memory: AST-aware Markdown indexing and hybrid retrieval over MCP."""

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

__version__ = "0.1.0"

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
