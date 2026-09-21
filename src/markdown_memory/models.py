"""Immutable, fully typed domain models.

Every entity is a frozen, slotted dataclass. ``to_dict`` methods produce the
JSON-serialisable payloads returned by the MCP tools.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

from pydantic import JsonValue

JsonDict: TypeAlias = dict[str, JsonValue]

PREAMBLE_TITLE = "[Overview / Preamble]"
PATH_SEPARATOR = " > "
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 characters per token), never below 1 for non-empty text."""
    if not text:
        return 0
    return max(1, -(-len(text) // CHARS_PER_TOKEN))


def part_path(base_path: str, part_index: int) -> str:
    """Breadcrumb for one part of an oversized section, e.g. ``A > B (Part 2)``."""
    return f"{base_path} (Part {part_index})"


@dataclass(slots=True, frozen=True)
class SectionDraft:
    """A section produced by the parser, not yet persisted.

    ``part_index`` is ``0`` for a section stored whole and ``1..n`` for the
    sequential parts of an oversized section. ``base_path`` is the breadcrumb
    without the ``(Part n)`` suffix. Line numbers are 1-based and inclusive.

    ``units`` are the section's passages as plain text - one per paragraph, list item,
    table row or code block. Each is embedded separately, because one vector for a
    whole section dilutes a single relevant table row beyond recognition. A section
    without units is heading-only and gets no vectors at all.
    """

    heading_title: str
    heading_level: int
    heading_path: str
    base_path: str
    content: str
    start_line: int
    end_line: int
    part_index: int = 0
    units: tuple[str, ...] = ()

    @property
    def embedding_text(self) -> str:
        """Text for the section-level vector: breadcrumb, then the body without markup."""
        return f"{self.heading_path}\n\n{' '.join(self.units)}"

    @property
    def unit_texts(self) -> tuple[str, ...]:
        """Texts for the passage-level vectors, each carrying the breadcrumb for context."""
        return tuple(f"{self.heading_path}: {unit}" for unit in self.units)


@dataclass(slots=True, frozen=True)
class SectionVectors:
    """Embeddings of one section: ``section`` is ``None`` for a heading-only section."""

    section: Sequence[float] | None
    units: tuple[Sequence[float], ...] = ()


@dataclass(slots=True, frozen=True)
class ParsedDocument:
    """Result of parsing one Markdown source."""

    title: str
    sections: tuple[SectionDraft, ...]
    line_count: int


@dataclass(slots=True, frozen=True)
class Document:
    """A persisted Markdown file."""

    id: int
    file_path: str
    title: str
    content_hash: str
    last_modified: int


@dataclass(slots=True, frozen=True)
class DocumentSummary:
    """A document together with the number of sections indexed for it."""

    file_path: str
    title: str
    section_count: int
    last_modified: int

    def to_dict(self) -> JsonDict:
        return {
            "file_path": self.file_path,
            "title": self.title,
            "section_count": self.section_count,
            "last_modified": self.last_modified,
        }


@dataclass(slots=True, frozen=True)
class Section:
    """A persisted section (or one part of an oversized section)."""

    id: int
    doc_id: int
    heading_title: str
    heading_level: int
    heading_path: str
    content: str
    start_line: int
    end_line: int
    part_index: int

    @property
    def base_path(self) -> str:
        """Breadcrumb with any ``(Part n)`` suffix removed."""
        if self.part_index <= 0:
            return self.heading_path
        suffix = part_path("", self.part_index)
        return self.heading_path.removesuffix(suffix)


@dataclass(slots=True, frozen=True)
class OutlineNode:
    """One heading in a document's hierarchical table of contents."""

    heading_title: str
    heading_level: int
    heading_path: str
    start_line: int
    end_line: int
    token_estimate: int
    part_count: int
    children: tuple[OutlineNode, ...] = ()

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "title": self.heading_title,
            "level": self.heading_level,
            "heading_path": self.heading_path,
            "lines": f"{self.start_line}-{self.end_line}",
            "tokens": self.token_estimate,
        }
        if self.part_count > 1:
            payload["parts"] = self.part_count
        if self.children:
            payload["children"] = [child.to_dict() for child in self.children]
        return payload


@dataclass(slots=True, frozen=True)
class SearchResult:
    """A section matched by hybrid search, with its fused and per-index ranks."""

    section_id: int
    file_path: str
    document_title: str
    heading_title: str
    heading_path: str
    content: str
    start_line: int
    end_line: int
    score: float
    fts_rank: int | None
    vec_rank: int | None
    matched_passage: str | None = None

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "file_path": self.file_path,
            "document_title": self.document_title,
            "heading_path": self.heading_path,
            "heading_title": self.heading_title,
            "lines": f"{self.start_line}-{self.end_line}",
            "score": round(self.score, 6),
            "fts_rank": self.fts_rank,
            "vec_rank": self.vec_rank,
            "tokens": estimate_tokens(self.content),
            "content": self.content,
        }
        if self.matched_passage is not None:
            payload["matched_passage"] = self.matched_passage
        return payload


@dataclass(slots=True, frozen=True)
class FileFailure:
    """A per-file failure recorded while indexing (the run itself continues)."""

    file_path: str
    message: str

    def to_dict(self) -> JsonDict:
        return {"file_path": self.file_path, "message": self.message}


#: Failures carried on a search answer. Enough to act on, few enough not to bury the
#: answer itself - the count in `message` says how many were left out.
MAX_REPORTED_FAILURES = 20


@dataclass(slots=True, frozen=True)
class IndexStatus:
    """Whether answers drawn from one tree can be trusted to be drawn from all of it.

    `verified` means a full walk of this scope finished and read every file it found. It
    is deliberately not a claim that the filesystem has stopped changing: a file created
    after its directory was walked is not in the index and not in `failures`, and the next
    run picks it up. A killed run and a tree nobody ever indexed both read unverified,
    which is the same answer because it is the same situation - nothing walked it whole.
    """

    verified: bool
    failures: tuple[FileFailure, ...] = ()
    #: Documents under this scope whose vectors were built by an older pooling scheme.
    #: They still answer, less well, and only a run over the directory holding them
    #: rebuilds - a parent run prunes `.venv`, `node_modules` and the like, so one indexed
    #: deliberately inside such a directory is never reached again.
    stale_vectors: int = 0

    def to_dict(self) -> JsonDict:
        shown = self.failures[:MAX_REPORTED_FAILURES]
        return {
            "coverage": "verified" if self.verified else "unknown",
            "failures": [failure.to_dict() for failure in shown],
            "message": self.message(),
        }

    def message(self) -> str | None:
        """One sentence, or nothing at all when there is nothing to act on."""
        if self.verified:
            return None
        if not self.failures:
            if self.stale_vectors:
                return (
                    f"{self.stale_vectors} document(s) here were indexed by an older "
                    "vector format and rank less well until the directory holding them is "
                    "indexed again."
                )
            return (
                "This documentation root has not been indexed end to end since it last "
                "changed, so an answer may be missing part of it. Run index_directory."
            )
        hidden = len(self.failures) - MAX_REPORTED_FAILURES
        more = f" (showing the first {MAX_REPORTED_FAILURES})" if hidden > 0 else ""
        return (
            f"{len(self.failures)} path(s) could not be indexed{more}; answers here are "
            "drawn from a tree that is missing them."
        )


@dataclass(slots=True, frozen=True)
class IndexReport:
    """Outcome of one ``index_directory`` run."""

    directory: str
    files_scanned: int
    files_indexed: int
    files_unchanged: int
    files_purged: int
    sections_indexed: int
    elapsed_seconds: float
    passages_indexed: int = 0
    errors: tuple[FileFailure, ...] = ()
    notes: tuple[str, ...] = ()

    def summary(self) -> str:
        lines = [
            f"Indexed {self.directory} in {self.elapsed_seconds:.2f}s: "
            f"{self.files_scanned} scanned, {self.files_indexed} (re)indexed, "
            f"{self.files_unchanged} unchanged, {self.files_purged} purged, "
            f"{self.sections_indexed} sections embedded ({self.passages_indexed} passages)."
        ]
        lines.extend(f"NOTE {note}" for note in self.notes)
        lines.extend(f"ERROR {error.file_path}: {error.message}" for error in self.errors)
        if self.errors:
            # Without this the run reads as a success with some noise attached, and an
            # index missing part of its tree answers questions as if it were whole.
            lines.append(
                f"INCOMPLETE: {len(self.errors)} file(s) could not be indexed; "
                "this documentation root is only partly searchable."
            )
        return "\n".join(lines)
