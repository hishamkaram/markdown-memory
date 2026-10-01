"""Immutable, fully typed domain models.

Every entity is a frozen, slotted dataclass. ``to_dict`` methods produce the
JSON-serialisable payloads returned by the MCP tools.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

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


PART_PREVIEW_CHARS = 100  # ~25 tokens: enough to tell one part of a split section from another


def preview(text: str, limit: int = PART_PREVIEW_CHARS) -> str:
    """``text`` cut to ``limit`` characters at a word boundary, marked with an ellipsis.

    Text without whitespace to cut at - a long URL or identifier - is cut at ``limit`` itself.
    """
    if len(text) <= limit:
        return text
    cut = next((i for i in range(limit, 0, -1) if text[i].isspace()), 0)
    return text[: cut or limit].rstrip() + "…"


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
    part_preview: str | None = None  # how one part of a split section begins

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
        # No `matched_passage` here: it restates a passage of `content`; it would be paid twice.
        return payload

    def to_pointer(self) -> JsonDict:
        """Where a lower-ranked hit is, what reading it costs, and why it matched - not its content.

        `heading_path` is what `read_section` takes, verbatim: for one part of a split section it
        names that part, while the base path would reassemble every part and cost more than
        `tokens` says. Nothing that ranks it (`score`, `fts_rank`, `vec_rank`) - its position in
        the list is its rank.
        """
        pointer: JsonDict = {
            "file_path": self.file_path,
            "heading_path": self.heading_path,
            "lines": f"{self.start_line}-{self.end_line}",
            "tokens": estimate_tokens(self.content),
        }
        if self.matched_passage is not None:
            pointer["matched_passage"] = self.matched_passage
        if self.part_preview is not None:
            pointer["part_preview"] = self.part_preview
        return pointer


KeywordMatch: TypeAlias = Literal["matched", "no_match", "filtered", "no_terms", "unavailable"]

# Only `no_match` may say that nothing contains the terms: `filtered` saw candidates the gate
# refused, `no_terms` searched nothing and `unavailable` could not look.
_KEYWORD_MESSAGES: dict[KeywordMatch, str] = {
    "no_match": "No section in this documentation root contains any of the searched terms: "
    "any hits are semantic neighbours, not matches.",
    "filtered": "The top keyword candidates each cover too little of the query to count as a "
    "match: any hits are semantic neighbours only.",
    "no_terms": "The query has no searchable terms: any hits are semantic neighbours only.",
    "unavailable": "Keyword search failed for this query: any hits are semantic neighbours only.",
}
# The page a lookup of identifiers nothing contains gets instead of their neighbours. It
# speaks of the index only: a file created since the last walk is in no status count.
_ABSTAINED = (
    "No section in the current index contains any of the searched terms, so no hits are "
    "returned: an identifier looked up this way is not in the indexed documentation (a file "
    "added since the last index run would not be searched yet). If it abbreviates a concept, "
    "search again in plain words."
)


@dataclass(slots=True, frozen=True)
class SearchPage:
    """One search's hits, and whether keyword search found the query's terms at all.

    `matched` speaks of the ranking, not the page: at least one keyword candidate survived the
    gate. With `limit >= 2` one is always on the page; at `limit = 1` it can lose a score tie
    to the best vector-only hit, and saying `no_match` then would be false.
    """

    results: tuple[SearchResult, ...]
    keyword_match: KeywordMatch

    def keyword_message(self) -> str | None:
        if self.keyword_match == "no_match" and not self.results:
            return _ABSTAINED
        return _KEYWORD_MESSAGES.get(self.keyword_match)


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
    #: Set when the weights behind the model name changed under an existing index: the
    #: stored vectors and the vectors a query would produce now come from different
    #: models. Nothing is discarded, and nothing new is written, until it is resolved.
    weights_mismatch: str | None = None
    #: Documents under this scope whose vectors were built by an older pooling scheme.
    #: They still answer, less well, and only a run over the directory holding them
    #: rebuilds - a parent run prunes `.venv`, `node_modules` and the like, so one indexed
    #: deliberately inside such a directory is never reached again.
    stale_vectors: int = 0
    #: Indexed documents a cheap probe could not confirm are still what was indexed: their
    #: bytes differ, or they are gone, unreadable, or no longer a regular file. It is
    #: best-effort in both directions. Counted over the rows the index holds, so a file
    #: nobody has indexed yet is not in it - finding those needs the directory walk, which
    #: is the expensive half. And the bytes are only read where the modification time moved,
    #: so an edit that restores a file's own timestamp is not seen. Zero means nothing was
    #: detected, not that every indexed file was hashed.
    changed_files: int = 0
    #: This server's own background run is indexing the tree right now. A hint about one
    #: process only: another process's run shows as coverage withdrawn, as it always did.
    indexing: bool = False
    #: What git said about the root the last time a walk of it finished: `applied`, `off`,
    #: `no_repository`, `unavailable` (git could not be asked, so what it ignores was indexed
    #: too), or `unknown` when no walk has finished since the index was built.
    gitignore: str = "unknown"

    def to_dict(self) -> JsonDict:
        shown = self.failures[:MAX_REPORTED_FAILURES]
        return {
            "coverage": "verified" if self.verified else "unknown",
            "failures": [failure.to_dict() for failure in shown],
            "changed_files": self.changed_files,
            "indexing": self.indexing,
            "gitignore": self.gitignore,
            "message": self.message(),
        }

    def message(self) -> str | None:
        """One sentence, or nothing at all when there is nothing to act on."""
        # First, because it is the only one that says the answers themselves may be
        # wrong rather than incomplete. Hoisted above `verified` rather than left below
        # it: the two cannot both hold today, and a reader should not have to know that.
        if self.weights_mismatch:
            return self.weights_mismatch
        if self.indexing:
            # Before everything that ends in "run index_directory": that run would be
            # refused as busy, and it would be refused for doing what is already being done.
            return (
                "An automatic index run is in progress, so an answer may be missing a file "
                "changed or added since the last one finished; there is no need to run "
                "index_directory."
            )
        if self.verified:
            # A walk that finished still describes the moment it finished. Files edited
            # since are the one thing a verified tree has left to say.
            if self.changed_files:
                return (
                    f"The last full index completed, but {self.changed_files} indexed "
                    "document(s) can no longer be confirmed to be what was indexed - "
                    "changed, unreadable or gone - so an answer may quote text that is no "
                    "longer there; run index_directory to refresh. The check is cheap and "
                    "best-effort: files created since that scan are not counted, and an "
                    "edit that puts a file's modification time back is not seen."
                )
            if self.gitignore == "unavailable":
                # Last: the tree is whole, only perhaps larger than the project's own docs.
                return (
                    "git could not list the files it ignores when this root was last indexed, "
                    "so generated or ignored Markdown may be among the results; the server "
                    "log says why."
                )
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
