"""Builders shared by the regression modules (importable via ``pythonpath = ["tests"]``)."""

from __future__ import annotations

from fakes import FakeEmbedder, vectors_for

from markdown_memory.db import Database
from markdown_memory.models import ParsedDocument, SectionDraft
from markdown_memory.parser import MarkdownParser


def parse(text: str) -> ParsedDocument:
    return MarkdownParser().parse(text, fallback_title="fallback")


def paths(document: ParsedDocument) -> list[str]:
    return [section.heading_path for section in document.sections]


def draft(title: str, content: str) -> SectionDraft:
    return SectionDraft(
        heading_title=title, heading_level=2, heading_path=f"Doc > {title}",
        base_path=f"Doc > {title}", content=content, start_line=1, end_line=1,
        units=(content.split("\n\n", 1)[-1],),
    )  # fmt: skip


def store(db: Database, embedder: FakeEmbedder, file_path: str, count: int = 3) -> None:
    sections = [draft(f"S{n}", f"## S{n}\n\nbody number {n}") for n in range(count)]
    db.replace_document(
        file_path=file_path, title="Doc", content_hash="h", last_modified=1,
        mtime_ns=1, sections=sections,
        vectors=vectors_for(embedder, sections),
    )  # fmt: skip
