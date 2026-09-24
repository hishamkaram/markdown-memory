"""Turning what an agent typed into the section it meant.

Heading paths arrive as breadcrumbs (``Root > Child``) and have to survive casing,
spacing and ambiguity; the outline is the same tree seen from above.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from markdown_memory.exceptions import (
    MarkdownMemoryError,
    SectionNotFoundError,
)
from markdown_memory.models import (
    PATH_SEPARATOR,
    OutlineNode,
    Section,
    estimate_tokens,
)

_PATH_SEPARATOR_PATTERN = re.compile(r"\s*>\s*")


_MAX_LISTED_PATHS = 40


# ---------------------------------------------------------------------- pure helpers


def _user_path(text: str, error: type[MarkdownMemoryError]) -> Path:
    """``Path(text).expanduser()``; pathlib's RuntimeError/ValueError become ``error``."""
    try:
        text.encode("utf-8")  # a lone surrogate cannot be bound as a SQLite parameter
        return Path(text).expanduser()
    except (RuntimeError, ValueError) as exc:  # unknown ~user, embedded NUL, bad encoding
        raise error(f"Invalid path {text!r}: {exc}") from exc


def _absolute(path: Path, error: type[MarkdownMemoryError]) -> Path:
    try:
        resolved = path.expanduser().resolve()
        str(resolved).encode("utf-8")
    except (OSError, RuntimeError, ValueError) as exc:  # symlink loop, NUL, bad encoding
        raise error(f"Invalid path {str(path)!r}: {exc}") from exc
    return resolved


def normalize_heading_path(heading_path: str) -> str:
    """Canonicalise breadcrumb spacing: ``A>B`` and ``A  >  B`` both become ``A > B``."""
    return PATH_SEPARATOR.join(
        segment.strip() for segment in _PATH_SEPARATOR_PATTERN.split(heading_path.strip())
    )


def _casefolded(heading_path: str) -> str:
    return normalize_heading_path(heading_path).casefold()


# Progressively looser ways to compare a requested path with a stored one. The same
# transformation is applied to BOTH sides, so a title containing '>' ("Step 1 -> Step 2",
# "Result<T, E>") still matches itself, and an exact-case request beats a case-folded one
# (sibling headings "Setup" and "SETUP" stay individually addressable).
_MATCH_KEYS: tuple[Callable[[str], str], ...] = (str, normalize_heading_path, _casefolded)


def resolve_heading_path(candidates: Sequence[str], requested: str) -> str:
    """The one stored path that ``requested`` designates.

    Whole-path matches are tried before trailing fragments (``Child > Subchild`` or the
    bare heading title); within each, stricter comparisons come first. The first tier
    with any match decides: one match wins, several are reported as ambiguous.
    """
    for as_suffix in (False, True):
        for key in _MATCH_KEYS:
            wanted = key(requested)
            if as_suffix:
                ending = PATH_SEPARATOR + wanted
                matches = [path for path in candidates if key(path).endswith(ending)]
            else:
                matches = [path for path in candidates if key(path) == wanted]
            if len(matches) == 1:
                return matches[0]
            if matches:
                raise SectionNotFoundError(
                    f"'{requested}' is ambiguous; use one of (exact spelling): "
                    + " | ".join(matches[:_MAX_LISTED_PATHS])
                )
    available = " | ".join(candidates[:_MAX_LISTED_PATHS]) or "(document has no sections)"
    raise SectionNotFoundError(f"No section '{requested}'. Available heading paths: {available}")


def select_sections(
    sections: Sequence[Section], heading_path: str, *, include_subsections: bool = False
) -> list[Section]:
    """Sections addressed by ``heading_path``, in source order.

    ``heading_path`` may name a whole section (every part of it is returned) or a single
    ``(Part n)`` of an oversized one. ``include_subsections`` adds the section's
    descendants; it has no meaning for a single part and is ignored there.
    """
    requested = heading_path.strip()
    if not requested:
        raise SectionNotFoundError("heading_path must not be empty")
    base_paths = list(dict.fromkeys(section.base_path for section in sections))
    parts = {s.heading_path: s for s in sections if s.part_index > 0}
    part_paths = [path for path in parts if path not in base_paths]
    chosen = resolve_heading_path([*base_paths, *part_paths], requested)
    if chosen not in base_paths:
        return [parts[chosen]]

    selected: list[Section] = []
    level: int | None = None
    for section in sections:
        if section.base_path == chosen:
            level = section.heading_level
            selected.append(section)
        elif level is not None:
            # Descendants are the sections that follow until a heading at the same or a
            # shallower level; walking the order is immune to '>' inside titles.
            if not include_subsections or level < 1 or section.heading_level <= level:
                break
            selected.append(section)
    return selected


def build_outline(sections: Sequence[Section]) -> list[OutlineNode]:
    """Nest a document's sections into a hierarchical table of contents."""

    @dataclass(slots=True)
    class _Pending:
        title: str
        level: int
        path: str
        start_line: int
        end_line: int
        tokens: int
        parts: int
        children: list[_Pending]

    roots: list[_Pending] = []
    stack: list[_Pending] = []
    by_path: dict[str, _Pending] = {}
    for section in sections:
        existing = by_path.get(section.base_path)
        if existing is not None:  # a further part of an oversized section
            existing.end_line = max(existing.end_line, section.end_line)
            existing.tokens += estimate_tokens(section.content)
            existing.parts += 1
            continue
        node = _Pending(
            title=section.heading_title,
            level=section.heading_level,
            path=section.base_path,
            start_line=section.start_line,
            end_line=section.end_line,
            tokens=estimate_tokens(section.content),
            parts=1,
            children=[],
        )
        by_path[section.base_path] = node
        if node.level < 1:  # the preamble is a sibling of the headings, never their parent
            roots.append(node)
            continue
        while stack and stack[-1].level >= node.level:
            stack.pop()
        (stack[-1].children if stack else roots).append(node)
        stack.append(node)

    def freeze(node: _Pending) -> OutlineNode:
        return OutlineNode(
            heading_title=node.title,
            heading_level=node.level,
            heading_path=node.path,
            start_line=node.start_line,
            end_line=node.end_line,
            token_estimate=node.tokens,
            part_count=node.parts,
            children=tuple(freeze(child) for child in node.children),
        )

    return [freeze(node) for node in roots]
