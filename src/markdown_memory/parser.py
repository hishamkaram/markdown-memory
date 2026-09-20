"""Defensive, AST-aware Markdown sectioniser.

The parser tokenises Markdown with ``markdown-it-py`` and uses the block tokens'
source maps to cut the *verbatim* source into heading-delimited sections. Only
headings that are real top-level AST nodes open a section, so ``#`` lines inside
code fences, block quotes or list items never do.

Defences against real-world documentation:

* **Skipped levels** (``#`` straight to ``####``) - a heading stack pops every
  entry with level >= L before pushing, so breadcrumbs stay well formed.
* **Preamble** - badges/summaries before the first heading become
  ``[Overview / Preamble]``.
* **Front matter** - a leading block that really is YAML is kept out of the AST
  (CommonMark would misread it as a thematic break plus a setext heading) and mined
  for ``title:``. A leading ``---`` rule followed by prose is left alone.
* **Unclosed code fences** - CommonMark runs such a fence to EOF (or to the closing
  marker of some *later* fence), swallowing the sections in between. The fence is
  instead closed before the next plausible heading and parsing resumes there.
* **Oversized / heading-less text** - split on paragraph boundaries into
  ``Path (Part n)`` parts that reassemble byte-for-byte. A fenced block is only cut
  when it exceeds the limit on its own.
* **Colliding breadcrumbs** - repeated paths get a ``[n]`` suffix, also against
  generated ``(Part n)`` paths, so every stored path addresses exactly one thing.

Every section is additionally broken into *units* (``extract_units``): the plain text of
each paragraph, list item, table row and code block. They feed passage-level embeddings.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from markdown_it import MarkdownIt
from markdown_it.token import Token

from markdown_memory.exceptions import ASTParseError
from markdown_memory.models import (
    PATH_SEPARATOR,
    PREAMBLE_TITLE,
    ParsedDocument,
    SectionDraft,
    part_path,
)

DEFAULT_MAX_SECTION_CHARS = 3200  # ~800 tokens
MAX_UNITS_PER_SECTION = 64
# Every fence repair re-tokenises the rest of the document. A budget keeps the worst case
# linear: once it is spent, the remaining fences are left as markdown-it paired them.
MAX_REPAIR_SCANS = 16
MAX_UNIT_CHARS = 600
UNTITLED_HEADING = "(untitled)"

_FENCE_OPEN = re.compile(r"^( *)(`{3,}|~{3,})(.*)$")
_MAX_TOP_LEVEL_INDENT = 3
# Fences whose purpose is to *show* Markdown: another fence inside them is sample text.
_MARKUP_SAMPLE_LANGUAGES = frozenset(
    {"", "markdown", "md", "mdx", "mdown", "text", "txt", "plain", "plaintext", "rst", "html"}
)
_ATX_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+\S")
_FRONT_MATTER_TITLE = re.compile(r"^title\s*:\s*(.+?)\s*$", re.IGNORECASE)
_YAML_KEY = re.compile(r"""^(?:"[^"]+"|'[^']+'|[A-Za-z_][^:#]*?)\s*:(\s|$)""")
_HTML_TAG_NAME = re.compile(r"^</?([A-Za-z][A-Za-z0-9-]*)")
# Inline HTML that only styles a heading. Any other "tag" is kept as title text: in
# technical docs `Option<T>` or `<details>` in a heading is almost always literal.
_FORMATTING_TAGS = frozenset(
    {
        "a", "abbr", "b", "br", "code", "del", "em", "font", "i", "img", "ins", "kbd",
        "mark", "s", "small", "span", "strong", "sub", "sup", "u",
    }
)  # fmt: skip
_WHITESPACE = re.compile(r"\s+")
_HTML_TAG = re.compile(r"<[^>]+>")
_BLOCK_CLOSERS = {
    "paragraph_open": "paragraph_close",
    "heading_open": "heading_close",
    "table_open": "table_close",
    "bullet_list_open": "bullet_list_close",
    "ordered_list_open": "ordered_list_close",
    "blockquote_open": "blockquote_close",
}

# Languages where a line starting with "# " cannot be a comment. Inside an unclosed
# fence of any *other* language a level-1 "# ..." line is assumed to be a comment.
_HASH_IS_NOT_COMMENT = frozenset(
    {
        "c", "cpp", "c++", "cs", "csharp", "css", "go", "golang", "html", "java",
        "javascript", "js", "json", "jsonc", "jsx", "kotlin", "lua", "rs", "rust",
        "scss", "sql", "swift", "ts", "tsx", "typescript", "xml",
    }
)  # fmt: skip


@dataclass(slots=True, frozen=True)
class _Heading:
    line: int  # 0-based index of the heading's first source line
    level: int
    title: str


@dataclass(slots=True, frozen=True)
class _Fence:
    """A top-level fenced block; line numbers are 0-based indexes into the document."""

    start: int  # the opening fence line
    stop: int  # exclusive: the line after the closing marker, or EOF when unclosed
    marker: str
    language: str
    has_info: bool
    closed: bool


@dataclass(slots=True, frozen=True)
class _Scan:
    """What one tokenisation pass found from some line to the end of the document."""

    headings: tuple[_Heading, ...]
    fences: tuple[_Fence, ...]

    @property
    def trailing_open_fence(self) -> _Fence | None:
        """An unclosed fence necessarily runs to EOF, so it can only be the last one."""
        if self.fences and not self.fences[-1].closed:
            return self.fences[-1]
        return None


@dataclass(slots=True)
class _ScanBudget:
    """How many more whole-document tokenisations fence repair may spend."""

    remaining: int

    def spend(self) -> bool:
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        return True


@dataclass(slots=True, frozen=True)
class _Block:
    """A blank-line-delimited run of text inside one section (character offsets)."""

    start: int
    end: int
    has_fence: bool


class MarkdownParser:
    """Turn Markdown source into breadcrumbed, size-bounded sections."""

    def __init__(self, max_section_chars: int = DEFAULT_MAX_SECTION_CHARS) -> None:
        if max_section_chars < 1:
            raise ValueError("max_section_chars must be positive")
        self._max_chars = max_section_chars
        self._md = MarkdownIt("commonmark").enable("table")

    def parse(self, text: str, *, fallback_title: str = "Untitled") -> ParsedDocument:
        """Parse ``text``; never raises for malformed Markdown, only for tokeniser failure."""
        lines = _split_lines(text)
        body_start, front_matter_title = _scan_front_matter(lines)
        headings = self._find_headings(lines, body_start)

        used_paths: set[str] = set()
        first_heading_line = headings[0].line if headings else len(lines)
        sections: list[SectionDraft] = self._build_section(
            lines,
            start=0,
            stop=first_heading_line,
            title=PREAMBLE_TITLE,
            level=0,
            path=PREAMBLE_TITLE,
            has_heading_line=False,
        )
        used_paths.update(section.heading_path for section in sections)
        used_paths.add(PREAMBLE_TITLE)

        stack: list[tuple[int, str]] = []
        next_occurrence: dict[str, int] = {}
        for index, heading in enumerate(headings):
            while stack and stack[-1][0] >= heading.level:
                stack.pop()
            parents = [title for _, title in stack]
            stop = headings[index + 1].line if index + 1 < len(headings) else len(lines)
            # Every stored path must address exactly one thing: a candidate is rejected
            # when it - or any "(Part n)" path generated from it - is already taken. The
            # section is built once; trying another name only renames the drafts.
            base_path = PATH_SEPARATOR.join([*parents, heading.title])
            drafts = self._build_section(
                lines,
                start=heading.line,
                stop=stop,
                title=heading.title,
                level=heading.level,
                path=base_path,
                has_heading_line=True,
            )
            occurrence = next_occurrence.get(base_path, 1)
            while True:
                title = heading.title if occurrence == 1 else f"{heading.title} [{occurrence}]"
                path = PATH_SEPARATOR.join([*parents, title])
                claimed = {path, *(part_path(path, d.part_index) for d in drafts if d.part_index)}
                if not claimed & used_paths:
                    break
                occurrence += 1
            next_occurrence[base_path] = occurrence + 1
            if occurrence > 1:
                drafts = [
                    replace(
                        draft,
                        heading_title=title,
                        base_path=path,
                        heading_path=part_path(path, draft.part_index)
                        if draft.part_index
                        else path,
                    )
                    for draft in drafts
                ]
            used_paths |= claimed
            stack.append((heading.level, title))
            sections.extend(drafts)

        return ParsedDocument(
            title=_pick_title(headings, front_matter_title, fallback_title),
            sections=tuple(sections),
            line_count=len(lines),
        )

    # ------------------------------------------------------------------ AST walk

    def _tokenize(self, source: str) -> list[Token]:
        try:
            return self._md.parse(source)
        except Exception as exc:  # markdown-it has no dedicated error type
            raise ASTParseError(f"markdown-it failed to tokenise the document: {exc}") from exc

    def _find_headings(self, lines: Sequence[str], body_start: int) -> list[_Heading]:
        """Collect top-level headings, re-scanning from wherever a broken fence is repaired."""
        headings: list[_Heading] = []
        offset = body_start
        budget = _ScanBudget(MAX_REPAIR_SCANS)
        while offset < len(lines):
            scan = self._scan(lines, offset)
            resume_at = self._resume_point(lines, scan, budget) if budget.spend() else None
            headings.extend(
                heading
                for heading in scan.headings
                if resume_at is None or heading.line < resume_at
            )
            if resume_at is None:
                break
            offset = resume_at  # always beyond `offset`: the loop terminates
        return headings

    def _scan(self, lines: Sequence[str], offset: int) -> _Scan:
        chunk = lines[offset:]
        tokens = self._tokenize("\n".join(chunk))
        headings: list[_Heading] = []
        fences: list[_Fence] = []
        for position, token in enumerate(tokens):
            if token.level != 0 or token.map is None:
                continue
            first, stop = token.map
            if token.type == "heading_open":
                inline = tokens[position + 1] if position + 1 < len(tokens) else None
                headings.append(
                    _Heading(
                        line=offset + first, level=int(token.tag[1:]), title=_inline_text(inline)
                    )
                )
            elif token.type == "fence":
                info = token.info.strip()
                closing = stop - 1
                fences.append(
                    _Fence(
                        start=offset + first,
                        stop=offset + stop,
                        marker=token.markup,
                        language=info.split(maxsplit=1)[0].lower() if info else "",
                        has_info=bool(info),
                        closed=closing > first
                        and closing < len(chunk)
                        and _is_closing_fence(chunk[closing], token.markup, _MAX_TOP_LEVEL_INDENT),
                    )
                )
        return _Scan(headings=tuple(headings), fences=tuple(fences))

    def _resume_point(self, lines: Sequence[str], scan: _Scan, budget: _ScanBudget) -> int | None:
        """Line at which a mis-paired fence should be cut and scanning restarted.

        Two symptoms give an unclosed fence away:

        1. A *code* fence contains an opening marker with an info string (a ``bash`` block
           holding a line `` ```python ``) of the same kind and at least the same length.
           That line cannot close the outer fence and is not code, so the outer fence was
           never closed and merely ended at the inner block's closing marker. Fences that
           exist to show Markdown (``markdown``, ``md``, ``text``, no language ...) are
           exempt: there an inner fence is the sample itself.
        2. The document ends inside a fence. If that last fence carries an info string it
           is the culprit. If it is a bare marker it is more likely the orphaned *closer*
           of an earlier block whose opener got paired with the wrong marker; the earliest
           fence whose repair makes the rest of the document well formed is then cut.

        Cutting a fence that merely *looks* closed re-pairs every later marker, which can
        turn real headings into code. Such a cut is therefore only accepted when it keeps
        every heading the unrepaired scan already found (``_keeps_headings``), and for
        symptom 2 only at a line followed by a blank line, as headings are and code
        comments are not.
        """
        for fence in scan.fences:
            if fence.language in _MARKUP_SAMPLE_LANGUAGES:
                continue
            nested = _nested_opener(lines, fence)
            if nested is not None:
                resume_at = _recovery_line(lines, fence, nested)
                if resume_at is not None and self._keeps_headings(lines, scan, resume_at, budget):
                    return resume_at
        last = scan.trailing_open_fence
        if last is None:
            return None
        if not last.has_info:
            for fence in scan.fences[:-1]:
                if fence.language in _MARKUP_SAMPLE_LANGUAGES:
                    continue  # a heading inside a Markdown sample is the sample, not a symptom
                resume_at = _recovery_line(lines, fence, fence.stop - 1, strict=True)
                if resume_at is not None and self._keeps_headings(
                    lines, scan, resume_at, budget, must_end_closed=True
                ):
                    return resume_at
        return _recovery_line(lines, last, len(lines))

    def _keeps_headings(
        self,
        lines: Sequence[str],
        scan: _Scan,
        resume_at: int,
        budget: _ScanBudget,
        *,
        must_end_closed: bool = False,
    ) -> bool:
        """True when re-scanning from ``resume_at`` loses no heading ``scan`` reported.

        Each call tokenises the rest of the document, so it draws on ``budget``; with the
        budget spent the candidate is rejected unchecked (no repair beats a wrong one).
        """
        if not budget.spend():
            return False
        repaired = self._scan(lines, resume_at)
        if must_end_closed and repaired.trailing_open_fence is not None:
            return False
        found = {heading.line for heading in repaired.headings}
        return all(heading.line in found for heading in scan.headings if heading.line >= resume_at)

    # ------------------------------------------------------------------ sections

    def _build_section(
        self,
        lines: Sequence[str],
        *,
        start: int,
        stop: int,
        title: str,
        level: int,
        path: str,
        has_heading_line: bool,
    ) -> list[SectionDraft]:
        """Build the section for ``lines[start:stop]``, split into parts if oversized."""
        while start < stop and not lines[start].strip():
            start += 1
        while stop > start and not lines[stop - 1].strip():
            stop -= 1
        if start >= stop:
            return []
        content = "\n".join(lines[start:stop])
        first_line = start + 1  # 1-based
        if len(content) <= self._max_chars:
            return [
                SectionDraft(
                    heading_title=title,
                    heading_level=level,
                    heading_path=path,
                    base_path=path,
                    content=content,
                    start_line=first_line,
                    end_line=stop,
                    units=self.extract_units(content, skip_heading=has_heading_line),
                )
            ]
        # Only trailing newlines are dropped from a part, and they are recoverable from
        # the line numbers, so join_parts() can rebuild the section byte-for-byte.
        spans = split_into_spans(content, self._max_chars, glue_first=has_heading_line)
        table_headers = _table_headers(lines[start:stop])
        drafts: list[SectionDraft] = []
        for number, (begin, end) in enumerate(spans, start=1):
            part = content[begin:end].rstrip("\n")
            first_part_line = content.count("\n", 0, begin)
            part_start = first_line + first_part_line
            # A part that starts in the middle of a table has lost the header row, and
            # without it the rows are just a paragraph. Units are extracted from the part
            # with its header restored; the stored content stays verbatim.
            header = (
                table_headers.get(first_part_line)
                if content[begin - 1 : begin] in {"", "\n"}
                else None
            )
            drafts.append(
                SectionDraft(
                    heading_title=title,
                    heading_level=level,
                    heading_path=part_path(path, number),
                    base_path=path,
                    content=part,
                    start_line=part_start,
                    end_line=part_start + part.count("\n"),
                    part_index=number,
                    units=self.extract_units(
                        f"{header}\n{part}" if header else part,
                        skip_heading=has_heading_line and number == 1,
                    ),
                )
            )
        return drafts

    # ------------------------------------------------------------------ units

    def extract_units(self, content: str, *, skip_heading: bool = False) -> tuple[str, ...]:
        """Plain-text passages of ``content``: paragraphs, list items, table rows, code blocks.

        Table rows are rendered as ``Header: cell; Header: cell`` so that a row keeps its
        meaning without the rest of the table. ``skip_heading`` drops the section's own
        heading (its text already lives in the breadcrumb).
        """
        tokens = self._tokenize(content)
        units: list[str] = []
        index = 0
        heading_skipped = not skip_heading
        while index < len(tokens) and len(units) < MAX_UNITS_PER_SECTION:
            token = tokens[index]
            if token.level != 0:
                index += 1
                continue
            end = _block_end(tokens, index)
            block = tokens[index : end + 1]
            if token.type == "heading_open" and not heading_skipped:
                heading_skipped = True
            elif token.type == "table_open":
                units.extend(_table_rows(block))
            elif token.type in {"bullet_list_open", "ordered_list_open"}:
                units.extend(_list_items(block))
            elif token.type == "blockquote_open":
                units.extend(_leaf_texts(block))  # one unit per quoted paragraph / code block
            else:
                units.append(" ".join(_leaf_texts(block)))
            index = end + 1
        cleaned = (_WHITESPACE.sub(" ", unit.replace("|", " ")).strip() for unit in units)
        return tuple(unit[:MAX_UNIT_CHARS] for unit in cleaned if unit)[:MAX_UNITS_PER_SECTION]


# ---------------------------------------------------------------------- helpers


_TABLE_DELIMITER = re.compile(
    r"^ {0,3}\|?[ \t]*:?-{1,}:?[ \t]*(\|[ \t]*:?-{1,}:?[ \t]*)*\|?[ \t]*$"
)


def _table_headers(lines: Sequence[str]) -> dict[int, str]:
    """Map each table *body* row (by line index) to its table's header + delimiter rows."""
    headers: dict[int, str] = {}
    index = 0
    while index + 1 < len(lines):
        is_header = "|" in lines[index] and "|" in lines[index + 1]
        if is_header and _TABLE_DELIMITER.match(lines[index + 1]) and lines[index].strip():
            header = f"{lines[index]}\n{lines[index + 1]}"
            index += 2
            while index < len(lines) and lines[index].strip() and "|" in lines[index]:
                headers[index] = header
                index += 1
        else:
            index += 1
    return headers


def _split_lines(text: str) -> list[str]:
    normalized = text.removeprefix("﻿").replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _scan_front_matter(lines: Sequence[str]) -> tuple[int, str | None]:
    """Return ``(first body line, front-matter title)`` for a leading YAML block."""
    if not lines or lines[0].rstrip() != "---":
        return 0, None
    for index in range(1, len(lines)):
        if lines[index].rstrip() in {"---", "..."}:
            block = lines[1:index]
            if not _looks_like_yaml(block):
                return 0, None
            title: str | None = None
            for line in block:
                match = _FRONT_MATTER_TITLE.match(line)
                if match:
                    title = match.group(1).strip("\"'") or None
                    break
            return index + 1, title
    return 0, None


# "Note: read this first" between two rules is a setext heading, not metadata. A lone
# `key: value` line is otherwise taken for front matter, as every site generator does.
_ADMONITION_KEYS = frozenset(
    {"note", "warning", "tip", "todo", "important", "caution", "hint", "see also", "example"}
)


def _looks_like_yaml(block: Sequence[str]) -> bool:
    """Distinguish front matter from a document that merely opens with a ``---`` rule.

    Every non-blank line must be a mapping key, a list item, a comment, a continuation
    or a closing flow bracket, and at least one top-level key must exist. A ``# ...``
    line followed by a blank line is taken for a Markdown heading, not a YAML comment.
    """
    has_key = False
    content_lines = [line for line in block if line.strip()]
    if len(content_lines) == 1:
        key = content_lines[0].split(":", 1)[0].strip().strip("\"'").lower()
        if key in _ADMONITION_KEYS:
            return False
    for index, line in enumerate(block):
        stripped = line.strip()
        if not stripped:
            continue
        if _YAML_KEY.match(line):
            has_key = True
        elif stripped.startswith("#"):
            followed_by_blank = index + 1 < len(block) and not block[index + 1].strip()
            if _ATX_HEADING.match(line) and followed_by_blank:
                return False
        elif not (line[0] in " \t" or stripped.startswith("- ") or stripped in {"-", "]", "}"}):
            return False
    return has_key


def _plain_inline(inline: Token | None) -> str:
    """Visible text of an inline token: markup stripped, code and image alt text kept."""
    if inline is None or inline.type != "inline":
        return ""
    fragments: list[str] = []
    for child in inline.children or []:
        if child.type in {"text", "code_inline", "image"}:
            fragments.append(child.content)
        elif child.type in {"softbreak", "hardbreak"}:
            fragments.append(" ")
        elif child.type == "html_inline":
            tag = _HTML_TAG_NAME.match(child.content)
            # Case-sensitive on purpose: `<u>` is underline, `<U>` is a type parameter.
            if tag is not None and tag.group(1) not in _FORMATTING_TAGS:
                fragments.append(child.content)  # `Option<T>`: a type parameter, not markup
            elif tag is not None and tag.group(1) == "br":
                fragments.append(" ")  # the only line break a table cell can hold
    return _WHITESPACE.sub(" ", "".join(fragments)).strip()


def _inline_text(inline: Token | None) -> str:
    """Plain text of a heading's inline token, never empty."""
    return _plain_inline(inline) or UNTITLED_HEADING


def _leaf_texts(block: Sequence[Token]) -> list[str]:
    """Text of every leaf in ``block``, at any depth: inline runs, code and raw HTML."""
    texts: list[str] = []
    for token in block:
        if token.type == "inline":
            texts.append(_plain_inline(token))
        elif token.type in {"fence", "code_block"}:
            texts.append(token.content)
        elif token.type == "html_block":
            texts.append(_HTML_TAG.sub(" ", token.content))
    return [text for text in texts if text.strip()]


def _block_end(tokens: Sequence[Token], start: int) -> int:
    """Index of the token closing the top-level block that opens at ``start``."""
    closer = _BLOCK_CLOSERS.get(tokens[start].type)
    if closer is None:
        return start
    for index in range(start + 1, len(tokens)):
        if tokens[index].type == closer and tokens[index].level == tokens[start].level:
            return index
    return len(tokens) - 1


def _table_rows(block: Sequence[Token]) -> list[str]:
    """One unit per body row, each cell labelled with its column header."""
    headers: list[str] = []
    rows: list[str] = []
    cells: list[str] = []
    in_head = False
    for position, token in enumerate(block):
        if token.type == "thead_open":
            in_head = True
        elif token.type == "thead_close":
            in_head = False
        elif token.type in {"th_open", "td_open"}:
            text = _plain_inline(block[position + 1] if position + 1 < len(block) else None)
            (headers if in_head else cells).append(text)
        elif token.type == "tr_close" and not in_head:
            labelled = [
                f"{header}: {cell}" if header else cell
                for header, cell in zip(headers + [""] * len(cells), cells, strict=False)
                if cell
            ]
            rows.append("; ".join(labelled))
            cells = []
    return rows


def _list_items(block: Sequence[Token]) -> list[str]:
    """One unit per top-level list item, nested content included."""
    items: list[str] = []
    fragments: list[str] = []
    for token in block:
        if token.type == "list_item_open" and token.level == 1:
            fragments = []
        elif token.type in {"inline", "fence", "code_block", "html_block"}:
            fragments.extend(_leaf_texts([token]))
        elif token.type == "list_item_close" and token.level == 1:
            items.append(" ".join(fragments))
    return items


def _pick_title(headings: Sequence[_Heading], front_matter_title: str | None, fallback: str) -> str:
    for heading in headings:
        if heading.level == 1 and heading.title != UNTITLED_HEADING:
            return heading.title
    if front_matter_title:
        return front_matter_title
    for heading in headings:
        if heading.title != UNTITLED_HEADING:
            return heading.title
    return fallback


def _is_closing_fence(line: str, marker: str, max_indent: int) -> bool:
    stripped = line.strip()
    indent = len(line) - len(line.lstrip(" "))
    return (
        indent <= max_indent
        and len(stripped) >= len(marker)
        and stripped == marker[0] * len(stripped)
    )


def _opening_fence(line: str) -> tuple[int, str, str] | None:
    """``(indent, marker, info)`` when ``line`` can open a fenced block."""
    match = _FENCE_OPEN.match(line)
    if match is None:
        return None
    indent, marker, info = len(match.group(1)), match.group(2), match.group(3).strip()
    if marker[0] == "`" and "`" in info:  # CommonMark: that is inline code, not a fence
        return None
    return indent, marker, info


def _nested_opener(lines: Sequence[str], fence: _Fence) -> int | None:
    """First line inside ``fence`` that is itself an opening fence with an info string."""
    last = fence.stop - 1 if fence.closed else fence.stop
    for index in range(fence.start + 1, min(last, len(lines))):
        opener = _opening_fence(lines[index])
        if opener is None:
            continue
        indent, marker, info = opener
        if (
            indent <= _MAX_TOP_LEVEL_INDENT
            and info
            and marker[0] == fence.marker[0]
            and len(marker) >= len(fence.marker)
        ):
            return index
    return None


def _recovery_line(
    lines: Sequence[str], fence: _Fence, stop: int, *, strict: bool = False
) -> int | None:
    """First line in ``(fence.start, stop)`` that should be read as a heading again.

    A candidate is an ATX heading preceded by a blank line. A level-1 candidate is
    only trusted when ``#`` cannot start a comment in the fence's language. ``strict``
    additionally requires a blank line (or EOF) after it.
    """
    for index in range(fence.start + 2, min(stop, len(lines))):
        match = _ATX_HEADING.match(lines[index])
        if match is None or lines[index - 1].strip():
            continue
        if strict and index + 1 < len(lines) and lines[index + 1].strip():
            continue
        if len(match.group(1)) >= 2 or fence.language in _HASH_IS_NOT_COMMENT:
            return index
    return None


# ---------------------------------------------------------------------- sub-chunker


class SectionPart(Protocol):
    """Anything carrying a part's text and its 1-based inclusive line range."""

    @property
    def content(self) -> str: ...
    @property
    def start_line(self) -> int: ...
    @property
    def end_line(self) -> int: ...


def join_parts(parts: Sequence[SectionPart]) -> str:
    """Reassemble consecutive parts (in source order) into the original verbatim text."""
    if not parts:
        return ""
    chunks = [parts[0].content]
    for previous, current in zip(parts, parts[1:], strict=False):
        chunks.append("\n" * max(0, current.start_line - previous.end_line))
        chunks.append(current.content)
    return "".join(chunks)


def split_into_spans(
    content: str, max_chars: int, *, glue_first: bool = False
) -> list[tuple[int, int]]:
    """Split ``content`` into contiguous ``(start, end)`` character spans of <= ``max_chars``.

    Spans tile the input exactly (``"".join(content[a:b]) == content``). Boundaries are
    paragraph breaks outside code fences; a single block that is still too large falls
    back to line boundaries, and a single over-long line to whitespace boundaries.
    ``glue_first`` keeps a heading attached to the block that follows it - unless that
    block holds a fence which fits in a part by itself but not together with the heading:
    a lone heading is harmless, a code block cut in two is not.
    """
    blocks = _paragraph_blocks(content)
    if glue_first and len(blocks) > 1:
        heading, following = blocks[0], blocks[1]
        cuts_a_fence = (
            following.has_fence
            and following.end - following.start <= max_chars < following.end - heading.start
        )
        if not cuts_a_fence:
            blocks[0:2] = [_Block(heading.start, following.end, following.has_fence)]
    pieces: list[tuple[int, int]] = []
    for block in blocks:
        if block.end - block.start <= max_chars:
            pieces.append((block.start, block.end))
        else:
            pieces.extend(_split_block(content, block.start, block.end, max_chars))

    spans: list[tuple[int, int]] = []
    span_start, span_end = pieces[0]
    for begin, end in pieces[1:]:
        if end - span_start > max_chars:
            spans.append((span_start, span_end))
            span_start = begin
        span_end = end
    spans.append((span_start, span_end))
    return spans


def _paragraph_blocks(content: str) -> list[_Block]:
    """Blank-line-separated blocks; a fenced block (at any indent) stays in one block."""
    blocks: list[_Block] = []
    raw_lines = content.split("\n")
    block_start = 0
    block_has_fence = False
    position = 0
    after_blank = False
    open_fence: tuple[int, str] | None = None  # (indent, marker) of the fence we are inside
    for index, line in enumerate(raw_lines):
        if open_fence is None:
            if not line.strip():
                after_blank = True
            else:
                if after_blank and position > block_start:
                    blocks.append(_Block(block_start, position, block_has_fence))
                    block_start, block_has_fence = position, False
                after_blank = False
                opener = _opening_fence(line)
                if opener is not None:
                    # Fences nested in list items are indented past column 3, so any
                    # indent opens one here; its closer may sit up to 3 columns deeper.
                    open_fence = (opener[0], opener[1])
                    block_has_fence = True
        elif _is_closing_fence(line, open_fence[1], open_fence[0] + _MAX_TOP_LEVEL_INDENT):
            open_fence = None
        position += len(line) + (1 if index < len(raw_lines) - 1 else 0)
    blocks.append(_Block(block_start, len(content), block_has_fence))
    return blocks


def _split_block(content: str, begin: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    """Split one oversized block at line boundaries, then at whitespace if needed."""
    pieces: list[tuple[int, int]] = []
    position = begin
    while position < end:
        newline = content.find("\n", position, end)
        line_end = end if newline == -1 else newline + 1
        # An over-long line is cut into pieces of at most half a part, so that whatever
        # precedes it (typically the heading) can still share a part with its first piece.
        piece = max(1, max_chars // 2) if line_end - position > max_chars else max_chars
        while line_end - position > piece:
            cut = content.rfind(" ", position + 1, position + piece)
            cut = position + piece if cut == -1 else cut + 1
            pieces.append((position, cut))
            position = cut
        pieces.append((position, line_end))
        position = line_end
    return pieces
