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
* **Unclosed code fences** - CommonMark runs such a fence to EOF, swallowing every
  heading after it. ``_resilient_fence`` replaces markdown-it's own fence rule and ends
  the fence at the next plausible heading instead, in the same pass. A fence that
  markdown-it closed normally is left exactly as CommonMark read it, unless it holds an
  opening fence of its own (```` ```bash ```` around ```` ```python ````), which means the
  markers were paired wrongly. Deciding on positive evidence only is what keeps a
  well-formed document - one that simply shows headings inside a fence - untouched; the
  cost is that a stray bare marker, which shifts every pairing after it, leaves the
  headings inside those fences hidden.
* **Oversized / heading-less text** - split on paragraph boundaries into
  ``Path (Part n)`` parts that reassemble byte-for-byte. A fenced block is only cut
  when it exceeds the limit on its own.
* **Colliding breadcrumbs** - repeated paths get a ``[n]`` suffix, also against
  generated ``(Part n)`` paths, so every stored path addresses exactly one thing.

Every section is additionally broken into *units* (``extract_units``): the plain text of
each paragraph, list item, table row and code block. They feed passage-level embeddings.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from markdown_it import MarkdownIt
from markdown_it.rules_block.fence import make_fence_rule
from markdown_it.rules_block.state_block import StateBlock
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
MAX_UNIT_CHARS = 600
UNTITLED_HEADING = "(untitled)"

_FENCE_OPEN = re.compile(r"^( *)(`{3,}|~{3,})(.*)$")
_MAX_TOP_LEVEL_INDENT = 3
logger = logging.getLogger(__name__)

_STOCK_FENCE = make_fence_rule()
_ATX_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+\S")
_FRONT_MATTER_TITLE = re.compile(r"^title\s*:\s*(.+?)\s*$", re.IGNORECASE)
# An unquoted key starts with a letter of any script or "_" (`[^\W\d]`), never a digit.
_YAML_KEY = re.compile(r"""^(?:"[^"]+"|'[^']+'|[^\W\d][^:#]*?)\s*:(\s|$)""")
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
# Removed before tags are: a ">" inside a comment would end the "tag" early and leak the
# rest of the comment as text. An unterminated comment hides everything after it.
_HTML_COMMENT = re.compile(r"<!--.*?(?:-->|\Z)", re.DOTALL)
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
        # The chain the stock rule is registered with. Dropping it would stop the fence rule
        # being consulted inside list items and block quotes, so a fence in a list would be
        # read as a paragraph and headings inside it would become real headings.
        self._md.block.ruler.at(
            "fence", _resilient_fence, {"alt": ["paragraph", "reference", "blockquote", "list"]}
        )
        # Passages are extracted with the stock rule. Fence recovery is there to rescue a
        # document's structure; inside one section an unclosed fence is usually a fenced
        # sample that oversized-section splitting cut in half, and reinterpreting its
        # contents as prose would index the template rather than the documentation.
        self._units_md = MarkdownIt("commonmark").enable("table")

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

    def _tokenize(self, source: str, *, recover_fences: bool = True) -> list[Token]:
        try:
            return (self._md if recover_fences else self._units_md).parse(source)
        except Exception as exc:  # markdown-it has no dedicated error type
            raise ASTParseError(f"markdown-it failed to tokenise the document: {exc}") from exc

    def _find_headings(self, lines: Sequence[str], body_start: int) -> list[_Heading]:
        """Collect the document's top-level headings in a single tokenisation pass.

        An unclosed fence no longer hides the rest of the document: ``_resilient_fence``
        ends it at the first plausible heading while the block parser is running, so the
        headings after it are simply there.
        """
        tokens = self._tokenize("\n".join(lines[body_start:]))
        headings: list[_Heading] = []
        for position, token in enumerate(tokens):
            if token.level != 0 or token.map is None or token.type != "heading_open":
                continue
            inline = tokens[position + 1] if position + 1 < len(tokens) else None
            headings.append(
                _Heading(
                    line=body_start + token.map[0],
                    level=int(token.tag[1:]),
                    title=_inline_text(inline),
                )
            )
        return headings

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
        tokens = self._tokenize(content, recover_fences=False)
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
            texts.append(_HTML_TAG.sub(" ", _HTML_COMMENT.sub(" ", token.content)))
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
    """One unit per body row, each cell labelled with its column header.

    A table without body rows yields its header cells instead: they are visible text,
    and a section holding nothing else would otherwise pass for a heading-only stub.
    """
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
    if rows:
        return rows
    header_row = "; ".join(header for header in headers if header)
    return [header_row] if header_row else []


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


def _recovery_line(state: StateBlock, start: int, stop: int, language: str) -> int | None:
    """First line in ``(start, stop)`` that should be read as a heading again.

    A candidate is an ATX heading preceded by a blank line. A level-1 candidate is only
    trusted when ``#`` cannot start a comment in the fence's language, because a lone
    ``# comment`` is far more common inside shell or Python than a heading is.
    """
    for index in range(start + 2, min(stop, len(state.bMarks) - 1)):
        line = state.src[state.bMarks[index] : state.eMarks[index]]
        previous = state.src[state.bMarks[index - 1] : state.eMarks[index - 1]]
        match = _ATX_HEADING.match(line)
        if match is None or previous.strip():
            continue
        if len(match.group(1)) >= 2 or language in _HASH_IS_NOT_COMMENT:
            return index
    return None


# Fences whose purpose is to *show* Markdown: a fence inside them is sample text, and a
# heading inside them is part of the sample, never a symptom of a broken document.
_MARKUP_SAMPLE_LANGUAGES = frozenset(
    {"", "markdown", "md", "mdx", "mdown", "text", "txt", "plain", "plaintext", "rst", "html"}
)


def _has_nested_opener(state: StateBlock, start: int, stop: int, markup: str) -> bool:
    """True when this fence contains a line that opens a fence of its own.

    ``` ```bash ... ```python ``` pairs the wrong markers: CommonMark reads the inner
    opener as body text and closes the outer fence somewhere later, hiding whatever lies
    between. An opener with an info string is the giveaway - a bare marker is ambiguous.
    """
    for index in range(start + 1, min(stop, len(state.bMarks) - 1)):
        opener = _opening_fence(state.src[state.bMarks[index] : state.eMarks[index]])
        if opener is None:
            continue
        indent, inner, info = opener
        if (
            indent <= _MAX_TOP_LEVEL_INDENT
            and info
            and inner[0] == markup[0]
            and len(inner) >= len(markup)
        ):
            return True
    return False


def _resilient_fence(state: StateBlock, start_line: int, end_line: int, silent: bool) -> bool:
    """markdown-it's fence rule, but an unclosed fence stops at the next heading.

    CommonMark runs an unclosed fence to the end of the document, so a single stray
    ``` in a long runbook deletes every heading after it from the outline - the document
    becomes one code block and search can never return those sections. Closing the fence
    at the first plausible heading costs a stray code block at worst; not closing it
    costs the whole tail of the document.

    A fence that markdown-it closed normally is never touched, so a well-formed document
    parses exactly as CommonMark says it should.
    """
    if not _STOCK_FENCE(state, start_line, end_line, silent):
        return False
    if silent:
        return True
    # Recovery is a top-level concern: resuming inside a list item or block quote would
    # hand the block parser a heading that does not belong to that container. A container
    # rewrites the line start past its own marker ("> ", list indent), so a line whose
    # parsed start is not its physical start is nested. `parentType` is no help here: it
    # is "paragraph" for an ordinary fence that ends a paragraph.
    physical_start = state.src.rfind("\n", 0, state.bMarks[start_line]) + 1
    if state.blkIndent > 0 or physical_start != state.bMarks[start_line]:
        return True
    token = state.tokens[-1]
    info = token.info.strip()
    language = info.split(maxsplit=1)[0].lower() if info else ""
    closing = state.src[state.bMarks[state.line - 1] : state.eMarks[state.line - 1]]
    if closing.strip().startswith(token.markup[0] * len(token.markup)):
        # This fence was closed. Its pairing is still suspect when it holds an opener of
        # its own: ```bash ... ```python pairs the wrong markers, so the closing marker
        # found somewhere later hides everything in between. Never suspect a fence that is
        # showing Markdown, where an inner fence is exactly what the sample is about.
        if language in _MARKUP_SAMPLE_LANGUAGES:
            return True
        if not _has_nested_opener(state, start_line, state.line, token.markup):
            return True
    cut = _recovery_line(state, start_line, state.line, language)
    if cut is None or cut <= start_line + 1:
        return True
    logger.warning(
        "Unclosed %s fence at line %d: closing it before the heading at line %d",
        language or "code",
        start_line + 1,
        cut + 1,
    )
    token.content = state.getLines(start_line + 1, cut, state.sCount[start_line], True)
    token.map = [start_line, cut]
    state.line = cut
    return True


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
    back to line boundaries, and a single over-long line to whitespace boundaries. Only
    blank lines may take a span past the limit: they stay with the text before them,
    because a span of their own would become a part with no content.
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
        if end - span_start > max_chars and content[begin:end].strip():
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
    """Split one oversized block at line boundaries, then at whitespace if needed.

    A block is oversized as a whole, yet a fence inside it (prose directly above it, a
    list whose items carry code) usually is not: a fenced run that fits in a part stays
    one piece, so it is only ever cut when it exceeds the limit by itself.
    """
    pieces: list[tuple[int, int]] = []
    position = begin
    for fence_start, fence_end in _fenced_runs(content, begin, end):
        pieces.extend(_line_pieces(content, position, fence_start, max_chars))
        if fence_end - fence_start <= max_chars:
            pieces.append((fence_start, fence_end))
        else:
            pieces.extend(_line_pieces(content, fence_start, fence_end, max_chars))
        position = fence_end
    pieces.extend(_line_pieces(content, position, end, max_chars))
    return pieces


def _fenced_runs(content: str, begin: int, end: int) -> list[tuple[int, int]]:
    """Character spans of the fenced runs in ``content[begin:end]``, closing line included.

    ``begin`` is a block boundary, so it is never inside a fence; a fence left open runs
    to ``end``.
    """
    runs: list[tuple[int, int]] = []
    open_fence: tuple[int, str] | None = None
    run_start = position = begin
    while position < end:
        newline = content.find("\n", position, end)
        line_end = end if newline == -1 else newline + 1
        line = content[position:line_end].rstrip("\n")
        if open_fence is None:
            opener = _opening_fence(line) if line.strip() else None
            if opener is not None:
                open_fence, run_start = (opener[0], opener[1]), position
        elif _is_closing_fence(line, open_fence[1], open_fence[0] + _MAX_TOP_LEVEL_INDENT):
            runs.append((run_start, line_end))
            open_fence = None
        position = line_end
    if open_fence is not None:
        runs.append((run_start, end))
    return runs


def _line_pieces(content: str, begin: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    """One piece per line of ``content[begin:end]``; an over-long line is cut at whitespace."""
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
