"""Invariants the parser must hold for any input, checked with generated documents.

These are written against the interface, not the implementation: they are the contract a
replacement sectioniser has to satisfy too. Generators build documents out of the pieces
that historically broke the parser - unbalanced fences, headings at every level, front
matter, tables, blockquotes, CRLF, and text that merely looks like markup.
"""

from __future__ import annotations

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from markdown_memory.models import PATH_SEPARATOR, SectionDraft
from markdown_memory.parser import (
    DEFAULT_MAX_SECTION_CHARS,
    MAX_UNIT_CHARS,
    MAX_UNITS_PER_SECTION,
    MarkdownParser,
    join_parts,
    split_into_spans,
)

# Lines that have each, at some point, made the parser do the wrong thing.
LINES = st.sampled_from(
    [
        "# Heading One",
        "## Heading Two",
        "### Heading Three",
        "###### Heading Six",
        "#NotAHeading",
        "   ## Indented Heading",
        "    # Four spaces: indented code, not a heading",
        "```",
        "```python",
        "```markdown",
        "~~~",
        "~~~~",
        "````",
        "text",
        "",
        "   ",
        "\t",
        "- list item",
        "  - nested item",
        "1. ordered item",
        "> quoted line",
        "> # quoted heading",
        "| a | b |",
        "|---|---|",
        "| 1 | 2 |",
        "---",
        "===",
        "<!-- comment -->",
        "<!-- unterminated",
        "<div>",
        "</div>",
        "title: front matter",
        "key: value",
        "a" * 200,
        "word " * 40,
    ]
)
DOCUMENTS = st.lists(LINES, min_size=0, max_size=40).map("\n".join)
PARSERS = st.integers(min_value=40, max_value=DEFAULT_MAX_SECTION_CHARS).map(MarkdownParser)
SLOW = settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def parts_of(sections: tuple[SectionDraft, ...], base_path: str) -> list[SectionDraft]:
    return [section for section in sections if section.base_path == base_path]


class TestParseNeverCrashes:
    @given(text=DOCUMENTS)
    @SLOW
    def test_any_document_parses(self, text: str) -> None:
        document = MarkdownParser().parse(text, fallback_title="fallback")
        assert document.title
        assert document.line_count >= 0

    @given(text=DOCUMENTS, newline=st.sampled_from(["\r\n", "\r", "\n"]))
    @SLOW
    def test_line_endings_do_not_change_the_heading_paths(self, text: str, newline: str) -> None:
        plain = MarkdownParser().parse(text, fallback_title="f")
        converted = MarkdownParser().parse(text.replace("\n", newline), fallback_title="f")
        assert [s.heading_path for s in plain.sections] == [
            s.heading_path for s in converted.sections
        ]

    @given(text=DOCUMENTS)
    @SLOW
    def test_a_null_byte_is_survivable(self, text: str) -> None:
        MarkdownParser().parse(text + "\x00tail", fallback_title="f")


class TestSectionsTileTheDocument:
    @given(text=DOCUMENTS, parser=PARSERS)
    @SLOW
    def test_line_ranges_advance_and_only_parts_may_share_a_line(
        self, text: str, parser: MarkdownParser
    ) -> None:
        """An over-long single line is cut into parts, so parts may share a line number.

        Sections that are not parts of one another must never overlap.
        """
        sections = parser.parse(text, fallback_title="f").sections
        assume(sections)
        for previous, section in zip(sections, sections[1:], strict=False):
            assert 1 <= section.start_line <= section.end_line
            assert section.start_line >= previous.start_line, section.heading_path
            if section.base_path != previous.base_path:
                assert section.start_line >= previous.end_line, section.heading_path

    @given(text=DOCUMENTS, parser=PARSERS)
    @SLOW
    def test_every_section_is_the_source_it_claims(self, text: str, parser: MarkdownParser) -> None:
        """A whole section is its source lines verbatim; a part is a slice of them."""
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        for section in parser.parse(text, fallback_title="f").sections:
            source = "\n".join(lines[section.start_line - 1 : section.end_line])
            if section.part_index == 0:
                assert section.content == source, section.heading_path
            else:
                assert section.content in source, section.heading_path

    @given(text=DOCUMENTS, parser=PARSERS)
    @SLOW
    def test_the_parts_of_a_section_reassemble_byte_for_byte(
        self, text: str, parser: MarkdownParser
    ) -> None:
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        sections = parser.parse(text, fallback_title="f").sections
        for base_path in {section.base_path for section in sections}:
            parts = parts_of(sections, base_path)
            whole = "\n".join(lines[parts[0].start_line - 1 : parts[-1].end_line])
            assert join_parts(parts) == whole, base_path


class TestHeadingPaths:
    @given(text=DOCUMENTS, parser=PARSERS)
    @SLOW
    def test_paths_are_unique_and_carry_their_title(
        self, text: str, parser: MarkdownParser
    ) -> None:
        sections = parser.parse(text, fallback_title="f").sections
        paths = [section.heading_path for section in sections]
        assert len(paths) == len(set(paths))
        for section in sections:
            assert section.heading_path.startswith(section.base_path)
            leaf = section.base_path.split(PATH_SEPARATOR)[-1]
            assert leaf.startswith(section.heading_title[: len(leaf)])

    @given(text=DOCUMENTS, parser=PARSERS)
    @SLOW
    def test_a_part_index_is_set_exactly_when_a_section_was_split(
        self, text: str, parser: MarkdownParser
    ) -> None:
        sections = parser.parse(text, fallback_title="f").sections
        for base_path in {section.base_path for section in sections}:
            parts = parts_of(sections, base_path)
            if len(parts) == 1:
                assert parts[0].part_index == 0
                assert parts[0].heading_path == parts[0].base_path
            else:
                assert [p.part_index for p in parts] == list(range(1, len(parts) + 1))


class TestUnits:
    @given(text=DOCUMENTS, parser=PARSERS)
    @SLOW
    def test_units_are_bounded_and_never_blank(self, text: str, parser: MarkdownParser) -> None:
        for section in parser.parse(text, fallback_title="f").sections:
            assert len(section.units) <= MAX_UNITS_PER_SECTION
            for unit in section.units:
                assert unit.strip()
                assert len(unit) <= MAX_UNIT_CHARS


class TestSpanTiling:
    @given(
        text=st.text(min_size=0, max_size=400),
        max_chars=st.integers(min_value=1, max_value=120),
        glue_first=st.booleans(),
    )
    @SLOW
    def test_spans_tile_the_input_exactly(
        self, text: str, max_chars: int, glue_first: bool
    ) -> None:
        spans = split_into_spans(text, max_chars, glue_first=glue_first)
        assert "".join(text[start:end] for start, end in spans) == text
        assert spans[0][0] == 0 and spans[-1][1] == len(text)
        for (_, end), (next_start, _) in zip(spans, spans[1:], strict=False):
            assert end == next_start

    @given(
        text=st.text(alphabet=st.sampled_from("ab \n"), min_size=0, max_size=400),
        max_chars=st.integers(min_value=8, max_value=60),
    )
    @SLOW
    def test_only_blank_text_may_exceed_the_limit(self, text: str, max_chars: int) -> None:
        """Trailing blank space rides along with the text above it and does not count."""
        for start, end in split_into_spans(text, max_chars):
            chunk = text[start:end]
            longest_line = max((len(line) for line in chunk.split("\n")), default=0)
            assert len(chunk.rstrip()) <= max_chars or not chunk.strip() or longest_line > max_chars
