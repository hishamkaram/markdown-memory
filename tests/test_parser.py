"""Parser tests: heading stack, preamble, malformed Markdown, sub-chunking."""

from __future__ import annotations

import pytest

from markdown_memory.models import PREAMBLE_TITLE, ParsedDocument, SectionDraft
from markdown_memory.parser import (
    DEFAULT_MAX_SECTION_CHARS,
    UNTITLED_HEADING,
    MarkdownParser,
    join_parts,
    split_into_spans,
)


def parse(text: str, **kwargs: int) -> ParsedDocument:
    return MarkdownParser(**kwargs).parse(text, fallback_title="fallback")


def paths(document: ParsedDocument) -> list[str]:
    return [section.heading_path for section in document.sections]


def by_path(document: ParsedDocument, path: str) -> SectionDraft:
    matches = [section for section in document.sections if section.heading_path == path]
    assert len(matches) == 1, f"{path!r} matched {len(matches)} sections: {paths(document)}"
    return matches[0]


def source_slice(text: str, section: SectionDraft) -> str:
    lines = text.split("\n")
    return "\n".join(lines[section.start_line - 1 : section.end_line])


# ---------------------------------------------------------------------- clean document


class TestCleanDocument:
    def test_breadcrumbs_follow_the_heading_hierarchy(self, clean_doc: str) -> None:
        assert paths(parse(clean_doc)) == [
            PREAMBLE_TITLE,
            "Orbit Gateway",
            "Orbit Gateway > Installation",
            "Orbit Gateway > Installation > From Source",
            "Orbit Gateway > Configuration",
            "Orbit Gateway > Configuration > Environment Variables",
            "Orbit Gateway > Configuration > Command Line Flags",
            "Orbit Gateway > Operations",
            "Orbit Gateway > Operations > Health Checks",
            "Orbit Gateway > Operations > Graceful Shutdown",
            "Orbit Gateway > License",
        ]

    def test_title_is_the_first_h1(self, clean_doc: str) -> None:
        assert parse(clean_doc).title == "Orbit Gateway"

    def test_levels_and_titles(self, clean_doc: str) -> None:
        section = by_path(parse(clean_doc), "Orbit Gateway > Configuration > Environment Variables")
        assert section.heading_level == 3
        assert section.heading_title == "Environment Variables"
        assert section.part_index == 0
        assert section.base_path == section.heading_path

    def test_every_section_is_a_verbatim_slice_of_its_line_range(self, clean_doc: str) -> None:
        document = parse(clean_doc)
        for section in document.sections:
            assert section.content == source_slice(clean_doc, section)
            assert section.start_line <= section.end_line

    def test_sections_do_not_overlap_and_cover_all_text(self, clean_doc: str) -> None:
        document = parse(clean_doc)
        covered: set[int] = set()
        for section in document.sections:
            span = set(range(section.start_line, section.end_line + 1))
            assert not covered & span
            covered |= span
        non_blank = {
            number for number, line in enumerate(clean_doc.split("\n"), start=1) if line.strip()
        }
        assert non_blank <= covered

    def test_table_stays_intact_in_its_section(self, clean_doc: str) -> None:
        section = by_path(parse(clean_doc), "Orbit Gateway > Configuration > Environment Variables")
        assert section.content.startswith("### Environment Variables")
        assert (
            "| `ORBIT_UPSTREAM_TIMEOUT_MS` | `3000` | Upstream request deadline |"
            in section.content
        )
        assert "--config" not in section.content  # next section's text

    def test_hash_comment_inside_closed_fence_is_not_a_heading(self, clean_doc: str) -> None:
        document = parse(clean_doc)
        assert not any("not a heading" in path for path in paths(document))
        install = by_path(document, "Orbit Gateway > Installation")
        assert "# this comment is inside a fence" in install.content
        assert install.content.rstrip().endswith("```")

    def test_embedding_text_prepends_the_breadcrumb(self, clean_doc: str) -> None:
        section = by_path(parse(clean_doc), "Orbit Gateway > Operations > Health Checks")
        assert section.units == ("GET /healthz returns 200 while the process accepts traffic.",)
        assert section.embedding_text == f"{section.heading_path}\n\n{section.units[0]}"
        assert section.unit_texts == (f"{section.heading_path}: {section.units[0]}",)


# ---------------------------------------------------------------------- heading stack


class TestHeadingStack:
    def test_level_one_straight_to_level_four(self) -> None:
        document = parse("# Root\n\n#### Deep\n\ntext\n")
        assert paths(document) == ["Root", "Root > Deep"]
        assert by_path(document, "Root > Deep").heading_level == 4

    def test_stack_pops_every_level_greater_or_equal(self) -> None:
        text = "# A\n#### D\n## B\n### C\n###### F\n### C2\n# Z\n## Y\n"
        assert paths(parse(text)) == [
            "A",
            "A > D",
            "A > B",
            "A > B > C",
            "A > B > C > F",
            "A > B > C2",
            "Z",
            "Z > Y",
        ]

    def test_document_starting_below_level_one(self) -> None:
        assert paths(parse("### Only\n\n## Up\n\n#### Down\n")) == ["Only", "Up", "Up > Down"]

    def test_duplicate_breadcrumbs_are_disambiguated(self) -> None:
        document = parse(
            "# Log\n## Fixed\na\n### Detail\nx\n## Fixed\nb\n### Detail\ny\n## Fixed\nc\n"
        )
        assert paths(document) == [
            "Log",
            "Log > Fixed",
            "Log > Fixed > Detail",
            "Log > Fixed [2]",
            "Log > Fixed [2] > Detail",
            "Log > Fixed [3]",
        ]
        assert len(set(paths(document))) == len(document.sections)

    def test_setext_headings(self) -> None:
        document = parse("Title\n=====\n\nbody\n\nSub\n---\n\nmore\n")
        assert paths(document) == ["Title", "Title > Sub"]
        assert by_path(document, "Title > Sub").content == "Sub\n---\n\nmore"

    def test_inline_markup_is_stripped_from_titles(self) -> None:
        document = parse("# The `orbit` **CLI** and [docs](http://x) ![logo](l.png) ##\n")
        assert paths(document) == ["The orbit CLI and docs logo"]

    def test_empty_heading_gets_a_placeholder_title(self) -> None:
        assert paths(parse("#\n\ntext\n")) == [UNTITLED_HEADING]

    def test_headings_nested_in_quotes_and_lists_do_not_open_sections(self) -> None:
        document = parse("# Real\n\n> ## Quoted\n> text\n\n- item\n\n  ## In List\n\ntail\n")
        assert paths(document) == ["Real"]
        assert "## Quoted" in document.sections[0].content

    def test_seven_hashes_is_not_a_heading(self) -> None:
        assert paths(parse("# A\n\n####### nope\n")) == ["A"]


# ---------------------------------------------------------------------- preamble


class TestPreamble:
    def test_content_before_first_heading_is_captured(self, clean_doc: str) -> None:
        preamble = parse(clean_doc).sections[0]
        assert preamble.heading_title == PREAMBLE_TITLE
        assert preamble.heading_path == PREAMBLE_TITLE
        assert preamble.heading_level == 0
        assert preamble.start_line == 1
        assert "img.shields.io" in preamble.content
        assert "# Orbit Gateway" not in preamble.content

    def test_no_preamble_when_document_starts_with_a_heading(self) -> None:
        assert paths(parse("\n\n# Title\n\ntext\n")) == ["Title"]

    def test_front_matter_is_not_misread_as_a_setext_heading(self) -> None:
        document = parse("---\ntitle: From Front Matter\ntags: [a, b]\n---\n\n## Section\n\nbody\n")
        assert paths(document) == [PREAMBLE_TITLE, "Section"]
        assert document.title == "From Front Matter"
        assert document.sections[0].content.startswith("---\ntitle: From Front Matter")

    def test_h1_wins_over_front_matter_title(self) -> None:
        assert parse("---\ntitle: FM\n---\n# Real Title\n").title == "Real Title"

    def test_unterminated_front_matter_is_plain_markdown(self) -> None:
        document = parse("---\n\n# Heading\n\ntext\n")
        assert "Heading" in paths(document)


# ---------------------------------------------------------------------- malformed input


class TestMalformedMarkdown:
    def test_empty_file(self) -> None:
        document = parse("")
        assert document.sections == ()
        assert document.title == "fallback"
        assert document.line_count == 0

    @pytest.mark.parametrize("text", [" ", "\n\n\n", " \t \n  \n"])
    def test_whitespace_only_file(self, text: str) -> None:
        assert parse(text).sections == ()

    def test_document_without_headings_becomes_one_preamble_section(self) -> None:
        document = parse("just a paragraph\n\nand another one\n")
        assert paths(document) == [PREAMBLE_TITLE]
        assert document.title == "fallback"
        assert document.sections[0].content == "just a paragraph\n\nand another one"

    def test_unclosed_fence_does_not_swallow_later_sections(self) -> None:
        text = (
            "# Doc\n\n## Setup\n\n```bash\nexport A=1\n\n# a shell comment\nrun\n\n"
            "## Usage\n\nuse it\n\n### Detail\n\nmore\n"
        )
        document = parse(text)
        assert paths(document) == ["Doc", "Doc > Setup", "Doc > Usage", "Doc > Usage > Detail"]
        setup = by_path(document, "Doc > Setup")
        assert "# a shell comment" in setup.content  # still code, never a heading
        assert "## Usage" not in setup.content
        assert by_path(document, "Doc > Usage").content == "## Usage\n\nuse it"

    def test_unclosed_tilde_fence(self) -> None:
        document = parse("# T\n\n~~~\ncode\n\n## After\n\ntext\n")
        assert paths(document) == ["T", "T > After"]

    def test_two_unclosed_fences_in_one_document(self) -> None:
        # A tilde fence cannot close a backtick fence, so both really are unclosed.
        text = "# D\n\n```\na\n\n## One\n\n~~~\nb\n\n## Two\n\nend\n"
        assert paths(parse(text)) == ["D", "D > One", "D > Two"]

    def test_second_backtick_fence_closes_the_first(self) -> None:
        # CommonMark: the second ``` is a closing marker, so "## One" is genuinely code.
        text = "# D\n\n```\na\n\n## One\n\n```\nb\n\n## Two\n\nend\n"
        assert paths(parse(text)) == ["D", "D > Two"]

    def test_unclosed_fence_at_end_of_file_without_later_heading(self) -> None:
        document = parse("# Doc\n\n```python\nprint('never closed')\n")
        assert paths(document) == ["Doc"]
        assert "never closed" in document.sections[0].content

    def test_unclosed_fence_followed_by_trailing_blank_lines(self) -> None:
        document = parse("# Doc\n\n```\ncode\n\n## Next\n\ntext\n\n\n\n")
        assert paths(document) == ["Doc", "Doc > Next"]

    def test_level_one_hash_is_a_comment_in_hash_comment_languages(self) -> None:
        document = parse("# Doc\n\n```python\nx = 1\n\n# Configure logging\ny = 2\n")
        assert paths(document) == ["Doc"]

    def test_level_one_hash_is_a_heading_after_an_unclosed_json_fence(self) -> None:
        document = parse('## Doc\n\n```json\n{"a": 1}\n\n# Next Chapter\n\ntext\n')
        assert paths(document) == ["Doc", "Next Chapter"]

    def test_closed_fence_at_end_of_file_keeps_its_headings_as_code(self) -> None:
        document = parse("# Doc\n\n```md\n\n## Example Heading\n\n```\n")
        assert paths(document) == ["Doc"]

    def test_closed_fence_in_the_middle_keeps_its_headings_as_code(self) -> None:
        document = parse("# Doc\n\n```md\n\n## Example Heading\n\n```\n\n## Real\n")
        assert paths(document) == ["Doc", "Doc > Real"]

    def test_windows_line_endings_and_bom(self) -> None:
        document = parse("﻿# Title\r\n\r\nbody\r\n\r\n## Sub\r\n\r\ntext\r\n")
        assert paths(document) == ["Title", "Title > Sub"]
        assert document.sections[0].content == "# Title\n\nbody"
        assert "\r" not in "".join(section.content for section in document.sections)

    def test_messy_fixture(self, messy_doc: str) -> None:
        document = parse(messy_doc)
        found = paths(document)
        assert found[0].startswith(PREAMBLE_TITLE)
        assert "Messy Service Notes > Deeply Nested Without Parents" in found
        assert "Messy Service Notes > Back To Level Two" in found
        assert "Messy Service Notes > Duplicate" in found
        assert "Messy Service Notes > Duplicate [2]" in found
        assert "Messy Service Notes > Section After Unclosed Fence" in found
        assert "Messy Service Notes > Section After Unclosed Fence > Child After Recovery" in found
        assert not any("this comment must not" in path for path in found)
        fence = by_path(document, "Messy Service Notes > Unclosed Fence Ahead")
        assert "export MESSY_FLAG=1" in fence.content
        assert "# this comment must not become a heading" in fence.content
        assert document.title == "Messy Service Notes"


# ---------------------------------------------------------------------- oversized sections


class TestOversizedSections:
    @staticmethod
    def big_section(paragraphs: int = 12) -> str:
        body = "\n\n".join(
            f"Paragraph {number}. " + "lorem ipsum dolor sit amet " * 20
            for number in range(1, paragraphs + 1)
        )
        return f"# Guide\n\n## Big\n\n{body}\n\n## Small\n\ntiny\n"

    def test_section_at_the_limit_is_not_split(self) -> None:
        text = "# T\n\n" + "x" * (DEFAULT_MAX_SECTION_CHARS - len("# T\n\n"))
        document = parse(text)
        assert len(document.sections) == 1
        assert document.sections[0].part_index == 0
        assert document.sections[0].heading_path == "T"

    def test_oversized_section_is_split_into_numbered_parts(self) -> None:
        document = parse(self.big_section())
        parts = [s for s in document.sections if s.base_path == "Guide > Big"]
        assert len(parts) >= 2
        assert [part.part_index for part in parts] == list(range(1, len(parts) + 1))
        assert [part.heading_path for part in parts] == [
            f"Guide > Big (Part {number})" for number in range(1, len(parts) + 1)
        ]
        for part in parts:
            assert len(part.content) <= DEFAULT_MAX_SECTION_CHARS
            assert part.heading_title == "Big"
            assert part.heading_level == 2
            assert part.embedding_text.startswith(part.heading_path + "\n\n")

    def test_neighbouring_sections_are_untouched_by_a_split(self) -> None:
        document = parse(self.big_section())
        small = by_path(document, "Guide > Small")
        assert small.part_index == 0
        assert small.content == "## Small\n\ntiny"

    def test_parts_break_on_paragraph_boundaries(self) -> None:
        text = self.big_section()
        document = parse(text)
        parts = [s for s in document.sections if s.base_path == "Guide > Big"]
        assert parts[0].content.startswith("## Big\n\nParagraph 1.")
        for part in parts[1:]:
            assert part.content.startswith("Paragraph ")
        for part in parts:
            assert part.content == source_slice(text, part)

    def test_parts_reassemble_to_the_exact_original_section(self) -> None:
        text = self.big_section()
        whole = by_path(parse(text, max_section_chars=1_000_000), "Guide > Big")
        parts = [s for s in parse(text).sections if s.base_path == "Guide > Big"]
        assert join_parts(parts) == whole.content
        assert parts[0].start_line == whole.start_line
        assert parts[-1].end_line == whole.end_line

    def test_heading_is_never_stranded_alone_in_part_one(self) -> None:
        paragraph = "word " * 630  # 3150 chars: fits alone, but not together with the heading
        document = parse(
            f"## A Rather Long Heading Title For This Section\n\n{paragraph}\n\n{paragraph}\n"
        )
        first = document.sections[0]
        assert first.part_index == 1
        assert len(first.content) > 200

    def test_heading_less_document_is_split_with_preamble_breadcrumb(self) -> None:
        text = "\n\n".join(f"sentence number {n} " + "filler text " * 40 for n in range(20))
        document = parse(text)
        assert len(document.sections) > 1
        assert paths(document) == [
            f"{PREAMBLE_TITLE} (Part {number})" for number in range(1, len(document.sections) + 1)
        ]
        assert join_parts(document.sections) == text

    def test_wall_of_text_on_a_single_line_is_split_at_whitespace(self) -> None:
        wall = "word " * 2000
        text = f"# Wall\n\n{wall.strip()}\n"
        document = parse(text)
        assert len(document.sections) >= 3
        for section in document.sections:
            assert len(section.content) <= DEFAULT_MAX_SECTION_CHARS
        assert join_parts(document.sections) == text.rstrip("\n")
        # no word is cut in half
        for section in document.sections[1:]:
            assert section.content.startswith("word")

    def test_unbreakable_line_is_hard_cut(self) -> None:
        text = "# X\n\n" + "z" * 10_000
        document = parse(text)
        assert all(len(s.content) <= DEFAULT_MAX_SECTION_CHARS for s in document.sections)
        assert join_parts(document.sections) == text

    def test_fenced_code_with_blank_lines_is_kept_in_one_part(self) -> None:
        code = (
            "```python\n" + "\n\n".join(f"def f{n}():\n    return {n}" for n in range(12)) + "\n```"
        )
        filler = "\n\n".join("filler paragraph " * 30 for _ in range(8))
        document = parse(f"# Doc\n\n{filler}\n\n{code}\n\n{filler}\n")
        holders = [s for s in document.sections if "```python" in s.content]
        assert len(holders) == 1
        assert code in holders[0].content

    def test_messy_wall_of_text_and_paragraphs_reassemble(self, messy_doc: str) -> None:
        unsplit = parse(messy_doc, max_section_chars=1_000_000)
        document = parse(messy_doc)
        for base in (
            "Messy Service Notes > Back To Level Two > Wall Of Text",
            "Messy Service Notes > Back To Level Two > Many Paragraphs",
        ):
            parts = [s for s in document.sections if s.base_path == base]
            assert len(parts) >= 2
            assert all(len(part.content) <= DEFAULT_MAX_SECTION_CHARS for part in parts)
            assert join_parts(parts) == by_path(unsplit, base).content

    @pytest.mark.parametrize("limit", [40, 97, 250, 1000])
    def test_any_limit_reassembles_both_fixtures(
        self, clean_doc: str, messy_doc: str, limit: int
    ) -> None:
        for text in (clean_doc, messy_doc):
            unsplit = parse(text, max_section_chars=1_000_000)
            document = parse(text, max_section_chars=limit)
            assert all(len(s.content) <= limit for s in document.sections)
            for whole in unsplit.sections:
                parts = [s for s in document.sections if s.base_path == whole.base_path]
                assert join_parts(parts) == whole.content


class TestSplitIntoSpans:
    def test_spans_tile_the_input_exactly(self) -> None:
        content = "alpha\n\nbeta beta\n\n\n\ngamma\n```\ncode\n\nmore\n```\n\ndelta " * 30
        spans = split_into_spans(content, 120)
        assert spans[0][0] == 0
        assert spans[-1][1] == len(content)
        for (_, end), (start, _) in zip(spans, spans[1:], strict=False):
            assert end == start
        assert all(end - start <= 120 for start, end in spans)

    def test_short_content_is_a_single_span(self) -> None:
        assert split_into_spans("short", 100) == [(0, 5)]

    def test_invalid_limit_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            MarkdownParser(max_section_chars=0)
