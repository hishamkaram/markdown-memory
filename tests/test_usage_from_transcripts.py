"""The transcript miner: the instrument that decides whether this tool is used at all.

Its numbers will be quoted to justify (or abandon) a day of benchmark work, so each one
is pinned here. Behaviour only - nothing in this module judges relevance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(scope="module")
def miner() -> Any:
    """Import by name, never by path.

    pytest's ``pythonpath`` puts ``scripts/`` on the path, and the mutation harness
    replaces that entry with a mutated copy. Loading the file by its absolute path would
    read the real source every time, so every mutation of this module would survive.
    """
    import usage_from_transcripts

    return usage_from_transcripts


def transcript(path: Path, *tool_calls: tuple[str, dict[str, object]]) -> Path:
    """Write a transcript whose assistant turns make ``tool_calls`` in order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for name, arguments in tool_calls:
        lines.append(
            json.dumps(
                {
                    "sessionId": path.stem,
                    "cwd": "/home/dev/project",
                    "timestamp": "2026-09-20T12:00:00Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "id": "x", "name": name, "input": arguments}
                        ],
                    },
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestWhatCountsAsUse:
    def test_a_project_that_never_calls_the_server_is_counted_and_reported(
        self, miner: Any, tmp_path: Path
    ) -> None:
        """Zero is the most important number this script can print."""
        transcript(tmp_path / "proj" / "a.jsonl", ("Read", {"file_path": "README.md"}))
        report = miner.summarise([miner.calls_in(p) for p in miner.transcripts(tmp_path, None)])
        assert report.sessions_scanned == 1
        assert report.sessions_using_server == 0
        assert report.searches == 0

    def test_the_server_is_recognised_under_its_mcp_prefix(
        self, miner: Any, tmp_path: Path
    ) -> None:
        transcript(
            tmp_path / "proj" / "a.jsonl",
            ("mcp__markdown-memory__search_docs", {"query": "retry policy"}),
            ("mcp__markdown-memory__read_section", {"file_path": "a.md", "heading_path": "A > B"}),
        )
        report = miner.summarise([miner.calls_in(p) for p in miner.transcripts(tmp_path, None)])
        assert report.sessions_using_server == 1
        assert report.server_calls == {"search_docs": 1, "read_section": 1}
        assert report.queries == ["retry policy"]

    def test_a_subagents_calls_count_too(self, miner: Any, tmp_path: Path) -> None:
        """Subagent transcripts live one directory deeper and call the same tools."""
        transcript(
            tmp_path / "proj" / "session" / "subagents" / "agent.jsonl",
            ("mcp__markdown-memory__search_docs", {"query": "from a subagent"}),
        )
        report = miner.summarise([miner.calls_in(p) for p in miner.transcripts(tmp_path, None)])
        assert report.searches == 1

    def test_another_servers_tools_are_not_counted_as_ours(
        self, miner: Any, tmp_path: Path
    ) -> None:
        transcript(tmp_path / "proj" / "a.jsonl", ("mcp__other-server__search", {"query": "x"}))
        report = miner.summarise([miner.calls_in(p) for p in miner.transcripts(tmp_path, None)])
        assert report.sessions_using_server == 0


class TestBehaviourAfterASearch:
    """The numbers that say whether a search was any use - none of them is a verdict."""

    def measure(self, miner: Any, tmp_path: Path, *calls: tuple[str, dict[str, object]]) -> Any:
        transcript(tmp_path / "proj" / "a.jsonl", *calls)
        return miner.summarise([miner.calls_in(p) for p in miner.transcripts(tmp_path, None)])

    def test_a_search_answered_by_reading_a_section(self, miner: Any, tmp_path: Path) -> None:
        report = self.measure(
            miner,
            tmp_path,
            ("mcp__markdown-memory__search_docs", {"query": "retry"}),
            ("mcp__markdown-memory__read_section", {"file_path": "a.md", "heading_path": "A"}),
        )
        assert report.searches_followed_by_read_section == 1
        assert report.searches_followed_by_file_access == 0

    def test_a_search_abandoned_for_the_file_system(self, miner: Any, tmp_path: Path) -> None:
        """The number a server-side log could never produce."""
        report = self.measure(
            miner,
            tmp_path,
            ("mcp__markdown-memory__search_docs", {"query": "retry"}),
            ("Read", {"file_path": "/docs/retry.md"}),
        )
        assert report.searches_followed_by_file_access == 1
        assert report.searches_followed_by_read_section == 0

    def test_a_search_abandoned_for_grep(self, miner: Any, tmp_path: Path) -> None:
        report = self.measure(
            miner,
            tmp_path,
            ("mcp__markdown-memory__search_docs", {"query": "retry"}),
            ("Bash", {"command": "rg -n 'retry' docs/guide.md"}),
        )
        assert report.searches_followed_by_file_access == 1

    def test_an_unrelated_shell_command_is_not_a_fallback(self, miner: Any, tmp_path: Path) -> None:
        report = self.measure(
            miner,
            tmp_path,
            ("mcp__markdown-memory__search_docs", {"query": "retry"}),
            ("Bash", {"command": "uv run pytest -q"}),
        )
        assert report.searches_followed_by_file_access == 0

    def test_a_reworded_second_search(self, miner: Any, tmp_path: Path) -> None:
        report = self.measure(
            miner,
            tmp_path,
            ("mcp__markdown-memory__search_docs", {"query": "retry"}),
            ("mcp__markdown-memory__search_docs", {"query": "backoff policy"}),
        )
        assert report.searches == 2
        assert report.searches_followed_by_another_search == 1

    def test_a_file_read_long_after_a_search_is_not_attributed_to_it(
        self, miner: Any, tmp_path: Path
    ) -> None:
        """Past the window, the link to the search is guesswork."""
        filler = [("Bash", {"command": "uv run pytest -q"})] * (miner.FALLBACK_WINDOW + 1)
        report = self.measure(
            miner,
            tmp_path,
            ("mcp__markdown-memory__search_docs", {"query": "retry"}),
            *filler,
            ("Read", {"file_path": "/docs/retry.md"}),
        )
        assert report.searches_followed_by_file_access == 0


class TestRobustness:
    def test_a_half_written_line_does_not_lose_the_session(
        self, miner: Any, tmp_path: Path
    ) -> None:
        """A live session's transcript is being appended to while this runs."""
        path = transcript(
            tmp_path / "proj" / "a.jsonl",
            ("mcp__markdown-memory__search_docs", {"query": "retry"}),
        )
        path.write_text(path.read_text() + '{"message": {"content": [{"type": "tool', "utf-8")
        report = miner.summarise([miner.calls_in(p) for p in miner.transcripts(tmp_path, None)])
        assert report.searches == 1

    def test_a_missing_transcript_directory_is_not_an_error(
        self, miner: Any, tmp_path: Path
    ) -> None:
        assert list(miner.transcripts(tmp_path / "nowhere", None)) == []
