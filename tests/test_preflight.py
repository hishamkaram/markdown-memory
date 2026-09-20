"""The preflight collector: real queries, their task context, and no opinions.

Two reviewers independently said not to write a family classifier before the data
exists, so these tests pin the opposite property: that the script collects faithfully,
attaches the request that prompted each query, and leaves judgement alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

SEARCH = "mcp__markdown-memory__search_docs"
READ = "mcp__markdown-memory__read_section"


@pytest.fixture(scope="module")
def preflight() -> Any:
    """By name, so the mutation harness's copy is the one under test."""
    import preflight as module

    return module


def session(path: Path, *turns: tuple[str, object]) -> Path:
    """Write a transcript. ``("user", text)`` or ``(tool_name, arguments)``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for role, payload in turns:
        if role == "user":
            message = {"role": "user", "content": [{"type": "text", "text": str(payload)}]}
        else:
            message = {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "x", "name": role, "input": payload}],
            }
        lines.append(
            json.dumps(
                {
                    "sessionId": path.stem,
                    "cwd": "/home/dev/project",
                    "timestamp": "2026-09-20T12:00:00Z",
                    "message": message,
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestCollecting:
    def test_nothing_is_invented_when_nothing_was_asked(
        self, preflight: Any, tmp_path: Path
    ) -> None:
        """A zero state has to stay a zero state; a stand-in would defeat the step."""
        session(tmp_path / "p" / "a.jsonl", ("user", "hello"), ("Read", {"file_path": "a.md"}))
        assert preflight.collect(tmp_path, None) == []

    def test_a_repeated_query_is_one_row_that_counts_its_repeats(
        self, preflight: Any, tmp_path: Path
    ) -> None:
        """Frequency is evidence: the question asked five times is not one of five."""
        session(
            tmp_path / "p" / "a.jsonl",
            (SEARCH, {"query": "retry policy"}),
            (SEARCH, {"query": "retry policy"}),
            (SEARCH, {"query": "backoff"}),
        )
        records = preflight.collect(tmp_path, None)
        assert [(r.query, r.count) for r in records] == [("retry policy", 2), ("backoff", 1)]

    def test_the_request_that_prompted_the_query_is_attached(
        self, preflight: Any, tmp_path: Path
    ) -> None:
        """A query alone cannot be judged; the task it served can."""
        session(
            tmp_path / "p" / "a.jsonl",
            ("user", "make the indexer skip vendored docs"),
            (SEARCH, {"query": "exclude directory"}),
        )
        record = preflight.collect(tmp_path, None)[0]
        assert record.occurrences[0].task == "make the indexer skip vendored docs"

    def test_an_older_request_does_not_leak_into_a_later_query(
        self, preflight: Any, tmp_path: Path
    ) -> None:
        """Attribution past a couple of turns is a guess, and a guess would read as fact."""
        session(
            tmp_path / "p" / "a.jsonl",
            ("user", "first task"),
            ("user", "second task"),
            ("user", "third task"),
            (SEARCH, {"query": "something"}),
        )
        record = preflight.collect(tmp_path, None)[0]
        assert record.occurrences[0].task == "third task"

    def test_a_tool_call_outside_an_assistant_turn_does_not_shift_attribution(
        self, preflight: Any, tmp_path: Path
    ) -> None:
        """The miner counts a tool call in any message; this must count the same ones.

        Counting only assistant turns here shifted every request by one, so each query
        was filed under someone else's task - silently, which is worse than filing it
        under nothing.
        """
        import json

        path = tmp_path / "p" / "a.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"message": {"role": "user", "content": [{"type": "text", "text": "the real task"}]}},
            {
                "message": {
                    "role": "system",
                    "content": [{"type": "tool_use", "id": "1", "name": "Bash", "input": {}}],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "2",
                            "name": SEARCH,
                            "input": {"query": "retry policy"},
                        }
                    ],
                }
            },
        ]
        path.write_text(
            "\n".join(
                json.dumps(row | {"sessionId": "s", "cwd": "/p", "timestamp": "t"}) for row in rows
            ),
            encoding="utf-8",
        )
        record = preflight.collect(tmp_path, None)[0]
        assert record.occurrences[0].task == "the real task"

    def test_what_the_agent_did_next_is_recorded(self, preflight: Any, tmp_path: Path) -> None:
        session(
            tmp_path / "p" / "a.jsonl",
            (SEARCH, {"query": "retry policy"}),
            (READ, {"file_path": "a.md", "heading_path": "A > B"}),
        )
        assert preflight.collect(tmp_path, None)[0].occurrences[0].followed_by == ["read_section"]

    def test_a_retreat_to_the_file_system_is_recorded(self, preflight: Any, tmp_path: Path) -> None:
        session(
            tmp_path / "p" / "a.jsonl",
            (SEARCH, {"query": "retry policy"}),
            ("Read", {"file_path": "/docs/retry.md"}),
        )
        assert preflight.collect(tmp_path, None)[0].occurrences[0].followed_by == ["Read"]

    def test_labels_ship_empty(self, preflight: Any, tmp_path: Path) -> None:
        """The script collects; a person decides. Categories invented before the data
        would decide in advance what anyone finds in it."""
        session(tmp_path / "p" / "a.jsonl", (SEARCH, {"query": "retry policy"}))
        record = preflight.collect(tmp_path, None)[0]
        assert (record.family, record.verdict, record.note) == ("", "", "")

    def test_rows_round_trip_as_json(self, preflight: Any, tmp_path: Path) -> None:
        session(
            tmp_path / "p" / "a.jsonl",
            ("user", "a task"),
            (SEARCH, {"query": "retry policy"}),
        )
        rows = list(preflight.rows(preflight.collect(tmp_path, None)))
        loaded = json.loads(rows[0])
        assert loaded["query"] == "retry policy"
        assert loaded["occurrences"][0]["task"] == "a task"
        assert loaded["family"] == ""
