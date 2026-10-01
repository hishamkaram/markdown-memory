"""The transcript miner: the instrument that decides whether this tool is used at all.

Its numbers will be quoted to justify (or abandon) a day of benchmark work, so each one
is pinned here. Behaviour only - nothing in this module judges relevance.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from markdown_memory.models import estimate_tokens

SEARCH = "mcp__markdown-memory__search_docs"
READ_SECTION = "mcp__markdown-memory__read_section"
CWD = "/home/dev/project"
_IDS = itertools.count()


@pytest.fixture(scope="module")
def miner() -> Any:
    """Import by name, never by path.

    pytest's ``pythonpath`` puts ``scripts/`` on the path, and the mutation harness
    replaces that entry with a mutated copy. Loading the file by its absolute path would
    read the real source every time, so every mutation of this module would survive.
    """
    import usage_from_transcripts

    return usage_from_transcripts


def _record(session: str, role: str, content: list[dict[str, object]]) -> str:
    return json.dumps(
        {
            "sessionId": session,
            "cwd": CWD,
            "timestamp": "2026-09-20T12:00:00Z",
            "message": {"role": role, "content": content},
        }
    )


def transcript(path: Path, *calls: tuple[Any, ...], session: str | None = None) -> Path:
    """Write a transcript whose assistant turns make ``calls`` in order.

    A call is ``(name, arguments)`` or ``(name, arguments, envelope)``; the envelope is the
    ``tool_result`` the client sent back (``content``, ``is_error``), written as the next
    record. Every call gets its own id.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    session = session or path.stem
    lines = []
    for name, arguments, *envelope in calls:
        tool_id = f"toolu_{next(_IDS)}"
        use = {"type": "tool_use", "id": tool_id, "name": name, "input": arguments}
        lines.append(_record(session, "assistant", [use]))
        if envelope:
            result = {"type": "tool_result", "tool_use_id": tool_id, **envelope[0]}
            lines.append(_record(session, "user", [result]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def page(*paths: str, keyword_match: str = "matched") -> dict[str, object]:
    """A ``search_docs`` result envelope whose hits are ``paths``."""
    results = [{"file_path": path, "heading_path": "A"} for path in paths]
    return {"content": json.dumps({"results": results, "keyword_match": keyword_match})}


def text(value: str) -> dict[str, object]:
    return {"content": value}


def measure(miner: Any, root: Path) -> Any:
    return miner.summarise([miner.read_transcript(p) for p in miner.transcripts(root, None)])


class TestWhatCountsAsUse:
    def test_a_project_that_never_calls_the_server_is_counted_and_reported(
        self, miner: Any, tmp_path: Path
    ) -> None:
        """Zero is the most important number this script can print."""
        transcript(tmp_path / "proj" / "a.jsonl", ("Read", {"file_path": "README.md"}))
        report = measure(miner, tmp_path)
        assert report.sessions_scanned == 1
        assert report.sessions_using_server == 0
        assert report.searches == 0

    def test_the_server_is_recognised_under_its_mcp_prefix(
        self, miner: Any, tmp_path: Path
    ) -> None:
        transcript(
            tmp_path / "proj" / "a.jsonl",
            (SEARCH, {"query": "retry policy"}),
            (READ_SECTION, {"file_path": "a.md", "heading_path": "A > B"}),
        )
        report = measure(miner, tmp_path)
        assert report.sessions_using_server == 1
        assert report.server_calls == {"search_docs": 1, "read_section": 1}
        assert report.queries == ["retry policy"]

    def test_a_subagents_calls_count_too(self, miner: Any, tmp_path: Path) -> None:
        """Subagent transcripts live one directory deeper and call the same tools."""
        transcript(
            tmp_path / "proj" / "session" / "subagents" / "agent.jsonl",
            (SEARCH, {"query": "from a subagent"}),
        )
        assert measure(miner, tmp_path).searches == 1

    def test_another_servers_tools_are_not_counted_as_ours(
        self, miner: Any, tmp_path: Path
    ) -> None:
        transcript(tmp_path / "proj" / "a.jsonl", ("mcp__other-server__search", {"query": "x"}))
        assert measure(miner, tmp_path).sessions_using_server == 0


class TestPairing:
    """A result belongs to its own call, or the sizes and fallbacks are someone else's."""

    def test_each_call_gets_its_own_result(self, miner: Any, tmp_path: Path) -> None:
        path = transcript(
            tmp_path / "p" / "a.jsonl",
            (SEARCH, {"query": "retry"}, page("/docs/a.md")),
            ("Read", {"file_path": "/docs/a.md"}, text("1\tretry")),
            ("Bash", {"command": "ls"}),
        )
        calls = miner.read_transcript(path).calls
        assert [c.index for c in calls] == [0, 1, 2]
        assert [c.paired for c in calls] == [True, True, False], "the tail is unpaired"
        assert calls[1].result == "1\tretry"

    def test_a_reused_id_pairs_first_in_first_out(self, miner: Any, tmp_path: Path) -> None:
        """One real transcript reuses one id 981 times; a dict would hand out the last."""
        path = tmp_path / "p" / "a.jsonl"
        path.parent.mkdir(parents=True)
        use = {"type": "tool_use", "id": "same", "name": "Read", "input": {}}
        lines = [_record("s", "assistant", [use]), _record("s", "assistant", [use])]
        for payload in ("first", "second"):
            result = {"type": "tool_result", "tool_use_id": "same", "content": payload}
            lines.append(_record("s", "user", [result]))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        assert [c.result for c in miner.read_transcript(path).calls] == ["first", "second"]

    def test_an_id_in_another_file_does_not_pair(self, miner: Any, tmp_path: Path) -> None:
        use = {"type": "tool_use", "id": "shared", "name": "Read", "input": {}}
        result = {"type": "tool_result", "tool_use_id": "shared", "content": "elsewhere"}
        (tmp_path / "a.jsonl").write_text(_record("s", "assistant", [use]) + "\n")
        (tmp_path / "b.jsonl").write_text(_record("s", "user", [result]) + "\n")
        assert miner.read_transcript(tmp_path / "a.jsonl").calls[0].paired is False

    def test_the_value_of_is_error_decides_a_failure(self, miner: Any, tmp_path: Path) -> None:
        transcript(
            tmp_path / "p" / "a.jsonl",
            (SEARCH, {"query": "a"}, {"content": "boom", "is_error": True}),
            (SEARCH, {"query": "b"}, {**page("/docs/b.md"), "is_error": False}),
        )
        report = measure(miner, tmp_path)
        assert report.failures == 1
        assert report.search_evidence == {"failed": 1, "parsed": 1}

    @pytest.mark.parametrize(
        ("content", "result", "non_text"),
        [
            ("plain", "plain", False),
            (
                [{"type": "text", "text": '{"a": '}, {"type": "text", "text": "1}"}],
                '{"a": 1}',
                False,
            ),
            ([{"type": "image", "source": {}}], "", True),
            ([{"type": "tool_reference", "tool_name": "x"}], "", True),
        ],
    )
    def test_result_content_shapes(
        self, miner: Any, tmp_path: Path, content: object, result: str, non_text: bool
    ) -> None:
        path = transcript(tmp_path / "a.jsonl", ("Read", {}, {"content": content}))
        call = miner.read_transcript(path).calls[0]
        assert (call.paired, call.result, call.non_text) == (True, result, non_text)


class TestSessions:
    def test_a_subagent_belongs_to_its_parents_session(self, miner: Any, tmp_path: Path) -> None:
        transcript(tmp_path / "p" / "s.jsonl", (SEARCH, {"query": "a"}), session="s")
        transcript(tmp_path / "p" / "s" / "subagents" / "x.jsonl", ("Bash", {}), session="s")
        report = measure(miner, tmp_path)
        assert report.sessions_scanned == 1
        assert report.documentation_sessions_with_a_call == 1

    def test_a_session_with_no_calls_is_still_a_session(self, miner: Any, tmp_path: Path) -> None:
        """530 real sessions never call a tool; they are part of the denominator."""
        (tmp_path / "p").mkdir()
        (tmp_path / "p" / "quiet.jsonl").write_text(_record("quiet", "user", []) + "\n")
        transcript(tmp_path / "p" / "busy.jsonl", (SEARCH, {"query": "a"}))
        assert measure(miner, tmp_path).sessions_scanned == 2

    def test_a_window_never_spans_two_files(self, miner: Any, tmp_path: Path) -> None:
        """A sidechain's calls run beside the main thread's, not after them."""
        transcript(tmp_path / "p" / "s.jsonl", (SEARCH, {"query": "a"}, page("/d/a.md")))
        transcript(
            tmp_path / "p" / "s" / "subagents" / "x.jsonl",
            ("Read", {"file_path": "/d/a.md"}),
            session="s",
        )
        report = measure(miner, tmp_path)
        assert report.searches_followed_by_same_file_read == 0
        assert report.searches_followed_by_other_markdown == 0


class TestBehaviourAfterASearch:
    """The numbers that say whether a search was any use - none of them is a verdict."""

    def test_a_search_answered_by_reading_a_section(self, miner: Any, tmp_path: Path) -> None:
        transcript(
            tmp_path / "proj" / "a.jsonl",
            (SEARCH, {"query": "retry"}, page("/docs/retry.md")),
            (READ_SECTION, {"file_path": "/docs/retry.md", "heading_path": "A"}),
        )
        report = measure(miner, tmp_path)
        assert report.searches_followed_by_read_section == 1
        assert report.searches_followed_by_same_file_read == 0
        assert report.searches_followed_by_other_markdown == 0

    def test_a_read_of_the_file_the_search_returned_is_a_fallback(
        self, miner: Any, tmp_path: Path
    ) -> None:
        """The number a server-side log could never produce."""
        transcript(
            tmp_path / "proj" / "a.jsonl",
            (SEARCH, {"query": "retry"}, page("/docs/retry.md")),
            ("Read", {"file_path": "/docs/retry.md"}),
        )
        report = measure(miner, tmp_path)
        assert report.searches_followed_by_same_file_read == 1
        assert report.searches_followed_by_other_markdown == 0

    def test_a_read_of_another_file_is_not_a_same_file_fallback(
        self, miner: Any, tmp_path: Path
    ) -> None:
        transcript(
            tmp_path / "proj" / "a.jsonl",
            (SEARCH, {"query": "retry"}, page("/docs/retry.md")),
            ("Read", {"file_path": "/docs/other.md"}),
        )
        report = measure(miner, tmp_path)
        assert report.searches_followed_by_same_file_read == 0
        assert report.searches_followed_by_other_markdown == 1

    def test_reading_back_a_hit_in_scratch_is_not_a_fallback(
        self, miner: Any, tmp_path: Path
    ) -> None:
        transcript(
            tmp_path / "proj" / "a.jsonl",
            (SEARCH, {"query": "retry"}, page("/tmp/scratchpad/retry.md")),
            ("Read", {"file_path": "/tmp/scratchpad/retry.md"}),
        )
        assert measure(miner, tmp_path).searches_followed_by_same_file_read == 0

    def test_same_file_wins_when_both_happen(self, miner: Any, tmp_path: Path) -> None:
        transcript(
            tmp_path / "proj" / "a.jsonl",
            (SEARCH, {"query": "retry"}, page("/docs/retry.md")),
            ("Read", {"file_path": "/docs/other.md"}),
            ("Read", {"file_path": "/docs/retry.md"}),
        )
        report = measure(miner, tmp_path)
        assert report.searches_followed_by_same_file_read == 1
        assert report.searches_followed_by_other_markdown == 0

    def test_a_relative_read_matches_the_absolute_hit(self, miner: Any, tmp_path: Path) -> None:
        transcript(
            tmp_path / "proj" / "a.jsonl",
            (SEARCH, {"query": "retry"}, page(f"{CWD}/docs/retry.md")),
            ("Read", {"file_path": "docs/../docs/retry.md"}),
        )
        assert measure(miner, tmp_path).searches_followed_by_same_file_read == 1

    def test_a_manual_read_after_a_failed_search_still_counts(
        self, miner: Any, tmp_path: Path
    ) -> None:
        """Only telling *which* file the search offered needs its result."""
        transcript(
            tmp_path / "proj" / "a.jsonl",
            (SEARCH, {"query": "retry"}, {"content": "boom", "is_error": True}),
            ("Read", {"file_path": "/docs/retry.md"}),
        )
        assert measure(miner, tmp_path).searches_followed_by_other_markdown == 1

    def test_a_search_abandoned_for_grep(self, miner: Any, tmp_path: Path) -> None:
        transcript(
            tmp_path / "proj" / "a.jsonl",
            (SEARCH, {"query": "retry"}),
            ("Bash", {"command": "rg -n 'retry' docs/guide.md"}),
        )
        assert measure(miner, tmp_path).searches_followed_by_other_markdown == 1

    def test_an_unrelated_shell_command_is_not_a_fallback(self, miner: Any, tmp_path: Path) -> None:
        transcript(
            tmp_path / "proj" / "a.jsonl",
            (SEARCH, {"query": "retry"}),
            ("Bash", {"command": "uv run pytest -q"}),
        )
        assert measure(miner, tmp_path).searches_followed_by_other_markdown == 0

    def test_a_reworded_second_search(self, miner: Any, tmp_path: Path) -> None:
        transcript(
            tmp_path / "proj" / "a.jsonl",
            (SEARCH, {"query": "retry"}),
            (SEARCH, {"query": "backoff policy"}),
        )
        report = measure(miner, tmp_path)
        assert report.searches == 2
        assert report.searches_followed_by_another_search == 1

    def test_reformulation_chains_are_counted_apart(self, miner: Any, tmp_path: Path) -> None:
        transcript(
            tmp_path / "proj" / "a.jsonl",
            (SEARCH, {"query": "a"}),
            (SEARCH, {"query": "b"}),
            (SEARCH, {"query": "c"}),
            *[("Bash", {"command": "ls"})] * (5),
            (SEARCH, {"query": "d"}),
            (SEARCH, {"query": "e"}),
            *[("Bash", {"command": "ls"})] * (5),
            (SEARCH, {"query": "alone"}),
        )
        assert measure(miner, tmp_path).reformulation_chains == [3, 2]

    def test_a_file_read_long_after_a_search_is_not_attributed_to_it(
        self, miner: Any, tmp_path: Path
    ) -> None:
        """Past the window, the link to the search is guesswork."""
        filler = [("Bash", {"command": "uv run pytest -q"})] * (miner.FALLBACK_WINDOW + 1)
        transcript(
            tmp_path / "proj" / "a.jsonl",
            (SEARCH, {"query": "retry"}, page("/docs/retry.md")),
            *filler,
            ("Read", {"file_path": "/docs/retry.md"}),
        )
        report = measure(miner, tmp_path)
        assert report.searches_followed_by_same_file_read == 0
        assert report.searches_followed_by_other_markdown == 0

    def test_every_search_has_exactly_one_evidence_state(self, miner: Any, tmp_path: Path) -> None:
        transcript(
            tmp_path / "proj" / "a.jsonl",
            (SEARCH, {"query": "a"}, page("/d/a.md")),
            (SEARCH, {"query": "b"}, {"content": "boom", "is_error": True}),
            (SEARCH, {"query": "c"}, text("not json")),
            (SEARCH, {"query": "d"}),
        )
        evidence = measure(miner, tmp_path).search_evidence
        assert evidence == {"parsed": 1, "failed": 1, "unparseable": 1, "unpaired": 1}

    def test_empty_pages_and_keyword_states_are_counted(self, miner: Any, tmp_path: Path) -> None:
        transcript(
            tmp_path / "proj" / "a.jsonl",
            (SEARCH, {"query": "maxItemErrors"}, page(keyword_match="no_match")),
            (SEARCH, {"query": "retry"}, page("/d/a.md")),
        )
        report = measure(miner, tmp_path)
        assert report.empty_results == 1
        assert report.keyword_match == {"no_match": 1, "matched": 1}


class TestWhatCountsAsMarkdown:
    """The fallback count used to score every Grep and Glob, whatever it searched."""

    def call(self, miner: Any, name: str, arguments: dict[str, object]) -> Any:
        return miner.Call(0, name, None, arguments, "s", CWD, "")

    @pytest.mark.parametrize(
        ("name", "arguments", "touches", "reads"),
        [
            ("Read", {"file_path": "/docs/a.md"}, True, True),
            ("Read", {"file_path": "/src/a.py"}, False, False),
            ("Glob", {"pattern": "**/*.js", "path": "node_modules"}, False, False),
            ("Glob", {"pattern": "docs/**/*.md"}, True, False),
            ("Grep", {"pattern": "retry", "path": "docs/guide.md"}, True, True),
            ("Grep", {"pattern": "retry", "glob": "*.md"}, True, True),
            ("Grep", {"pattern": "retry", "type": "md"}, True, True),
            ("Grep", {"pattern": r"README\.md", "path": "src/"}, False, False),
            ("Grep", {"pattern": "retry"}, False, False),
            ("Bash", {"command": "cat docs/guide.md"}, True, True),
            ("Bash", {"command": "uv run pytest -q"}, False, False),
        ],
    )
    def test_only_fields_that_name_files_count(
        self,
        miner: Any,
        name: str,
        arguments: dict[str, object],
        touches: bool,
        reads: bool,
    ) -> None:
        call = self.call(miner, name, arguments)
        assert miner._touches_markdown(call) is touches
        assert miner._reads_markdown(call) is reads

    @pytest.mark.parametrize(
        ("name", "arguments"),
        [
            ("Read", {"file_path": "/tmp/scratchpad/notes.md"}),
            ("Read", {"file_path": os.path.expanduser("~/.claude/plans/plan.md")}),
            ("Read", {"file_path": "~/.claude/plans/plan.md"}),
            ("Bash", {"command": "cat /tmp/scratchpad/notes.md"}),
            ("Grep", {"pattern": "retry", "type": "md", "path": "/tmp/scratchpad"}),
            ("Glob", {"pattern": "**/*.md", "path": "/tmp/scratchpad"}),
        ],
    )
    def test_an_agents_own_notes_are_not_documentation(
        self, miner: Any, name: str, arguments: dict[str, object]
    ) -> None:
        assert miner._touches_markdown(self.call(miner, name, arguments)) is False


class TestDocumentationSessions:
    """Who needed documentation, and whether they asked the server for it."""

    def test_a_session_that_read_docs_by_hand_is_silence(self, miner: Any, tmp_path: Path) -> None:
        transcript(tmp_path / "p" / "a.jsonl", ("Read", {"file_path": "/docs/guide.md"}))
        report = measure(miner, tmp_path)
        assert report.documentation_sessions == 1
        assert report.documentation_sessions_with_a_call == 0
        assert report.silent_sessions == 1

    def test_a_retrieval_call_is_adoption(self, miner: Any, tmp_path: Path) -> None:
        transcript(tmp_path / "p" / "a.jsonl", (SEARCH, {"query": "a"}))
        report = measure(miner, tmp_path)
        assert report.documentation_sessions == report.documentation_sessions_with_a_call == 1
        assert report.retrieval_calls_in_documentation_sessions == 1

    def test_indexing_is_not_looking_anything_up(self, miner: Any, tmp_path: Path) -> None:
        transcript(tmp_path / "p" / "a.jsonl", ("mcp__markdown-memory__index_directory", {}))
        report = measure(miner, tmp_path)
        assert report.sessions_using_server == 1
        assert report.documentation_sessions == 0

    def test_listing_markdown_files_reads_none(self, miner: Any, tmp_path: Path) -> None:
        transcript(tmp_path / "p" / "a.jsonl", ("Glob", {"pattern": "**/*.md"}))
        assert measure(miner, tmp_path).documentation_sessions == 0

    @pytest.mark.parametrize("edit_first", [False, True])
    def test_reading_the_file_you_edit_is_maintenance(
        self, miner: Any, tmp_path: Path, edit_first: bool
    ) -> None:
        read = ("Read", {"file_path": "README.md"})
        edit = ("Edit", {"file_path": f"{CWD}/README.md"})
        transcript(tmp_path / "p" / "a.jsonl", *((edit, read) if edit_first else (read, edit)))
        assert measure(miner, tmp_path).documentation_sessions == 0

    def test_an_edit_in_a_subagent_counts_for_the_session(self, miner: Any, tmp_path: Path) -> None:
        transcript(tmp_path / "p" / "s.jsonl", ("Read", {"file_path": "/d/a.md"}), session="s")
        transcript(
            tmp_path / "p" / "s" / "subagents" / "x.jsonl",
            ("Write", {"file_path": "/d/a.md"}),
            session="s",
        )
        assert measure(miner, tmp_path).documentation_sessions == 0


class TestResultSizes:
    def test_each_tool_row_is_sized_with_the_shared_estimator(
        self, miner: Any, tmp_path: Path
    ) -> None:
        body = "x" * 401
        transcript(
            tmp_path / "p" / "a.jsonl",
            (SEARCH, {"query": "a"}, page("/d/a.md")),
            ("Read", {"file_path": "/d/a.md"}, text(body)),
            ("Read", {"file_path": "/d/b.md"}, {"content": [{"type": "image", "source": {}}]}),
            ("Read", {"file_path": "/src/a.py"}, text(body)),
        )
        report = measure(miner, tmp_path)
        row = report.rows["Read (Markdown)"]
        assert (row.calls, row.paired, row.non_text) == (2, 2, 1)
        assert row.tokens == [estimate_tokens(body)], "the image and the .py stay out"
        search = report.as_dict()["result_tokens"]["search_docs"]
        assert search["total"] == estimate_tokens(page("/d/a.md")["content"])  # type: ignore[arg-type]
        assert report.as_dict()["result_tokens"]["read_section"]["n"] == 0

    def test_statistics_survive_empty_and_single_samples(self, miner: Any) -> None:
        assert miner._stats([]) == {"n": 0, "total": 0, "median": 0, "p90": 0, "max": 0}
        assert miner._stats([7]) == {"n": 1, "total": 7, "median": 7, "p90": 7, "max": 7}
        assert miner._stats(list(range(1, 11)))["p90"] == 9


class TestTheCommandLine:
    def run(self, miner: Any, monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
        monkeypatch.setattr(sys, "argv", ["usage_from_transcripts.py", *argv])
        assert miner.main() == 0

    def test_zero_adoption_still_reports_who_needed_docs(
        self, miner: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        transcript(tmp_path / "p" / "a.jsonl", ("Read", {"file_path": "/docs/guide.md"}))
        self.run(miner, monkeypatch, "--root", str(tmp_path))
        out = capsys.readouterr().out
        assert "documentation-seeking sessions   1" in out
        assert "read Markdown, never called       1" in out
        assert "not billed" in out

    def test_the_json_report_carries_the_estimator_and_failures(
        self, miner: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        transcript(
            tmp_path / "p" / "a.jsonl", (SEARCH, {"query": "a"}, {"content": "x", "is_error": True})
        )
        out = tmp_path / "report.json"
        self.run(miner, monkeypatch, "--root", str(tmp_path), "--json", str(out))
        data = json.loads(out.read_text())
        assert data["failures"] == 1
        assert "not billed" in data["estimator"]
        assert data["search_evidence"] == {"failed": 1}
        assert "failed retrieval calls           1" in capsys.readouterr().out

    def test_a_server_registered_under_another_name(
        self, miner: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transcript(tmp_path / "p" / "a.jsonl", ("mcp__mm__search_docs", {"query": "a"}))
        out = tmp_path / "report.json"
        self.run(miner, monkeypatch, "--root", str(tmp_path), "--server", "mm", "--json", str(out))
        assert json.loads(out.read_text())["searches"] == 1


class TestRobustness:
    def test_a_half_written_line_does_not_lose_the_session(
        self, miner: Any, tmp_path: Path
    ) -> None:
        """A live session's transcript is being appended to while this runs."""
        path = transcript(tmp_path / "proj" / "a.jsonl", (SEARCH, {"query": "retry"}))
        path.write_text(path.read_text() + '{"message": {"content": [{"type": "tool', "utf-8")
        assert measure(miner, tmp_path).searches == 1

    def test_a_missing_transcript_directory_is_not_an_error(
        self, miner: Any, tmp_path: Path
    ) -> None:
        assert list(miner.transcripts(tmp_path / "nowhere", None)) == []
