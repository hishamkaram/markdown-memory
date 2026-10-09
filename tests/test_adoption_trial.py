"""The Step C harness (#126): what it reads from each agent, and the rules that decide.

The decision rules are frozen in `docs/adoption-trial/step-c.md` before any session runs,
so each boundary is pinned here: a rule that drifts by one would decide the trial.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

CWD = "/work/repo"


@pytest.fixture(scope="module")
def trial() -> Any:
    """Import by name, never by path, so the mutation harness's copy is the one tested."""
    import adoption_trial

    return adoption_trial


def lines(*records: dict[str, Any]) -> list[str]:
    return [json.dumps(record) for record in records]


def done(item: dict[str, Any]) -> dict[str, Any]:
    return {"type": "item.completed", "item": item}


def search(item_id: str, **extra: Any) -> dict[str, Any]:
    text = {"type": "text", "text": json.dumps({"results": [], "keyword_match": "matched"})}
    return done(
        {
            "id": item_id,
            "type": "mcp_tool_call",
            "server": "markdown-memory",
            "tool": "search_docs",
            "arguments": {"query": "cache"},
            "result": {"content": [text]},
            "error": None,
            "status": "completed",
        }
        | extra
    )


def message(text: str) -> dict[str, Any]:
    return done({"id": text, "type": "agent_message", "text": text})


def transcript(tmp_path: Path, records: list[dict[str, Any]]) -> Any:
    from usage_from_transcripts import read_transcript

    path = tmp_path / "t.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return read_transcript(path)


class TestCodexEvents:
    def test_a_markdown_cat_lands_in_the_shell_channel_with_its_output(
        self, trial: Any, tmp_path: Path
    ) -> None:
        from usage_from_transcripts import summarise

        events = lines(
            {"type": "thread.started", "thread_id": "t1"},
            done(
                {
                    "id": "c1",
                    "type": "command_execution",
                    "command": "/bin/bash -lc 'cat docs/cache.md'",
                    "aggregated_output": "x" * 400,
                    "exit_code": 0,
                    "status": "completed",
                }
            ),
            message("done"),
        )
        read = transcript(tmp_path, trial.convert_codex(events, CWD))
        assert [call.name for call in read.calls] == ["Bash"]
        assert read.calls[0].result == "x" * 400
        assert summarise([read]).shell_tokens_per_session == [100]

    def test_a_search_becomes_a_retrieval_call_of_the_session_in_its_directory(
        self, trial: Any, tmp_path: Path
    ) -> None:
        events = lines({"type": "thread.started", "thread_id": "t1"}, search("m1"))
        (call,) = transcript(tmp_path, trial.convert_codex(events, CWD)).calls
        assert (call.tool, call.paired, call.failed) == ("search_docs", True, False)
        assert (call.session, call.project, call.query) == ("t1", CWD, "cache")
        assert json.loads(call.result)["keyword_match"] == "matched"

    def test_a_failed_search_is_failed(self, trial: Any, tmp_path: Path) -> None:
        events = lines(search("m1", result=None, error={"message": "boom"}, status="failed"))
        (call,) = transcript(tmp_path, trial.convert_codex(events, CWD)).calls
        assert call.failed and call.result == "boom"

    def test_the_order_of_calls_is_kept(self, trial: Any, tmp_path: Path) -> None:
        events = lines(
            search("m1"),
            done({"id": "c1", "type": "command_execution", "command": "ls", "exit_code": 0}),
            search("m2"),
        )
        calls = transcript(tmp_path, trial.convert_codex(events, CWD)).calls
        assert [call.name for call in calls] == [
            "mcp__markdown-memory__search_docs",
            "Bash",
            "mcp__markdown-memory__search_docs",
        ]

    def test_a_session_without_tool_calls_still_has_its_last_answer(self, trial: Any) -> None:
        events = lines(message("thinking"), message("The docs say X."))
        assert trial.answer("codex", events) == "The docs say X."
        assert trial.convert_codex(events, CWD)[-1]["message"]["content"][0]["type"] == "text"


class TestClaudeStream:
    INIT = {
        "type": "system",
        "subtype": "init",
        "session_id": "s1",
        "cwd": CWD,
        "mcp_servers": [{"name": "markdown-memory", "status": "connected"}],
        "plugins": [],
    }

    def test_every_record_carries_the_session_and_directory(
        self, trial: Any, tmp_path: Path
    ) -> None:
        use = {"type": "tool_use", "id": "u1", "name": "Read", "input": {"file_path": "a.md"}}
        back = {"type": "tool_result", "tool_use_id": "u1", "content": "text"}
        stream = lines(
            self.INIT,
            {"type": "assistant", "session_id": "s1", "message": {"content": [use]}},
            {"type": "user", "session_id": "s1", "message": {"content": [back]}},
        )
        (call,) = transcript(tmp_path, trial.normalize_claude(stream)).calls
        assert (call.session, call.project, call.paired) == ("s1", CWD, True)

    def test_the_answer_is_the_last_result(self, trial: Any) -> None:
        stream = lines(
            self.INIT, {"type": "result", "result": "first"}, {"type": "result", "result": "last"}
        )
        assert trial.answer("claude", stream) == "last"

    @pytest.mark.parametrize(
        ("change", "ok"),
        [
            ({}, True),
            ({"mcp_servers": [{"name": "markdown-memory", "status": "failed"}]}, False),
            (
                {
                    "mcp_servers": [
                        {"name": "markdown-memory", "status": "connected"},
                        {"name": "context7", "status": "connected"},
                    ]
                },
                False,
            ),
            ({"plugins": [{"name": "caveman"}]}, False),
        ],
    )
    def test_init_must_show_only_our_server_and_no_plugin(
        self, trial: Any, change: dict[str, Any], ok: bool
    ) -> None:
        assert trial.init_ok([self.INIT | change]) is ok


def session(trial: Any, agent: str, arm: str, task: str, attempt: int = 1, **extra: Any) -> Any:
    fields: dict[str, Any] = {
        "exit_code": 0,
        "seconds": 1.0,
        "timed_out": False,
        "init_ok": True,
        "has_answer": True,
        "raw": "",
        "transcript": "",
    }
    return trial.Session(agent, arm, task, attempt, **(fields | extra))


class TestPairs:
    def test_a_failed_arm_drops_its_pair_from_both_arms(self, trial: Any) -> None:
        sessions = [
            session(trial, "claude", "A", "t1"),
            session(trial, "claude", "B", "t1", exit_code=1),
            session(trial, "claude", "A", "t2"),
            session(trial, "claude", "B", "t2"),
        ]
        kept, dropped = trial.kept_pairs(sessions)
        assert list(kept) == [("claude", "t2")] and dropped == 1

    def test_a_failure_of_arm_a_drops_the_pair_too(self, trial: Any) -> None:
        sessions = [
            session(trial, "codex", "A", "t1", has_answer=False),
            session(trial, "codex", "B", "t1"),
        ]
        assert trial.kept_pairs(sessions) == ({}, 1)

    def test_a_pair_whose_rerun_succeeds_is_kept_at_its_rerun(self, trial: Any) -> None:
        sessions = [
            session(trial, "codex", "A", "t1", timed_out=True),
            session(trial, "codex", "B", "t1"),
            session(trial, "codex", "A", "t1", attempt=2),
            session(trial, "codex", "B", "t1", attempt=2),
        ]
        kept, dropped = trial.kept_pairs(sessions)
        assert dropped == 0
        assert {s.attempt for s in kept["codex", "t1"].values()} == {2}

    def test_a_frozen_pair_that_did_not_run_in_both_arms_is_named(self, trial: Any) -> None:
        sessions = [
            session(trial, "claude", "A", "t1"),
            session(trial, "claude", "B", "t1"),
            session(trial, "codex", "B", "t1"),
        ]
        assert trial.missing_pairs(sessions, ["t1", "t2"]) == [
            "codex t1",
            "claude t2",
            "codex t2",
        ]

    def test_more_than_10_percent_dropped_stops_the_analysis(self, trial: Any) -> None:
        assert not trial.too_many_dropped(8, 80)
        assert trial.too_many_dropped(9, 80)
        assert trial.too_many_dropped(0, 0)

    def test_the_answerable_labels_are_the_frozen_ones(self, trial: Any) -> None:
        frozen = trial.tasks_in(trial.FROZEN_TASKS)
        assert len(frozen) == 40
        assert sum(not task["answerable"] for task in frozen.values()) == 8


class TestDecision:
    @pytest.mark.parametrize(
        ("b", "c", "holds"),
        [(8, 2, False), (9, 1, True), (6, 0, True), (5, 0, False), (15, 5, True), (14, 6, False)],
    )
    def test_the_sign_test_holds_at_alpha_0_025_exactly_where_the_protocol_says(
        self, trial: Any, b: int, c: int, holds: bool
    ) -> None:
        assert trial.effect(b, c) is holds

    def test_the_sign_test_is_exact(self, trial: Any) -> None:
        from fractions import Fraction

        assert trial.sign_test(9, 1) == Fraction(11, 1024)
        assert trial.sign_test(0, 0) == 1

    @staticmethod
    def grades(a: list[str], b: list[str]) -> dict[str, dict[str, str]]:
        return {f"t{i}": {"A": x, "B": y} for i, (x, y) in enumerate(zip(a, b, strict=True))}

    def test_harm_fires_at_exactly_15_points_fewer_correct(self, trial: Any) -> None:
        asked = {f"t{i}": True for i in range(20)}
        a = ["correct"] * 10 + ["wrong"] * 10
        assert trial.harm(self.grades(a, ["correct"] * 7 + ["wrong"] * 13), asked)
        assert not trial.harm(self.grades(a, ["correct"] * 8 + ["wrong"] * 12), asked)

    def test_partial_is_not_correct(self, trial: Any) -> None:
        asked = {f"t{i}": True for i in range(4)}
        assert trial.harm(self.grades(["correct"] * 4, ["correct"] * 3 + ["partial"]), asked)

    def test_harm_fires_at_two_fewer_correct_abstentions(self, trial: Any) -> None:
        unanswerable = {f"t{i}": False for i in range(4)}
        a = ["abstained"] * 4
        assert trial.harm(self.grades(a, ["abstained"] * 2 + ["answered"] * 2), unanswerable)
        assert not trial.harm(self.grades(a, ["abstained"] * 3 + ["answered"]), unanswerable)

    def test_it_ships_only_on_an_effect_with_no_harm(self, trial: Any) -> None:
        assert trial.ship({"claude": True, "codex": False}, {"claude": False, "codex": False})
        assert not trial.ship({"claude": True, "codex": False}, {"claude": False, "codex": True})
        assert not trial.ship({"claude": False, "codex": False}, {"claude": False, "codex": False})

    def test_a_disagreement_takes_the_lower_grade_unless_resolved(self, trial: Any) -> None:
        first = {"g1": "correct", "g2": "partial", "g3": "abstained", "g4": "wrong"}
        second = {"g1": "partial", "g2": "partial", "g3": "answered", "g4": "correct"}
        merged, listed = trial.merge_grades(first, second, {"g4": "correct"})
        assert merged == {"g1": "partial", "g2": "partial", "g3": "answered", "g4": "correct"}
        assert len(listed) == 3


def test_codex_may_call_only_the_retrieval_tools_without_asking(trial: Any, tmp_path: Path) -> None:
    arm = trial.Arm("A", tmp_path / "wt", tmp_path / "a.db")
    argv = trial.command("codex", arm, tmp_path, tmp_path / "s.mcp.json")
    approved = sorted(part for part in argv if part.endswith('approval_mode = "approve"'))
    assert len(approved) == 4 and not any("index_directory" in part for part in approved)
    assert "mcp_servers.markdown-memory.required = true" in argv
    assert 'MARKDOWN_MEMORY_AUTO_INDEX = "0"' in " ".join(argv)


def test_claude_sees_only_its_own_server_and_project_settings(trial: Any, tmp_path: Path) -> None:
    arm = trial.Arm("B", tmp_path / "wt", tmp_path / "b.db")
    own = tmp_path / "claude-B-t1-1.mcp.json"  # one file per session: none is rewritten mid-read
    argv = trial.command("claude", arm, tmp_path, own)
    assert argv[argv.index("--setting-sources") + 1] == "project"
    assert "--strict-mcp-config" in argv and argv[argv.index("--mcp-config") + 1] == str(own)
    config = json.loads(own.read_text(encoding="utf-8"))
    assert list(config["mcpServers"]) == ["markdown-memory"]
    assert config["mcpServers"]["markdown-memory"]["env"]["MARKDOWN_MEMORY_AUTO_INDEX"] == "0"
    allowed = argv[argv.index("--allowedTools") + 1].split(",")
    assert "mcp__markdown-memory__index_directory" not in allowed


def test_every_documented_command_line_parses(trial: Any) -> None:
    usage = [
        line.split("scripts/adoption_trial.py ", 1)[1]
        for line in trial.__doc__.splitlines()
        if "scripts/adoption_trial.py " in line
    ]
    assert len(usage) == 4
    for line in usage:
        argv = [word for word in line.replace("[", "").replace("]", "").split() if word != "..."]
        trial.make_parser().parse_args(argv)
