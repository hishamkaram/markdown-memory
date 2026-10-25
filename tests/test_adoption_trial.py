"""The Step C and Step D harness (#126, #134): what it reads, and the rules that decide.

The decision rules are frozen in `docs/adoption-trial/step-c.md` and `step-d.md` before any
session runs, so each boundary is pinned here: a rule that drifts by one would decide the trial.
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
            ({"plugins": [{"name": "caveman", "path": "/plugins/caveman"}]}, False),
            (
                {
                    "plugins": [
                        {
                            "name": "cc-plugin-agents-md",
                            "path": "builtin",
                            "source": "cc-plugin-agents-md@builtin",
                        }
                    ]
                },
                True,
            ),
            ({"plugins": [{"name": "x", "path": "builtin", "source": "x@market"}]}, False),
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


# --- Step D (#134): the protocol preset, Codex's own home, and its delivery check --------------

# What today's `report()` printed for `trial_run` before Step D changed the harness: Step C's
# report must not change by a character.
STEP_C_REPORT = """pairs: 80 kept, 0 dropped of 80

claude: 40 kept pairs
  arm A: adoption 30/40; correct 31/32; abstained 8/8; calls per adopting session 1.0; same-file fallback 0/30; chains 0; median documentation tokens 11.0, shell 0.0
  arm B: adoption 40/40; correct 32/32; abstained 7/8; calls per adopting session 1.0; same-file fallback 0/40; chains 0; median documentation tokens 11.0, shell 0.0
  sign test: b=10 c=0 n=10 p=0.0010 -> effect yes (alpha 0.025); harm no

codex: 40 kept pairs
  arm A: adoption 0/40; correct 31/32; abstained 8/8; calls per adopting session 0.0; same-file fallback 0/0; chains 0; median documentation tokens -, shell -
  arm B: adoption 3/40; correct 32/32; abstained 7/8; calls per adopting session 1.0; same-file fallback 0/3; chains 0; median documentation tokens 11, shell 0
  sign test: b=3 c=0 n=3 p=0.1250 -> effect no (alpha 0.025); harm no

grader disagreements: 4
  g037: correct vs partial -> partial
  g074: abstained vs answered -> answered
  g111: correct vs partial -> partial
  g148: abstained vs answered -> answered

ship: yes
"""  # noqa: E501 - the report's own lines, verbatim


def trial_run(
    trial: Any, root: Path, codex_b: int = 3, codex_failed: int | None = 2, claude_wrong: int = 0
) -> tuple[Path, list[Path]]:
    """All 80 frozen pairs: Claude searches in B always and in A on 30 tasks; Codex in B on the
    first `codex_b`, one of whose searches failed. Every 37th item's second grader is one grade
    lower; Claude's B answers to the first `claude_wrong` answerable tasks are wrong."""
    frozen = trial.tasks_in(trial.FROZEN_TASKS)
    sessions, blind, first, second = [], {}, {}, {}
    for i, task in enumerate(sorted(frozen)):
        for agent in trial.AGENTS:
            for arm in trial.ARMS:
                events = [{"type": "thread.started", "thread_id": f"{agent}-{arm}-{task}"}]
                if agent == "claude" and (arm == "B" or i < 30):
                    events.append(search("m1"))
                if agent == "codex" and arm == "B" and i < codex_b:
                    failed = {"result": None, "error": {"message": "boom"}, "status": "failed"}
                    events.append(search("m1", **(failed if i == codex_failed else {})))
                events.append(message("answer"))
                path = root / f"{agent}-{arm}-{task}-1.transcript.jsonl"
                records = trial.convert_codex(lines(*events), CWD)
                path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
                sessions.append(session(trial, agent, arm, task, transcript=str(path)))
                item = f"g{len(sessions):03d}"
                blind[item] = {"agent": agent, "arm": arm, "task": task}
                grade = "correct" if frozen[task]["answerable"] else "abstained"
                lower = "partial" if frozen[task]["answerable"] else "answered"
                asked = [t for t in sorted(frozen) if frozen[t]["answerable"]][:claude_wrong]
                if (agent, arm) == ("claude", "B") and task in asked:
                    grade = "wrong"
                first[item], second[item] = grade, lower if len(sessions) % 37 == 0 else grade
    (root / "sessions.json").write_text(
        json.dumps([trial.asdict(s) for s in sessions]), encoding="utf-8"
    )
    for name, value in (("blind", blind), ("g1", first), ("g2", second)):
        (root / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
    return root / "blind.json", [root / "g1.json", root / "g2.json"]


class TestStepCReportIsUnchanged:
    def test_a_run_without_a_manifest_reports_as_step_c_did(
        self, trial: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        blind, grades = trial_run(trial, tmp_path)
        assert trial.report([tmp_path], blind, grades, None) == 0
        assert capsys.readouterr().out == STEP_C_REPORT

    def test_a_step_c_manifest_reports_the_same(
        self, trial: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        blind, grades = trial_run(trial, tmp_path)
        trial.write_manifest(tmp_path, trial.Setup())
        assert trial.report([tmp_path], blind, grades, None) == 0
        assert capsys.readouterr().out == STEP_C_REPORT


class TestStepDReport:
    def run(self, trial: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> str:
        blind, grades = trial_run(trial, tmp_path)
        trial.write_manifest(
            tmp_path, trial.Setup(trial.STEP_D, tmp_path / "home", "model-x", None)
        )
        assert trial.report([tmp_path], blind, grades, None) == 0
        return capsys.readouterr().out

    def test_only_codex_is_tested_and_a_failed_search_is_not_adoption(
        self, trial: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = self.run(trial, tmp_path, capsys)
        assert "effect not tested (step-d tests codex)" in out
        assert "  arm B: adoption 2/40;" in out  # the failed search does not count
        assert "with failed calls 3/40" in out
        assert "b=2 c=0 n=2 p=0.2500 -> effect no (alpha 0.05)" in out
        assert "paired difference B - A: +2 of 40 pairs" in out
        assert "by repository: uv 2/20" in out

    def test_claude_s_effect_alone_does_not_ship_step_d(
        self, trial: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = self.run(trial, tmp_path, capsys)
        assert "b=10 c=0" in out and out.endswith("ship: no\n")

    def test_calls_per_adopting_session_count_only_the_adopters(
        self, trial: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = self.run(trial, tmp_path, capsys)
        codex_b = next(line for line in out.split("codex:")[1].splitlines() if "arm B:" in line)
        assert "calls per adopting session 1.0;" in codex_b  # the failed call is not theirs
        assert "B - A: +2 of 40 pairs; uv +2 of 20, vite +0 of 20" in out

    @pytest.mark.parametrize(("wrong", "ships"), [(8, False), (0, True)])
    def test_claude_s_harm_vetoes_a_significant_codex_effect(
        self,
        trial: Any,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        wrong: int,
        ships: bool,
    ) -> None:
        blind, grades = trial_run(trial, tmp_path, codex_b=8, codex_failed=None, claude_wrong=wrong)
        trial.write_manifest(tmp_path, trial.Setup(trial.STEP_D, tmp_path, "model-x"))
        assert trial.report([tmp_path], blind, grades, None) == 0
        out = capsys.readouterr().out
        assert "b=8 c=0 n=8 p=0.0039 -> effect yes (alpha 0.05)" in out
        assert out.endswith(f"ship: {'yes' if ships else 'no'}\n")


class TestManifest:
    def test_it_round_trips_the_protocol(self, trial: Any, tmp_path: Path) -> None:
        trial.write_manifest(tmp_path, trial.Setup(trial.STEP_D, tmp_path, "m", "high"))
        assert trial.protocol_of(tmp_path) is trial.STEP_D
        fields = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert (fields["codex_model"], fields["codex_effort"]) == ("m", "high")
        assert fields["rule"] == trial.RULE

    def test_no_manifest_is_step_c(self, trial: Any, tmp_path: Path) -> None:
        assert trial.protocol_of(tmp_path) is trial.STEP_C

    @pytest.mark.parametrize("text", ["{not json", '{"protocol": "step-z"}', "[]"])
    def test_an_unreadable_manifest_is_refused(
        self, trial: Any, tmp_path: Path, text: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        blind, grades = trial_run(trial, tmp_path)
        (tmp_path / "manifest.json").write_text(text, encoding="utf-8")
        assert trial.protocol_of(tmp_path) is None
        assert trial.report([tmp_path], blind, grades, None) == 2
        assert "REFUSED" in capsys.readouterr().out

    def test_runs_of_two_protocols_are_refused(
        self, trial: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        uv, vite = tmp_path / "uv", tmp_path / "vite"
        uv.mkdir(), vite.mkdir()
        blind, grades = trial_run(trial, uv)
        trial_run(trial, vite)  # complete on its own: only the protocols differ
        trial.write_manifest(vite, trial.Setup(trial.STEP_D, tmp_path, "m"))
        assert trial.report([uv, vite], blind, grades, None) == 2
        assert "REFUSED" in capsys.readouterr().out


class TestProtocols:
    def test_alpha_is_the_family_s_5_percent_split_over_the_agents_tested(self, trial: Any) -> None:
        from fractions import Fraction

        assert trial.STEP_C.alpha == Fraction(1, 40)
        assert trial.STEP_D.alpha == Fraction(1, 20)

    @pytest.mark.parametrize(("b", "c", "holds"), [(5, 0, True), (4, 0, False), (8, 2, False)])
    def test_step_d_s_one_test_holds_at_0_05(self, trial: Any, b: int, c: int, holds: bool) -> None:
        assert trial.effect(b, c, trial.STEP_D.alpha) is holds

    def test_adoption_under_step_d_needs_an_answered_call_that_did_not_fail(
        self, trial: Any, tmp_path: Path
    ) -> None:
        failed = search("m1", result=None, error={"message": "boom"}, status="failed")
        empty = transcript(tmp_path, trial.convert_codex(lines(search("m1")), CWD))
        broken = transcript(tmp_path, trial.convert_codex(lines(failed), CWD))
        assert trial.adopted(empty, successful_only=True)  # an empty result is still an answer
        assert not trial.adopted(broken, successful_only=True)
        assert trial.adopted(broken, successful_only=False)  # Step C counted the attempt

    def test_an_unanswered_call_is_not_adoption_under_step_d(
        self, trial: Any, tmp_path: Path
    ) -> None:
        use = {"type": "tool_use", "id": "u1", "name": "mcp__markdown-memory__search_docs"}
        alone = transcript(
            tmp_path,
            [{"type": "assistant", "sessionId": "s", "cwd": CWD, "message": {"content": [use]}}],
        )
        assert not trial.adopted(alone, successful_only=True)
        assert trial.adopted(alone, successful_only=False)

    def test_search_first_means_before_any_markdown_read(self, trial: Any, tmp_path: Path) -> None:
        cat = done({"id": "c1", "type": "command_execution", "command": "cat docs/a.md"})
        ls = done({"id": "c0", "type": "command_execution", "command": "ls"})
        before = transcript(tmp_path, trial.convert_codex(lines(ls, search("m1"), cat), CWD))
        after = transcript(tmp_path, trial.convert_codex(lines(cat, search("m1")), CWD))
        assert trial.searched_first(before) and not trial.searched_first(after)

    def test_wilson_matches_its_closed_form(self, trial: Any) -> None:
        low, high = trial.wilson(0, 40)
        assert low == 0.0 and high == pytest.approx(0.0876, abs=1e-4)
        assert trial.wilson(20, 40) == pytest.approx((0.3520, 0.6480), abs=1e-4)
        assert trial.wilson(0, 0) == (0.0, 1.0)


class TestCodexHome:
    def setup_d(self, trial: Any, home: Path, effort: str | None = None) -> Any:
        return trial.Setup(trial.STEP_D, home, "model-x", effort)

    def test_step_c_inherits_the_environment(self, trial: Any) -> None:
        assert trial.child_env("codex", trial.Setup()) is None

    def test_codex_gets_its_own_home_and_os_environ_is_untouched(
        self, trial: Any, tmp_path: Path
    ) -> None:
        import os

        before = dict(os.environ)
        env = trial.child_env("codex", self.setup_d(trial, tmp_path))
        assert (env["CODEX_HOME"], env["HOME"]) == (str(tmp_path), str(tmp_path))
        assert dict(os.environ) == before
        claude = trial.child_env("claude", self.setup_d(trial, tmp_path))
        assert claude["DISABLE_AUTOUPDATER"] == "1" and claude["HOME"] == before["HOME"]

    def test_codex_runs_without_apps_plugins_or_hooks_on_the_pinned_model(
        self, trial: Any, tmp_path: Path
    ) -> None:
        arm = trial.Arm("B", tmp_path / "wt", tmp_path / "b.db", tmp_path / "clone")
        argv = trial.command("codex", arm, tmp_path, tmp_path / "x", self.setup_d(trial, tmp_path))
        settings = [argv[i + 1] for i, part in enumerate(argv) if part == "-c"]
        assert set(trial.ISOLATION) <= set(settings)
        assert 'model="model-x"' in settings
        assert not any(s.startswith("model_reasoning_effort") for s in settings)
        assert "--ignore-user-config" in argv
        server_env = next(s for s in settings if s.startswith("mcp_servers.markdown-memory.env"))
        assert f'HOME = "{Path.home()}"' in server_env  # uv and the model cache live there

    def test_an_effort_is_pinned_only_when_given(self, trial: Any, tmp_path: Path) -> None:
        arm = trial.Arm("A", tmp_path, tmp_path / "a.db", tmp_path)
        argv = trial.command(
            "codex", arm, tmp_path, tmp_path / "x", self.setup_d(trial, tmp_path, "high")
        )
        assert 'model_reasoning_effort="high"' in argv

    def test_step_c_s_codex_command_is_unchanged(self, trial: Any, tmp_path: Path) -> None:
        arm = trial.Arm("A", tmp_path / "wt", tmp_path / "a.db")
        argv = trial.command("codex", arm, tmp_path, tmp_path / "x")
        assert not set(trial.ISOLATION) & set(argv)
        assert not any(part.startswith("model=") for part in argv)


class TestArms:
    def test_an_arm_may_name_its_own_clone(self, trial: Any, tmp_path: Path) -> None:
        arm = trial.Arm.parse(f"B={tmp_path}/wt:{tmp_path}/b.db:{tmp_path}/clone")
        assert arm.clone == (tmp_path / "clone").resolve()
        assert trial.Arm.parse(f"A={tmp_path}/wt:{tmp_path}/a.db").clone is None

    def test_a_fourth_field_is_refused(self, trial: Any) -> None:
        import argparse

        with pytest.raises(argparse.ArgumentTypeError):
            trial.Arm.parse("A=w:d:c:extra")

    @pytest.mark.parametrize(
        ("argv", "message"),
        [
            (["--protocol", "step-d", "--codex-model", "m"], "needs --codex-home"),
            (["--codex-home", "/h"], "belong to step-d"),
            (["--protocol", "step-d", "--codex-home", "/h", "--codex-model", "m"], "own clone"),
        ],
    )
    def test_step_d_refuses_to_start_without_what_it_pins(
        self, trial: Any, argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        parser = trial.make_parser()
        arguments = parser.parse_args(
            ["run", "t.json", "--arm", "A=w:d", "--arm", "B=w:d", *argv, "o"]
        )
        with pytest.raises(SystemExit):
            trial.setup_of(arguments, parser)
        assert message in capsys.readouterr().err

    def test_a_step_d_probe_runs_in_its_arm_s_clone(
        self, trial: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        parser = trial.make_parser()
        flags = ["--protocol", "step-d", "--codex-home", "/h", "--codex-model", "m"]
        arm = f"A=w:d:{tmp_path}/clone"
        ok = parser.parse_args(
            ["probe", "--agent", "codex", "--arm", arm, "--repo", f"{tmp_path}/clone", *flags, "o"]
        )
        assert trial.setup_of(ok, parser).protocol is trial.STEP_D
        bad = parser.parse_args(
            ["probe", "--agent", "codex", "--arm", arm, "--repo", str(tmp_path), *flags, "o"]
        )
        with pytest.raises(SystemExit):
            trial.setup_of(bad, parser)
        assert "--repo is that clone" in capsys.readouterr().err


def item(payload: dict[str, Any]) -> dict[str, Any]:
    return {"type": "response_item", "payload": payload}


def said(role: str, *texts: str) -> dict[str, Any]:
    kind = "output_text" if role == "assistant" else "input_text"
    return item(
        {"type": "message", "role": role, "content": [{"type": kind, "text": t} for t in texts]}
    )


class TestDelivery:
    """Every predicate of step-d.md's delivery check, each failing on its own."""

    BRIEF = "You are the assigned worker; answer the question."

    def expect(self, trial: Any, tmp_path: Path, ruled: bool = True, **change: Any) -> Any:
        fields = {
            "thread": "t1",
            "cwd": tmp_path / "clone",
            "brief": self.BRIEF,
            "ruled": ruled,
            "model": "model-x",
            "effort": None,
            "home": tmp_path / "home",
        }
        return trial.Delivery(**(fields | change))

    def records(
        self, trial: Any, tmp_path: Path, agents: str | None = "rule", skills: tuple[str, ...] = ()
    ) -> list[dict[str, Any]]:
        roots = "\n".join(f"- `r{i}` = `{root}`" for i, root in enumerate(skills))
        startup = [said("developer", f"<skills_instructions>\n### Skill roots\n{roots}\n")]
        if agents is not None:
            body = trial.RULE if agents == "rule" else agents
            startup.append(
                said(
                    "user",
                    f"# AGENTS.md instructions for {tmp_path}/clone\n\n"
                    f"<INSTRUCTIONS>\n{body}\n</INSTRUCTIONS>",
                )
            )
        return [
            {
                "type": "session_meta",
                "payload": {
                    "id": "t1",
                    "cwd": str(tmp_path / "clone"),
                    "base_instructions": {"text": "You are Codex."},
                },
            },
            *startup,
            {"type": "turn_context", "payload": {"model": "model-x", "effort": None}},
            said("user", self.BRIEF),
            {"type": "event_msg", "payload": {"type": "user_message", "message": self.BRIEF}},
            said("assistant", "I looked at docs that mention codebase-memory and context7."),
        ]

    def test_arm_b_with_the_rule_passes(self, trial: Any, tmp_path: Path) -> None:
        assert trial.delivered(self.records(trial, tmp_path), self.expect(trial, tmp_path))

    def test_arm_a_without_any_agents_md_passes(self, trial: Any, tmp_path: Path) -> None:
        # vite has no AGENTS.md, so its arm A gets no AGENTS.md message at all.
        records = self.records(trial, tmp_path, agents=None)
        assert trial.delivered(records, self.expect(trial, tmp_path, ruled=False))

    def test_arm_a_with_its_own_agents_md_passes(self, trial: Any, tmp_path: Path) -> None:
        records = self.records(trial, tmp_path, agents="- ALWAYS run the tests")
        assert trial.delivered(records, self.expect(trial, tmp_path, ruled=False))

    def test_the_rule_in_arm_a_fails(self, trial: Any, tmp_path: Path) -> None:
        records = self.records(trial, tmp_path)
        assert not trial.delivered(records, self.expect(trial, tmp_path, ruled=False))

    def test_arm_b_without_the_rule_fails(self, trial: Any, tmp_path: Path) -> None:
        records = self.records(trial, tmp_path, agents="- ALWAYS run the tests")
        assert not trial.delivered(records, self.expect(trial, tmp_path))

    @pytest.mark.parametrize(
        "change",
        [
            {"thread": "t2"},
            {"cwd": Path("/elsewhere")},
            {"model": "model-y"},
            {"effort": "high"},
            {"brief": "another brief"},
        ],
    )
    def test_another_session_model_effort_or_brief_fails(
        self, trial: Any, tmp_path: Path, change: dict[str, Any]
    ) -> None:
        assert not trial.delivered(
            self.records(trial, tmp_path), self.expect(trial, tmp_path, **change)
        )

    def test_a_later_turn_on_another_model_fails(self, trial: Any, tmp_path: Path) -> None:
        records = [
            *self.records(trial, tmp_path),
            {"type": "turn_context", "payload": {"model": "model-y", "effort": None}},
        ]
        assert not trial.delivered(records, self.expect(trial, tmp_path))

    def test_a_turn_without_its_effort_fails(self, trial: Any, tmp_path: Path) -> None:
        records = self.records(trial, tmp_path)
        context = next(r for r in records if r["type"] == "turn_context")
        del context["payload"]["effort"]  # absent is not "the default": it is no evidence
        assert not trial.delivered(records, self.expect(trial, tmp_path))

    def test_no_turn_context_fails(self, trial: Any, tmp_path: Path) -> None:
        records = [r for r in self.records(trial, tmp_path) if r["type"] != "turn_context"]
        assert not trial.delivered(records, self.expect(trial, tmp_path))

    def test_two_session_records_fail(self, trial: Any, tmp_path: Path) -> None:
        records = self.records(trial, tmp_path)
        assert not trial.delivered([records[0], *records], self.expect(trial, tmp_path))

    def test_the_brief_given_twice_fails(self, trial: Any, tmp_path: Path) -> None:
        records = [*self.records(trial, tmp_path), said("user", self.BRIEF)]
        assert not trial.delivered(records, self.expect(trial, tmp_path))

    @pytest.mark.parametrize("word", ["codebase-memory", "context7", "Personal rules"])
    def test_the_operator_s_instructions_at_startup_fail(
        self, trial: Any, tmp_path: Path, word: str
    ) -> None:
        records = self.records(trial, tmp_path, agents=f"{trial.RULE}\n# {word}")
        assert not trial.delivered(records, self.expect(trial, tmp_path))

    def test_the_operator_s_instructions_in_the_base_instructions_fail(
        self, trial: Any, tmp_path: Path
    ) -> None:
        records = self.records(trial, tmp_path)
        records[0]["payload"]["base_instructions"] = {"text": "Use context7."}
        assert not trial.delivered(records, self.expect(trial, tmp_path))

    @pytest.mark.parametrize("base", [None, "You are Codex.", {"text": ""}, {"provenance": {}}])
    def test_missing_base_instructions_fail(self, trial: Any, tmp_path: Path, base: Any) -> None:
        records = self.records(trial, tmp_path)
        if base is None:
            del records[0]["payload"]["base_instructions"]
        else:
            records[0]["payload"]["base_instructions"] = base
        assert not trial.delivered(records, self.expect(trial, tmp_path))

    def test_what_the_session_read_after_the_brief_does_not_count(
        self, trial: Any, tmp_path: Path
    ) -> None:
        # records() ends with an answer naming two of the words; it still passes.
        assert trial.delivered(self.records(trial, tmp_path), self.expect(trial, tmp_path))

    def test_skill_roots_in_the_clone_and_codex_s_own_pass(
        self, trial: Any, tmp_path: Path
    ) -> None:
        roots = (f"{tmp_path}/clone/.codex/skills", f"{tmp_path}/home/skills/.system")
        records = self.records(trial, tmp_path, skills=roots)
        assert trial.delivered(records, self.expect(trial, tmp_path))

    @pytest.mark.parametrize(
        "root", ["~/.agents/skills", "/home/someone/.codex/skills", "{tmp}/home/skills"]
    )
    def test_any_other_skill_root_fails(self, trial: Any, tmp_path: Path, root: str) -> None:
        records = self.records(trial, tmp_path, skills=(root.format(tmp=tmp_path),))
        assert not trial.delivered(records, self.expect(trial, tmp_path))

    def test_a_skill_root_that_links_out_of_the_clone_fails(
        self, trial: Any, tmp_path: Path
    ) -> None:
        (tmp_path / "clone").mkdir()
        (tmp_path / "outside").mkdir()
        (tmp_path / "clone" / "skills").symlink_to(tmp_path / "outside")
        records = self.records(trial, tmp_path, skills=(f"{tmp_path}/clone/skills",))
        assert not trial.delivered(records, self.expect(trial, tmp_path))


class TestRollout:
    def write(self, home: Path, name: str, text: str) -> None:
        day = home / "sessions" / "2026" / "10" / "25"
        day.mkdir(parents=True, exist_ok=True)
        (day / name).write_text(text, encoding="utf-8")

    def test_the_one_rollout_of_the_thread_is_read(self, trial: Any, tmp_path: Path) -> None:
        # Codex leaves raw control characters inside strings; they still parse.
        self.write(tmp_path, "rollout-2026-10-25T10-00-00-t1.jsonl", '{"type": "x", "v": "a\tb"}\n')
        assert trial.rollout(tmp_path, "t1") == [{"type": "x", "v": "a\tb"}]

    def test_none_two_or_a_broken_rollout_is_no_evidence(self, trial: Any, tmp_path: Path) -> None:
        assert trial.rollout(tmp_path, "t1") is None
        assert trial.rollout(tmp_path, "") is None
        self.write(tmp_path, "rollout-a-t1.jsonl", "{}\n")
        self.write(tmp_path, "rollout-b-t1.jsonl", "{}\n")
        assert trial.rollout(tmp_path, "t1") is None
        self.write(tmp_path, "rollout-c-t2.jsonl", "{}\n{broken\n")
        assert trial.rollout(tmp_path, "t2") is None
        self.write(tmp_path, "rollout-d-t3.jsonl", "[1]\n")
        assert trial.rollout(tmp_path, "t3") is None

    def test_a_rollout_that_is_not_text_is_no_evidence(self, trial: Any, tmp_path: Path) -> None:
        day = tmp_path / "sessions"
        day.mkdir()
        (day / "rollout-a-t1.jsonl").write_bytes(b'{"v": "\xff"}\n')
        assert trial.rollout(tmp_path, "t1") is None


class TestInventory:
    def records(
        self,
        trial: Any,
        code: str | None = None,
        status: str = "Script completed\n",
        count: int = 1,
    ) -> list[dict[str, Any]]:
        names = ["exec_command", "mcp__markdown_memory__search_docs"]
        calls = [
            item(
                {
                    "type": "custom_tool_call",
                    "name": "exec",
                    "call_id": f"c{i}",
                    "input": code or trial.INVENTORY,
                }
            )
            for i in range(count)
        ]
        outputs = [
            item(
                {
                    "type": "custom_tool_call_output",
                    "call_id": f"c{i}",
                    "output": [
                        {"type": "input_text", "text": status},
                        {"type": "input_text", "text": json.dumps(names)},
                    ],
                }
            )
            for i in range(count)
        ]
        return [*calls, *outputs]

    def test_the_exact_call_s_own_output_is_the_inventory(self, trial: Any) -> None:
        assert trial.inventory(self.records(trial)) == [
            "exec_command",
            "mcp__markdown_memory__search_docs",
        ]

    def test_another_script_is_not_evidence(self, trial: Any) -> None:
        assert (
            trial.inventory(
                self.records(
                    trial,
                    code="text(JSON.stringify(ALL_TOOLS.filter(t => t.name !== 'x')))",
                )
            )
            is None
        )

    def test_a_script_that_did_not_complete_is_not_evidence(self, trial: Any) -> None:
        assert trial.inventory(self.records(trial, status="Script failed\n")) is None

    def test_two_listings_are_not_evidence(self, trial: Any) -> None:
        assert trial.inventory(self.records(trial, count=2)) is None

    def test_a_listing_without_its_output_is_not_evidence(self, trial: Any) -> None:
        assert trial.inventory(self.records(trial)[:1]) is None


class TestInventoryVerdict:
    OURS = [
        f"mcp__markdown_memory__{tool}"
        for tool in (
            "search_docs",
            "read_section",
            "get_document_outline",
            "list_documents",
            "index_directory",
        )
    ]
    NATIVE = ["exec_command", "apply_patch", "view_image"]

    def test_the_server_s_five_tools_and_the_first_probe_s_natives_pass(self, trial: Any) -> None:
        assert trial.inventory_ok(self.NATIVE + self.OURS, None)
        assert trial.inventory_ok(self.OURS + self.NATIVE, self.NATIVE + self.OURS)

    @pytest.mark.parametrize(
        "names",
        [
            NATIVE + OURS + ["web_search"],  # a native tool the first probe did not have
            NATIVE[1:] + OURS,  # one it had, gone
            NATIVE + OURS + ["mcp__codex_apps__gmail_send"],
            NATIVE + OURS[1:],
        ],
    )
    def test_any_other_tool_set_fails(self, trial: Any, names: list[str]) -> None:
        assert not trial.inventory_ok(names, self.NATIVE + self.OURS)

    def test_no_listing_fails(self, trial: Any) -> None:
        assert not trial.inventory_ok(None, None)


def valid_rollout(trial: Any, home: Path, thread: str, cwd: Path, brief: str, ruled: bool) -> None:
    """A rollout that passes the delivery check, where Codex would write it."""
    agents = trial.RULE if ruled else "- ALWAYS run the tests"
    records = [
        {
            "type": "session_meta",
            "payload": {
                "id": thread,
                "cwd": str(cwd),
                "base_instructions": {"text": "You are Codex."},
            },
        },
        said(
            "user",
            f"# AGENTS.md instructions for {cwd}\n\n<INSTRUCTIONS>\n{agents}\n</INSTRUCTIONS>",
        ),
        {"type": "turn_context", "payload": {"model": "model-x", "effort": None}},
        said("user", brief),
    ]
    day = home / "sessions" / "2026" / "10" / "25"
    day.mkdir(parents=True, exist_ok=True)
    (day / f"rollout-2026-10-25T10-00-00-{thread}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )


class TestStepDWiring:
    """The run itself, with the agents replaced: env, delivery, retries and clones are wired."""

    TASK = {"id": "uv-01", "question": "How do I pin a package?"}

    def fake_agents(
        self, trial: Any, monkeypatch: pytest.MonkeyPatch, home: Path, delivered_on: set[int]
    ) -> list[dict[str, Any]]:
        import subprocess

        calls: list[dict[str, Any]] = []

        def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            calls.append({"argv": argv, **kwargs})
            if argv[0] == "claude":
                init = {
                    "type": "system",
                    "subtype": "init",
                    "session_id": "s",
                    "cwd": str(kwargs["cwd"]),
                    "mcp_servers": [{"name": "markdown-memory", "status": "connected"}],
                    "plugins": [],
                }
                out = lines(init, {"type": "result", "result": "answer"})
            else:
                thread = f"t{len(calls)}"
                if len(calls) in delivered_on:
                    arm_b = kwargs["cwd"].name == "clone-B"
                    valid_rollout(trial, home, thread, kwargs["cwd"], argv[-1], arm_b)
                out = lines({"type": "thread.started", "thread_id": thread}, message("answer"))
            return subprocess.CompletedProcess(argv, 0, "\n".join(out), "")

        monkeypatch.setattr(trial.subprocess, "run", run)
        return calls

    def setup(self, trial: Any, home: Path) -> Any:
        return trial.Setup(trial.STEP_D, home, "model-x", None)

    def arm(self, trial: Any, tmp_path: Path, name: str) -> Any:
        clone = tmp_path / f"clone-{name}"
        clone.mkdir(exist_ok=True)
        return trial.Arm(name, tmp_path / "wt", tmp_path / f"{name}.db", clone)

    def test_a_codex_session_runs_in_its_own_home_and_is_vouched_for_by_its_rollout(
        self, trial: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        calls = self.fake_agents(trial, monkeypatch, home, delivered_on={1})
        arm = self.arm(trial, tmp_path, "A")
        good = trial.run_session(
            "codex", arm, self.TASK, 1, arm.clone, tmp_path, setup=self.setup(trial, home)
        )
        assert good.init_ok and not good.failed
        assert (calls[0].get("env") or {}).get("CODEX_HOME") == str(home)
        assert calls[0]["cwd"] == arm.clone
        bad = trial.run_session(
            "codex", arm, self.TASK, 1, arm.clone, tmp_path, setup=self.setup(trial, home)
        )
        assert not bad.init_ok and bad.failed  # no rollout for this thread

    def test_an_undelivered_pair_runs_again_then_drops_and_each_arm_uses_its_clone(
        self, trial: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        calls = self.fake_agents(trial, monkeypatch, home, delivered_on=set())
        checked: list[Path] = []
        monkeypatch.setattr(trial, "tracked_changes", lambda clone: checked.append(clone) or "")
        tasks = tmp_path / "tasks.json"
        tasks.write_text(json.dumps({"tasks": [self.TASK]}), encoding="utf-8")
        arms = {name: self.arm(trial, tmp_path, name) for name in trial.ARMS}
        out = tmp_path / "out"
        out.mkdir()
        assert trial.run(tasks, None, arms, out, self.setup(trial, home)) == 0
        sessions = trial.load_sessions([out])
        codex = sorted((s.attempt, s.arm) for s in sessions if s.agent == "codex")
        assert codex == [(1, "A"), (1, "B"), (2, "A"), (2, "B")]  # retried once, both arms
        assert trial.kept_pairs(sessions)[1] == 1  # then dropped
        assert {c["cwd"] for c in calls} == {arms["A"].clone, arms["B"].clone}
        assert set(checked) == {arms["A"].clone, arms["B"].clone}
        assert trial.protocol_of(out) is trial.STEP_D
        claude = [c for c in calls if c["argv"][0] == "claude"]
        assert all(c["env"]["DISABLE_AUTOUPDATER"] == "1" for c in claude)


def test_the_readme_recommends_the_rule_step_d_measured(trial: Any) -> None:
    # A reworded README line is not the rule the trial measured (#134).
    readme = (Path(__file__).parent.parent / "README.md").read_text(encoding="utf-8")
    assert trial.RULE in readme
