"""Step C of the adoption trial: does the server's discovery text change adoption? (#126)

    uv run python scripts/adoption_trial.py probe --agent claude --arm A=WORKTREE:DB --repo DIR OUT
    uv run python scripts/adoption_trial.py run TASKS --clone DIR --arm A=WT:DB --arm B=WT:DB OUT
    uv run python scripts/adoption_trial.py answers GRADING OUT [OUT ...] --keys KEY [KEY ...]
    uv run python scripts/adoption_trial.py report OUT [OUT ...] --blind MAP --grades G1 G2

Two arms serve the same server build with different discovery text, and every task runs in
both arms back to back, with both agents. `run` writes each session's raw output, its
transcript in the shape `usage_from_transcripts.read_transcript` reads (Claude's stream-json
with the session and directory on every record; Codex's events converted), and
`sessions.json`. A pair whose session failed runs again once, both arms; if it fails again
it is dropped from both, so the arms always share a denominator. Any tracked change to the
clone stops the run.

`answers` writes every kept session's final answer under a shuffled id, beside the key, for
graders who see neither the arm nor the tool calls; the map back stays in its own file.
`report` merges two graders' grades, then applies the decision rules frozen in
`docs/adoption-trial/step-c.md`: an exact one-sided sign test per agent on the discordant
pairs at alpha 0.025, a harm guard on accuracy and abstention, and the ship rule. It refuses
a trial that is missing any frozen task, and stops before a verdict when more than 10% of the
pairs were dropped.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from usage_from_transcripts import RETRIEVAL_TOOLS, read_transcript, summarise

ROOT = Path(__file__).resolve().parent.parent
FROZEN_TASKS = tuple(ROOT / "docs" / "adoption-trial" / f"tasks-{r}.json" for r in ("uv", "vite"))
SERVER = "markdown-memory"
AGENTS = ("claude", "codex")
ARMS = ("A", "B")
SEED = 43  # Step B's task order
BUDGET_SECONDS = 600
CONCURRENCY = 4
# Two agents are tested, so each is held to half of 0.05 (Bonferroni): a false release
# stays at or below 5% (#126).
ALPHA = Fraction(1, 40)
HARM_POINTS = Fraction(15, 100)
HARM_ABSTENTIONS = 2
MAX_DROPPED = Fraction(1, 10)
# Worst first: a disagreement between graders takes the lower grade.
ANSWERABLE_GRADES = ("wrong", "partial", "correct")
NO_ANSWER_GRADES = ("answered", "abstained")

BRIEF = """You are the assigned worker; do not delegate this assignment and do not modify any files.

A developer working in this repository asks: "{question}"

Answer from this repository's own documentation. Cite the file path and section for each claim.
If the documentation does not cover it, say so plainly instead of guessing. Keep the answer under
150 words."""

ALLOWED_TOOLS = (
    *(f"mcp__{SERVER}__{tool}" for tool in RETRIEVAL_TOOLS),
    "Read",
    "Grep",
    "Glob",
    *(
        f"Bash({command}:*)"
        for command in ("rg", "grep", "sed -n", "head", "cat", "wc", "ls", "find")
    ),
)
DENIED_TOOLS = (f"mcp__{SERVER}__index_directory", "Edit", "Write", "NotebookEdit")


@dataclass(slots=True, frozen=True)
class Session:
    """One agent session in one arm, as `sessions.json` records it."""

    agent: str
    arm: str
    task: str
    attempt: int
    exit_code: int
    seconds: float
    timed_out: bool
    init_ok: bool
    has_answer: bool
    raw: str  # the agent's own output
    transcript: str  # the same session in the shape read_transcript reads

    @property
    def failed(self) -> bool:
        return self.exit_code != 0 or self.timed_out or not self.init_ok or not self.has_answer


# --- reading what the agents wrote ---------------------------------------------------------


def _records(lines: Iterable[str]) -> list[dict[str, Any]]:
    found = []
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            found.append(record)
    return found


def claude_init(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return next(
        (r for r in records if r.get("type") == "system" and r.get("subtype") == "init"), {}
    )


def init_ok(records: Sequence[Mapping[str, Any]]) -> bool:
    """Our server is the only MCP server, it connected, and no plugin is loaded."""
    init = claude_init(records)
    servers = [(s.get("name"), s.get("status")) for s in init.get("mcp_servers") or []]
    return servers == [(SERVER, "connected")] and not init.get("plugins")


def normalize_claude(lines: Iterable[str]) -> list[dict[str, Any]]:
    """Stream-json records, each carrying the session id and directory read_transcript reads.

    Stream-json names the session `session_id` and gives the directory only on `init`; the
    transcript parser reads `sessionId` and a `cwd` on every record.
    """
    records = _records(lines)
    init = claude_init(records)
    session, cwd = str(init.get("session_id", "")), str(init.get("cwd", ""))
    return [dict(r, sessionId=session, cwd=cwd) for r in records]


def convert_codex(lines: Iterable[str], cwd: str) -> list[dict[str, Any]]:
    """Codex `exec --json` events as Claude-shape tool_use and tool_result records."""
    converted: list[dict[str, Any]] = []
    session = ""

    def add(role: str, part: dict[str, Any]) -> None:
        message = {"role": role, "content": [part]}
        converted.append({"type": role, "sessionId": session, "cwd": cwd, "message": message})

    def pair(call: str, name: str, arguments: object, content: object, failed: bool) -> None:
        arguments = arguments if isinstance(arguments, dict) else {}
        add("assistant", {"type": "tool_use", "id": call, "name": name, "input": arguments})
        add(
            "user",
            {"type": "tool_result", "tool_use_id": call, "content": content, "is_error": failed},
        )

    for event in _records(lines):
        if event.get("type") == "thread.started":
            session = str(event.get("thread_id", ""))
        item = event.get("item")
        if event.get("type") != "item.completed" or not isinstance(item, dict):
            continue
        kind, call = item.get("type"), str(item.get("id", ""))
        if kind == "agent_message":
            add("assistant", {"type": "text", "text": str(item.get("text", ""))})
        elif kind == "command_execution":
            output = str(item.get("aggregated_output") or "")
            failed = item.get("exit_code") not in (0, None)
            pair(call, "Bash", {"command": str(item.get("command", ""))}, output, failed)
        elif kind == "mcp_tool_call":
            result, error = item.get("result"), item.get("error")
            content: object = result.get("content", []) if isinstance(result, dict) else []
            if error:
                content = str(error.get("message", error) if isinstance(error, dict) else error)
            failed = bool(error) or item.get("status") == "failed"
            name = f"mcp__{item.get('server')}__{item.get('tool')}"
            pair(call, name, item.get("arguments"), content, failed)
    return converted


def answer(agent: str, lines: Iterable[str]) -> str:
    """The session's final answer: Claude's last result, Codex's last agent message."""
    text = ""
    for record in _records(lines):
        item = record.get("item")
        if agent == "claude" and record.get("type") == "result":
            text = str(record.get("result") or "")
        elif (
            agent == "codex"
            and record.get("type") == "item.completed"
            and isinstance(item, dict)
            and item.get("type") == "agent_message"
        ):
            text = str(item.get("text") or "")
    return text.strip()


# --- the decision rules ----------------------------------------------------------------------


def kept_pairs(
    sessions: Sequence[Session],
) -> tuple[dict[tuple[str, str], dict[str, Session]], int]:
    """Each (agent, task) pair's last attempt in both arms, if neither failed; and the drops."""
    latest: dict[tuple[str, str], dict[str, Session]] = {}
    for session in sorted(sessions, key=lambda s: s.attempt):
        latest.setdefault((session.agent, session.task), {})[session.arm] = session
    kept = {
        key: arms
        for key, arms in latest.items()
        if set(arms) == set(ARMS) and not any(arms[arm].failed for arm in ARMS)
    }
    return kept, len(latest) - len(kept)


def missing_pairs(sessions: Sequence[Session], tasks: Iterable[str]) -> list[str]:
    """The (agent, task) pairs of the frozen task sets that did not run in both arms."""
    arms: dict[tuple[str, str], set[str]] = {}
    for session in sessions:
        arms.setdefault((session.agent, session.task), set()).add(session.arm)
    return [
        f"{agent} {task}"
        for task in tasks
        for agent in AGENTS
        if arms.get((agent, task), set()) != set(ARMS)
    ]


def too_many_dropped(dropped: int, total: int) -> bool:
    return Fraction(dropped, total) > MAX_DROPPED if total else True


def sign_test(b: int, c: int) -> Fraction:
    """Exact one-sided p of b or more B-only adoptions among n = b + c discordant pairs."""
    n = b + c
    return Fraction(sum(math.comb(n, k) for k in range(b, n + 1)), 2**n)


def effect(b: int, c: int) -> bool:
    return sign_test(b, c) < ALPHA


def merge_grades(
    first: Mapping[str, str], second: Mapping[str, str], resolved: Mapping[str, str]
) -> tuple[dict[str, str], list[str]]:
    """One grade per item: agreed, else the lower of the two unless resolved by the key."""
    merged, disagreements = {}, []
    for item, grade in first.items():
        other = second[item]
        if grade == other:
            merged[item] = grade
            continue
        scale = ANSWERABLE_GRADES if grade in ANSWERABLE_GRADES else NO_ANSWER_GRADES
        merged[item] = resolved.get(item, min(grade, other, key=scale.index))
        disagreements.append(f"{item}: {grade} vs {other} -> {merged[item]}")
    return merged, disagreements


def harm(grades: Mapping[str, Mapping[str, str]], answerable: Mapping[str, bool]) -> bool:
    """Per kept task, each arm's grade. Partial is not correct.

    Fires when B's share of correct answers is 15 points or more below A's, or when B
    abstains correctly on at least two fewer no-answer tasks than A.
    """
    asked = [task for task in grades if answerable[task]]
    if asked:
        drop = Fraction(
            sum(grades[t]["A"] == "correct" for t in asked)
            - sum(grades[t]["B"] == "correct" for t in asked),
            len(asked),
        )
        if drop >= HARM_POINTS:
            return True
    unanswerable = [task for task in grades if not answerable[task]]
    abstained = {arm: sum(grades[t][arm] == "abstained" for t in unanswerable) for arm in ARMS}
    return abstained["A"] - abstained["B"] >= HARM_ABSTENTIONS


def ship(effects: Mapping[str, bool], harms: Mapping[str, bool]) -> bool:
    """The text ships only on an effect for at least one agent and harm for neither."""
    return any(effects.values()) and not any(harms.values())


# --- running sessions --------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Arm:
    name: str
    worktree: Path
    db: Path

    @classmethod
    def parse(cls, text: str) -> Arm:
        name, _, rest = text.partition("=")
        worktree, _, db = rest.partition(":")
        return cls(name, Path(worktree).resolve(), Path(db).resolve())

    def env(self, docs: Path) -> dict[str, str]:
        return {
            "MARKDOWN_MEMORY_DB": str(self.db),
            "MARKDOWN_MEMORY_DOCS_DIR": str(docs),
            "MARKDOWN_MEMORY_AUTO_INDEX": "0",
        }


def command(agent: str, arm: Arm, docs: Path, config: Path) -> list[str]:
    """The session's argv; Claude's MCP config is written to ``config``, one per session."""
    args = ["run", "--directory", str(arm.worktree), SERVER]
    if agent == "claude":
        server = {"command": "uv", "args": args, "env": arm.env(docs)}
        config.write_text(json.dumps({"mcpServers": {SERVER: server}}), encoding="utf-8")
        return [
            "claude",
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--setting-sources",
            "project",
            "--strict-mcp-config",
            "--mcp-config",
            str(config),
            "--permission-mode",
            "default",
            "--allowedTools",
            ",".join(ALLOWED_TOOLS),
            "--disallowedTools",
            ",".join(DENIED_TOOLS),
        ]
    env = ", ".join(f"{key} = {json.dumps(value)}" for key, value in arm.env(docs).items())
    settings = [
        'command = "uv"',
        f"args = {json.dumps(args)}",
        f"env = {{ {env} }}",
        "required = true",  # a server that fails to start fails the session
        *(f'tools.{tool}.approval_mode = "approve"' for tool in RETRIEVAL_TOOLS),
    ]
    overrides = [part for line in settings for part in ("-c", f"mcp_servers.{SERVER}.{line}")]
    return [
        "codex",
        "exec",
        "--json",
        "--ignore-user-config",
        "-s",
        "read-only",
        "-C",
        str(docs),
        *overrides,
    ]


def run_session(
    agent: str,
    arm: Arm,
    task: Mapping[str, Any],
    attempt: int,
    docs: Path,
    out: Path,
    prompt: str | None = None,
) -> Session:
    stem = out / f"{agent}-{arm.name}-{task['id']}-{attempt}"
    text = prompt or BRIEF.format(question=task["question"])
    argv = command(agent, arm, docs, stem.with_suffix(".mcp.json"))
    if agent == "codex":
        argv.append(text)
    started = time.monotonic()
    try:
        done = subprocess.run(
            argv,
            cwd=docs,
            input=text if agent == "claude" else None,
            stdin=subprocess.DEVNULL if agent == "codex" else None,
            capture_output=True,
            text=True,
            timeout=BUDGET_SECONDS,
            check=False,
        )
        stdout, stderr, code, timed_out = done.stdout, done.stderr, done.returncode, False
    except subprocess.TimeoutExpired as expired:
        stdout = expired.stdout.decode() if isinstance(expired.stdout, bytes) else ""
        stderr, code, timed_out = "", -1, True
    seconds = time.monotonic() - started
    raw = stem.with_suffix(".raw.jsonl")
    raw.write_text(stdout, encoding="utf-8")
    stem.with_suffix(".err").write_text(stderr, encoding="utf-8")
    lines = stdout.splitlines()
    if agent == "claude":
        records, ok = normalize_claude(lines), init_ok(_records(lines))
    else:
        records, ok = convert_codex(lines, str(docs)), True  # required = true stands in
    transcript = stem.with_suffix(".transcript.jsonl")
    transcript.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return Session(
        agent,
        arm.name,
        str(task["id"]),
        attempt,
        code,
        round(seconds, 1),
        timed_out,
        ok,
        bool(answer(agent, lines)),
        str(raw),
        str(transcript),
    )


def tracked_changes(clone: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(clone), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def order(tasks: Sequence[Mapping[str, Any]]) -> list[tuple[Mapping[str, Any], tuple[str, str]]]:
    """Step B's shuffle; the arm that runs first alternates from one task to the next."""
    shuffled = list(tasks)
    random.Random(SEED).shuffle(shuffled)
    return [(task, ARMS if i % 2 == 0 else ARMS[::-1]) for i, task in enumerate(shuffled)]


def run(tasks_file: Path, clone: Path, arms: Mapping[str, Arm], out: Path) -> int:
    tasks = json.loads(tasks_file.read_text(encoding="utf-8"))["tasks"]
    sessions: list[Session] = []
    stop = threading.Event()
    lock = threading.Lock()

    def pair(task: Mapping[str, Any], arm_order: tuple[str, str], agent: str) -> None:
        for attempt in (1, 2):
            done = []
            for name in arm_order:
                if stop.is_set():
                    return
                session = run_session(agent, arms[name], task, attempt, clone, out)
                changed = tracked_changes(clone)
                with lock:
                    sessions.append(session)
                    if changed:
                        print(f"ABORT: tracked change after {session.raw}:\n{changed}")
                        stop.set()
                done.append(session)
            if not any(session.failed for session in done):
                return

    with ThreadPoolExecutor(CONCURRENCY) as pool:
        jobs = [
            pool.submit(pair, task, arm_order, agent)
            for task, arm_order in order(tasks)
            for agent in AGENTS
        ]
        for job in jobs:
            job.result()
    (out / "sessions.json").write_text(
        json.dumps([asdict(s) for s in sessions], indent=2) + "\n", encoding="utf-8"
    )
    print(f"{len(sessions)} sessions written to {out / 'sessions.json'}")
    return 1 if stop.is_set() else 0


def probe(agent: str, arm: Arm, repo: Path, out: Path) -> int:
    """A session told to search: proves the server answers and the transcript parses."""
    task = {"id": "probe", "question": ""}
    prompt = "Call the search_docs tool with the query 'install', then quote its first result."
    session = run_session(agent, arm, task, 1, repo, out, prompt=prompt)
    calls = read_transcript(Path(session.transcript)).calls
    searched = [c for c in calls if c.tool == "search_docs" and c.paired and not c.failed]
    projects = {c.project for c in searched}
    # read_transcript falls back to the file name for a session id the records lack.
    ids = {c.session for c in searched}
    good = (
        not session.failed
        and bool(searched)
        and projects == {str(repo)}
        and len(ids) == 1
        and Path(session.transcript).stem not in ids
    )
    print(
        f"{agent} arm {arm.name}: exit {session.exit_code}, init {session.init_ok}, "
        f"answer {session.has_answer}, searches {len(searched)}, cwd {sorted(projects)}, "
        f"session {sorted(ids)} -> {'OK' if good else 'FAIL'}"
    )
    return 0 if good else 1


# --- grading and the report --------------------------------------------------------------------


def load_sessions(outs: Sequence[Path]) -> list[Session]:
    return [
        Session(**record)
        for out in outs
        for record in json.loads((out / "sessions.json").read_text(encoding="utf-8"))
    ]


def tasks_in(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """The tasks of these task files, by id."""
    return {
        task["id"]: task
        for path in paths
        for task in json.loads(path.read_text(encoding="utf-8"))["tasks"]
    }


def answers(outs: Sequence[Path], keys: Sequence[Path], grading: Path) -> int:
    kept, _ = kept_pairs(load_sessions(outs))
    key = tasks_in(keys)
    questions = {task: entry["question"] for task, entry in tasks_in(FROZEN_TASKS).items()}
    items = [session for arms in kept.values() for session in arms.values()]
    random.Random(126).shuffle(items)
    blind, rows = {}, []
    for number, session in enumerate(items, 1):
        item = f"g{number:03d}"
        blind[item] = {"agent": session.agent, "arm": session.arm, "task": session.task}
        lines = Path(session.raw).read_text(encoding="utf-8").splitlines()
        entry = key[session.task]
        rows.append(
            {
                "item": item,
                "question": questions[session.task],
                "answerable": entry["answerable"],
                "answer": answer(session.agent, lines),
                "key": {k: entry.get(k) for k in ("answer", "sources", "absence", "note")},
            }
        )
    grading.mkdir(parents=True, exist_ok=True)
    (grading / "items.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (grading / "blind-map.json").write_text(json.dumps(blind, indent=2), encoding="utf-8")
    print(f"{len(rows)} answers to grade in {grading / 'items.jsonl'}")
    return 0


def _median(values: Sequence[int]) -> str:
    return str(statistics.median(values)) if values else "-"


def report(
    outs: Sequence[Path], blind: Path, graders: Sequence[Path], resolved: Path | None
) -> int:
    sessions = load_sessions(outs)
    frozen = tasks_in(FROZEN_TASKS)
    missing = missing_pairs(sessions, frozen)
    if missing:
        print(f"REFUSED: {len(missing)} pairs of the frozen tasks never ran: {missing}")
        return 2
    kept, dropped = kept_pairs(sessions)
    total = len(kept) + dropped
    print(f"pairs: {len(kept)} kept, {dropped} dropped of {total}")
    if too_many_dropped(dropped, total):
        print("STOPPED: more than 10% of pairs were dropped; the protocol says redesign")
        return 1
    answerable = {task: entry["answerable"] for task, entry in frozen.items()}
    first, second = (json.loads(path.read_text(encoding="utf-8")) for path in graders)
    fixes = json.loads(resolved.read_text(encoding="utf-8")) if resolved else {}
    merged, disagreements = merge_grades(first, second, fixes)
    where = json.loads(blind.read_text(encoding="utf-8"))
    graded: dict[tuple[str, str], dict[str, str]] = {}
    for item, grade in merged.items():
        place = where[item]
        graded.setdefault((place["agent"], place["task"]), {})[place["arm"]] = grade
    effects, harms = {}, {}
    for agent in AGENTS:
        mine = {task: arms for (who, task), arms in kept.items() if who == agent}
        adopted = {
            (task, arm): any(
                call.tool in RETRIEVAL_TOOLS
                for call in read_transcript(Path(arms[arm].transcript)).calls
            )
            for task, arms in mine.items()
            for arm in ARMS
        }
        b = sum(adopted[t, "B"] and not adopted[t, "A"] for t in mine)
        c = sum(adopted[t, "A"] and not adopted[t, "B"] for t in mine)
        p = sign_test(b, c)
        effects[agent] = effect(b, c)
        grades = {task: graded[agent, task] for task in mine}
        harms[agent] = harm(grades, answerable)
        asked = [t for t in mine if answerable[t]]
        print(f"\n{agent}: {len(mine)} kept pairs")
        for arm in ARMS:
            transcripts = [read_transcript(Path(mine[t][arm].transcript)) for t in mine]
            usage = summarise(transcripts)
            users = sum(adopted[t, arm] for t in mine)
            calls = sum(1 for tr in transcripts for c in tr.calls if c.tool in RETRIEVAL_TOOLS)
            correct = sum(grades[t][arm] == "correct" for t in asked)
            abstained = sum(grades[t][arm] == "abstained" for t in mine if not answerable[t])
            print(
                f"  arm {arm}: adoption {users}/{len(mine)}; correct {correct}/{len(asked)}; "
                f"abstained {abstained}/{len(mine) - len(asked)}; "
                f"calls per adopting session {calls / users if users else 0:.1f}; "
                f"same-file fallback {usage.searches_followed_by_same_file_read}/"
                f"{usage.searches}; chains {len(usage.reformulation_chains)}; "
                f"median documentation tokens {_median(usage.documentation_tokens_per_session)}"
                f", shell {_median(usage.shell_tokens_per_session)}"
            )
        print(
            f"  sign test: b={b} c={c} n={b + c} p={float(p):.4f} "
            f"-> effect {'yes' if effects[agent] else 'no'} (alpha {float(ALPHA)}); "
            f"harm {'yes' if harms[agent] else 'no'}"
        )
    print(f"\ngrader disagreements: {len(disagreements)}")
    for line in disagreements:
        print(f"  {line}")
    print(f"\nship: {'yes' if ship(effects, harms) else 'no'}")
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("probe")
    one.add_argument("--agent", choices=AGENTS, required=True)
    one.add_argument("--arm", type=Arm.parse, required=True)
    one.add_argument("--repo", type=Path, required=True)
    one.add_argument("out", type=Path)
    trial = commands.add_parser("run")
    trial.add_argument("tasks", type=Path)
    trial.add_argument("--clone", type=Path, required=True)
    trial.add_argument("--arm", type=Arm.parse, action="append", required=True)
    trial.add_argument("out", type=Path)
    collect = commands.add_parser("answers")
    collect.add_argument("grading", type=Path)
    collect.add_argument("outs", type=Path, nargs="+")
    collect.add_argument("--keys", type=Path, nargs="+", required=True)
    table = commands.add_parser("report")
    table.add_argument("outs", type=Path, nargs="+")
    table.add_argument("--blind", type=Path, required=True)
    table.add_argument("--grades", type=Path, nargs=2, required=True)
    table.add_argument("--resolved", type=Path)
    return parser


def main() -> int:
    parser = make_parser()
    arguments = parser.parse_args()

    if arguments.command in ("probe", "run"):
        out = arguments.out.resolve()
        out.mkdir(parents=True, exist_ok=True)
    if arguments.command == "probe":
        return probe(arguments.agent, arguments.arm, arguments.repo.resolve(), out)
    if arguments.command == "run":
        arms = {arm.name: arm for arm in arguments.arm}
        if set(arms) != set(ARMS):
            parser.error("give --arm A=... and --arm B=...")
        return run(arguments.tasks, arguments.clone.resolve(), arms, out)
    if arguments.command == "answers":
        return answers(arguments.outs, arguments.keys, arguments.grading)
    return report(arguments.outs, arguments.blind, arguments.grades, arguments.resolved)


if __name__ == "__main__":
    sys.exit(main())
