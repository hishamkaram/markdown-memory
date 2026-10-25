"""Steps C and D of the adoption trial: what makes an agent use the server? (#126, #134)

    uv run python scripts/adoption_trial.py probe --agent claude --arm A=WORKTREE:DB --repo DIR OUT
    uv run python scripts/adoption_trial.py run TASKS --clone DIR --arm A=WT:DB --arm B=WT:DB OUT
    uv run python scripts/adoption_trial.py answers GRADING OUT [OUT ...] --keys KEY [KEY ...]
    uv run python scripts/adoption_trial.py report OUT [OUT ...] --blind MAP --grades G1 G2

Step C (`docs/adoption-trial/step-c.md`, the default protocol): two arms serve the same server
build with different discovery text. Step D (`step-d.md`): the same server, and two clones of
the repository, one of them carrying a navigation rule in its `AGENTS.md`; Codex runs in a home
of its own, and every Codex session's rollout must show what reached the model. `run` and
`probe` take it as `--protocol step-d --codex-home HOME --codex-model MODEL [--codex-effort E]`,
with each arm naming its own clone (`--arm A=WORKTREE:DB:CLONE`); `probe --inventory` has Codex
list every tool it can call (`--baseline FILE`: the first probe writes it, the rest must match).

Every task runs in both arms back to back, with both agents. `run` writes each session's raw
output, its transcript in the shape `usage_from_transcripts.read_transcript` reads (Claude's
stream-json with the session and directory on every record; Codex's events converted),
`sessions.json`, and `manifest.json`, which names the protocol. A pair whose session failed runs
again once, both arms; if it fails again it is dropped from both, so the arms always share a
denominator. Any tracked change to a clone stops the run.

`answers` writes every kept session's final answer under a shuffled id, beside the key, for
graders who see neither the arm nor the tool calls; the map back stays in its own file.
`report` reads the protocol from the runs' manifests (none: Step C), merges two graders'
grades, then applies that protocol's frozen decision rules: an exact one-sided sign test on the
discordant pairs of each agent it tests, a harm guard on accuracy and abstention for both
agents, and the ship rule. It refuses a trial that is missing any frozen task or mixes
protocols, and stops before a verdict when more than 10% of the pairs were dropped.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
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

from usage_from_transcripts import (
    RETRIEVAL_TOOLS,
    Transcript,
    _reads_markdown,
    read_transcript,
    summarise,
)

ROOT = Path(__file__).resolve().parent.parent
FROZEN_TASKS = tuple(ROOT / "docs" / "adoption-trial" / f"tasks-{r}.json" for r in ("uv", "vite"))
SERVER = "markdown-memory"
AGENTS = ("claude", "codex")
ARMS = ("A", "B")
SEED = 43  # Step B's task order
BUDGET_SECONDS = 600
CONCURRENCY = 4
# A false release stays at or below 5%: each tested agent is held to its share of it
# (Bonferroni), 0.025 for Step C's two (#126), 0.05 for Step D's one (#134).
FAMILY_ALPHA = Fraction(1, 20)
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

# Step D's navigation rule, frozen in step-d.md: arm B's clone carries it as one line of its
# AGENTS.md, and arm B's rollouts must show it reached the model.
RULE = (
    "- For questions about this repository's documentation, call the `search_docs` tool "
    "(markdown-memory MCP server) before grepping or reading Markdown files; use `read_section` "
    "only when the text it returns is not enough."
)
# Codex in its own home sees no apps, plugins or hooks; it logs in to that home's own file.
ISOLATION = (
    "features.apps=false",
    "features.plugins=false",
    "features.hooks=false",
    'cli_auth_credentials_store="file"',
)
# The operator's own instructions: none may reach a Step D session (step-d.md).
FOREIGN = ("codebase-memory", "context7", "Personal rules")
AGENTS_HEADER = "# AGENTS.md instructions for "
SKILL_ROOT = re.compile(r"^- `r\d+` = `(?P<path>[^`]+)`$", re.MULTILINE)
INVENTORY = "text(JSON.stringify(ALL_TOOLS.map(t => t.name)))"


@dataclass(slots=True, frozen=True)
class Protocol:
    """What a step tests and how its sessions run; `run` records it, `report` reads it back."""

    name: str
    effect_agents: tuple[str, ...]
    successful_only: bool  # adoption needs a paired retrieval call that did not fail
    isolated: bool  # Codex runs in its own home, and its delivery is checked

    @property
    def alpha(self) -> Fraction:
        return FAMILY_ALPHA / len(self.effect_agents)


STEP_C = Protocol("step-c", AGENTS, successful_only=False, isolated=False)
STEP_D = Protocol("step-d", ("codex",), successful_only=True, isolated=True)
PROTOCOLS = {protocol.name: protocol for protocol in (STEP_C, STEP_D)}


@dataclass(slots=True, frozen=True)
class Setup:
    """How every session of a run starts; `manifest.json` records it."""

    protocol: Protocol = STEP_C
    codex_home: Path | None = None
    codex_model: str | None = None
    codex_effort: str | None = None  # None: the model's default, which turn_context reports


STEP_C_SETUP = Setup()


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
    """Our server is the only MCP server, it connected, and no plugin is loaded.

    Claude Code's own built-in plugins (path `builtin`) ship with the agent build and load in
    both arms alike, so they do not count (Amendment 1 of step-c.md).
    """
    init = claude_init(records)
    servers = [(s.get("name"), s.get("status")) for s in init.get("mcp_servers") or []]
    added = [
        p
        for p in init.get("plugins") or []
        if p.get("path") != "builtin" or not str(p.get("source", "")).endswith("@builtin")
    ]
    return servers == [(SERVER, "connected")] and not added


@dataclass(slots=True, frozen=True)
class Delivery:
    """What one Codex session's rollout must show (step-d.md, the delivery check)."""

    thread: str
    cwd: Path  # the arm's clone
    brief: str  # the text the session was given
    ruled: bool  # arm B: the rule is in the AGENTS.md message
    model: str
    effort: str | None
    home: Path  # the Codex home the session ran in


def _payload(record: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = record.get("payload")
    return payload if isinstance(payload, dict) else {}


def _text(message: Mapping[str, Any]) -> str:
    parts = message.get("content")
    if not isinstance(parts, list):
        return ""
    return "\n".join(str(p.get("text", "")) for p in parts if isinstance(p, dict))


def rollout(home: Path, thread: str) -> list[dict[str, Any]] | None:
    """The records of the one rollout Codex wrote for this thread; None if not exactly that.

    Codex names a rollout `rollout-<time>-<thread id>.jsonl` under its home's `sessions/`. A
    missing, duplicated or unparsable rollout is evidence the session cannot be vouched for.
    """
    found = sorted((home / "sessions").rglob(f"rollout-*-{thread}.jsonl")) if thread else []
    if len(found) != 1:
        return None
    try:
        text = found[0].read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):  # gone, unreadable or not text: no evidence either
        return None
    records = []
    for line in text.splitlines():
        try:
            record = json.loads(line, strict=False)  # Codex leaves raw control characters in
        except ValueError:
            return None
        if not isinstance(record, dict):
            return None
        records.append(record)
    return records


def delivered(records: Sequence[Mapping[str, Any]], expect: Delivery) -> bool:
    """Whether the rollout shows this session started as Step D says, and nothing else reached it.

    Only the startup records are read: the base instructions, and the messages before the
    brief (exactly one user message equals it). What a session read later can quote anything.
    """
    meta = [_payload(r) for r in records if r.get("type") == "session_meta"]
    if len(meta) != 1 or (meta[0].get("id"), meta[0].get("cwd")) != (
        expect.thread,
        str(expect.cwd),
    ):
        return False
    contexts = [_payload(r) for r in records if r.get("type") == "turn_context"]
    pinned = (expect.model, expect.effort)
    # The effort must be reported, not merely absent: None is a value Codex writes for "default".
    if not contexts or any(
        "effort" not in c or (c.get("model"), c.get("effort")) != pinned for c in contexts
    ):
        return False
    messages = [
        (str(_payload(r).get("role")), _text(_payload(r)))
        for r in records
        if r.get("type") == "response_item" and _payload(r).get("type") == "message"
    ]
    briefs = [
        i for i, (role, text) in enumerate(messages) if role == "user" and text == expect.brief
    ]
    if len(briefs) != 1:
        return False
    base = meta[0].get("base_instructions")
    base_text = base.get("text") if isinstance(base, dict) else None
    if not isinstance(base_text, str) or not base_text:
        return False
    startup = [text for _, text in messages[: briefs[0]]]
    startup.append(base_text)
    if any(word in text for text in startup for word in FOREIGN):
        return False
    instructions = [text for text in startup if text.startswith(AGENTS_HEADER)]
    if expect.ruled != any(RULE in text for text in instructions):
        return False
    system = Path(os.path.realpath(expect.home / "skills" / ".system"))
    clone = Path(os.path.realpath(expect.cwd))
    for text in startup:
        for match in SKILL_ROOT.finditer(text):
            root = Path(os.path.realpath(os.path.expanduser(match["path"])))
            if root != system and not root.is_relative_to(clone):
                return False
    return True


def inventory(records: Sequence[Mapping[str, Any]]) -> list[str] | None:
    """The tool names one exact `INVENTORY` exec call printed; None if the evidence is not clean.

    The names come from the call's own paired output in the rollout, never from the answer.
    """
    items = [_payload(r) for r in records if r.get("type") == "response_item"]
    calls = [
        item
        for item in items
        if item.get("type") == "custom_tool_call"
        and item.get("name") == "exec"
        and str(item.get("input", "")).strip() == INVENTORY
    ]
    if len(calls) != 1:
        return None
    outputs = [
        item.get("output")
        for item in items
        if item.get("type") == "custom_tool_call_output"
        and item.get("call_id") == calls[0].get("call_id")
    ]
    if len(outputs) != 1 or not isinstance(outputs[0], list) or len(outputs[0]) != 2:
        return None
    status, listing = (part.get("text") if isinstance(part, dict) else None for part in outputs[0])
    if not (isinstance(status, str) and status.startswith("Script completed")):
        return None
    try:
        names = json.loads(str(listing))
    except ValueError:
        return None
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        return None
    return names


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


def effect(b: int, c: int, alpha: Fraction = STEP_C.alpha) -> bool:
    return sign_test(b, c) < alpha


def adopted(transcript: Transcript, successful_only: bool) -> bool:
    """At least one retrieval call; under Step D, one that was answered and did not fail.

    An empty search still counts: the agent used the server, and the server answered.
    """
    return any(
        call.tool in RETRIEVAL_TOOLS and (not successful_only or (call.paired and not call.failed))
        for call in transcript.calls
    )


def searched_first(transcript: Transcript) -> bool:
    """Whether `search_docs` came before the session's first hand read of Markdown."""
    for call in transcript.calls:
        if call.tool == "search_docs":
            return True
        if call.tool is None and _reads_markdown(call):
            return False
    return False


def wilson(k: int, n: int) -> tuple[float, float]:
    """The 95% Wilson interval of k in n; the whole range when there is nothing to estimate."""
    if not n:
        return 0.0, 1.0
    z = 1.959963984540054
    share, spread = k / n, z * z / n
    centre = (share + spread / 2) / (1 + spread)
    half = z * math.sqrt(share * (1 - share) / n + spread / (4 * n)) / (1 + spread)
    return max(0.0, centre - half), min(1.0, centre + half)


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
    clone: Path | None = None  # Step D: the arm's own clone; Step C shares one

    @classmethod
    def parse(cls, text: str) -> Arm:
        name, _, rest = text.partition("=")
        worktree, db, *clone = rest.split(":")
        if len(clone) > 1:
            raise argparse.ArgumentTypeError(f"not NAME=WORKTREE:DB[:CLONE]: {text}")
        own = Path(clone[0]).resolve() if clone else None
        return cls(name, Path(worktree).resolve(), Path(db).resolve(), own)

    def env(self, docs: Path) -> dict[str, str]:
        return {
            "MARKDOWN_MEMORY_DB": str(self.db),
            "MARKDOWN_MEMORY_DOCS_DIR": str(docs),
            "MARKDOWN_MEMORY_AUTO_INDEX": "0",
        }


def child_env(agent: str, setup: Setup) -> dict[str, str] | None:
    """The session's environment, built for it alone; None inherits ours, as Step C did.

    Step D gives Codex a home of its own, so neither the operator's `~/.codex` (instructions,
    skills, hooks, login) nor `$HOME/.agents/skills` reaches it. `os.environ` is never changed:
    sessions run four at a time.
    """
    if not setup.protocol.isolated:
        return None
    env = dict(os.environ)
    if agent == "codex":
        env |= {"CODEX_HOME": str(setup.codex_home), "HOME": str(setup.codex_home)}
    else:
        env["DISABLE_AUTOUPDATER"] = "1"  # the build recorded is the build that runs
    return env


def command(
    agent: str, arm: Arm, docs: Path, config: Path, setup: Setup = STEP_C_SETUP
) -> list[str]:
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
    server_env = arm.env(docs)
    if setup.protocol.isolated:
        server_env["HOME"] = str(Path.home())  # uv and the model cache live in the real home
    env = ", ".join(f"{key} = {json.dumps(value)}" for key, value in server_env.items())
    settings = [
        'command = "uv"',
        f"args = {json.dumps(args)}",
        f"env = {{ {env} }}",
        "required = true",  # a server that fails to start fails the session
        *(f'tools.{tool}.approval_mode = "approve"' for tool in RETRIEVAL_TOOLS),
    ]
    overrides = [part for line in settings for part in ("-c", f"mcp_servers.{SERVER}.{line}")]
    if setup.protocol.isolated:
        pinned = [f"model={json.dumps(setup.codex_model)}"]
        if setup.codex_effort is not None:
            pinned.append(f"model_reasoning_effort={json.dumps(setup.codex_effort)}")
        overrides += [part for line in (*ISOLATION, *pinned) for part in ("-c", line)]
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
    setup: Setup = STEP_C_SETUP,
) -> Session:
    stem = out / f"{agent}-{arm.name}-{task['id']}-{attempt}"
    text = prompt or BRIEF.format(question=task["question"])
    argv = command(agent, arm, docs, stem.with_suffix(".mcp.json"), setup)
    if agent == "codex":
        argv.append(text)
    started = time.monotonic()
    try:
        done = subprocess.run(
            argv,
            cwd=docs,
            env=child_env(agent, setup),
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
    elif setup.protocol.isolated:
        records, ok = convert_codex(lines, str(docs)), codex_ok(lines, arm, docs, text, setup)
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


def thread_of(lines: Iterable[str]) -> str:
    started = (r for r in _records(lines) if r.get("type") == "thread.started")
    return str(next(started, {}).get("thread_id", ""))


def codex_ok(lines: Sequence[str], arm: Arm, docs: Path, text: str, setup: Setup) -> bool:
    """Step D's availability check for Codex: its rollout shows what reached the model."""
    if setup.codex_home is None or setup.codex_model is None:
        raise ValueError("step-d runs Codex only with --codex-home and --codex-model")
    thread = thread_of(lines)
    records = rollout(setup.codex_home, thread)
    expect = Delivery(
        thread,
        docs,
        text,
        arm.name == "B",
        setup.codex_model,
        setup.codex_effort,
        setup.codex_home,
    )
    return records is not None and delivered(records, expect)


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


MANIFEST = "manifest.json"


def write_manifest(out: Path, setup: Setup) -> None:
    """What the run was, so `report` reads its rules from the run and never from a flag."""
    fields = {
        "protocol": setup.protocol.name,
        "codex_home": str(setup.codex_home) if setup.codex_home else None,
        "codex_model": setup.codex_model,
        "codex_effort": setup.codex_effort,
        "rule": RULE if setup.protocol.isolated else None,
    }
    (out / MANIFEST).write_text(json.dumps(fields, indent=2) + "\n", encoding="utf-8")


def protocol_of(out: Path) -> Protocol | None:
    """The run's protocol: Step C without a manifest (Step C wrote none); None if unreadable."""
    path = out / MANIFEST
    if not path.exists():
        return STEP_C
    try:
        fields = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    name = fields.get("protocol") if isinstance(fields, dict) else None
    return PROTOCOLS.get(name) if isinstance(name, str) else None


def run(
    tasks_file: Path,
    clone: Path | None,
    arms: Mapping[str, Arm],
    out: Path,
    setup: Setup = STEP_C_SETUP,
) -> int:
    tasks = json.loads(tasks_file.read_text(encoding="utf-8"))["tasks"]
    sessions: list[Session] = []
    stop = threading.Event()
    lock = threading.Lock()
    write_manifest(out, setup)

    def pair(task: Mapping[str, Any], arm_order: tuple[str, str], agent: str) -> None:
        for attempt in (1, 2):
            done = []
            for name in arm_order:
                if stop.is_set():
                    return
                docs = arms[name].clone or clone
                if docs is None:
                    raise ValueError(f"arm {name} has no clone, and no --clone was given")
                session = run_session(agent, arms[name], task, attempt, docs, out, setup=setup)
                changed = tracked_changes(docs)
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


OUR_TOOLS = frozenset(
    f"mcp__{SERVER.replace('-', '_')}__{tool}" for tool in (*RETRIEVAL_TOOLS, "index_directory")
)


def inventory_ok(names: Sequence[str] | None, baseline: Sequence[str] | None) -> bool:
    """The MCP tools are exactly the server's five, and every name equals the first probe's.

    A native tool added, removed or renamed between probes fails as surely as another server.
    """
    if names is None or {name for name in names if name.startswith("mcp__")} != OUR_TOOLS:
        return False
    return baseline is None or sorted(names) == sorted(baseline)


def probe_inventory(arm: Arm, repo: Path, out: Path, setup: Setup, baseline: Path) -> int:
    """A Codex session told to print `ALL_TOOLS`: every tool it can call, read from its rollout.

    Step D's isolation is checked here rather than in the trial's transcripts, which cannot show
    a tool that was loaded and never called. The first probe that passes writes ``baseline``;
    every later one must list exactly the same tools.
    """
    if setup.codex_home is None:
        raise ValueError("--inventory reads the rollout in --codex-home")
    task = {"id": "inventory", "question": ""}
    prompt = f'Run exactly this JavaScript in one exec call, then reply "done":\n{INVENTORY}'
    session = run_session("codex", arm, task, 1, repo, out, prompt=prompt, setup=setup)
    lines = Path(session.raw).read_text(encoding="utf-8").splitlines()
    records = rollout(setup.codex_home, thread_of(lines))
    names = inventory(records) if records is not None else None
    first = None if not baseline.exists() else json.loads(baseline.read_text(encoding="utf-8"))
    good = not session.failed and inventory_ok(names, first)
    if good and first is None:
        baseline.write_text(json.dumps(sorted(names or []), indent=2) + "\n", encoding="utf-8")
    print(f"codex arm {arm.name} inventory: {json.dumps(sorted(names or []))}")
    print(f"  init {session.init_ok}, baseline {baseline} -> {'OK' if good else 'FAIL'}")
    return 0 if good else 1


def probe(agent: str, arm: Arm, repo: Path, out: Path, setup: Setup = STEP_C_SETUP) -> int:
    """A session told to search: proves the server answers and the transcript parses."""
    task = {"id": "probe", "question": ""}
    prompt = "Call the search_docs tool with the query 'install', then quote its first result."
    session = run_session(agent, arm, task, 1, repo, out, prompt=prompt, setup=setup)
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


def _count(
    used: Mapping[tuple[str, str], bool], tasks: Iterable[str], arm: str, repo: str
) -> tuple[int, int]:
    """Adopting sessions of one arm in one repository (task ids start with it), and the total."""
    mine = [task for task in tasks if task.split("-")[0] == repo]
    return sum(used[task, arm] for task in mine), len(mine)


def _difference(used: Mapping[tuple[str, str], bool], tasks: Iterable[str], repo: str) -> str:
    """B-only minus A-only pairs of one repository, out of its pairs."""
    mine = [task for task in tasks if task.split("-")[0] == repo]
    b = sum(used[task, "B"] and not used[task, "A"] for task in mine)
    c = sum(used[task, "A"] and not used[task, "B"] for task in mine)
    return f"{b - c:+d} of {len(mine)}"


def _interval(k: int, n: int) -> str:
    low, high = wilson(k, n)
    return f"{k}/{n} [{low:.0%}, {high:.0%}]"


def report(
    outs: Sequence[Path], blind: Path, graders: Sequence[Path], resolved: Path | None
) -> int:
    found = [protocol_of(out) for out in outs]
    protocol = found[0] if found else None
    if protocol is None or any(other != protocol for other in found):
        print(f"REFUSED: the runs do not share one readable protocol: {sorted(map(str, outs))}")
        return 2
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
        read = {
            (task, arm): read_transcript(Path(arms[arm].transcript))
            for task, arms in mine.items()
            for arm in ARMS
        }
        used = {
            key: adopted(transcript, protocol.successful_only) for key, transcript in read.items()
        }
        b = sum(used[t, "B"] and not used[t, "A"] for t in mine)
        c = sum(used[t, "A"] and not used[t, "B"] for t in mine)
        p = sign_test(b, c)
        if agent in protocol.effect_agents:
            effects[agent] = effect(b, c, protocol.alpha)
        grades = {task: graded[agent, task] for task in mine}
        harms[agent] = harm(grades, answerable)  # every agent, whatever the protocol tests
        asked = [t for t in mine if answerable[t]]
        print(f"\n{agent}: {len(mine)} kept pairs")
        for arm in ARMS:
            transcripts = [read[t, arm] for t in mine]
            usage = summarise(transcripts)
            users = sum(used[t, arm] for t in mine)
            # Step C counted every session's calls; Step D counts only the adopters' own.
            counted = [read[t, arm] for t in mine if used[t, arm] or not protocol.successful_only]
            calls = sum(1 for tr in counted for c in tr.calls if c.tool in RETRIEVAL_TOOLS)
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
            if protocol.successful_only:
                attempted = sum(adopted(read[t, arm], successful_only=False) for t in mine)
                first = sum(searched_first(read[t, arm]) for t in mine)
                repos = sorted({t.split("-")[0] for t in mine})
                per_repo = ", ".join(
                    f"{repo} {_interval(*_count(used, mine, arm, repo))}" for repo in repos
                )
                print(
                    f"    adoption {_interval(users, len(mine))} (95% Wilson); with failed calls "
                    f"{attempted}/{len(mine)}; search_docs before any Markdown read "
                    f"{first}/{len(mine)}; by repository: {per_repo}"
                )
        if agent in effects:
            verdict = f"effect {'yes' if effects[agent] else 'no'} (alpha {float(protocol.alpha)})"
        else:
            verdict = (
                f"effect not tested ({protocol.name} tests {', '.join(protocol.effect_agents)})"
            )
        print(
            f"  sign test: b={b} c={c} n={b + c} p={float(p):.4f} "
            f"-> {verdict}; "
            f"harm {'yes' if harms[agent] else 'no'}"
        )
        if protocol.successful_only:
            repos = sorted({t.split("-")[0] for t in mine})
            by_repo = ", ".join(f"{repo} {_difference(used, mine, repo)}" for repo in repos)
            print(f"  paired difference B - A: {b - c:+d} of {len(mine)} pairs; {by_repo}")
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
    one.add_argument("--inventory", action="store_true", help="Codex lists every tool it has")
    one.add_argument("--baseline", type=Path, help="--inventory: the first probe's list")
    one.add_argument("out", type=Path)
    trial = commands.add_parser("run")
    trial.add_argument("tasks", type=Path)
    trial.add_argument("--clone", type=Path, help="the clone every arm shares (Step C)")
    trial.add_argument("--arm", type=Arm.parse, action="append", required=True)
    trial.add_argument("out", type=Path)
    for sub in (one, trial):
        sub.add_argument("--protocol", choices=sorted(PROTOCOLS), default=STEP_C.name)
        sub.add_argument("--codex-home", type=Path, help="Step D: Codex's own home")
        sub.add_argument("--codex-model", help="Step D: the model every Codex session runs")
        sub.add_argument("--codex-effort", help="Step D: its effort, when not the default")
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


def setup_of(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> Setup:
    """The run's setup from its flags; Step D refuses to start without what it pins."""
    protocol = PROTOCOLS[arguments.protocol]
    pinned = (arguments.codex_home, arguments.codex_model, arguments.codex_effort)
    if not protocol.isolated:
        if (
            any(value is not None for value in pinned)
            or arguments.command == "probe"
            and arguments.inventory
        ):
            parser.error(f"--codex-* and --inventory belong to step-d, not {protocol.name}")
        return Setup(protocol)
    if arguments.codex_home is None or arguments.codex_model is None:
        parser.error("step-d needs --codex-home and --codex-model")
    arms = arguments.arm if arguments.command == "run" else [arguments.arm]
    if any(arm.clone is None for arm in arms):
        parser.error("step-d arms name their own clone: --arm A=WORKTREE:DB:CLONE")
    if arguments.command == "probe" and arguments.repo.resolve() != arguments.arm.clone:
        parser.error("a step-d probe runs in its arm's clone: --repo is that clone")
    return Setup(protocol, arguments.codex_home.resolve(), *pinned[1:])


def main() -> int:
    parser = make_parser()
    arguments = parser.parse_args()

    if arguments.command in ("probe", "run"):
        setup = setup_of(arguments, parser)
        out = arguments.out.resolve()
        out.mkdir(parents=True, exist_ok=True)
    if arguments.command == "probe":
        repo = arguments.repo.resolve()
        if arguments.inventory:
            if arguments.agent != "codex" or arguments.baseline is None:
                parser.error("--inventory lists Codex's tools: --agent codex --baseline FILE")
            return probe_inventory(arguments.arm, repo, out, setup, arguments.baseline.resolve())
        return probe(arguments.agent, arguments.arm, repo, out, setup)
    if arguments.command == "run":
        arms = {arm.name: arm for arm in arguments.arm}
        if set(arms) != set(ARMS):
            parser.error("give --arm A=... and --arm B=...")
        clone = arguments.clone.resolve() if arguments.clone else None
        if clone is None and any(arm.clone is None for arm in arms.values()):
            parser.error("give --clone, or a clone in every arm")
        return run(arguments.tasks, clone, arms, out, setup)
    if arguments.command == "answers":
        return answers(arguments.outs, arguments.keys, arguments.grading)
    return report(arguments.outs, arguments.blind, arguments.grades, arguments.resolved)


if __name__ == "__main__":
    sys.exit(main())
