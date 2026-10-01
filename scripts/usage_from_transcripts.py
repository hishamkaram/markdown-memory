"""What actually happened when an agent needed documentation.

    uv run python scripts/usage_from_transcripts.py                # every project
    uv run python scripts/usage_from_transcripts.py --project markdown-memory
    uv run python scripts/usage_from_transcripts.py --server mm    # registered as "mm"
    uv run python scripts/usage_from_transcripts.py --queries      # one query per line
    uv run python scripts/usage_from_transcripts.py --json report.json

Claude Code already records every tool call it makes, with arguments, results and order,
in ``~/.claude/projects/**/*.jsonl``. That is a better source than anything this server
could log about itself, for one reason: it also records the calls that went somewhere
else. A server-side log can say how often ``search_docs`` was answered; only the
transcript can say how often the agent asked it, disliked the answer, and read the file
with ``Read`` instead - which is the number that decides whether this tool is worth
having.

Reported, per Claude session (a subagent's transcript belongs to its parent's session):

* the *documentation-seeking* sessions - those that made a retrieval call or read Markdown
  they did not also edit - how many of them used the server, and how many read the files
  by hand and never called it (the *silence*);
* every ``mcp__*markdown*`` call, and the retrieval calls that failed;
* after each search: a ``Read`` of a file the search itself returned (the *same-file
  fallback*), any other Markdown access, another search, and reformulation *chains*;
* the estimated size of every result that entered the agent's context, per tool.

Nothing here is a relevance judgement. A read after a search means the agent opened
something, not that the answer was right; these are behaviours, and the labels come from
reading them. Every size is an estimate of context exposure, not of billed tokens.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import tempfile
from collections import Counter, deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from markdown_memory.models import estimate_tokens

TRANSCRIPTS = Path.home() / ".claude" / "projects"
SERVER_TOOL = re.compile(r"^mcp__[^_]*markdown[^_]*__(?P<tool>\w+)$", re.IGNORECASE)
# The tools that fetch documentation. `index_directory` maintains the index and says
# nothing about whether the agent wanted to read anything.
RETRIEVAL_TOOLS = ("search_docs", "read_section", "get_document_outline", "list_documents")
# A search is "abandoned" when the agent reaches for the file system instead, within this
# many tool calls. Beyond that the connection to the search is guesswork.
FALLBACK_WINDOW = 4
FILE_TOOLS = ("Read", "Grep", "Glob", "NotebookRead")
READ_TOOLS = ("Read", "NotebookRead")
EDIT_TOOLS = ("Edit", "MultiEdit", "Write", "NotebookEdit")
GREP_LIKE = re.compile(r"\b(rg|grep|ag|sed -n|head|cat|awk)\b")
MARKDOWN = re.compile(r"\.mdx?\b", re.IGNORECASE)
MARKDOWN_TYPES = ("md", "markdown")
MARKDOWN_TOKEN = re.compile(r"[^\s'\"`;|&<>()]+\.mdx?\b", re.IGNORECASE)
# Where an agent keeps its own notes - plans, memory, scratchpads. Markdown there is the
# agent's working state, and the most-read Markdown in real transcripts; it is not the
# documentation anyone wanted to look up.
SCRATCH = tuple(
    sorted({"/tmp", tempfile.gettempdir(), os.path.join(os.path.expanduser("~"), ".claude")})
)
ROWS = (*RETRIEVAL_TOOLS, "Read (Markdown)", "Grep (Markdown)")
ESTIMATOR = (
    "Sizes are estimates of what entered the agent's context: characters / 4, the "
    "estimate_tokens rule the server and the eval use - not billed tokens, and not a "
    "saving. It under-counts JSON, paths and code; a Read result includes its line "
    "numbers; a client-capped listing measures the cap; non-text and unpaired results are "
    "left out, not counted as zero; Markdown access through Bash is matched approximately."
)


@dataclass(slots=True, frozen=True)
class Call:
    """One tool call, in the order the session made it, with its result if one arrived."""

    index: int
    name: str
    tool: str | None  # the server's tool name, when this is one of ours
    arguments: dict[str, object]
    session: str
    project: str  # the working directory the call ran in
    timestamp: str
    paired: bool = False
    result: str = ""  # the result's text
    non_text: bool = False  # a result with no text at all: an image, a tool reference
    failed: bool = False

    @property
    def query(self) -> str:
        value = self.arguments.get("query")
        return value if isinstance(value, str) else ""


@dataclass(slots=True, frozen=True)
class Transcript:
    """One JSONL file: a session's main thread, or one of its subagents."""

    session: str
    calls: list[Call]


@dataclass
class Row:
    """The results one tool returned."""

    calls: int = 0
    paired: int = 0
    non_text: int = 0
    tokens: list[int] = field(default_factory=list)


@dataclass
class Report:
    """Counts over every transcript that was read."""

    sessions_scanned: int = 0
    sessions_using_server: int = 0
    documentation_sessions: int = 0
    documentation_sessions_with_a_call: int = 0
    silent_sessions: int = 0
    retrieval_calls_in_documentation_sessions: int = 0
    documentation_tokens_per_session: list[int] = field(default_factory=list)
    server_calls: Counter[str] = field(default_factory=Counter)
    failures: int = 0
    searches: int = 0
    search_evidence: Counter[str] = field(default_factory=Counter)
    searches_followed_by_read_section: int = 0
    searches_followed_by_another_search: int = 0
    searches_followed_by_same_file_read: int = 0
    searches_followed_by_other_markdown: int = 0
    reformulation_chains: list[int] = field(default_factory=list)
    keyword_match: Counter[str] = field(default_factory=Counter)
    empty_results: int = 0
    rows: dict[str, Row] = field(default_factory=lambda: {name: Row() for name in ROWS})
    projects: Counter[str] = field(default_factory=Counter)
    queries: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "estimator": ESTIMATOR,
            "sessions_scanned": self.sessions_scanned,
            "sessions_using_server": self.sessions_using_server,
            "documentation_sessions": self.documentation_sessions,
            "documentation_sessions_with_a_call": self.documentation_sessions_with_a_call,
            "silent_sessions": self.silent_sessions,
            "retrieval_calls_in_documentation_sessions": (
                self.retrieval_calls_in_documentation_sessions
            ),
            "documentation_tokens_per_session": _stats(self.documentation_tokens_per_session),
            "server_calls": dict(self.server_calls),
            "failures": self.failures,
            "searches": self.searches,
            "search_evidence": dict(self.search_evidence),
            "searches_followed_by_read_section": self.searches_followed_by_read_section,
            "searches_followed_by_another_search": self.searches_followed_by_another_search,
            "searches_followed_by_same_file_read": self.searches_followed_by_same_file_read,
            "searches_followed_by_other_markdown": self.searches_followed_by_other_markdown,
            "reformulation_chains": len(self.reformulation_chains),
            "longest_reformulation_chain": max(self.reformulation_chains, default=0),
            "keyword_match": dict(self.keyword_match),
            "empty_results": self.empty_results,
            "result_tokens": {
                name: {
                    "calls": row.calls,
                    "paired": row.paired,
                    "non_text": row.non_text,
                    **_stats(row.tokens),
                }
                for name, row in self.rows.items()
            },
            "projects": dict(self.projects),
            "queries": self.queries,
        }


def _stats(values: Sequence[int]) -> dict[str, int]:
    """Median and p90 rather than a mean: result sizes are heavy-tailed."""
    if not values:
        return {"n": 0, "total": 0, "median": 0, "p90": 0, "max": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "total": sum(ordered),
        "median": round(statistics.median(ordered)),
        "p90": ordered[math.ceil(0.9 * len(ordered)) - 1],
        "max": ordered[-1],
    }


def transcripts(root: Path, project: str | None) -> Iterator[Path]:
    if not root.is_dir():
        return
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        if project and project.lower() not in directory.name.lower():
            continue
        # Recursive: a subagent's transcript lives in <session>/subagents/ and its tool
        # calls are as real as the main thread's.
        yield from sorted(directory.rglob("*.jsonl"))


def _payload(content: object) -> tuple[str, bool]:
    """A result's text, and whether it had content but no text at all."""
    if isinstance(content, str):
        return content, False
    if not isinstance(content, list):
        return "", False
    # Joined with nothing between: a JSON payload split across parts stays parseable.
    texts = [
        str(part.get("text", ""))
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    ]
    return "".join(texts), bool(content) and not texts


def read_transcript(path: Path, server: re.Pattern[str] = SERVER_TOOL) -> Transcript:
    """Every tool call in one transcript, in order, each paired with its own result.

    Unreadable lines are skipped. A result is matched to the oldest call still waiting
    under its id, inside this file only: ids are not unique - one real transcript reuses
    one 981 times, and the same id turns up in several files.
    """
    found: list[Call] = []
    session = ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return Transcript(path.stem, found)
    waiting: dict[str, deque[int]] = {}
    for line in text.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue  # a transcript being written to can end mid-line
        if not isinstance(record, dict):
            continue
        session = session or str(record.get("sessionId") or "")
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "tool_result":
                queue = waiting.get(str(part.get("tool_use_id", "")))
                if queue:
                    position = queue.popleft()
                    text_of, non_text = _payload(part.get("content"))
                    found[position] = replace(
                        found[position],
                        paired=True,
                        result=text_of,
                        non_text=non_text,
                        failed=part.get("is_error") is True,
                    )
                continue
            if part.get("type") != "tool_use":
                continue
            name = str(part.get("name", ""))
            arguments = part.get("input")
            match = server.match(name)
            waiting.setdefault(str(part.get("id", "")), deque()).append(len(found))
            found.append(
                Call(
                    index=len(found),
                    name=name,
                    tool=match.group("tool") if match else None,
                    arguments=arguments if isinstance(arguments, dict) else {},
                    session=str(record.get("sessionId") or path.stem),
                    project=str(record.get("cwd", path.parent.name)),
                    timestamp=str(record.get("timestamp", "")),
                )
            )
    return Transcript(session or path.stem, found)


def calls_in(path: Path) -> list[Call]:
    """Every tool call in one transcript, in order. Unreadable lines are skipped."""
    return read_transcript(path).calls


def _norm(path: str, cwd: str) -> str:
    """One spelling per path, by text alone.

    The files a transcript names may be gone, so the file system is never asked: two
    spellings of one file through a symlink stay two paths.
    """
    path = os.path.expanduser(path)
    if not os.path.isabs(path) and cwd:
        path = os.path.join(cwd, path)
    return os.path.normpath(path)


def _scratch(path: str) -> bool:
    return any(path == root or path.startswith(root + os.sep) for root in SCRATCH)


def _markdown_path(value: object, cwd: str) -> bool:
    return (
        isinstance(value, str) and bool(MARKDOWN.search(value)) and not _scratch(_norm(value, cwd))
    )


def _read_path(call: Call) -> str | None:
    """The file a ``Read`` opened, or an edit changed, spelled as ``_norm`` spells it."""
    value = call.arguments.get("file_path") or call.arguments.get("notebook_path")
    return _norm(value, call.project) if isinstance(value, str) else None


def _searched_scratch(call: Call) -> bool:
    """A ``Grep`` or ``Glob`` rooted in the agent's own notes, whatever it looked for."""
    path = call.arguments.get("path")
    return isinstance(path, str) and _scratch(_norm(path, call.project))


def _reads_markdown(call: Call) -> bool:
    """Whether a non-server call read Markdown by hand: its content, not just its name.

    Only the fields that name files count. A ``Grep`` *pattern* that mentions ``.md``
    searches for that text wherever it is, and a ``Grep`` with no path, glob or type says
    nothing about what it searched.
    """
    arguments = call.arguments
    if call.name in READ_TOOLS:
        return _markdown_path(
            arguments.get("file_path") or arguments.get("notebook_path"), call.project
        )
    if call.name == "Grep":
        return not _searched_scratch(call) and (
            arguments.get("type") in MARKDOWN_TYPES
            or any(_markdown_path(arguments.get(key), call.project) for key in ("path", "glob"))
        )
    if call.name == "Bash":
        command = arguments.get("command")
        return (
            isinstance(command, str)
            and bool(GREP_LIKE.search(command))
            and any(
                not _scratch(_norm(token, call.project))
                for token in MARKDOWN_TOKEN.findall(command)
            )
        )
    return False


def _touches_markdown(call: Call) -> bool:
    """Whether a non-server call went looking in Markdown by hand - by name or content."""
    if call.name == "Glob":
        return not _searched_scratch(call) and any(
            _markdown_path(call.arguments.get(key), call.project) for key in ("pattern", "path")
        )
    return _reads_markdown(call)


def _page(call: Call) -> dict[str, object] | None:
    """A ``search_docs`` result, when one arrived and it parses."""
    if not call.paired or call.failed:
        return None
    try:
        page = json.loads(call.result)
    except ValueError:
        return None
    if not isinstance(page, dict) or not isinstance(page.get("results"), list):
        return None
    return page


def _hits(page: dict[str, object], cwd: str) -> set[str]:
    results = page["results"]
    assert isinstance(results, list)
    return {
        _norm(hit["file_path"], cwd)
        for hit in results
        if isinstance(hit, dict) and isinstance(hit.get("file_path"), str)
    }


def _row(call: Call) -> str | None:
    if call.tool in RETRIEVAL_TOOLS:
        return call.tool
    if call.name in READ_TOOLS and _reads_markdown(call):
        return "Read (Markdown)"
    if call.name == "Grep" and _reads_markdown(call):
        return "Grep (Markdown)"
    return None


def _tokens(calls: Sequence[Call]) -> int:
    return sum(estimate_tokens(call.result) for call in calls if call.paired and call.result)


def _searches(report: Report, calls: Sequence[Call]) -> None:
    """What each search in one file was followed by. Windows never cross files."""
    chain = 0
    for position, call in enumerate(calls):
        if call.tool != "search_docs":
            continue
        report.searches += 1
        if call.query:
            report.queries.append(call.query)
        window = calls[position + 1 : position + 1 + FALLBACK_WINDOW]
        page = _page(call)
        hits: set[str] = set()
        if page is not None:
            report.search_evidence["parsed"] += 1
            report.keyword_match[str(page.get("keyword_match", "not reported"))] += 1
            if not page["results"]:
                report.empty_results += 1
            hits = _hits(page, call.project)
        elif not call.paired:
            report.search_evidence["unpaired"] += 1
        elif call.failed:
            report.search_evidence["failed"] += 1
        else:
            report.search_evidence["unparseable"] += 1
        if any(later.tool == "read_section" for later in window):
            report.searches_followed_by_read_section += 1
        chain += 1
        if any(later.tool == "search_docs" for later in window):
            report.searches_followed_by_another_search += 1
        else:
            if chain >= 2:
                report.reformulation_chains.append(chain)
            chain = 0
        # A manual read after a failed or unreadable search is still a manual read: only
        # telling which file the search had offered needs the result.
        reads = [later for later in window if later.name in READ_TOOLS and _reads_markdown(later)]
        if any(_read_path(later) in hits for later in reads):
            report.searches_followed_by_same_file_read += 1
        elif any(_touches_markdown(later) for later in window):
            report.searches_followed_by_other_markdown += 1


def summarise(transcripts: Sequence[Transcript]) -> Report:
    report = Report()
    sessions: dict[str, list[Transcript]] = {}
    for transcript in transcripts:
        sessions.setdefault(transcript.session, []).append(transcript)
    for files in sessions.values():
        report.sessions_scanned += 1
        calls = [call for transcript in files for call in transcript.calls]
        ours = [call for call in calls if call.tool]
        if ours:
            report.sessions_using_server += 1
            report.projects[Path(ours[0].project).name] += 1
        for call in ours:
            report.server_calls[str(call.tool)] += 1
        retrieval = [call for call in calls if call.tool in RETRIEVAL_TOOLS]
        report.failures += sum(call.failed for call in retrieval)
        # Reading the README you are about to edit is maintenance, not looking something up.
        edited = {_read_path(call) for call in calls if call.name in EDIT_TOOLS} - {None}
        by_hand = [
            call for call in calls if _reads_markdown(call) and _read_path(call) not in edited
        ]
        if retrieval or by_hand:
            report.documentation_sessions += 1
            report.retrieval_calls_in_documentation_sessions += len(retrieval)
            # Only the rows' results: a shell command's output is mostly not the Markdown it
            # happened to name, so Bash reads count towards eligibility but not exposure.
            report.documentation_tokens_per_session.append(
                _tokens([call for call in retrieval + by_hand if _row(call)])
            )
            if retrieval:
                report.documentation_sessions_with_a_call += 1
            else:
                report.silent_sessions += 1
        for call in calls:
            name = _row(call)
            if name is None:
                continue
            row = report.rows[name]
            row.calls += 1
            row.paired += call.paired
            row.non_text += call.non_text
            if call.paired and not call.non_text:
                row.tokens.append(estimate_tokens(call.result) if call.result else 0)
        for transcript in files:
            _searches(report, transcript.calls)
    return report


def _share(count: int, total: int) -> str:
    return f"{count:>5}  ({count / total:.0%})" if total else f"{count:>5}"


def print_report(report: Report) -> None:
    print(f"sessions scanned                 {report.sessions_scanned}")
    docs = report.documentation_sessions
    print(f"documentation-seeking sessions   {docs}")
    print(
        f"  with a retrieval call         {_share(report.documentation_sessions_with_a_call, docs)}"
    )
    print(f"  read Markdown, never called   {_share(report.silent_sessions, docs)}")
    if docs:
        per_session = report.retrieval_calls_in_documentation_sessions / docs
        print(f"  retrieval calls per session   {per_session:>7.1f}")
    print("  (a behavioural proxy: a retrieval call, or Markdown read and not edited)")
    print(f"\nsessions that used the server    {report.sessions_using_server}")
    print(f"calls by tool                    {dict(report.server_calls)}")
    print(f"failed retrieval calls           {report.failures}")
    print(f"\nsearches                         {report.searches}")
    if report.searches:
        print(f"  evidence                       {dict(report.search_evidence)}")
        for label, count in (
            ("followed by read_section", report.searches_followed_by_read_section),
            ("followed by a new search", report.searches_followed_by_another_search),
            ("same-file Read (fallback)", report.searches_followed_by_same_file_read),
            ("other Markdown access", report.searches_followed_by_other_markdown),
        ):
            print(f"  {label:<29} {_share(count, report.searches)}")
        chains = report.reformulation_chains
        print(
            f"  reformulation chains          {len(chains):>5}  (longest {max(chains, default=0)})"
        )
        print(f"  keyword_match                  {dict(report.keyword_match)}")
        print(f"  empty results                 {report.empty_results:>5}")
    print("\nestimated result tokens per tool (context exposure, not billed)")
    header = f"{'tool':<22} {'calls':>6} {'paired':>6} {'text':>6} {'total':>8} {'median':>7}"
    print(header + f" {'p90':>7} {'max':>7}")
    for name, row in report.rows.items():
        stats = _stats(row.tokens)
        print(
            f"{name:<22} {row.calls:>6} {row.paired:>6} {stats['n']:>6} {stats['total']:>8}"
            f" {stats['median']:>7} {stats['p90']:>7} {stats['max']:>7}"
        )
    exposure = _stats(report.documentation_tokens_per_session)
    print(f"documentation tokens per documentation-seeking session, median {exposure['median']}")
    print("\n" + ESTIMATOR)
    print("\nNone of these is a relevance judgement: a read means the agent opened")
    print("something, not that it was right. Read them before labelling them.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--project", help="only transcripts whose directory matches this")
    parser.add_argument("--root", type=Path, default=TRANSCRIPTS, help="transcript directory")
    parser.add_argument(
        "--server", help="the name the server was registered under (default: *markdown*)"
    )
    parser.add_argument("--queries", action="store_true", help="print the queries, one per line")
    parser.add_argument("--json", type=Path, help="write the full report here")
    arguments = parser.parse_args()

    server = SERVER_TOOL
    if arguments.server:
        server = re.compile(rf"^mcp__{re.escape(arguments.server)}__(?P<tool>\w+)$")
    found = [
        read_transcript(path, server) for path in transcripts(arguments.root, arguments.project)
    ]
    report = summarise(found)

    if arguments.queries:
        for query in report.queries:
            print(query)
        return 0
    if arguments.json:
        arguments.json.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
        print(f"report written to {arguments.json}")
    print_report(report)
    if not report.sessions_using_server:
        print("\nThe server has never been called. Every accuracy number this project reports")
        print("comes from a benchmark; none of it comes from use. That is the finding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
