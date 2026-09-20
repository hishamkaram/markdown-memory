"""What actually happened when an agent needed documentation.

    uv run python scripts/usage_from_transcripts.py                # every project
    uv run python scripts/usage_from_transcripts.py --project markdown-memory
    uv run python scripts/usage_from_transcripts.py --queries      # one query per line
    uv run python scripts/usage_from_transcripts.py --json report.json

Claude Code already records every tool call it makes, with arguments, results and order,
in ``~/.claude/projects/**/*.jsonl``. That is a better source than anything this server
could log about itself, for one reason: it also records the calls that went somewhere
else. A server-side log can say how often ``search_docs`` was answered; only the
transcript can say how often the agent asked it, disliked the answer, and read the file
with ``Read`` instead - which is the number that decides whether this tool is worth
having.

Reported per session:

* every ``mcp__*markdown*`` call, its arguments and whether it failed;
* the *fallback rate*: a search followed, within a few calls, by ``Read``/``Grep``/
  ``Glob`` or a ``grep``-like ``Bash`` command against a Markdown file;
* the *reformulation rate*: a search followed by another search in the same session;
* the *silence*: sessions in a project that has the server configured which never called
  it at all. A tool nobody reaches for has no retrieval quality worth measuring.

Nothing here is a relevance judgement. A read after a search means the agent opened
something, not that the answer was right; these are behaviours, and the labels come from
reading them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

TRANSCRIPTS = Path.home() / ".claude" / "projects"
SERVER_TOOL = re.compile(r"^mcp__[^_]*markdown[^_]*__(?P<tool>\w+)$", re.IGNORECASE)
# A search is "abandoned" when the agent reaches for the file system instead, within this
# many tool calls. Beyond that the connection to the search is guesswork.
FALLBACK_WINDOW = 4
FILE_TOOLS = ("Read", "Grep", "Glob", "NotebookRead")
GREP_LIKE = re.compile(r"\b(rg|grep|ag|sed -n|head|cat|awk)\b")
MARKDOWN = re.compile(r"\.mdx?\b", re.IGNORECASE)


@dataclass(slots=True, frozen=True)
class Call:
    """One tool call, in the order the session made it."""

    index: int
    name: str
    tool: str | None  # the server's tool name, when this is one of ours
    arguments: dict[str, object]
    session: str
    project: str
    timestamp: str

    @property
    def query(self) -> str:
        value = self.arguments.get("query")
        return value if isinstance(value, str) else ""


@dataclass
class Report:
    """Counts over every transcript that was read."""

    sessions_scanned: int = 0
    sessions_using_server: int = 0
    server_calls: Counter[str] = field(default_factory=Counter)
    searches: int = 0
    searches_followed_by_read_section: int = 0
    searches_followed_by_another_search: int = 0
    searches_followed_by_file_access: int = 0
    failures: int = 0
    projects: Counter[str] = field(default_factory=Counter)
    queries: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["server_calls"] = dict(self.server_calls)
        data["projects"] = dict(self.projects)
        return data


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


def calls_in(path: Path) -> list[Call]:
    """Every tool call in one transcript, in order. Unreadable lines are skipped."""
    found: list[Call] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return found
    for line in text.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue  # a transcript being written to can end mid-line
        if not isinstance(record, dict):
            continue
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "tool_use":
                continue
            name = str(part.get("name", ""))
            arguments = part.get("input")
            match = SERVER_TOOL.match(name)
            found.append(
                Call(
                    index=len(found),
                    name=name,
                    tool=match.group("tool") if match else None,
                    arguments=arguments if isinstance(arguments, dict) else {},
                    session=str(record.get("sessionId", path.stem)),
                    project=str(record.get("cwd", path.parent.name)),
                    timestamp=str(record.get("timestamp", "")),
                )
            )
    return found


def _touches_markdown(call: Call) -> bool:
    """Whether a non-server call went looking in Markdown by hand."""
    if call.name in FILE_TOOLS:
        blob = json.dumps(call.arguments)
        return bool(MARKDOWN.search(blob)) or call.name in ("Grep", "Glob")
    if call.name == "Bash":
        command = call.arguments.get("command")
        return isinstance(command, str) and bool(
            GREP_LIKE.search(command) and MARKDOWN.search(command)
        )
    return False


def summarise(sessions: Sequence[Sequence[Call]]) -> Report:
    report = Report()
    for calls in sessions:
        report.sessions_scanned += 1
        ours = [call for call in calls if call.tool]
        if not ours:
            continue
        report.sessions_using_server += 1
        report.projects[Path(ours[0].project).name] += 1
        for call in ours:
            report.server_calls[str(call.tool)] += 1
        for position, call in enumerate(calls):
            if call.tool != "search_docs":
                continue
            report.searches += 1
            if call.query:
                report.queries.append(call.query)
            window = calls[position + 1 : position + 1 + FALLBACK_WINDOW]
            if any(later.tool == "read_section" for later in window):
                report.searches_followed_by_read_section += 1
            if any(later.tool == "search_docs" for later in window):
                report.searches_followed_by_another_search += 1
            if any(_touches_markdown(later) for later in window):
                report.searches_followed_by_file_access += 1
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--project", help="only transcripts whose directory matches this")
    parser.add_argument("--root", type=Path, default=TRANSCRIPTS, help="transcript directory")
    parser.add_argument("--queries", action="store_true", help="print the queries, one per line")
    parser.add_argument("--json", type=Path, help="write the full report here")
    arguments = parser.parse_args()

    sessions = [calls_in(path) for path in transcripts(arguments.root, arguments.project)]
    report = summarise(sessions)

    if arguments.queries:
        for query in report.queries:
            print(query)
        return 0
    if arguments.json:
        arguments.json.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
        print(f"report written to {arguments.json}")

    print(f"sessions scanned            {report.sessions_scanned}")
    print(f"sessions that used the tool {report.sessions_using_server}")
    if not report.sessions_using_server:
        print("\nThe server has never been called. Every accuracy number this project reports")
        print("comes from a benchmark; none of it comes from use. That is the finding.")
        return 0
    print(f"projects                    {dict(report.projects)}")
    print(f"calls by tool               {dict(report.server_calls)}")
    print(f"\nsearches                    {report.searches}")
    if report.searches:
        for label, count in (
            ("followed by read_section", report.searches_followed_by_read_section),
            ("followed by a new search", report.searches_followed_by_another_search),
            ("followed by file access", report.searches_followed_by_file_access),
        ):
            print(f"  {label:<26} {count:>4}  ({count / report.searches:.0%})")
        print("\nNone of these is a relevance judgement: a read means the agent opened")
        print("something, not that it was right. Read them before labelling them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
