"""Collect real queries for review. No classifier: a person labels them.

    uv run python scripts/preflight.py                      # summary
    uv run python scripts/preflight.py --out review.jsonl   # one row per query to label
    uv run python scripts/preflight.py --project markdown-memory

Step 3 of the plan asks for the query families, the fallback rate, and the families the
planned benchmark has no stratum for. Two reviewers, asked independently, said the same
thing about how to get them: **do not write a classifier**. At a hundred queries a person
labels the set in half an hour and sees what the categories should have been, while
regexes written before the data arrives decide in advance what anyone will find in it.
"unanswerable" is not even decidable from the query string - it depends on what the
corpus holds.

So this script does the part a script does well - find every query, count repeats, attach
the request that prompted it, record what the agent did next - and stops exactly where
judgement starts. The ``family`` and ``verdict`` fields ship empty.

The task context is the point. A query like "retry policy" tells you nothing on its own;
the user's request two turns earlier tells you what the agent was trying to do, and only
that makes the retrieval judgeable.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import usage_from_transcripts as miner

# How far back to look for the request that prompted a query. Beyond a couple of turns
# the connection is a guess, and a guess in the context column would be labelled as fact.
CONTEXT_TURNS = 2
CONTEXT_CHARS = 400


@dataclass(slots=True)
class Occurrence:
    """One search, with what prompted it and what happened next."""

    session: str
    project: str
    timestamp: str
    task: str
    followed_by: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QueryRecord:
    """One distinct query, however many times it was asked."""

    query: str
    count: int
    occurrences: list[Occurrence]
    family: str = ""  # a person fills these in
    verdict: str = ""
    note: str = ""


def _text_of(message: object) -> str:
    """The human-readable text of a transcript message, or ''."""
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = [
        str(part.get("text", ""))
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    ]
    return " ".join(part for part in parts if part).strip()


def read_session(path: Path) -> tuple[list[miner.Call], dict[int, str]]:
    """Every tool call in a transcript, plus the user request in force at each one.

    Calls and requests are counted in the same pass, in the same order the miner uses,
    because two loops with slightly different rules drift: the miner takes a tool call
    from any message, so counting only assistant messages here would shift every request
    by one and attribute each query to the wrong task - silently, and worse than not
    attributing it at all.
    """
    calls = miner.calls_in(path)
    requests: dict[int, str] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return calls, requests
    recent: list[str] = []
    index = 0
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if message.get("role") == "user" and record.get("type") != "tool_result":
            text = _text_of(message)
            if text and not text.startswith("<"):  # skip system-injected blocks
                recent.append(text)
                recent[:] = recent[-CONTEXT_TURNS:]
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "tool_use":
                requests[index] = recent[-1][:CONTEXT_CHARS] if recent else ""
                index += 1
    return calls, requests


def collect(root: Path, project: str | None) -> list[QueryRecord]:
    """Every search ever made, grouped by query, newest occurrence last."""
    grouped: dict[str, QueryRecord] = {}
    for path in miner.transcripts(root, project):
        calls, requests = read_session(path)
        for position, call in enumerate(calls):
            if call.tool != "search_docs" or not call.query:
                continue
            window = calls[position + 1 : position + 1 + miner.FALLBACK_WINDOW]
            followed_by = [
                str(later.tool) if later.tool else later.name
                for later in window
                if later.tool or miner._touches_markdown(later)
            ]
            occurrence = Occurrence(
                session=call.session,
                project=Path(call.project).name,
                timestamp=call.timestamp,
                task=requests.get(call.index, ""),
                followed_by=followed_by,
            )
            record = grouped.setdefault(call.query, QueryRecord(call.query, 0, []))
            record.count += 1
            record.occurrences.append(occurrence)
    return sorted(grouped.values(), key=lambda record: (-record.count, record.query))


def families_seen(records: Sequence[QueryRecord]) -> Counter[str]:
    """Whatever a person has written in the family column so far."""
    return Counter(record.family for record in records if record.family)


def rows(records: Sequence[QueryRecord]) -> Iterator[str]:
    for record in records:
        yield json.dumps(asdict(record), ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--root", type=Path, default=miner.TRANSCRIPTS)
    parser.add_argument("--project", help="only transcripts whose directory matches this")
    parser.add_argument("--out", type=Path, help="write one JSON row per query, ready to label")
    parser.add_argument("--target", type=int, default=100, help="queries the preflight needs")
    arguments = parser.parse_args()

    records = collect(arguments.root, arguments.project)
    total = sum(record.count for record in records)

    if not records:
        print("0 queries across the transcripts searched.")
        print("\nNothing to analyse yet, and no synthetic stand-in will do: the point of this")
        print("step is to find out what real queries look like. Use the server for real work")
        print("first - see the plan's Step 2 - then run this again.")
        return 0

    print(f"{total} searches, {len(records)} distinct queries")
    fallbacks = sum(
        1
        for record in records
        for occurrence in record.occurrences
        # Only a step off the server is a retreat: an outline after a search is navigation.
        if any(step in miner.FILE_TOOLS or step == "Bash" for step in occurrence.followed_by)
    )
    reads = sum(
        1
        for record in records
        for occurrence in record.occurrences
        if "read_section" in occurrence.followed_by
    )
    print(f"  followed by read_section   {reads:>4}  ({reads / total:.0%})")
    print(f"  followed by file access    {fallbacks:>4}  ({fallbacks / total:.0%})")
    with_task = sum(1 for r in records for o in r.occurrences if o.task)
    print(f"  with the request attached  {with_task:>4}  ({with_task / total:.0%})")

    labelled = families_seen(records)
    print(f"\nfamilies labelled so far: {dict(labelled) if labelled else 'none - label them'}")
    if total < arguments.target:
        print(f"\n{total} of {arguments.target} queries. The preflight is not due yet.")

    print("\nmost asked:")
    for record in records[:10]:
        print(f"  {record.count:>3}x  {record.query[:80]}")

    if arguments.out:
        arguments.out.write_text("\n".join(rows(records)) + "\n", encoding="utf-8")
        print(f"\n{len(records)} rows written to {arguments.out}")
        print("Fill in `family`, `verdict` and `note` by reading each query against its task.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
