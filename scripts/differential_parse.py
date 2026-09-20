"""Compare two parser implementations over a corpus of real Markdown files.

    uv run python scripts/differential_parse.py ~/Active-Projects
    uv run python scripts/differential_parse.py ~/docs --baseline /tmp/before.json
    uv run python scripts/differential_parse.py ~/docs --record /tmp/before.json

The parser decides what gets embedded, so a change to it can move retrieval without
changing a single heading. This records every section a parser produces - breadcrumb,
line span, verbatim content and the passage units - and diffs two runs byte-for-byte.

Record a baseline with the current parser, change the parser, then run again with
``--baseline``: the exit code is non-zero when any section differs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path

from markdown_memory.parser import MarkdownParser, join_parts

MAX_BYTES = 4_000_000
SKIPPED = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".mypy_cache"}


def markdown_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*.md")):
        if any(part in SKIPPED for part in path.parts):
            continue
        if path.is_file() and path.stat().st_size <= MAX_BYTES:
            yield path


def fingerprint(path: Path, parser: MarkdownParser) -> dict[str, object]:
    """Everything about a document that could change what is indexed or returned."""
    text = path.read_text(encoding="utf-8", errors="replace")
    started = time.perf_counter()
    document = parser.parse(text, fallback_title=path.stem)
    elapsed_ms = (time.perf_counter() - started) * 1000
    sections = [
        {
            "path": section.heading_path,
            "base": section.base_path,
            "level": section.heading_level,
            "part": section.part_index,
            "lines": [section.start_line, section.end_line],
            "content": hashlib.sha256(section.content.encode()).hexdigest()[:16],
            "chars": len(section.content),
            "units": [hashlib.sha256(unit.encode()).hexdigest()[:16] for unit in section.units],
        }
        for section in document.sections
    ]
    # Parts must still rebuild their section exactly, whatever the implementation.
    reassembled = {}
    for base in {s.base_path for s in document.sections}:
        parts = [s for s in document.sections if s.base_path == base]
        reassembled[base] = hashlib.sha256(join_parts(parts).encode()).hexdigest()[:16]
    return {
        "title": document.title,
        "line_count": document.line_count,
        "sections": sections,
        "reassembled": reassembled,
        "elapsed_ms": round(elapsed_ms, 3),
    }


def scan(root: Path) -> dict[str, dict[str, object]]:
    parser = MarkdownParser()
    result: dict[str, dict[str, object]] = {}
    for number, path in enumerate(markdown_files(root), start=1):
        try:
            result[str(path)] = fingerprint(path, parser)
        except Exception as exc:  # a crash is itself a finding
            result[str(path)] = {"error": f"{type(exc).__name__}: {exc}"}
        if number % 250 == 0:
            print(f"  ... {number} files", flush=True)
    return result


def report_slowest(scanned: dict[str, dict[str, object]], limit: float) -> int:
    timed = [
        (float(data["elapsed_ms"]), name)
        for name, data in scanned.items()
        if isinstance(data.get("elapsed_ms"), float)
    ]
    timed.sort(reverse=True)
    over = [(ms, name) for ms, name in timed if ms > limit]
    print(f"\nslowest parse: {timed[0][0]:.1f} ms ({Path(timed[0][1]).name})" if timed else "")
    for ms, name in over[:10]:
        print(f"  SLOW {ms:8.1f} ms  {name}")
    return len(over)


def compare(before: dict[str, object], after: dict[str, object]) -> list[str]:
    differences: list[str] = []
    for name in sorted(set(before) | set(after)):
        old, new = before.get(name), after.get(name)
        if old is None or new is None:
            differences.append(f"{name}: only in {'after' if old is None else 'before'}")
            continue
        assert isinstance(old, dict) and isinstance(new, dict)
        for field in ("title", "line_count", "sections", "reassembled", "error"):
            if old.get(field) != new.get(field):
                differences.append(f"{name}: {field} differs")
                if field == "sections":
                    differences.extend(section_diff(old[field], new[field]))
    return differences


def section_diff(old: object, new: object) -> list[str]:
    assert isinstance(old, list) and isinstance(new, list)
    lines: list[str] = []
    by_path_old = {s["path"]: s for s in old}
    by_path_new = {s["path"]: s for s in new}
    for path in sorted(set(by_path_old) - set(by_path_new)):
        lines.append(f"    - lost:  {path}")
    for path in sorted(set(by_path_new) - set(by_path_old)):
        lines.append(f"    + new:   {path}")
    for path in sorted(set(by_path_old) & set(by_path_new)):
        a, b = by_path_old[path], by_path_new[path]
        for field in ("lines", "content", "units", "chars", "level", "part"):
            if a[field] != b[field]:
                lines.append(f"    ~ {path}: {field} {a[field]!r} -> {b[field]!r}")
    return lines[:20]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("root", type=Path, help="directory to scan recursively for *.md")
    parser.add_argument("--record", type=Path, help="write this run's fingerprints and stop")
    parser.add_argument("--baseline", type=Path, help="compare this run against a recorded run")
    parser.add_argument("--slow-ms", type=float, default=5.0, help="report parses slower than this")
    arguments = parser.parse_args()

    root = arguments.root.expanduser().resolve()
    print(f"scanning {root}")
    scanned = scan(root)
    crashed = [name for name, data in scanned.items() if "error" in data]
    print(f"parsed {len(scanned)} files, {len(crashed)} crashed")
    for name in crashed[:10]:
        print(f"  CRASH {name}: {scanned[name]['error']}")
    slow = report_slowest(scanned, arguments.slow_ms)

    if arguments.record:
        arguments.record.write_text(json.dumps(scanned), encoding="utf-8")
        print(f"\nrecorded {len(scanned)} fingerprints to {arguments.record}")
        return 1 if crashed else 0

    if not arguments.baseline:
        print("\nnothing to compare against; pass --record or --baseline")
        return 1 if crashed else 0

    before = json.loads(arguments.baseline.read_text(encoding="utf-8"))
    differences = compare(before, scanned)
    if differences:
        print(f"\n{len(differences)} difference(s):")
        for line in differences[:60]:
            print(f"  {line}")
        if len(differences) > 60:
            print(f"  ... and {len(differences) - 60} more")
        return 1
    print(f"\nIDENTICAL: {len(scanned)} files produce byte-for-byte the same sections and units")
    if slow:
        print(f"note: {slow} file(s) parsed slower than {arguments.slow_ms} ms")
    return 1 if crashed else 0


if __name__ == "__main__":
    sys.exit(main())
