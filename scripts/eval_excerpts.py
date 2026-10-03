"""Does the top hit's excerpt keep the answer, and what does it save? (#76)

`search_docs` may send its top hit as an excerpt - the passage that matched and its
neighbours - instead of the whole section. This script is how that change is judged:

    uv run python scripts/eval_excerpts.py rankings --corpus v1 OUT.jsonl
    uv run python scripts/eval_excerpts.py build --corpus v2 --split dev OUTDIR
    uv run python scripts/eval_excerpts.py prompt
    uv run python scripts/eval_excerpts.py disagree OUTDIR/key.jsonl CODEX.json GEMINI.json OUT
    uv run python scripts/eval_excerpts.py score KEY [KEY ...] --codex C.json --gemini G.json
    uv run python scripts/eval_excerpts.py latency --corpus v1

`rankings` writes what the searcher ranks for every query (20 hits, as the retrieval gate
searches) so that a snapshot before and after a change can be compared byte for byte.
`build` writes, for every labelled query, the top hit as sent and the whole section it
comes from, as A/B texts in a seeded random order, plus a key that says which is which and
what each costs. Two judges answer the prompt `prompt` prints; `score` turns their verdicts
into answer retention and expected cost. Tune on `dev` only; `held_out` is run once, last.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import random
import statistics
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import eval_cache
from eval_retrieval import CORPUS_NAMES, Corpus, _p95, _probes, corpora, open_service

from markdown_memory import search as search_module
from markdown_memory.config import ServerConfig
from markdown_memory.embedders import DEFAULT_EMBEDDER
from markdown_memory.models import estimate_tokens
from markdown_memory.server import MarkdownMemoryService, create_server

SEED = 76
RANKING_LIMIT = 20  # what eval_retrieval.evaluate searches
KINDS = ("paraphrase", "identifier")
VERDICTS = ("yes", "partial", "no")
FLOOR_RETENTION = 0.95
CEILING_COST_RATIO = 0.80  # on corpus v2 only; v1 is reported
COST_GATED = ("v2",)

JUDGE_PROMPT = """\
You are judging a documentation-retrieval experiment. Do NOT run tests and change NO files.
Only read the input file.

Each line of the input is one JSON object with an "id", a "query" a coding agent sent to a
documentation search tool, and "texts": one or two candidate texts ("A", and sometimes "B")
the tool could return as its top answer. The texts are verbatim Markdown from the docs.

For each item and each text present, judge whether an agent reading ONLY that text could act
on the query without reading more:
- "yes": it answers the query
- "partial": relevant, but the agent would need to read more to act
- "no": does not answer

Judge each text independently; do not assume a longer text is better. Be strict and literal.

Output ONLY a JSON array, one object per item, nothing else (no prose, no code fences):
{"id": <id>, "A": "yes|partial|no", "B": "yes|partial|no"}
with "B" present exactly when the item has a text "B".
"""


@dataclass(slots=True, frozen=True)
class Item:
    """One judged query: the texts the judges see, and the key they must not."""

    judge: dict[str, Any]
    key: dict[str, Any]


def _cases(
    queries: Mapping[str, Any], split: str, kinds: Sequence[str] = KINDS
) -> Iterator[tuple[str, int, dict[str, Any]]]:
    for kind in kinds:
        for index, case in enumerate(queries[split].get(kind, [])):
            yield kind, index, case


def _relative(file_path: str, root: str) -> str:
    path = Path(file_path)
    return path.relative_to(root).as_posix() if path.is_absolute() else path.as_posix()


# ---------------------------------------------------------------------- rankings


def rankings(service: MarkdownMemoryService, queries: Mapping[str, Any]) -> list[str]:
    """One sorted-key JSON line per query: what was ranked, in order, and the keyword verdict."""
    lines: list[str] = []
    for split in ("dev", "held_out"):
        for kind, index, case in _cases(queries, split, (*KINDS, "no_answer")):
            page = service.search_page(str(case["query"]), RANKING_LIMIT)
            record = {
                "id": f"{split}/{kind}/{index}",
                "keyword_match": page.keyword_match,
                "query": case["query"],
                "ranked": [
                    [_relative(hit.file_path, service.root), hit.heading_path]
                    for hit in page.results
                ],
            }
            lines.append(json.dumps(record, sort_keys=True, ensure_ascii=False))
    return lines


# ---------------------------------------------------------------------- build


def _call(service: MarkdownMemoryService, tool: str, arguments: dict[str, Any]) -> str:
    """The text block a tool sends - what a client puts in the model's context."""
    server = create_server(service=service)

    async def call() -> str:
        outcome = await server.call_tool(tool, arguments)
        text = getattr(outcome.content[0], "text", None)
        if outcome.is_error or not isinstance(text, str):
            raise RuntimeError(f"{tool} failed for {arguments!r}: {text}")
        return text

    return asyncio.run(call())


def _compact(value: object) -> str:
    """The server's own serialisation (`server._json`), so token counts compare like for like."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def make_item(
    item_id: str, query: str, payload_text: str, section_text: str, rng: random.Random
) -> Item | None:
    """The judge's view and the key of one query, from what `search_docs` sent.

    `section_text` is `read_section` of the top hit: its whole section, or its part for a
    `(Part n)`. The baseline payload is the same page with that text as the top hit's content,
    so the two payloads differ by the excerpt alone.
    """
    page = json.loads(payload_text)
    if not page["results"]:
        return None
    top = page["results"][0]
    whole = {key: value for key, value in top.items() if key != "excerpt"}
    whole["content"] = section_text
    baseline = dict(page, results=[whole, *page["results"][1:]])
    key: dict[str, Any] = {
        "id": item_id,
        "payload_tokens": estimate_tokens(payload_text),
        "section_payload_tokens": estimate_tokens(_compact(baseline)),
        "read_tokens": estimate_tokens(section_text),
    }
    if top.get("excerpt"):
        excerpt_first = rng.random() < 0.5
        key["excerpt"], key["section"] = ("A", "B") if excerpt_first else ("B", "A")
        texts = {key["excerpt"]: top["content"], key["section"]: section_text}
        texts = {letter: texts[letter] for letter in ("A", "B")}
    else:
        key["excerpt"], key["section"] = None, "A"
        texts = {"A": top["content"]}
    return Item({"id": item_id, "query": query, "texts": texts}, key)


def build(
    service: MarkdownMemoryService, queries: Mapping[str, Any], corpus: str, split: str
) -> list[Item]:
    rng = random.Random(f"{SEED}/{corpus}/{split}")
    items: list[Item] = []
    for kind, index, case in _cases(queries, split):
        query = str(case["query"])
        payload_text = _call(service, "search_docs", {"query": query})
        top = (json.loads(payload_text)["results"] or [None])[0]
        if top is None:
            continue
        section_text = _call(
            service,
            "read_section",
            {"file_path": top["file_path"], "heading_path": top["heading_path"]},
        )
        item = make_item(
            f"{corpus}-{split}-{kind}-{index:02d}", query, payload_text, section_text, rng
        )
        if item is not None:
            items.append(item)
    return items


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ---------------------------------------------------------------------- score


def read_verdicts(path: Path, keys: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    """A judge's answer, checked against the key: one object per item, the letters it showed."""
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    answers = json.loads(text)
    verdicts: dict[str, dict[str, str]] = {}
    for answer in answers:
        item_id = answer.get("id")
        if item_id not in keys:
            continue  # a judge given several builds' inputs at once answers for all of them
        letters = ("A",) if keys[item_id]["excerpt"] is None else ("A", "B")
        shown = {letter: answer.get(letter) for letter in letters}
        extra = set(answer) - {"id", "note", *letters}
        if any(value not in VERDICTS for value in shown.values()) or extra:
            raise SystemExit(f"{path}: malformed verdict {answer!r}")
        verdicts[item_id] = {letter: str(value) for letter, value in shown.items()}
    return verdicts


def _agreed(
    item_id: str,
    letter: str,
    codex: Mapping[str, Mapping[str, str]],
    gemini: Mapping[str, Mapping[str, str]],
    joint: Mapping[str, Mapping[str, str]],
) -> str | None:
    """Both judges' verdict on one text, else the joint pass's, else unresolved (None)."""
    first, second = codex[item_id][letter], gemini[item_id][letter]
    if first == second:
        return first
    return joint.get(item_id, {}).get(letter)


@dataclass(slots=True, frozen=True)
class Outcome:
    eligible: bool  # the whole section answers
    kept: bool
    cost: int  # payload, plus the read the agent then needs
    baseline: int
    excerpted: bool


def outcome(
    key: Mapping[str, Any],
    codex: Mapping[str, Mapping[str, str]],
    gemini: Mapping[str, Mapping[str, str]],
    joint: Mapping[str, Mapping[str, str]],
) -> Outcome:
    item_id = key["id"]
    section = _agreed(item_id, key["section"], codex, gemini, joint) == "yes"
    if key["excerpt"] is None:
        # One text, one verdict: it decides eligibility and retention, and the payload already
        # is the whole section, so no read is ever charged.
        return Outcome(
            section, section, key["payload_tokens"], key["section_payload_tokens"], False
        )
    kept = section and _agreed(item_id, key["excerpt"], codex, gemini, joint) == "yes"
    cost = key["payload_tokens"] + (key["read_tokens"] if section and not kept else 0)
    return Outcome(section, kept, cost, key["section_payload_tokens"], True)


def disagreements(
    keys: Sequence[Mapping[str, Any]],
    codex: Mapping[str, Mapping[str, str]],
    gemini: Mapping[str, Mapping[str, str]],
) -> list[str]:
    return [key["id"] for key in keys if codex[key["id"]] != gemini[key["id"]]]


def score(
    keys: Sequence[Mapping[str, Any]],
    codex: Mapping[str, Mapping[str, str]],
    gemini: Mapping[str, Mapping[str, str]],
    joint: Mapping[str, Mapping[str, str]],
) -> tuple[list[str], bool]:
    """The report lines, and whether every gate held."""
    missing = [key["id"] for key in keys if key["id"] not in codex or key["id"] not in gemini]
    if missing:
        raise SystemExit(f"no verdict from both judges for: {', '.join(missing)}")
    lines: list[str] = []
    passed = True
    by_corpus: dict[str, list[Outcome]] = {}
    for key in keys:
        by_corpus.setdefault(key["id"].split("-")[0], []).append(outcome(key, codex, gemini, joint))
    for corpus, outcomes in sorted(by_corpus.items()):
        eligible = [o for o in outcomes if o.eligible]
        kept = sum(o.kept for o in eligible)
        retention = kept / len(eligible) if eligible else 0.0
        cost = statistics.mean(o.cost for o in eligible) if eligible else 0.0
        baseline = statistics.mean(o.baseline for o in eligible) if eligible else 0.0
        ratio = cost / baseline if baseline else 0.0
        others = [o for o in outcomes if not o.eligible]
        lines.append(
            f"{corpus}: retention {kept}/{len(eligible)} = {retention:.1%} "
            f"(floor {FLOOR_RETENTION:.0%}); expected cost {cost:.0f} vs {baseline:.0f} "
            f"= {ratio:.1%}"
            + (f" (ceiling {CEILING_COST_RATIO:.0%})" if corpus in COST_GATED else " (reported)")
        )
        if eligible:
            lines.append(
                f"  cost median {statistics.median(o.cost for o in eligible):.0f} "
                f"p95 {_p95([float(o.cost) for o in eligible]):.0f}; "
                f"excerpted {sum(o.excerpted for o in outcomes)}/{len(outcomes)}"
            )
        if others:
            lines.append(
                f"  {len(others)} items whose section does not answer (outside the denominator): "
                f"payload {statistics.mean(o.cost for o in others):.0f} "
                f"vs {statistics.mean(o.baseline for o in others):.0f}"
            )
        passed &= bool(eligible) and retention >= FLOOR_RETENTION
        if corpus in COST_GATED:
            passed &= bool(eligible) and ratio <= CEILING_COST_RATIO
    return lines, passed


# ---------------------------------------------------------------------- latency


@contextlib.contextmanager
def excerpts_off() -> Iterator[None]:
    """The whole-section path: the anchor is never found. Nothing in `src/` exists for this."""
    original = getattr(search_module, "select_anchor", None)
    if original is None:  # before the change there is nothing to turn off
        yield
        return
    search_module.select_anchor = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    try:
        yield
    finally:
        search_module.select_anchor = original  # type: ignore[attr-defined]


def latency(service: MarkdownMemoryService, queries: Mapping[str, Any], rounds: int) -> list[str]:
    texts = [
        str(case["query"]) for split in ("dev", "held_out") for _, _, case in _cases(queries, split)
    ]
    rng = random.Random(SEED)
    service.search_docs("warm up", 5)
    times: dict[str, list[float]] = {"off": [], "on": []}
    for _ in range(rounds):
        for query in texts:
            for mode in rng.sample(["off", "on"], 2):
                context = excerpts_off() if mode == "off" else contextlib.nullcontext()
                with context:
                    started = time.perf_counter()
                    service.search_docs(query, 5)
                    times[mode].append((time.perf_counter() - started) * 1000)
    off, on = times["off"], times["on"]
    median_off, median_on = statistics.median(off), statistics.median(on)
    p95_off, p95_on = _p95(off), _p95(on)
    return [
        f"n={len(off)} per mode ({rounds} rounds x {len(texts)} queries)",
        f"off: median {median_off:.1f}ms p95 {p95_off:.1f}ms",
        f"on:  median {median_on:.1f}ms p95 {p95_on:.1f}ms",
        f"delta: median {median_on / median_off - 1:+.1%} (ceiling +10%), "
        f"p95 {p95_on / p95_off - 1:+.1%} (ceiling +20%)",
    ]


# ---------------------------------------------------------------------- main


def _open(arguments: argparse.Namespace, corpus: Corpus) -> MarkdownMemoryService:
    base = ServerConfig.from_env()
    service, built = open_service(arguments, base, _probes(corpus.root), corpus)
    print(f"index: {'built' if built else 'reused from cache'} ({corpus.name})", file=sys.stderr)
    return service


def _with_service(arguments: argparse.Namespace, work: Any) -> int:
    corpus = corpora()[arguments.corpus]
    queries = json.loads(corpus.queries.read_text(encoding="utf-8"))
    try:
        with eval_cache.lock(eval_cache.cache_root()):
            service = _open(arguments, corpus)
            try:
                return int(work(service, queries, corpus))
            finally:
                service.close()
    except eval_cache.BusyError as busy:
        print(f"REFUSING TO RUN: {busy}", file=sys.stderr)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def indexed(name: str) -> argparse.ArgumentParser:
        command = commands.add_parser(name)
        command.add_argument("--corpus", default="v1", choices=CORPUS_NAMES)
        command.add_argument("--embedder", default=DEFAULT_EMBEDDER)
        command.add_argument("--rebuild", action="store_true")
        return command

    indexed("rankings").add_argument("out", type=Path)
    build_command = indexed("build")
    build_command.add_argument("--split", required=True, choices=("dev", "held_out"))
    build_command.add_argument("out", type=Path)
    indexed("latency").add_argument("--rounds", type=int, default=10)
    commands.add_parser("prompt")
    disagree_command = commands.add_parser("disagree")
    for name in ("key", "codex", "gemini", "out"):
        disagree_command.add_argument(name, type=Path)
    score_command = commands.add_parser("score")
    score_command.add_argument("keys", type=Path, nargs="+")
    score_command.add_argument("--codex", type=Path, required=True)
    score_command.add_argument("--gemini", type=Path, required=True)
    score_command.add_argument("--joint", type=Path)
    arguments = parser.parse_args(argv)
    logging.disable(logging.CRITICAL)

    if arguments.command == "prompt":
        print(JUDGE_PROMPT, end="")
        return 0
    if arguments.command in ("score", "disagree"):
        paths = arguments.keys if arguments.command == "score" else [arguments.key]
        keys = [key for path in paths for key in _read_jsonl(path)]
        by_id = {key["id"]: key for key in keys}
        codex = read_verdicts(arguments.codex, by_id)
        gemini = read_verdicts(arguments.gemini, by_id)
        if arguments.command == "disagree":
            split = set(disagreements(keys, codex, gemini))
            judged = _read_jsonl(arguments.key.with_name("judge_input.jsonl"))
            _write_jsonl(arguments.out, [item for item in judged if item["id"] in split])
            print(f"{len(split)} items disagree; joint-pass input written to {arguments.out}")
            return 0
        joint = read_verdicts(arguments.joint, by_id) if arguments.joint else {}
        lines, passed = score(keys, codex, gemini, joint)
        print("\n".join(lines))
        print("PASS" if passed else "FAIL")
        return 0 if passed else 1

    def run(service: MarkdownMemoryService, queries: Mapping[str, Any], corpus: Corpus) -> int:
        if arguments.command == "rankings":
            lines = rankings(service, queries)
            arguments.out.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
            print(f"{len(lines)} rankings written to {arguments.out}")
        elif arguments.command == "build":
            items = build(service, queries, corpus.name, arguments.split)
            arguments.out.mkdir(parents=True, exist_ok=True)
            _write_jsonl(arguments.out / "judge_input.jsonl", [item.judge for item in items])
            _write_jsonl(arguments.out / "key.jsonl", [item.key for item in items])
            excerpted = sum(item.key["excerpt"] is not None for item in items)
            print(f"{len(items)} items ({excerpted} excerpted) written to {arguments.out}")
        else:
            print("\n".join(latency(service, queries, arguments.rounds)))
        return 0

    return _with_service(arguments, run)


if __name__ == "__main__":
    raise SystemExit(main())
