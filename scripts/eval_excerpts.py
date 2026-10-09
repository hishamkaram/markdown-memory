"""Does the top hit's excerpt keep the answer, and what does it save? (#76)

`search_docs` may send its top hit as an excerpt - the passage that matched and its
neighbours - instead of the whole section. This script is how that change is judged:

    uv run python scripts/eval_excerpts.py rankings --corpus v1 OUT.jsonl
    uv run python scripts/eval_excerpts.py build --corpus v2 --split dev OUTDIR
    uv run python scripts/eval_excerpts.py texts OUTDIR section|excerpt OUT.jsonl
    uv run python scripts/eval_excerpts.py prompt
    uv run python scripts/eval_excerpts.py freeze KEY [KEY ...] --judge J.json ... --out E.json
    uv run python scripts/eval_excerpts.py score KEY [KEY ...] --eligible E.json --judge J.json ...
    uv run python scripts/eval_excerpts.py latency --corpus v1

`rankings` writes what the searcher ranks for every query (20 hits, as the retrieval gate
searches) so that a snapshot before and after a change can be compared byte for byte.
`build` writes, for every labelled query, the top hit as sent and the whole section it
comes from, plus a key that says what each costs; `--queries` reads a query file other than
the corpus's own, such as a sealed set. `texts` turns a build into what a judge sees: one
text per item, either the whole section or the excerpt. Judges answer the prompt `prompt`
prints, each in isolation. `freeze` records, once and before any change is tuned, which
items' whole sections answer - a strict majority of the judges saying "yes" - so the
denominator does not move with the change being measured. `score` then judges each excerpt
by the same majority and reports answer retention and expected cost against the gates.
`latency` times search with excerpts on and off, and exits non-zero when the median or p95
rises past its ceiling.
Tune on `dev` only; a held-out set is run once, last.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
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
# Gated on corpus v2 only, over every frozen eligible item; v1 is reported. The cost gate asks
# for a material net saving; the 80% target is what #76 hoped for, and is reported, not gated.
GATED = ("v2",)
FLOOR_RETENTION = 0.95
FLOOR_RETENTION_BOUND = 0.90  # one-sided 95% Wilson lower bound on retention
CEILING_COST_RATIO = 0.90
CEILING_COST_BOUND = 0.95  # one-sided 95% bootstrap upper bound, resampling sections
CEILING_LATENCY_MEDIAN = 0.10  # excerpts on against off
CEILING_LATENCY_P95 = 0.20
TARGET_COST_RATIO = 0.80
TARGET_COST_BOUND = 0.85
Z_95 = 1.6448536269514722  # one-sided 95%
BOOTSTRAP_ROUNDS = 2000
JUDGES = 3  # a strict majority of three: no split is left for a joint pass to settle

JUDGE_PROMPT = """\
You are judging a documentation-retrieval experiment. Do NOT run tests and change NO files.
Only read the input file.

Each line of the input is one JSON object with an "id", a "query" a coding agent sent to a
documentation search tool, "hit": the file and heading path the tool named for its top answer,
and "texts": {"A": ...}, the text it returned for that hit, verbatim Markdown from the docs.

For each item, judge whether an agent reading ONLY that hit - its file, heading path and text
- could act on the query without reading more:
- "yes": it answers the query
- "partial": relevant, but the agent would need to read more to act
- "no": does not answer

Be strict and literal.

Output ONLY a JSON array, one object per item, nothing else (no prose, no code fences):
{"id": <id>, "A": "yes|partial|no"}
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
    for split in (name for name, cases in queries.items() if isinstance(cases, dict)):
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


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_item(
    item_id: str,
    query: str,
    payload_text: str,
    baseline_text: str,
    section_text: str,
    rng: random.Random,
) -> Item | None:
    """The judge's view and the key of one query, from what `search_docs` sent.

    `baseline_text` is what `search_docs` sends for the same query with the excerpt switched
    off, and `section_text` is `read_section` of the top hit: its whole section, or its part
    for a `(Part n)`. The two responses must name the same top hit, whose whole content is
    that section, or the baseline would price something else.
    """
    page = json.loads(payload_text)
    if not page["results"]:
        return None
    top = page["results"][0]
    whole = json.loads(baseline_text)["results"][0]
    if (whole["file_path"], whole["heading_path"], whole["content"]) != (
        top["file_path"],
        top["heading_path"],
        section_text,
    ):
        raise RuntimeError(f"{item_id}: the whole-section response names another top hit")
    key: dict[str, Any] = {
        "id": item_id,
        # Queries answered by one section are not independent: the bootstrap resamples these.
        "cluster": f"{top['file_path']} :: {top['heading_path']}",
        "payload_tokens": estimate_tokens(payload_text),
        "section_payload_tokens": estimate_tokens(baseline_text),
        "read_tokens": estimate_tokens(section_text),
        # What the judges read, so a frozen verdict is only ever scored against the same text
        # (#87): the cluster names a heading, and a heading can hold other words next time.
        "query_sha256": _sha256(query),
        "section_sha256": _sha256(section_text),
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
        with excerpts_off():
            baseline_text = _call(service, "search_docs", {"query": query})
        top = (json.loads(payload_text)["results"] or [None])[0]
        if top is None:
            continue
        section_text = _call(
            service,
            "read_section",
            {"file_path": top["file_path"], "heading_path": top["heading_path"]},
        )
        item_id = f"{corpus}-{split}-{kind}-{index:02d}"
        item = make_item(item_id, query, payload_text, baseline_text, section_text, rng)
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


def judge_texts(
    judged: Sequence[Mapping[str, Any]], keys: Sequence[Mapping[str, Any]], which: str
) -> list[dict[str, Any]]:
    """One text per item for a judge: every whole section, or every excerpt there is.

    Each comes with the file and heading path of its hit, which the agent sees beside the
    text: a release named only by its heading is still named.
    """
    by_id = {key["id"]: key for key in keys}
    texts: list[dict[str, Any]] = []
    for item in judged:
        key = by_id[item["id"]]
        letter = key[which]
        if letter is not None:
            file_path, heading_path = key["cluster"].split(" :: ", 1)
            texts.append(
                {
                    "id": item["id"],
                    "query": item["query"],
                    "hit": {"file_path": file_path, "heading_path": heading_path},
                    "texts": {"A": item["texts"][letter]},
                }
            )
    return texts


def read_verdicts(path: Path, ids: Sequence[str]) -> dict[str, str]:
    """One judge's answer: exactly one well-formed verdict for each of ``ids``, nothing else."""
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    wanted = set(ids)
    verdicts: dict[str, str] = {}
    for answer in json.loads(text):
        item_id = answer.get("id")
        if item_id not in wanted:
            raise SystemExit(f"{path}: a verdict for {item_id!r}, which it was not given")
        if (
            answer.get("A") not in VERDICTS
            or set(answer) - {"id", "A", "note"}
            or item_id in verdicts
        ):
            raise SystemExit(f"{path}: malformed verdict {answer!r}")
        verdicts[item_id] = str(answer["A"])
    missing = wanted - verdicts.keys()
    if missing:
        raise SystemExit(f"{path}: no verdict for {', '.join(sorted(missing))}")
    return verdicts


def drifted(keys: Sequence[Mapping[str, Any]], frozen: Mapping[str, Any]) -> str:
    """Why a build cannot be scored against a frozen denominator, or "" when it can.

    Every frozen eligible item must be in the build exactly once, answered by the section it
    was frozen with, for the same query and the same section text: a changed ranking, a
    rebuilt index, an edited query or a document edited under the same heading would
    otherwise score a verdict against text the judges never read (#87). The two payload
    costs are not frozen - they are what the change being scored is meant to move - and
    `read_tokens` follows from the section text. `judge_input.jsonl` is not read here: that
    the judges were given this build's texts rests on both files coming from one `build`.
    """
    counts: dict[str, int] = {}
    for key in keys:
        counts[key["id"]] = counts.get(key["id"], 0) + 1
    problems = [
        f"{item_id} {'missing' if item_id not in counts else 'repeated'}"
        for item_id in frozen["eligible"]
        if counts.get(item_id) != 1
    ]
    clusters = frozen.get("clusters")
    if clusters is None:
        return "the frozen file records no sections; freeze it again"
    queries, sections = frozen.get("queries"), frozen.get("sections")
    if queries is None or sections is None:
        return "the frozen file records no texts; freeze it again"
    eligible = set(frozen["eligible"])
    for key in keys:
        if key["id"] not in eligible:
            continue
        if clusters.get(key["id"]) != key["cluster"]:
            problems.append(f"{key['id']} now answered by another section")
        if unhashed(key):
            problems.append(f"{key['id']} records no text hashes; build it again")
            continue
        if queries.get(key["id"]) != key.get("query_sha256"):
            problems.append(f"{key['id']} asks another query")
        if sections.get(key["id"]) != key.get("section_sha256"):
            problems.append(f"{key['id']} answered by other section text")
    return "; ".join(problems)


def unhashed(key: Mapping[str, Any]) -> bool:
    """A key built before #87: it cannot say which texts its verdicts were about."""
    return key.get("query_sha256") is None or key.get("section_sha256") is None


def majority_yes(item_id: str, judges: Sequence[Mapping[str, str]]) -> bool:
    """A strict majority of the judges said "yes": with two, both; a split is not a yes."""
    return 2 * sum(judge[item_id] == "yes" for judge in judges) > len(judges)


def wilson_lower(kept: int, total: int, z: float = Z_95) -> float:
    if total == 0:
        return 0.0
    share = kept / total
    centre = share + z * z / (2 * total)
    spread = z * (share * (1 - share) / total + z * z / (4 * total * total)) ** 0.5
    return (centre - spread) / (1 + z * z / total)


@dataclass(slots=True, frozen=True)
class Outcome:
    kept: bool
    cost: int  # payload, plus the read the agent then needs
    baseline: int
    excerpted: bool
    cluster: str


def outcome(key: Mapping[str, Any], judges: Sequence[Mapping[str, str]]) -> Outcome:
    """One eligible item: an excerpt is kept when a majority says it answers on its own."""
    if key["excerpt"] is None:
        return Outcome(
            True, key["payload_tokens"], key["section_payload_tokens"], False, key["cluster"]
        )
    kept = majority_yes(key["id"], judges)
    cost = key["payload_tokens"] + (0 if kept else key["read_tokens"])
    return Outcome(kept, cost, key["section_payload_tokens"], True, key["cluster"])


def cost_upper(outcomes: Sequence[Outcome], seed: str) -> float:
    """The 95th percentile of the cost ratio over resampled sections, not resampled queries."""
    clusters: dict[str, list[Outcome]] = {}
    for item in outcomes:
        clusters.setdefault(item.cluster, []).append(item)
    groups = [(sum(o.cost for o in g), sum(o.baseline for o in g)) for g in clusters.values()]
    rng = random.Random(seed)
    ratios = []
    for _ in range(BOOTSTRAP_ROUNDS):
        drawn = [groups[rng.randrange(len(groups))] for _ in groups]
        ratios.append(sum(c for c, _ in drawn) / sum(b for _, b in drawn))
    ratios.sort()
    return ratios[int(0.95 * len(ratios)) - 1]


def score(
    keys: Sequence[Mapping[str, Any]],
    eligible: Sequence[str],
    judges: Sequence[Mapping[str, str]],
) -> tuple[list[str], bool]:
    """The report lines, and whether every gate held."""
    lines: list[str] = []
    passed = True
    frozen = set(eligible)
    by_corpus: dict[str, list[Outcome]] = {}
    for key in keys:
        if key["id"] in frozen:
            by_corpus.setdefault(key["id"].split("-")[0], []).append(outcome(key, judges))
    for corpus, outcomes in sorted(by_corpus.items()):
        kept = sum(o.kept for o in outcomes)
        retention = kept / len(outcomes)
        bound = wilson_lower(kept, len(outcomes))
        ratio = sum(o.cost for o in outcomes) / sum(o.baseline for o in outcomes)
        upper = cost_upper(outcomes, f"{SEED}/{corpus}")
        gated = corpus in GATED
        lines.append(
            f"{corpus}: retention {kept}/{len(outcomes)} = {retention:.1%} (floor "
            f"{FLOOR_RETENTION:.0%}), Wilson lower bound {bound:.1%} (floor "
            f"{FLOOR_RETENTION_BOUND:.0%}); cost {ratio:.1%}, upper bound {upper:.1%}"
            + (
                f" (ceilings {CEILING_COST_RATIO:.0%}, {CEILING_COST_BOUND:.0%})"
                if gated
                else " (reported, not gated)"
            )
        )
        if gated:
            met = ratio <= TARGET_COST_RATIO and upper <= TARGET_COST_BOUND
            lines.append(
                f"  cost target {TARGET_COST_RATIO:.0%} / {TARGET_COST_BOUND:.0%}: "
                + ("met" if met else "missed (reported, not gated)")
            )
        lines.append(
            f"  cost median {statistics.median(o.cost for o in outcomes):.0f} "
            f"p95 {_p95([float(o.cost) for o in outcomes]):.0f}; "
            f"excerpted {sum(o.excerpted for o in outcomes)}/{len(outcomes)} eligible"
        )
        if gated:
            passed &= retention >= FLOOR_RETENTION and bound >= FLOOR_RETENTION_BOUND
            passed &= ratio <= CEILING_COST_RATIO and upper < CEILING_COST_BOUND
    if not any(corpus in by_corpus for corpus in GATED):
        lines.append("no eligible item of a gated corpus")
        passed = False
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


def latency_verdict(off: Sequence[float], on: Sequence[float]) -> tuple[list[str], bool]:
    """The report lines, and whether both ceilings held: excerpts on against off."""
    median_off, median_on = statistics.median(off), statistics.median(on)
    p95_off, p95_on = _p95(off), _p95(on)
    # Ratios, not deltas: 110 / 100 - 1 is 0.10000000000000009, over the +10% it meets.
    median, p95 = median_on / median_off, p95_on / p95_off
    lines = [
        f"off: median {median_off:.1f}ms p95 {p95_off:.1f}ms",
        f"on:  median {median_on:.1f}ms p95 {p95_on:.1f}ms",
        f"delta: median {median - 1:+.1%} (ceiling +{CEILING_LATENCY_MEDIAN:.0%}), "
        f"p95 {p95 - 1:+.1%} (ceiling +{CEILING_LATENCY_P95:.0%})",
    ]
    return lines, median <= 1 + CEILING_LATENCY_MEDIAN and p95 <= 1 + CEILING_LATENCY_P95


def latency(
    service: MarkdownMemoryService, queries: Mapping[str, Any], rounds: int
) -> tuple[list[str], bool]:
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
    lines, passed = latency_verdict(off, on)
    return [f"n={len(off)} per mode ({rounds} rounds x {len(texts)} queries)", *lines], passed


# ---------------------------------------------------------------------- main


def _open(arguments: argparse.Namespace, corpus: Corpus) -> MarkdownMemoryService:
    base = ServerConfig.from_env()
    service, built = open_service(arguments, base, _probes(corpus.root), corpus)
    print(f"index: {'built' if built else 'reused from cache'} ({corpus.name})", file=sys.stderr)
    return service


def _with_service(arguments: argparse.Namespace, work: Any) -> int:
    corpus = corpora()[arguments.corpus]
    source = getattr(arguments, "queries", None) or corpus.queries
    queries = json.loads(source.read_text(encoding="utf-8"))
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

    for name in ("rankings", "build"):
        command = indexed(name)
        command.add_argument("--queries", type=Path, help="a query file other than the corpus's")
        if name == "build":
            command.add_argument(
                "--split", required=True, help="dev, held_out, or a split --queries names"
            )
        command.add_argument("out", type=Path)
    indexed("latency").add_argument("--rounds", type=int, default=10)
    commands.add_parser("prompt")
    texts_command = commands.add_parser("texts")
    texts_command.add_argument("build_dir", type=Path)
    texts_command.add_argument("which", choices=("section", "excerpt"))
    texts_command.add_argument("out", type=Path)
    for name in ("freeze", "score"):
        command = commands.add_parser(name)
        command.add_argument("keys", type=Path, nargs="+")
        command.add_argument("--judge", type=Path, action="append", required=True)
        if name == "freeze":
            command.add_argument("--out", type=Path, required=True)
        else:
            command.add_argument("--eligible", type=Path, required=True)
    arguments = parser.parse_args(argv)
    logging.disable(logging.CRITICAL)

    if arguments.command == "prompt":
        print(JUDGE_PROMPT, end="")
        return 0
    if arguments.command == "texts":
        keys = _read_jsonl(arguments.build_dir / "key.jsonl")
        judged = _read_jsonl(arguments.build_dir / "judge_input.jsonl")
        texts = judge_texts(judged, keys, arguments.which)
        _write_jsonl(arguments.out, texts)
        print(f"{len(texts)} {arguments.which} texts written to {arguments.out}")
        return 0
    if arguments.command in ("freeze", "score"):
        if len(arguments.judge) != JUDGES:
            raise SystemExit(f"exactly {JUDGES} judges are needed")
        keys = [key for path in arguments.keys for key in _read_jsonl(path)]
        if len({key["id"] for key in keys}) != len(keys):
            raise SystemExit("an item id appears twice in the keys")
        if arguments.command == "freeze":
            old = [key["id"] for key in keys if unhashed(key)]
            if old:
                raise SystemExit(f"{', '.join(old)} records no text hashes; build it again")
            ids = [key["id"] for key in keys]
            judges = [read_verdicts(path, ids) for path in arguments.judge]
            frozen = {
                "eligible": [item_id for item_id in ids if majority_yes(item_id, judges)],
                "items": len(ids),
                "clusters": {key["id"]: key["cluster"] for key in keys},
                "queries": {key["id"]: key.get("query_sha256") for key in keys},
                "sections": {key["id"]: key.get("section_sha256") for key in keys},
            }
            arguments.out.write_text(json.dumps(frozen, indent=1) + "\n", encoding="utf-8")
            print(
                f"{len(frozen['eligible'])}/{len(ids)} items eligible, written to {arguments.out}"
            )
            return 0
        frozen = json.loads(arguments.eligible.read_text(encoding="utf-8"))
        drift = drifted(keys, frozen)
        if drift:
            raise SystemExit(f"the build no longer matches the frozen denominator: {drift}")
        eligible = frozen["eligible"]
        # Every excerpt is judged, eligible or not, so a verdict file matches its input exactly.
        excerpted = [key["id"] for key in keys if key["excerpt"] is not None]
        judges = [read_verdicts(path, excerpted) for path in arguments.judge]
        lines, passed = score(keys, eligible, judges)
        for path in [arguments.eligible, *arguments.judge]:
            lines.append(f"  sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}  {path}")
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
            lines, passed = latency(service, queries, arguments.rounds)
            print("\n".join(lines))
            print("PASS" if passed else "FAIL")
            return 0 if passed else 1
        return 0

    return _with_service(arguments, run)


if __name__ == "__main__":
    raise SystemExit(main())
