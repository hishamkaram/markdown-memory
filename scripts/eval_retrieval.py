"""Reproducible retrieval-quality benchmark for the shipped search pipeline.

Indexes the frozen corpus in ``scripts/eval_data/corpus`` and scores ``search_docs``
against the labelled queries in ``scripts/eval_data/queries.json``:

    uv run python scripts/eval_retrieval.py                 # default embedder
    uv run python scripts/eval_retrieval.py --embedder bge-small
    uv run python scripts/eval_retrieval.py --show-misses   # list queries missed at Top-1
    uv run python scripts/eval_retrieval.py --show-costs    # what each default call costs
    uv run python scripts/eval_retrieval.py --update-baseline   # after an ACCEPTED change
    uv run python scripts/eval_retrieval.py --corpus v2     # real upstream docs, report-only
    uv run python scripts/eval_retrieval.py --split dev --record dev.json   # tune: dev only

Scores are compared with the frozen baseline in ``scripts/eval_data/baseline.json``
(accuracy in percentage points, latency in ms; latency is informational - it depends on
the machine - and never gates).

Exits non-zero when the held-out set regresses below the documented floor, so it can
gate a change to the parser, the embedder or the ranking. Beside accuracy it reports what the
default `search_docs` call costs in estimated tokens against the section that answers it;
that is informational and never gates, never enters the baseline. So is the no-answer stratum:
queries the corpus cannot answer, scored by how often the default call abstains. The held-out
queries were written before any tuning: tune on ``dev`` only, never on ``held_out``.

``--corpus v2`` scores the pinned upstream documentation in ``scripts/eval_data/corpus_v2``
against ``queries_v2.json`` instead (#75). It has its own baseline entry and no floors: it
reports, and the gate stays the v1 corpus until v2's numbers have held across runs.

``--split dev`` scores, validates and shows the dev queries only, and ``--record`` keeps every
query's outcome so that ``scripts/eval_compare.py`` can hold two revisions to a per-query
no-regression rule (#94). A run that scores less than both splits, or labels from another
``--queries`` file, is report-only: it checks no floor and cannot write the baseline.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import eval_cache

import markdown_memory
from markdown_memory.config import ServerConfig
from markdown_memory.embedders import DEFAULT_EMBEDDER, Embedder, create_embedder
from markdown_memory.models import OutlineNode, SearchResult, estimate_tokens
from markdown_memory.server import MarkdownMemoryService, create_server

DATA = Path(__file__).parent / "eval_data"
BASELINE = DATA / "baseline.json"
ACCURACY_FIELDS = ("top1", "top3", "top5", "any_valid_top1")
PRIMARY_GRADE = 3
# Floors for the default embedder on the held-out set (measured: 88% / 97% / 100%).
FLOOR_PARAPHRASE_TOP1 = 0.80
FLOOR_PARAPHRASE_TOP5 = 0.90
FLOOR_IDENTIFIER_TOP1 = 1.00
NO_ANSWER_SHAPES = ("identifier", "question")
# Answerable strata. `mixed` - a question naming an identifier - exists only in corpus_v2 (#78):
# a query file without it is scored as before.
ANSWERABLE = ("paraphrase", "identifier", "mixed")
OPTIONAL_KINDS = ("mixed",)
SPLITS = ("dev", "held_out")
RECORD_SCHEMA = 1


@dataclass(slots=True, frozen=True)
class Corpus:
    """One labelled corpus: its documents, its queries, and where its index is cached."""

    name: str
    root: Path
    queries: Path

    def cache(self) -> Path:
        """v1 keeps the directory it always had; any other corpus gets one of its own.

        Each directory keeps its most recently used indexes (`eval_cache.prune`), so two
        corpora sharing one would evict each other's.
        """
        root = eval_cache.cache_root()
        return root if self.name == "v1" else root.with_name(f"eval-{self.name}")

    def baseline_key(self, preset: str) -> str:
        return preset if self.name == "v1" else f"{preset}@{self.name}"


CORPUS = DATA / "corpus"
CORPUS_V2 = DATA / "corpus_v2"
CORPUS_NAMES = ("v1", "v2")


def corpora() -> dict[str, Corpus]:
    """Built when asked, not at import: a caller that points `CORPUS` elsewhere is honoured."""
    return {
        "v1": Corpus("v1", CORPUS, DATA / "queries.json"),
        "v2": Corpus("v2", CORPUS_V2, DATA / "queries_v2.json"),
    }


@dataclass(slots=True, frozen=True)
class Outcome:
    """How one labelled query was answered: what `--record` keeps and the comparison reads."""

    query: str
    rank: int | None  # of the `expected` section among the first 20 results
    any_valid: bool  # the first result carries any grade
    ndcg5: float
    top: tuple[str, ...]  # the first five results' labels, as `path::heading`


@dataclass(slots=True, frozen=True)
class Scores:
    top1: float
    top3: float
    top5: float
    any_valid_top1: float
    ndcg5: float
    median_ms: float
    p95_ms: float
    cases: tuple[Outcome, ...] = field(default=())

    @property
    def misses(self) -> tuple[tuple[str, int | None], ...]:
        return tuple((case.query, case.rank) for case in self.cases if case.rank != 1)


@dataclass(slots=True, frozen=True)
class Answer:
    """The section a case's `expected` label names, and what reading it costs."""

    file_path: str
    heading_path: str
    tokens: int


@dataclass(slots=True, frozen=True)
class Cost:
    """What the default `search_docs` call cost for each case of one set, in order."""

    queries: tuple[str, ...]
    payloads: tuple[int, ...]
    answers: tuple[int, ...]

    @property
    def ratios(self) -> list[float]:
        return [
            payload / answer for payload, answer in zip(self.payloads, self.answers, strict=True)
        ]


@dataclass(slots=True, frozen=True)
class NoAnswer:
    """How the default call answered one set of queries the corpus cannot answer, in order."""

    queries: tuple[str, ...]
    hits: tuple[int, ...]
    no_match: tuple[bool, ...]
    payloads: tuple[int, ...]  # what each call cost, its keyword_message included


def _base(heading_path: str) -> str:
    return heading_path.split(" (Part ")[0]


def _names(file_path: str, root: str) -> tuple[str, str]:
    """What a label may name a file by: its name, or its whole path from the corpus root.

    `configuration.md::X` names a file by its name; `gh/docs/README.md::X` by its path -
    corpus_v2 holds several README.md files. Nothing in between: `docs/README.md` would name
    every `docs/README.md` at any depth.
    """
    path = Path(file_path)
    return path.name, path.relative_to(root).as_posix()


def _labels(result: SearchResult, root: str) -> set[str]:
    """The bare heading path, and that path qualified by each name of its file."""
    base = _base(result.heading_path)
    return {base} | {f"{name}::{base}" for name in _names(result.file_path, root)}


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def _outline_paths(nodes: Sequence[OutlineNode]) -> Iterator[str]:
    for node in nodes:
        yield _base(node.heading_path)
        yield from _outline_paths(node.children)


def _kinds(queries: Mapping[str, Mapping[str, object]], split: str) -> tuple[str, ...]:
    """The answerable strata ``split`` holds: every required one, and an optional one if present."""
    return tuple(k for k in ANSWERABLE if k not in OPTIONAL_KINDS or k in queries[split])


def resolve_answers(
    service: MarkdownMemoryService,
    queries: dict[str, dict[str, list[dict[str, object]]]],
    splits: Sequence[str] = SPLITS,
) -> dict[str, Answer]:
    """Every `expected` label, resolved to the one section it names - before anything is searched.

    Part of loading the fixture, like parsing it: a label that names no section, or more than
    one, is a broken fixture, and scoring it would score nothing.
    `file.md::path` (or `dir/file.md::path`) names the file by the end of its path; a bare path
    must be unique across the corpus.
    """
    owners: dict[str, list[str]] = {}
    for document in service.list_documents():
        for path in _outline_paths(service.get_document_outline(document.file_path)):
            owners.setdefault(path, []).append(document.file_path)
    answers: dict[str, Answer] = {}
    for split in splits:  # only the splits scored: a dev run learns nothing from held-out
        for kind in _kinds(queries, split):  # no-answer cases have nothing to resolve
            for case in queries[split][kind]:
                label = str(case["expected"])
                if label in answers:
                    continue
                name, qualified, path = label.rpartition("::")
                found = [
                    file_path
                    for file_path in owners.get(path, [])
                    if not qualified or name in _names(file_path, service.root)
                ]
                if len(found) != 1:
                    raise SystemExit(f"fixture: {label!r} names {len(found)} sections, not one")
                # Never zero: a section's text starts with its own heading line.
                tokens = estimate_tokens(service.read_section(found[0], path))
                answers[label] = Answer(found[0], path, tokens)
    return answers


def _tokens(text: str) -> list[str]:
    return re.findall(r"[0-9a-z]+", text.lower())


def check_no_answer(
    queries: dict[str, dict[str, list[dict[str, object]]]],
    corpus: Path,
    splits: Sequence[str] = SPLITS,
) -> None:
    """Every no-answer case is well formed and absent from the corpus - before anything is searched.

    Absent as the keyword index would look for it: the query's tokens as a consecutive run, so
    `--enable-tls` is found in a corpus that says `enable_tls`, while a longer identifier that
    merely contains it is not. Read only when there are cases to check.
    """
    cases = [case for split in splits for case in queries[split]["no_answer"]]
    if not cases:
        return
    documents = [_tokens(path.read_text(encoding="utf-8")) for path in sorted(corpus.rglob("*.md"))]
    for case in cases:
        phrase = _tokens(str(case.get("query", "")))
        if "expected" in case or case.get("shape") not in NO_ANSWER_SHAPES or not phrase:
            raise SystemExit(f"fixture: malformed no-answer case {case!r}")
        width = len(phrase)
        for words in documents:
            if any(words[i : i + width] == phrase for i in range(len(words) - width + 1)):
                raise SystemExit(f"fixture: no-answer query {case['query']!r} is in the corpus")


def measure_no_answer(
    service: MarkdownMemoryService,
    queries: dict[str, dict[str, list[dict[str, object]]]],
    splits: Sequence[str] = SPLITS,
) -> dict[str, NoAnswer]:
    """What the default call returns for each no-answer case, per split and shape.

    On `search_page`, the layer `evaluate` reads and the tool serialises: an abstention made
    there is what an agent receives.
    """
    measured: dict[str, NoAnswer] = {}
    for split in splits:
        for shape in NO_ANSWER_SHAPES:
            cases = [case for case in queries[split]["no_answer"] if case["shape"] == shape]
            pages = [service.search_page(str(case["query"]), 5) for case in cases]
            measured[f"{split}/{shape}"] = NoAnswer(
                queries=tuple(str(case["query"]) for case in cases),
                hits=tuple(len(page.results) for page in pages),
                no_match=tuple(page.keyword_match == "no_match" for page in pages),
                payloads=_payloads(service, [str(case["query"]) for case in cases]),
            )
    return measured


def print_no_answer(measured: dict[str, NoAnswer], *, show_misses: bool) -> None:
    print("\nno-answer stratum (informational; abstention never gates; n is small, low power)")
    header = f"{'set':<22} {'n':>3}  abstained  no_match  payload"
    print(header + "\n" + "-" * len(header))
    for split in dict.fromkeys(name.split("/")[0] for name in measured):
        rows = [(shape, measured[f"{split}/{shape}"]) for shape in NO_ANSWER_SHAPES]
        rows.append(
            (
                "all",
                NoAnswer(
                    queries=tuple(q for _, part in rows for q in part.queries),
                    hits=tuple(h for _, part in rows for h in part.hits),
                    no_match=tuple(m for _, part in rows for m in part.no_match),
                    payloads=tuple(t for _, part in rows for t in part.payloads),
                ),
            )
        )
        for shape, result in rows:
            if not result.queries:
                continue  # a rate over nothing is not zero
            n = len(result.queries)
            abstained = sum(hits == 0 for hits in result.hits) / n
            no_match = sum(result.no_match) / n
            payload = statistics.median(result.payloads)
            row = f"{split + ' ' + shape:<22} {n:>3}  {abstained:9.0%}  {no_match:8.0%}"
            print(f"{row}  {payload:7.0f}")
    if show_misses:
        for name, result in measured.items():
            split, shape = name.split("/")
            for query, hits in zip(result.queries, result.hits, strict=True):
                if hits:
                    print(f"  miss [{split}/no_answer {shape}] {hits} hits: {query}")


def _payloads(service: MarkdownMemoryService, queries: Sequence[str]) -> tuple[int, ...]:
    """The default call's estimated cost for each query, measured on the text the server sends.

    That text is what a client puts in the model's context, so it is measured as sent - escaping
    included - rather than re-serialised here. Calls go one at a time: they share one service
    and one SQLite connection.
    """
    server = create_server(service=service)

    async def measure() -> tuple[int, ...]:
        payloads: list[int] = []
        for query in queries:
            outcome = await server.call_tool("search_docs", {"query": query})
            text = getattr(outcome.content[0], "text", None)
            if outcome.is_error or not isinstance(text, str):
                raise RuntimeError(f"search_docs failed for {query!r}")
            payloads.append(estimate_tokens(text))
        return tuple(payloads)

    return asyncio.run(measure())


def measure_costs(
    service: MarkdownMemoryService,
    queries: dict[str, dict[str, list[dict[str, object]]]],
    answers: dict[str, Answer],
    splits: Sequence[str] = SPLITS,
) -> dict[str, Cost]:
    """What the default call cost against the section that answers it, per split and kind."""
    costs: dict[str, Cost] = {}
    for split in splits:
        for kind in _kinds(queries, split):
            cases = queries[split][kind]
            costs[f"{split}/{kind}"] = Cost(
                queries=tuple(str(case["query"]) for case in cases),
                payloads=_payloads(service, [str(case["query"]) for case in cases]),
                answers=tuple(answers[str(case["expected"])].tokens for case in cases),
            )
    return costs


def print_costs(costs: dict[str, Cost], *, per_query: bool) -> None:
    print("\ndefault search_docs call, estimated text-payload tokens (informational)")
    header = f"{'set':<22} {'n':>3}  payload    p95  answer  payload/answer"
    print(header + "\n" + "-" * len(header))
    for name, cost in costs.items():
        print(
            f"{name.replace('/', ' '):<22} {len(cost.payloads):>3}  "
            f"{statistics.median(cost.payloads):7.0f}  {_p95(cost.payloads):5.0f}  "
            f"{statistics.median(cost.answers):6.0f}  {statistics.median(cost.ratios):13.1f}x"
        )
    if per_query:
        for name, cost in costs.items():
            for query, payload, answer, ratio in zip(
                cost.queries, cost.payloads, cost.answers, cost.ratios, strict=True
            ):
                print(f"  cost [{name}] {payload:5d} / {answer:4d} = {ratio:5.1f}x  {query}")


def _dcg(grades: list[int]) -> float:
    return float(sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, 1)))


def _grades(case: dict[str, object]) -> dict[str, int]:
    also_valid = case.get("also_valid", {})
    extra = also_valid if isinstance(also_valid, dict) else {}
    return {str(case["expected"]): PRIMARY_GRADE, **{str(k): int(v) for k, v in extra.items()}}


def _grade(results: Sequence[SearchResult], case: dict[str, object], root: str) -> Outcome:
    """Score one query's ranked results against its labels."""
    expected, grades = str(case["expected"]), _grades(case)
    # A label is credited once: the parts of one oversized section share a label, and
    # counting each would push nDCG above 1.
    credited: set[str] = set()
    relevance: list[int] = []
    for result in results:
        labels = _labels(result, root)
        relevance.append(max((grades.get(label, 0) for label in labels - credited), default=0))
        credited |= labels
    ideal = sorted(grades.values(), reverse=True)[:5]
    return Outcome(
        query=str(case["query"]),
        rank=next((i for i, r in enumerate(results, 1) if expected in _labels(r, root)), None),
        any_valid=bool(relevance and relevance[0] > 0),
        ndcg5=_dcg(relevance[:5]) / _dcg(ideal),
        top=tuple(
            f"{Path(r.file_path).relative_to(root).as_posix()}::{r.heading_path}"
            for r in results[:5]
        ),
    )


def evaluate(service: MarkdownMemoryService, cases: list[dict[str, object]]) -> Scores:
    outcomes: list[Outcome] = []
    latencies: list[float] = []
    for case in cases:
        started = time.perf_counter()
        results = service.search_docs(str(case["query"]), 20)
        latencies.append((time.perf_counter() - started) * 1000)
        outcomes.append(_grade(results, case, service.root))
    total = len(cases)

    def within(k: int) -> float:
        return sum(case.rank is not None and case.rank <= k for case in outcomes) / total

    return Scores(
        top1=within(1),
        top3=within(3),
        top5=within(5),
        any_valid_top1=sum(case.any_valid for case in outcomes) / total,
        ndcg5=statistics.mean(case.ndcg5 for case in outcomes),
        median_ms=statistics.median(latencies),
        p95_ms=_p95(latencies),
        cases=tuple(outcomes),
    )


def cases_sha256(queries: Mapping[str, Mapping[str, object]], name: str) -> str:
    """The fingerprint of the cases set ``name`` (``split/kind``) holds in ``queries``."""
    split, kind = name.split("/")
    return _sha256(json.dumps(queries[split][kind], sort_keys=True).encode())


def print_deltas(
    preset: str, scores: dict[str, Scores], queries: Mapping[str, Mapping[str, object]]
) -> None:
    """Compare ``scores`` with the frozen baseline for ``preset`` (if one is recorded).

    Only a set scored on the same cases has a delta (#116): #105 replaced two held-out sets
    in place, and their baseline, recorded on the retired queries, read as a 30-point drop.
    """
    recorded = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}
    baseline = recorded.get(preset)
    if baseline is None:
        print(f"\n(no baseline recorded for {preset!r}; run with --update-baseline)")
        return
    print("\ndelta vs frozen baseline (percentage points; latency in ms, informational)")
    for name, result in scores.items():
        before = baseline.get(name)
        if before is None:
            continue
        scored_on = before.get("cases_sha256")
        if scored_on is None:
            print(f"  {name:<22} no delta: the baseline does not say which queries it scored")
            continue
        if scored_on != cases_sha256(queries, name):
            print(f"  {name:<22} no delta: the baseline was scored on other queries (re-record it)")
            continue
        accuracy = "  ".join(
            f"{field} {(getattr(result, field) - before[field]) * 100:+.0f}pp"
            for field in ACCURACY_FIELDS
        )
        latency = result.median_ms - before["median_ms"]
        print(f"  {name:<22} {accuracy}  median {latency:+.0f}ms")


def update_baseline(
    preset: str, scores: dict[str, Scores], queries: Mapping[str, Mapping[str, object]]
) -> None:
    recorded = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}
    recorded[preset] = {
        name: {
            **{k: round(v, 4) for k, v in asdict(result).items() if k != "cases"},
            "cases_sha256": cases_sha256(queries, name),
        }
        for name, result in scores.items()
    }
    BASELINE.write_text(json.dumps(recorded, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nbaseline for {preset!r} written to {BASELINE}")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _code_identity() -> dict[str, object]:
    """The package under test as imported - `PYTHONPATH` may name another revision's `src`.

    The commit alone cannot tell two uncommitted states of `search.py` apart; the digest of
    every module's bytes can.
    """
    package = Path(markdown_memory.__file__).resolve().parent
    digest = hashlib.sha256()
    for module in sorted(package.glob("*.py")):
        digest.update(module.name.encode("utf-8") + b"\0" + module.read_bytes())
    revision = subprocess.run(
        ["git", "-C", str(package), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return {
        "package": str(package),
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "sha256": digest.hexdigest(),
    }


def record_cases(
    service: MarkdownMemoryService,
    queries: dict[str, dict[str, list[dict[str, object]]]],
    splits: Sequence[str],
    scores: dict[str, Scores],
) -> dict[str, dict[str, object]]:
    """Every scored case's outcome, plus what the default page returned for it.

    The page is what an agent receives, so a query whose page empties is a loss even when no
    label ranked before: `scripts/eval_compare.py` reads `hits` for exactly that.
    """
    cases: dict[str, dict[str, object]] = {}

    def put(key: str, value: dict[str, object]) -> None:
        if key in cases:
            raise ValueError(f"two cases share the key {key!r}; a record must name each once")
        cases[key] = value

    for split in splits:
        for kind in _kinds(queries, split):
            labelled = queries[split][kind]
            outcomes = scores[f"{split}/{kind}"].cases
            for case, outcome in zip(labelled, outcomes, strict=True):
                if not math.isfinite(outcome.ndcg5):
                    raise ValueError(f"nDCG@5 of {outcome.query!r} is {outcome.ndcg5}")
                page = service.search_page(outcome.query, 5)
                put(f"{split}|{kind}|{outcome.query}", {
                    "expected": str(case["expected"]),
                    "also_valid": {k: v for k, v in _grades(case).items() if k != case["expected"]},
                    "rank": outcome.rank, "any_valid": outcome.any_valid, "ndcg5": outcome.ndcg5,
                    "top": list(outcome.top),
                    "hits": len(page.results), "keyword_match": page.keyword_match,
                })  # fmt: skip
        for case in queries[split]["no_answer"]:
            page = service.search_page(str(case["query"]), 5)
            put(f"{split}|no_answer|{case['query']}", {
                "shape": str(case["shape"]),
                "hits": len(page.results), "keyword_match": page.keyword_match,
            })  # fmt: skip
    return cases


def write_record(path: Path, record: dict[str, object]) -> None:
    """Write ``record`` whole or not at all: a failed run leaves what was there before."""
    text = json.dumps(record, indent=1, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _probes(corpus: Path) -> tuple[eval_cache.Probe, ...]:
    """Passages of the corpus, used to ask the cached index whose vectors it holds."""
    probes = eval_cache.probe_passages(corpus)
    if not probes:
        raise SystemExit(f"the corpus at {corpus} has no indexable passage")
    return probes


def open_service(
    arguments: argparse.Namespace,
    base: ServerConfig,
    probes: Sequence[eval_cache.Probe],
    corpus: Corpus | None = None,
) -> tuple[MarkdownMemoryService, bool]:
    """Return a service over the cached index, building it only when it cannot be reused.

    The checks are deliberately redundant with the cache key: the key says what the inputs
    were, these say what the artifact is.
    """
    # The model is loaded before the key is computed, so the key sees the files that a
    # first run downloads - otherwise every machine's second run would find a different
    # key and rebuild the index it just built.
    embedder = create_embedder(arguments.embedder, cache_dir=base.model_cache_dir)
    corpus = corpus or corpora()["v1"]
    embedder.embed_query("load the model")  # downloads it on a first run; `warm_up` is
    # not part of the Embedder protocol, and one query is enough to put the files on disk
    key = eval_cache.build_key(
        corpus.root, arguments.embedder, model_cache_dir=base.model_cache_dir
    )
    root = corpus.cache()
    workspace = root / key.digest
    workspace.mkdir(parents=True, exist_ok=True)
    eval_cache.drop_incomplete(root, key.digest)
    db_path = workspace / "eval.db"
    fingerprint = eval_cache.parse_fingerprint(corpus.root)
    if arguments.rebuild:
        eval_cache.discard(db_path)
        reason: str | None = "--rebuild"
    else:
        try:
            eval_cache.validate(db_path, fingerprint, key)
            reason = None
        except eval_cache.StaleCacheError as stale:
            eval_cache.discard(db_path)
            reason = str(stale)

    service = _service(arguments, base, db_path, embedder, corpus.root)
    if reason is None:
        try:
            eval_cache.check_integrity(service.db)
            eval_cache.check_vectors(service.db, service.embedder, probes)
            eval_cache.prune(root, key.digest)  # only on success: a failure evicts nothing
            return service, False
        except eval_cache.StaleCacheError as stale:
            reason = str(stale)
            service.close()
            eval_cache.discard(db_path)
            service = _service(arguments, base, db_path, embedder, corpus.root)
    print(f"index: building ({reason})")
    report = service.index_directory()
    print(report.summary())
    if report.errors:
        # Scoring a partial index reports a number for a system that was never built,
        # and caching it would keep reporting it.
        service.close()
        eval_cache.discard(db_path)
        raise SystemExit(f"indexing failed for {len(report.errors)} file(s); not scoring this run")
    eval_cache.check_vectors(service.db, service.embedder, probes)
    try:
        eval_cache.confirm_stable(corpus.root, fingerprint)
    except eval_cache.StaleCacheError as unstable:
        service.close()
        eval_cache.discard(db_path)
        raise SystemExit(str(unstable)) from unstable
    eval_cache.record(db_path, key, fingerprint)
    eval_cache.prune(root, key.digest)
    return service, True


def _service(
    arguments: argparse.Namespace,
    base: ServerConfig,
    db_path: Path,
    embedder: Embedder,
    docs_dir: Path | None = None,
) -> MarkdownMemoryService:
    """The service an evaluation indexes with: four settings, each already in the cache key.

    The database, corpus, preset and model cache are keyed as data; every other setting is
    `ServerConfig`'s default, whose source (`config.py`) is keyed. Passing anything else here
    - `exclude`, `gitignore` - changes what is indexed without changing the key (#99), so it
    has to be keyed in the same change.
    """
    return MarkdownMemoryService(
        ServerConfig(
            db_path=db_path,
            docs_dir=CORPUS if docs_dir is None else docs_dir,
            embedder=arguments.embedder,
            model_cache_dir=base.model_cache_dir,
        ),
        embedder=embedder,
    )


def _splits(arguments: argparse.Namespace) -> tuple[str, ...]:
    split = getattr(arguments, "split", "all")
    return SPLITS if split == "all" else (split,)


def run(
    queries: dict[str, dict[str, list[dict[str, object]]]],
    arguments: argparse.Namespace,
    base: ServerConfig,
    probes: Sequence[eval_cache.Probe],
    corpus: Corpus | None = None,
) -> dict[str, Scores]:
    corpus = corpus or corpora()["v1"]
    splits = _splits(arguments)
    service, built = open_service(arguments, base, probes, corpus)
    try:
        answers = resolve_answers(service, queries, splits)
        check_no_answer(queries, corpus.root, splits)
        print(f"embedder: {service.embedder.model_name}")
        if not built:
            print("index: reused from cache (fingerprint, integrity and vectors verified)")
        service.search_docs("warm up", 1)
        header = f"{'set':<22} {'n':>3}  Top-1  Top-3  Top-5  any-valid@1  nDCG@5  median   p95"
        print("\n" + header + "\n" + "-" * len(header))
        scores: dict[str, Scores] = {}
        for split in splits:
            for kind in _kinds(queries, split):
                cases = queries[split][kind]
                result = scores[f"{split}/{kind}"] = evaluate(service, cases)
                print(
                    f"{split + ' ' + kind:<22} {len(cases):>3}  {result.top1:5.0%}  "
                    f"{result.top3:5.0%}  {result.top5:5.0%}  {result.any_valid_top1:11.0%}  "
                    f"{result.ndcg5:6.2f}  {result.median_ms:4.0f}ms  {result.p95_ms:4.0f}ms"
                )
        if arguments.show_misses:
            for name, result in scores.items():
                for query, rank in result.misses:
                    print(f"  miss [{name}] rank={rank}: {query}")
        destination = getattr(arguments, "record", None)
        if destination is not None:
            # Outside the informational guards below: a record that is missing a pass would
            # compare as if that pass had nothing to say.
            write_record(destination, {
                "schema": RECORD_SCHEMA, "corpus": corpus.name, "preset": arguments.embedder,
                "splits": list(splits), "queries": str(arguments.queries_path),
                "queries_sha256": _sha256(arguments.queries_path.read_bytes()),
                "corpus_sha256": eval_cache.corpus_digest(corpus.root),
                "parse_fingerprint": eval_cache.parse_fingerprint(corpus.root),
                "code": _code_identity(), "evaluator": _sha256(Path(__file__).read_bytes()),
                "counts": {
                    f"{split}|{kind}": len(queries[split][kind])
                    for split in splits for kind in (*_kinds(queries, split), "no_answer")
                },
                "cases": record_cases(service, queries, splits, scores),
            })  # fmt: skip
            print(f"\nrecord: {destination}")
        try:
            print_no_answer(
                measure_no_answer(service, queries, splits), show_misses=arguments.show_misses
            )
        except Exception as exc:  # informational, like the cost pass below
            print(f"\nno-answer stratum: not measured ({exc})")
        try:
            print_costs(
                measure_costs(service, queries, answers, splits), per_query=arguments.show_costs
            )
        except Exception as exc:  # informational: it must never decide how the run ends
            print(f"\ncost: not measured ({exc})")
        return scores
    finally:
        service.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--embedder",
        default=DEFAULT_EMBEDDER,
        choices=("embeddinggemma", "bge-small"),
        help="model to score; MARKDOWN_MEMORY_EMBEDDER is deliberately ignored so that the "
        "gate always judges the default model unless this flag says otherwise",
    )
    parser.add_argument("--show-misses", action="store_true", help="list queries missed at Top-1")
    parser.add_argument(
        "--show-costs", action="store_true", help="list what each query's default call costs"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="discard the cached index and build it again (about a minute on this corpus)",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="record these scores as the new frozen baseline for this embedder",
    )
    parser.add_argument(
        "--corpus",
        default="v1",
        choices=CORPUS_NAMES,
        help="v1 (default) is the gate; v2 is the pinned upstream documentation, report-only",
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=("all", *SPLITS),
        help="score only these queries; anything but 'all' is report-only (tune on dev)",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        help="label file to score instead of the corpus's own (report-only), e.g. the labels "
        "a base revision's sections have",
    )
    parser.add_argument(
        "--record",
        type=Path,
        help="write every case's outcome here as JSON, for scripts/eval_compare.py",
    )
    arguments = parser.parse_args()
    logging.disable(logging.CRITICAL)

    corpus = corpora()[arguments.corpus]
    report_only = [
        reason
        for applies, reason in (
            (arguments.split != "all", f"--split {arguments.split}"),
            (arguments.queries is not None, f"--queries {arguments.queries}"),
        )
        if applies
    ]
    if report_only and arguments.update_baseline:
        # The baseline holds all four sets, scored against the corpus's own labels; neither
        # a partial run nor another label file may stand in for it.
        parser.error(f"--update-baseline cannot record a report-only run ({report_only[0]})")
    arguments.queries_path = (arguments.queries or corpus.queries).resolve()
    queries = json.loads(arguments.queries_path.read_text(encoding="utf-8"))
    base = ServerConfig.from_env()
    probes = _probes(corpus.root)
    try:
        # One lock for every corpus: a second evaluation on the same CPU moves the latency
        # either one reports, whichever corpus it scores.
        with eval_cache.lock(eval_cache.cache_root()):
            scores = run(queries, arguments, base, probes, corpus)
    except eval_cache.BusyError as busy:
        print(f"REFUSING TO RUN: {busy}", file=sys.stderr)
        return 1

    preset = arguments.embedder
    print_deltas(corpus.baseline_key(preset), scores, queries)
    if report_only:
        print(f"\nGATES NOT CHECKED: report-only run ({'; '.join(report_only)})")
        return 0
    if corpus.name != "v1":
        if arguments.update_baseline:
            update_baseline(corpus.baseline_key(preset), scores, queries)
        print(f"\nREPORT ONLY: corpus {corpus.name!r} has no floors; the gate is corpus 'v1'")
        return 0
    if preset != DEFAULT_EMBEDDER:
        if arguments.update_baseline:
            update_baseline(preset, scores, queries)
        # The floors are calibrated for the default embedder. Say so: a silent exit 0
        # would read as "gates passed".
        print(f"\nGATES NOT CHECKED: floors apply to {DEFAULT_EMBEDDER!r} only, not {preset!r}")
        return 0
    held_out, identifiers = scores["held_out/paraphrase"], scores["held_out/identifier"]
    failures = [
        message
        for ok, message in (
            (held_out.top1 >= FLOOR_PARAPHRASE_TOP1, f"held-out Top-1 {held_out.top1:.0%}"),
            (held_out.top5 >= FLOOR_PARAPHRASE_TOP5, f"held-out Top-5 {held_out.top5:.0%}"),
            (
                identifiers.top1 >= FLOOR_IDENTIFIER_TOP1
                and scores["dev/identifier"].top1 >= FLOOR_IDENTIFIER_TOP1,
                "identifier Top-1 below 100%",
            ),
        )
        if not ok
    ]
    if failures:
        print("\nREGRESSION: " + "; ".join(failures))
        if arguments.update_baseline:
            # Recording these numbers would make the regression the reference the next run
            # measures against, and the delta would then read +0pp.
            print("not recording a baseline for a run that fails the floors")
        return 1
    if arguments.update_baseline:
        update_baseline(preset, scores, queries)
    print("\nOK: held-out floors met (Top-1 >= 80%, Top-5 >= 90%, identifiers 100%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
