"""Reproducible retrieval-quality benchmark for the shipped search pipeline.

Indexes the frozen corpus in ``scripts/eval_data/corpus`` and scores ``search_docs``
against the labelled queries in ``scripts/eval_data/queries.json``:

    uv run python scripts/eval_retrieval.py                 # default embedder
    uv run python scripts/eval_retrieval.py --embedder bge-small
    uv run python scripts/eval_retrieval.py --show-misses   # list queries missed at Top-1
    uv run python scripts/eval_retrieval.py --update-baseline   # after an ACCEPTED change

Scores are compared with the frozen baseline in ``scripts/eval_data/baseline.json``
(accuracy in percentage points, latency in ms; latency is informational - it depends on
the machine - and never gates).

Exits non-zero when the held-out set regresses below the documented floor, so it can
gate a change to the parser, the embedder or the ranking. The held-out queries were
written before any tuning: tune on ``dev`` only, never on ``held_out``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import eval_cache

from markdown_memory.indexer import DEFAULT_EMBEDDER, Embedder, create_embedder
from markdown_memory.models import SearchResult
from markdown_memory.server import MarkdownMemoryService, ServerConfig

DATA = Path(__file__).parent / "eval_data"
CORPUS = DATA / "corpus"
BASELINE = DATA / "baseline.json"
ACCURACY_FIELDS = ("top1", "top3", "top5", "any_valid_top1")
PRIMARY_GRADE = 3
# Floors for the default embedder on the held-out set (measured: 85% / 97% / 100%).
FLOOR_PARAPHRASE_TOP1 = 0.80
FLOOR_PARAPHRASE_TOP5 = 0.90
FLOOR_IDENTIFIER_TOP1 = 1.00


@dataclass(slots=True, frozen=True)
class Scores:
    top1: float
    top3: float
    top5: float
    any_valid_top1: float
    ndcg5: float
    median_ms: float
    p95_ms: float
    misses: tuple[tuple[str, int | None], ...]


def _labels(result: SearchResult) -> set[str]:
    base = result.heading_path.split(" (Part ")[0]
    return {base, f"{Path(result.file_path).name}::{base}"}


def _dcg(grades: list[int]) -> float:
    return float(sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, 1)))


def evaluate(service: MarkdownMemoryService, cases: list[dict[str, object]]) -> Scores:
    ranks: list[int | None] = []
    valid_first = 0
    ndcg: list[float] = []
    latencies: list[float] = []
    for case in cases:
        query, expected = str(case["query"]), str(case["expected"])
        also_valid = case.get("also_valid", {})
        grades = {expected: PRIMARY_GRADE, **(also_valid if isinstance(also_valid, dict) else {})}
        started = time.perf_counter()
        results = service.search_docs(query, 20)
        latencies.append((time.perf_counter() - started) * 1000)
        # A label is credited once: the parts of one oversized section share a label, and
        # counting each would push nDCG above 1.
        credited: set[str] = set()
        relevance: list[int] = []
        for result in results:
            fresh = _labels(result) - credited
            relevance.append(max((grades.get(label, 0) for label in fresh), default=0))
            credited |= _labels(result)
        ranks.append(next((i for i, r in enumerate(results, 1) if expected in _labels(r)), None))
        valid_first += bool(relevance and relevance[0] > 0)
        ideal = sorted(grades.values(), reverse=True)[:5]
        ndcg.append(_dcg(relevance[:5]) / _dcg(ideal))
    total = len(cases)

    def within(k: int) -> float:
        return sum(rank is not None and rank <= k for rank in ranks) / total

    ordered = sorted(latencies)
    return Scores(
        top1=within(1),
        top3=within(3),
        top5=within(5),
        any_valid_top1=valid_first / total,
        ndcg5=statistics.mean(ndcg),
        median_ms=statistics.median(latencies),
        p95_ms=ordered[max(0, math.ceil(total * 0.95) - 1)],
        misses=tuple(
            (str(case["query"]), rank) for case, rank in zip(cases, ranks, strict=True) if rank != 1
        ),
    )


def print_deltas(preset: str, scores: dict[str, Scores]) -> None:
    """Compare ``scores`` with the frozen baseline for ``preset`` (if one is recorded)."""
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
        accuracy = "  ".join(
            f"{field} {(getattr(result, field) - before[field]) * 100:+.0f}pp"
            for field in ACCURACY_FIELDS
        )
        latency = result.median_ms - before["median_ms"]
        print(f"  {name:<22} {accuracy}  median {latency:+.0f}ms")


def update_baseline(preset: str, scores: dict[str, Scores]) -> None:
    recorded = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}
    recorded[preset] = {
        name: {k: round(v, 4) for k, v in asdict(result).items() if k != "misses"}
        for name, result in scores.items()
    }
    BASELINE.write_text(json.dumps(recorded, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nbaseline for {preset!r} written to {BASELINE}")


def _probes(corpus: Path) -> tuple[eval_cache.Probe, ...]:
    """Passages of the corpus, used to ask the cached index whose vectors it holds."""
    probes = eval_cache.probe_passages(corpus)
    if not probes:
        raise SystemExit(f"the corpus at {corpus} has no indexable passage")
    return probes


def open_service(
    arguments: argparse.Namespace, base: ServerConfig, probes: Sequence[eval_cache.Probe]
) -> tuple[MarkdownMemoryService, bool]:
    """Return a service over the cached index, building it only when it cannot be reused.

    The checks are deliberately redundant with the cache key: the key says what the inputs
    were, these say what the artifact is.
    """
    # The model is loaded before the key is computed, so the key sees the files that a
    # first run downloads - otherwise every machine's second run would find a different
    # key and rebuild the index it just built.
    embedder = create_embedder(arguments.embedder, cache_dir=base.model_cache_dir)
    embedder.embed_query("load the model")  # downloads it on a first run; `warm_up` is
    # not part of the Embedder protocol, and one query is enough to put the files on disk
    key = eval_cache.build_key(CORPUS, arguments.embedder, model_cache_dir=base.model_cache_dir)
    root = eval_cache.cache_root()
    workspace = root / key.digest
    workspace.mkdir(parents=True, exist_ok=True)
    eval_cache.prune(root, key.digest)
    db_path = workspace / "eval.db"
    fingerprint = eval_cache.parse_fingerprint(CORPUS)
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

    service = _service(arguments, base, db_path, embedder)
    if reason is None:
        try:
            eval_cache.check_integrity(service.db)
            eval_cache.check_vectors(service.db, service.embedder, probes)
            return service, False
        except eval_cache.StaleCacheError as stale:
            reason = str(stale)
            service.close()
            eval_cache.discard(db_path)
            service = _service(arguments, base, db_path, embedder)
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
        eval_cache.confirm_stable(CORPUS, fingerprint)
    except eval_cache.StaleCacheError as unstable:
        service.close()
        eval_cache.discard(db_path)
        raise SystemExit(str(unstable)) from unstable
    eval_cache.record(db_path, key, fingerprint)
    return service, True


def _service(
    arguments: argparse.Namespace, base: ServerConfig, db_path: Path, embedder: Embedder
) -> MarkdownMemoryService:
    return MarkdownMemoryService(
        ServerConfig(
            db_path=db_path,
            docs_dir=CORPUS,
            embedder=arguments.embedder,
            model_cache_dir=base.model_cache_dir,
        ),
        embedder=embedder,
    )


def run(
    queries: dict[str, dict[str, list[dict[str, object]]]],
    arguments: argparse.Namespace,
    base: ServerConfig,
    probes: Sequence[eval_cache.Probe],
) -> dict[str, Scores]:
    service, built = open_service(arguments, base, probes)
    try:
        print(f"embedder: {service.embedder.model_name}")
        if not built:
            print("index: reused from cache (fingerprint, integrity and vectors verified)")
        service.search_docs("warm up", 1)
        header = f"{'set':<22} {'n':>3}  Top-1  Top-3  Top-5  any-valid@1  nDCG@5  median   p95"
        print("\n" + header + "\n" + "-" * len(header))
        scores: dict[str, Scores] = {}
        for split in ("dev", "held_out"):
            for kind in ("paraphrase", "identifier"):
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
        "--rebuild",
        action="store_true",
        help="discard the cached index and build it again (about a minute on this corpus)",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="record these scores as the new frozen baseline for this embedder",
    )
    arguments = parser.parse_args()
    logging.disable(logging.CRITICAL)

    queries = json.loads((DATA / "queries.json").read_text(encoding="utf-8"))
    base = ServerConfig.from_env()
    probes = _probes(CORPUS)
    try:
        with eval_cache.lock(eval_cache.cache_root()):
            scores = run(queries, arguments, base, probes)
    except eval_cache.BusyError as busy:
        print(f"REFUSING TO RUN: {busy}", file=sys.stderr)
        return 1

    preset = arguments.embedder
    print_deltas(preset, scores)
    if preset != DEFAULT_EMBEDDER:
        if arguments.update_baseline:
            update_baseline(preset, scores)
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
        update_baseline(preset, scores)
    print("\nOK: held-out floors met (Top-1 >= 80%, Top-5 >= 90%, identifiers 100%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
