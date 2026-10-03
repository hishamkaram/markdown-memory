#!/usr/bin/env python3
"""What does searching an index from another CPU actually cost? Measure it on the real pipeline.

#21 measured the compute paths and found four of them. The same text embeds up to 1.6e-3
cosine apart between x86-64 and arm64, and `weights_revision` names the revision and the
graph but not the arithmetic, so an index copied between two machines is searched with
vectors from one kernel and queries from another while reporting itself healthy.

What #21 could *not* measure is what that costs an answer. Its churn figures come from
forcing the fp32 kernel on one machine, which moves a vector 2.1e-4 to 6.1e-4 - the size
of the VNNI difference, and seven times smaller than the arm64 one - and it ranked by
vector distance alone. Production does not: it fuses BM25 with vector ranking through RRF
and takes a max over per-passage vectors, all of which should damp a disagreement that
vector ranking alone would show. So the number that decides issue #23 is this one.

**It has now been produced, and it is zero.** Replaying the queries of an Apple M2 Pro
(1.645e-3 from the reference) and a Neoverse-N2 (1.626e-3) against an index built on
x86_64 moved **no top-1 result** on any of the 86 labelled queries, left the labelled
section in the top five in every cell, and changed neither count. What moved is the tail:
the top-five list differed on 22 of 86 queries from Apple Silicon and 19 of 86 from
Neoverse-N2, never at the first position and never at the labelled section. Replaying
x86_64's own export against the x86_64 index gives zero everywhere, which is the control
that says the harness reports no difference where there is none. #23 is closed on that.

Re-run it rather than trusting the numbers: a different corpus, a further-out compute path
or a new onnxruntime could all move them, and this instrument exists so that costs one
dispatch rather than an argument.

The trick that makes it cheap: a query is one vector. Rather than shipping an index
between machines, each host exports the 86 labelled queries as it embeds them - about
1.6 MB - and this script replays those vectors against an index built here, through
`search_docs` itself. Same corpus, same ranking, same fusion; only the arithmetic behind
the query vector changes.

    uv run python scripts/cross_host_churn.py --export vectors-$(uname -m).json
    gh run download <run-id>            # one directory per runner
    uv run python scripts/cross_host_churn.py --score vectors-*/*.json

An export also carries the golden probe's distance from the committed reference, which is
what says *which* compute path produced it - the CPU model does not, since the same chip
appeared in CI both with and without the VNNI flag.

One direction, stated: the index is built here and the queries come from there. The mirror
- an index built there, queried here - needs every corpus vector from the other host rather
than 86, and #21 found the two directions indistinguishable (zero churn both ways) when it
could run both. Read this as the cost of carrying an index to another machine, which is the
case a user meets.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import eval_retrieval
from eval_retrieval import CORPUS, DATA, _labels

from markdown_memory.config import ServerConfig
from markdown_memory.embedders import (
    DEFAULT_EMBEDDER,
    GEMMA_QUERY_PROMPT,
    Embedder,
    create_embedder,
)
from markdown_memory.server import MarkdownMemoryService

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "tests" / "fixtures" / "gemma_q4_reference.json"
TOP_N = 5


class ReplayEmbedder:
    """The real embedder with someone else's arithmetic for the queries.

    Everything but `embed_query` delegates, so the index's provenance checks see the
    embedder that actually built it and the run is not refused for the very difference
    it exists to measure. A query this was not given is an error rather than a silent
    fallback to local numbers - a partial replay would report a blend of two hosts and
    call it one.
    """

    def __init__(self, inner: Embedder, vectors: dict[str, list[float]]) -> None:
        self._inner = inner
        self._vectors = vectors
        self.replayed = 0

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    @property
    def weights_revision(self) -> str | None:
        return self._inner.weights_revision

    def warm_up(self) -> None:
        self._inner.warm_up()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        try:
            vector = self._vectors[text]
        except KeyError:
            raise SystemExit(f"the export does not carry a vector for {text!r}") from None
        self.replayed += 1
        return vector


def _validate(payload: dict[str, Any], embedder: Embedder, texts: Sequence[str]) -> None:
    """Refuse an export that would be scored without being used.

    `vec_search` raising is not loud: `HybridSearcher` answers a failed vector ranking with
    the keyword half alone, which is right for a server and wrong for a measurement - the
    run would finish and print churn figures for a search that never saw these vectors. So
    every one is checked before any of them ranks.
    """
    vectors = payload.get("vectors")
    if not isinstance(vectors, dict):
        raise SystemExit("the export carries no `vectors` object")
    missing = [text for text in texts if text not in vectors]
    if missing:
        raise SystemExit(f"the export is missing {len(missing)} of {len(texts)} queries")
    for text in texts:
        vector = vectors[text]
        if not isinstance(vector, list) or len(vector) != embedder.dimension:
            raise SystemExit(f"the vector for {text!r} is not {embedder.dimension} numbers")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector):
            raise SystemExit(f"the vector for {text!r} holds a value that is not finite")
        norm = math.sqrt(sum(float(value) * float(value) for value in vector))
        if abs(norm - 1.0) > 1e-3:
            raise SystemExit(
                f"the vector for {text!r} has norm {norm:.6f}, so it is not the unit vector "
                "`embed_query` returns - cosine against the index would measure nothing"
            )


@dataclass(slots=True, frozen=True)
class Ranking:
    """What one host's queries did to the ranking, per split and kind."""

    total: int
    top1_changed: int
    top5_changed: int
    top1_correct_before: int
    top1_correct_after: int
    recall5_before: int
    recall5_after: int


def _queries() -> dict[str, dict[str, list[dict[str, Any]]]]:
    loaded: dict[str, dict[str, list[dict[str, Any]]]] = json.loads(
        (DATA / "queries.json").read_text(encoding="utf-8")
    )
    return loaded


def _texts(queries: dict[str, dict[str, list[dict[str, Any]]]]) -> list[str]:
    return [
        str(case["query"])
        for split in ("dev", "held_out")
        for kind in ("paraphrase", "identifier")
        for case in queries[split][kind]
    ]


def _probe_distance(embedder: Embedder) -> float | None:
    """How far this host sits from the committed reference vector, or None if unknown.

    The same number the golden test reads, recorded so an export carries some account of
    which arithmetic produced it - a CPU model does not, since CI saw one chip report the
    VNNI flag on one run and not on another.

    It is a band rather than a name. One scalar cannot separate hosts that agree within
    the match radius, and two already do: the recorded aarch64 and Apple Silicon paths sit
    1.9e-5 apart against a 5e-5 radius. Read it as "which group of paths", and the CPU
    lines the workflow prints for the rest.

    None when the reference describes something else - a moved revision, graph or prompt -
    because a distance to a vector from another model is not a distance between paths.
    """
    if not REFERENCE.exists():
        return None
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    revision = embedder.weights_revision or ""
    if f"{reference['revision']}/{reference['graph']}" != revision:
        return None
    if reference["prompt"] != GEMMA_QUERY_PROMPT:
        return None
    actual = embedder.embed_query(reference["text"])
    return 1.0 - sum(a * b for a, b in zip(reference["vector"], actual, strict=True))


def export(path: Path, embedder_name: str) -> int:
    embedder = create_embedder(embedder_name, cache_dir=ServerConfig.from_env().model_cache_dir)
    queries = _queries()
    texts = _texts(queries)
    payload = {
        "_about": (
            "Query vectors as one host embeds them, for scripts/cross_host_churn.py --score. "
            "Not a fixture: nothing asserts on these, they are one machine's arithmetic."
        ),
        "host": {
            "machine": platform.machine(),
            "system": platform.system(),
            "processor": platform.processor() or "unknown",
        },
        "embedder": embedder.model_name,
        "weights_revision": embedder.weights_revision,
        "golden_probe_distance": _probe_distance(embedder),
        "vectors": {text: embedder.embed_query(text) for text in texts},
    }
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    probe = payload["golden_probe_distance"]
    shown = "unknown" if probe is None else f"{probe:.6e}"
    print(f"wrote {path} - {len(texts)} queries, golden probe {shown}")
    return 0


def _rank(service: MarkdownMemoryService, query: str) -> list[set[str]]:
    return [_labels(hit, service.root) for hit in service.search_docs(query, TOP_N)]


def _compare(
    local: MarkdownMemoryService,
    replayed: MarkdownMemoryService,
    cases: list[dict[str, Any]],
) -> Ranking:
    counts = dict.fromkeys(
        ("top1_changed", "top5_changed", "before_top1", "after_top1", "before_5", "after_5"), 0
    )
    for case in cases:
        query, expected = str(case["query"]), str(case["expected"])
        before, after = _rank(local, query), _rank(replayed, query)
        if not before or not after:
            continue
        counts["top1_changed"] += before[0] != after[0]
        counts["top5_changed"] += [sorted(labels) for labels in before] != [
            sorted(labels) for labels in after
        ]
        counts["before_top1"] += expected in before[0]
        counts["after_top1"] += expected in after[0]
        counts["before_5"] += any(expected in labels for labels in before)
        counts["after_5"] += any(expected in labels for labels in after)
    return Ranking(
        len(cases),
        counts["top1_changed"],
        counts["top5_changed"],
        counts["before_top1"],
        counts["after_top1"],
        counts["before_5"],
        counts["after_5"],
    )


def _open(db_path: Path, embedder_name: str, embedder: Embedder) -> MarkdownMemoryService:
    return MarkdownMemoryService(
        ServerConfig(
            db_path=db_path,
            docs_dir=CORPUS,
            embedder=embedder_name,
            model_cache_dir=ServerConfig.from_env().model_cache_dir,
        ),
        embedder=embedder,
    )


def score(paths: Sequence[Path], arguments: argparse.Namespace) -> int:
    import eval_cache

    queries = _queries()
    texts = _texts(queries)
    base = ServerConfig.from_env()
    probes = eval_retrieval._probes(CORPUS)
    with eval_cache.lock(eval_cache.cache_root()):
        # Always rebuilt, never reused. The eval cache key names the corpus, the module
        # sources and the model files, but deliberately not the arithmetic - which is the
        # one thing this measurement varies. A cache carried from another host, or written
        # by this machine on another compute path, passes its probes (the tolerance is
        # 1e-3, and the paths sit inside it) and would be scored as "built here". Eleven
        # seconds on this corpus is a cheap way not to measure the wrong index.
        arguments.rebuild = True
        service, built = eval_retrieval.open_service(arguments, base, probes)
        if not built:
            raise SystemExit("the index was reused despite --rebuild; refusing to score it")
        embedder = service.embedder
        # The same path `open_service` chose, recomputed rather than read off the service,
        # which keeps its config to itself. Both calls are pure functions of the corpus,
        # the embedder name and the model files, so they cannot disagree.
        key = eval_cache.build_key(CORPUS, arguments.embedder, model_cache_dir=base.model_cache_dir)
        db_path = eval_cache.cache_root() / key.digest / "eval.db"
        here = _probe_distance(embedder)
        print(f"index built here: {embedder.weights_revision}")
        print(f"this host's golden probe: {'unknown' if here is None else f'{here:.6e}'}\n")
        try:
            for path in paths:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("weights_revision") != embedder.weights_revision:
                    raise SystemExit(
                        f"{path} was made by {payload.get('weights_revision')!r}, and this index "
                        f"by {embedder.weights_revision!r}: that is a different model, not a "
                        "different compute path, and comparing them measures nothing"
                    )
                _validate(payload, embedder, texts)
                replay = ReplayEmbedder(embedder, payload["vectors"])
                replayed = _open(db_path, arguments.embedder, replay)
                try:
                    _report(path, payload, here, queries, service, replayed)
                finally:
                    replayed.close()
                # A search that fell back to keyword ranking would still print numbers.
                # This says the imported vectors were asked for, once per scored query.
                if replay.replayed < len(texts):
                    raise SystemExit(
                        f"only {replay.replayed} of {len(texts)} queries used an imported "
                        "vector; those figures are not a cross-host measurement"
                    )
        finally:
            service.close()
    return 0


def _report(
    path: Path,
    payload: dict[str, Any],
    here: float | None,
    queries: dict[str, dict[str, list[dict[str, Any]]]],
    local: MarkdownMemoryService,
    replayed: MarkdownMemoryService,
) -> None:
    host, there = payload["host"], payload.get("golden_probe_distance")
    apart = "unknown" if None in (here, there) else f"{abs(there - here):.6e}"
    print(f"== {path.name}: {host['machine']} / {host['system']} ({host['processor']})")
    print(f"   golden probe {'unknown' if there is None else f'{there:.6e}'}, {apart} from here")
    header = f"   {'set':<22} {'n':>3}  top-1 moved  top-5 moved  top-1 right  labelled in 5"
    print(header + "\n   " + "-" * (len(header) - 3))
    for split in ("dev", "held_out"):
        for kind in ("paraphrase", "identifier"):
            result = _compare(local, replayed, queries[split][kind])
            print(
                f"   {split + ' ' + kind:<22} {result.total:>3}  {result.top1_changed:>11}  "
                f"{result.top5_changed:>11}  "
                f"{result.top1_correct_before:>5} -> {result.top1_correct_after:<5}  "
                f"{result.recall5_before:>6} -> {result.recall5_after}"
            )
    print()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--export", type=Path, help="write this host's query vectors here")
    parser.add_argument(
        "--score", type=Path, nargs="+", help="replay these exports against an index built here"
    )
    parser.add_argument("--embedder", default=DEFAULT_EMBEDDER, choices=("embeddinggemma",))
    parser.add_argument("--show-misses", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    # `open_service` reads it; scoring sets it, and the export path never builds an index.
    arguments.rebuild = False
    if bool(arguments.export) == bool(arguments.score):
        parser.error("pass exactly one of --export and --score")
    if arguments.export:
        return export(arguments.export, arguments.embedder)
    return score(arguments.score, arguments)


if __name__ == "__main__":
    sys.exit(main())
