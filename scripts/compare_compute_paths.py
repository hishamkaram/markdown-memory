#!/usr/bin/env python3
"""Does the 4-bit graph compute the same numbers on every CPU? Measure it, do not assume.

Every `MatMulNBits` node in the pinned graph carries `accuracy_level=4`, which asks
onnxruntime for the int8 compute kernel. It gets it only where MLAS has one: on arm64 that
needs FEAT_DotProd, and where it is missing `MatMulNBits::GetComputeType` silently returns
`SQNBIT_CompFp32` instead. Same graph file, different arithmetic, no error and no log. The
index's provenance key names the revision and the graph, so a change of compute path is
invisible to it - which is the question this script exists to size.

It answers that by making the fallback happen on purpose: `accuracy_level` is rewritten from
4 to 1 in a copy of the graph, which is the one thing that selects `CompFp32` on a machine
that could have run int8. That is a compute-*type* comparison, not a simulation of an ARM
host - the ISA is still this one - so it bounds the effect rather than reproducing it.

Two numbers come out, and the first is the one that matters:

  * **rank churn** - the labelled eval queries, embedded by one path and ranked against
    sections embedded by the other. Cosine is a proxy for harm; this is harm.
  * **cosine distance and max coordinate movement**, against a noise floor measured by
    running the unpatched graph against itself and again at a different thread count. A
    delta means nothing without the noise it has to beat.

On the reference machine (x86_64, onnxruntime 1.30.0) the numbers are unambiguous: forcing
fp32 moves a vector by 2.1e-4 to 6.1e-4 cosine against a ~1e-7 noise floor, and across both
directions and both query splits it changed the top result for none of the 86 labelled
queries. The labelled section stayed inside the top five in every cell (40/44 dev, 39/42
held out) - a proxy for the Recall@5 that `docs/evaluation-protocol.md` makes primary, not
that metric itself. One query in 44 saw a different fifth section, and it was not the
labelled one.

Note what that claim is: no observed retrieval loss, on this frozen corpus, on this
machine, through vector ranking alone. It is not a proof that no CPU anywhere can do worse,
and it is not production either - the server fuses these ranks with BM25, which should damp
a disagreement, but it also embeds passages and takes a max over them, which is a different
calculation and not one this bounds.

**And the gate has since shown that bound is not the whole range.** Running the golden
vector on four kinds of host gave four values, each repeating to every printed digit rather
than drifting, because the kernel is chosen by the instruction set and not by luck:

    0.0        x86_64 without VNNI (AMD EPYC 7763 / 9V74 as avx2) - the reference
    2.414e-4   x86_64 with VNNI (Intel Xeon 6973P-C; AMD EPYC 9V74) - identical on both
    1.626e-3   aarch64 with dot product (Neoverse-N2)
    1.645e-3   arm64 Darwin (Apple M2 Pro)

So the effect this script sizes by forcing fp32 - 2.1e-4 to 6.1e-4 - is about the size of
the VNNI difference and roughly seven times smaller than the arm64 one. **Nothing has
measured what 1.6e-3 does to rank**, because the churn above needs both sessions on one
machine and no arm64 host here can produce the x86 side. That gap, not the numbers above,
is what decides whether the provenance key should grow to name the compute path.

Re-run it rather than trusting those numbers on other hardware - and note the script
refuses to pretend: if this host has no int8 kernel to lose, both sessions compute the same
way and it says so instead of reporting agreement.

    uv run python scripts/compare_compute_paths.py
    uv run python scripts/compare_compute_paths.py --update-reference
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from markdown_memory.config import ServerConfig  # noqa: E402
from markdown_memory.embedders import (  # noqa: E402
    GEMMA_DIMENSION,
    GEMMA_DOCUMENT_PROMPT,
    GEMMA_MAX_TOKENS,
    GEMMA_QUERY_PROMPT,
)
from markdown_memory.model_cache import (  # noqa: E402
    GEMMA_MODEL_FILE,
    GEMMA_REVISION,
    _stamp_is_current,
    gemma_model_dir,
)
from markdown_memory.parser import MarkdownParser  # noqa: E402

CORPUS = ROOT / "scripts/eval_data/corpus"
QUERIES = ROOT / "scripts/eval_data/queries.json"
REFERENCE = ROOT / "tests/fixtures/gemma_q4_reference.json"

#: Serialised `AttributeProto` for `accuracy_level = 4`, as the node embeds it: field 5 of
#: NodeProto (`2a`), 21 bytes (`15`), then name (field 1, 14 bytes), `i` (field 3, varint)
#: and `type` (field 20, INT). 23 bytes in all. Both `4` and `1` are one-byte varints, so
#: the substitution below cannot move an enclosing length prefix.
_LEVEL_4 = b"\x2a\x15\x0a\x0eaccuracy_level\x18\x04\xa0\x01\x02"
_LEVEL_1 = _LEVEL_4.replace(b"\x18\x04", b"\x18\x01")

#: Every projection in the graph. Checked, not assumed: a byte pattern that stopped matching
#: would otherwise patch nothing and report a delta of zero, which reads as good news.
_EXPECTED_NODES = 170

#: How far over the noise the patch must move a vector before this host is believed to have
#: had an int8 kernel to lose. A heuristic, not a proof: onnxruntime reports nothing about
#: which kernel it chose, so the only evidence available is whether asking for a different
#: one changed anything. Measured at 1645x on the reference machine, so the order of
#: magnitude here is not a close call.
_OVER_NOISE = 10

#: The probe the golden fixture pins. Short, and made of the identifiers this project's own
#: documentation is full of, so a tokenizer change shows up here too.
_REFERENCE_PROBE = "MARKDOWN_MEMORY_THREADS caps onnxruntime's intra-op pool."

_PROBES = (
    ("identifier", "HELIOS_BATCH"),
    ("sentence", _REFERENCE_PROBE),
    (
        "paragraph",
        "The ingest pipeline applies backpressure when consumer lag exceeds the configured "
        "ceiling, pausing the producer until the flush queue drains below half of it. "
        "Operators see this as a rising lag metric and a flat error rate.",
    ),
    ("long", "section heading and body text. " * 120),  # over 512 tokens: exercises truncation
)


class Probe(NamedTuple):
    """One text embedded down both compute paths."""

    label: str
    prompt: str
    cosine: float
    max_coordinate: float


def _varint(buffer: bytes, index: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = buffer[index]
        index += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, index
        shift += 7


def _fields(buffer: bytes, start: int, end: int) -> Iterator[tuple[int, int, int]]:
    """`(field_number, payload_start, payload_end)` for each length-delimited field."""
    index = start
    while index < end:
        tag, index = _varint(buffer, index)
        wire = tag & 0x07
        if wire == 2:
            length, index = _varint(buffer, index)
            yield tag >> 3, index, index + length
            index += length
        elif wire == 0:
            _, index = _varint(buffer, index)
        elif wire == 5:
            index += 4
        elif wire == 1:
            index += 8
        else:  # groups: this graph has none, and guessing past one would desynchronise
            raise ValueError(f"unsupported protobuf wire type {wire}")


def _levels_by_op(raw: bytes) -> dict[str, list[int]]:
    """Every node's `accuracy_level`, grouped by `op_type`.

    A count of byte matches proves only that the bytes are there. This walks ModelProto ->
    GraphProto -> NodeProto so the patch can be checked against the operator it belongs to.
    """
    levels: dict[str, list[int]] = {}
    for number, start, end in _fields(raw, 0, len(raw)):
        if number != 7:  # ModelProto.graph
            continue
        for node_number, node_start, node_end in _fields(raw, start, end):
            if node_number != 1:  # GraphProto.node
                continue
            op_type = ""
            level: int | None = None
            for part, part_start, part_end in _fields(raw, node_start, node_end):
                if part == 4:  # NodeProto.op_type
                    op_type = raw[part_start:part_end].decode("utf-8")
                elif part == 5:  # NodeProto.attribute
                    name = ""
                    value: int | None = None
                    index = part_start
                    while index < part_end:
                        tag, index = _varint(raw, index)
                        field, wire = tag >> 3, tag & 0x07
                        if wire == 2:
                            length, index = _varint(raw, index)
                            if field == 1:  # AttributeProto.name
                                name = raw[index : index + length].decode("utf-8")
                            index += length
                        elif wire == 0:
                            read, index = _varint(raw, index)
                            if field == 3:  # AttributeProto.i
                                value = read
                        elif wire == 5:  # AttributeProto.f, and other 32-bit fields
                            index += 4
                        elif wire == 1:
                            index += 8
                        else:
                            raise ValueError(f"unsupported attribute wire type {wire}")
                    if name == "accuracy_level" and value is not None:
                        level = value
            if level is not None:
                levels.setdefault(op_type, []).append(level)
    return levels


def force_fp32_compute(raw: bytes) -> bytes:
    """The graph with every `accuracy_level` set to 1, or a refusal.

    Fails closed on every count, because the failure this guards against is silent: a
    pattern that no longer matches patches nothing, the two sessions run identical
    arithmetic, and the script reports perfect agreement.
    """
    found = raw.count(_LEVEL_4)
    nodes = raw.count(b"MatMulNBits")
    if found != _EXPECTED_NODES or nodes != _EXPECTED_NODES:
        raise SystemExit(
            f"expected {_EXPECTED_NODES} accuracy_level attributes and MatMulNBits nodes, "
            f"found {found} and {nodes}: the graph or its serialisation moved, and this "
            "patch must be re-derived rather than trusted"
        )
    patched = raw.replace(_LEVEL_4, _LEVEL_1)
    if len(patched) != len(raw):
        raise SystemExit("the patch changed the file length, which it must never do")
    before, after = _levels_by_op(raw), _levels_by_op(patched)
    if set(before) != {"MatMulNBits"} or set(after) != {"MatMulNBits"}:
        raise SystemExit(f"accuracy_level appears on operators other than MatMulNBits: {before}")
    if before["MatMulNBits"] != [4] * _EXPECTED_NODES:
        raise SystemExit(f"the graph no longer asks for int8 compute: {sorted(set(before))}")
    if after["MatMulNBits"] != [1] * _EXPECTED_NODES:
        raise SystemExit("the patch did not reach every node")
    return patched


def _assert_int8_was_actually_used(against_fp32: Sequence[Probe], noise: Sequence[Probe]) -> bool:
    """Did the unpatched graph really run the int8 kernel here? Say so, do not assume it.

    onnxruntime reports nothing about which `MatMulNBits` kernel it chose, and asking for
    `accuracy_level=4` is a request, not a guarantee. A CPU that cannot honour it - arm64
    without FEAT_DotProd - computes in fp32 for the unpatched graph too, and then both
    sessions here are the same session wearing two labels. The comparison collapses to the
    noise floor and prints perfect agreement, which reads exactly like good news.

    So the collapse is the signal: the patch is known to change the arithmetic on a machine
    that had int8 to lose, and a delta indistinguishable from running one graph twice means
    there was none.
    """
    signal = min(abs(probe.cosine) for probe in against_fp32)
    floor = max(abs(probe.cosine) for probe in noise)
    if signal > floor * _OVER_NOISE:
        # A floor of exactly zero is the good case - every coordinate reproduced bit for
        # bit - and dividing by it to say how good would be the one way to fail here.
        margin = f"{signal / floor:.0f}x the noise" if floor else "against no noise at all"
        print(f"\nint8 kernel in use: the patch moved vectors {margin}")
        return True
    print(
        f"\n*** This host did not take the int8 path. Patching accuracy_level moved vectors "
        f"by {signal:.3e}, against a noise floor of {floor:.3e} - the two graphs computed "
        f"the same way, so nothing here measures the fallback. On arm64 that means no "
        f"FEAT_DotProd. The numbers above are not evidence about {platform.machine()}."
    )
    return False


def _model_dir() -> Path:
    """The verified model cache, or a refusal.

    Verified, not merely present: every number this script prints is labelled with the
    pinned revision, and a cache holding something else would put that label on evidence
    about a different model. `_stamp_is_current` is the same check the embedder makes
    before it will load these files.
    """
    directory = gemma_model_dir(ServerConfig.from_env().model_cache_dir)
    if not (directory / GEMMA_MODEL_FILE).is_file():
        raise SystemExit(f"the embedding model is not cached at {directory}")
    if not _stamp_is_current(directory):
        raise SystemExit(
            f"the model cache at {directory} is not verified against the manifest. Run the "
            "server or the test suite once to repair it; this script will not label a "
            f"measurement with {GEMMA_REVISION[:12]} without knowing that is what ran."
        )
    return directory


def _scratch_graph(source: Path, scratch: Path, raw: bytes) -> Path:
    """The patched graph, with its weights beside it under the name the graph records.

    Hard-linked rather than copied: 197 MB, and a link is a regular file, which is what
    onnxruntime demands of external data (it refuses a symlink out of the model directory).
    """
    target = scratch / GEMMA_MODEL_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    weights = source.with_name(source.name + "_data")
    link = target.with_name(target.name + "_data")
    try:
        os.link(weights, link)
    except OSError:  # a scratch directory on another filesystem
        shutil.copy2(weights, link)
    return target


class Session:
    """One graph, loaded the way the embedder loads it, with the thread count pinned."""

    def __init__(self, graph: Path, tokenizer_source: Path, *, threads: int) -> None:
        import onnxruntime
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(tokenizer_source / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=GEMMA_MAX_TOKENS)
        self.tokenizer.enable_padding()
        options = onnxruntime.SessionOptions()
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.intra_op_num_threads = threads
        self.session = onnxruntime.InferenceSession(
            str(graph), options, providers=["CPUExecutionProvider"]
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Normalised vectors, by the same arithmetic as `EmbeddingGemmaEmbedder._embed`."""
        import numpy as np

        encodings = self.tokenizer.encode_batch(list(texts))
        outputs = self.session.run(
            ["sentence_embedding"],
            {
                "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            },
        )[0]
        norms = np.maximum(np.linalg.norm(outputs, axis=-1, keepdims=True), 1e-12)
        result: list[list[float]] = (outputs / norms).tolist()
        return result


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return 1.0 - sum(a * b for a, b in zip(left, right, strict=True))


def _compare(one: Session, other: Session) -> list[Probe]:
    probes: list[Probe] = []
    for prompt_name, prompt in (("query", GEMMA_QUERY_PROMPT), ("document", GEMMA_DOCUMENT_PROMPT)):
        for label, text in _PROBES:
            left = one.embed([prompt + text])[0]
            right = other.embed([prompt + text])[0]
            probes.append(
                Probe(
                    label=f"{prompt_name}/{label}",
                    prompt=prompt_name,
                    cosine=_cosine(left, right),
                    max_coordinate=max(abs(a - b) for a, b in zip(left, right, strict=True)),
                )
            )
    return probes


def _sections() -> list[tuple[str, str]]:
    """`(heading_path, content)` for the frozen eval corpus, as the indexer would cut it."""
    parser = MarkdownParser()
    sections: list[tuple[str, str]] = []
    for path in sorted(CORPUS.rglob("*.md")):
        for section in parser.parse(path.read_text(encoding="utf-8")).sections:
            sections.append((section.heading_path, section.content))
    return sections


def _labelled() -> list[tuple[str, str, str]]:
    """`(split, query, expected_heading)`. The split is carried because the policy needs it.

    `CLAUDE.md` says the held-out queries must never choose a parameter. A decision about
    whether the compute path deserves a provenance key is exactly such a choice, so the two
    splits are reported apart and the decision is made on `dev`.
    """
    data = json.loads(QUERIES.read_text(encoding="utf-8"))
    rows: list[tuple[str, str, str]] = []
    for split in ("dev", "held_out"):
        for kind in ("paraphrase", "identifier"):
            for row in data[split].get(kind, []):
                rows.append((split, row["query"], row["expected"].split("::")[-1]))
    return rows


class Churn(NamedTuple):
    """What changed when the query's compute path was not the index's."""

    queries: int
    top1: int
    members: int
    order: int
    correct_before: int
    correct_after: int
    recall_before: int
    recall_after: int


def _rank_churn(indexed: Session, querying: Session, split: str) -> Churn:
    """Sections embedded by one path, queries by the other, ranked on vectors alone.

    The sensitive measurement rather than the realistic one, deliberately. The server
    fuses these ranks with BM25, and fusion damps a disagreement that shows up here, so a
    zero here is a stronger statement than a zero through the real searcher would be. It
    is not the production path: no breadcrumb prefixes, no passage pooling, no RRF.

    Accuracy against the labels is counted too, because stability and correctness are
    different questions - a reordering that fixes a wrong answer and one that breaks a right
    answer both show up as churn, and only one of them matters.

    That count is deliberately *not* called Recall@5. `docs/evaluation-protocol.md` defines
    Recall@5 over the graded `also_valid` labels, against production's passage vectors and
    reassembled oversized sections; this matches one exact heading path against raw whole
    sections. It moves for the same reasons the real metric would and is a fair witness to
    a change, but it is neither an upper nor a lower bound on it - passage max-sim is a
    different calculation, not a quieter version of this one. `scripts/eval_retrieval.py`
    remains the gate.
    """
    import numpy as np

    sections = _sections()
    matrix = np.array(indexed.embed([GEMMA_DOCUMENT_PROMPT + body for _, body in sections]))
    paths = [path for path, _ in sections]
    rows = [row for row in _labelled() if row[0] == split]
    top1 = members = order = before = after = 0
    recalled_before = recalled_after = 0
    for _split, text, expected in rows:
        prompt = GEMMA_QUERY_PROMPT + text
        ranked = []
        for session in (indexed, querying):
            scores = matrix @ np.array(session.embed([prompt])[0])
            ranked.append([paths[i] for i in np.argsort(-scores)[:5]])
        top1 += ranked[0][0] != ranked[1][0]
        members += set(ranked[0]) != set(ranked[1])
        order += ranked[0] != ranked[1]
        before += ranked[0][0] == expected
        after += ranked[1][0] == expected
        recalled_before += expected in ranked[0]
        recalled_after += expected in ranked[1]
    return Churn(len(rows), top1, members, order, before, after, recalled_before, recalled_after)


def _write_reference(session: Session) -> None:
    import onnxruntime

    vector = session.embed([GEMMA_QUERY_PROMPT + _REFERENCE_PROBE])[0]
    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE.write_text(
        json.dumps(
            {
                "_about": (
                    "What the pinned graph returns for one fixed string, and the distances "
                    "the other compute paths return for it. Regenerate with `uv run python "
                    "scripts/compare_compute_paths.py --update-reference` when the revision "
                    "or the graph moves - never to quiet a failing test. Regenerating resets "
                    "`compute_paths` to the host that ran it: every distance is measured "
                    "against `vector`, so a new vector retires all of them."
                ),
                "revision": GEMMA_REVISION,
                "graph": GEMMA_MODEL_FILE,
                "prompt": GEMMA_QUERY_PROMPT,
                "text": _REFERENCE_PROBE,
                "produced_on": {
                    "machine": platform.machine(),
                    "system": platform.system(),
                    "onnxruntime": onnxruntime.__version__,
                },
                # Reset, not preserved: the recorded distances are all relative to
                # `vector`, so writing a new one makes every other entry a number about a
                # vector that no longer exists. The gate re-measures them on its own hosts.
                "compute_paths": [
                    {
                        "distance": 0.0,
                        "cpu": f"{platform.machine()} ({platform.processor() or 'unknown'})",
                        "note": "the path this vector was produced on",
                    }
                ],
                "vector": vector,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {REFERENCE.relative_to(ROOT)} ({GEMMA_DIMENSION} dimensions)")
    print("compute_paths now holds this host only; the other paths have to be re-measured")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-reference",
        action="store_true",
        help="rewrite the golden fixture from the unpatched, cached model",
    )
    parser.add_argument(
        "--skip-rank-churn", action="store_true", help="the vector report only (much faster)"
    )
    arguments = parser.parse_args(argv)

    directory = _model_dir()
    graph = directory / GEMMA_MODEL_FILE
    raw = graph.read_bytes()
    patched_bytes = force_fp32_compute(raw)
    print(f"machine     {platform.machine()} ({platform.system()})")
    print(f"graph       {GEMMA_MODEL_FILE} at {GEMMA_REVISION[:12]}")
    print(
        f"patched     {_EXPECTED_NODES} MatMulNBits nodes, accuracy_level 4 -> 1, "
        f"{len(raw)} bytes unchanged"
    )

    scratch = Path(tempfile.mkdtemp(prefix="mdmem-compute-paths-"))
    try:
        patched_graph = _scratch_graph(graph, scratch, patched_bytes)
        int8 = Session(graph, directory, threads=1)
        fp32 = Session(patched_graph, directory, threads=1)

        print("\nint8 (accuracy_level=4) against fp32 (accuracy_level=1)")
        print(f"  {'probe':<22}{'cosine distance':>18}{'max coordinate':>18}")
        against_fp32 = _compare(int8, fp32)
        for probe in against_fp32:
            print(f"  {probe.label:<22}{probe.cosine:>18.3e}{probe.max_coordinate:>18.3e}")

        print("\nnoise floor: the same graph against itself")
        again = Session(graph, directory, threads=1)
        noise = _compare(int8, again)
        for probe in noise:
            print(f"  {probe.label:<22}{probe.cosine:>18.3e}{probe.max_coordinate:>18.3e}")
        print("\nnoise floor: the same graph at a different thread count")
        threaded = Session(graph, directory, threads=4)
        across_threads = _compare(int8, threaded)
        noise += across_threads
        for probe in across_threads:
            print(f"  {probe.label:<22}{probe.cosine:>18.3e}{probe.max_coordinate:>18.3e}")

        separated = _assert_int8_was_actually_used(against_fp32, noise)
        if arguments.update_reference:
            if not separated:
                raise SystemExit(
                    "refusing to write the reference: this host did not take the int8 path, "
                    "so the vector would be baselined on the fallback"
                )
            _write_reference(int8)

        if not arguments.skip_rank_churn:
            print("\nrank churn: sections embedded by one path, queries by the other.")
            print("Vector ranking only - the server fuses with BM25, which damps all of this.")
            for indexed, querying, direction in (
                (int8, fp32, "int8 index, fp32 queries"),
                (fp32, int8, "fp32 index, int8 queries"),
            ):
                for split in ("dev", "held_out"):
                    churn = _rank_churn(indexed, querying, split)
                    total = churn.queries
                    print(f"\n  {direction}, {split} ({total} queries)")
                    print(f"    Top-1 changed        {churn.top1:>4}  ({churn.top1 / total:.1%})")
                    print(
                        f"    Top-5 membership     {churn.members:>4}  "
                        f"({churn.members / total:.1%})"
                    )
                    print(
                        f"    Top-5 order only     {churn.order - churn.members:>4}  "
                        f"({(churn.order - churn.members) / total:.1%})"
                    )
                    print(
                        f"    exact label in top 5 {churn.recall_before:>4} -> "
                        f"{churn.recall_after}  (of {total})"
                    )
                    print(
                        f"    Top-1 correct        {churn.correct_before:>4} -> "
                        f"{churn.correct_after}  (of {total})"
                    )
            print("\nDecide on `dev`: the held-out queries must never choose a parameter.")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
