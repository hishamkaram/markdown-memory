"""The gather-before-dequantize rewrite: what it changes, and what it refuses to touch.

The graphs here are hand-encoded and small enough to run, so "the rewrite preserves the
values" is asserted by running both of them, not by reading the bytes. The real model is
covered by the `embedding`-marked tests at the end.
"""

from __future__ import annotations

import hashlib
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from markdown_memory.graph_patch import MAX_GRAPH_BYTES, gather_before_dequantize

FLOAT, INT8, INT64 = 1, 3, 7


def varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | 0x80 if value else byte)
        if not value:
            return bytes(out)


def field(number: int, payload: bytes) -> bytes:
    return varint(number << 3 | 2) + varint(len(payload)) + payload


def number_field(number: int, value: int) -> bytes:
    return varint(number << 3) + varint(value)


def tensor(name: str, data_type: int, dims: list[int], raw: bytes) -> bytes:
    body = b"".join(number_field(1, dim) for dim in dims)
    body += number_field(2, data_type) + field(8, name.encode()) + field(9, raw)
    return body


def value_info(name: str, elem_type: int, dims: list[int]) -> bytes:
    shape = b"".join(field(1, number_field(1, dim)) for dim in dims)
    tensor_type = number_field(1, elem_type) + field(2, shape)
    return field(1, name.encode()) + field(2, field(1, tensor_type))


def node(
    op_type: str, inputs: list[str], outputs: list[str], name: str, attributes: bytes = b""
) -> bytes:
    body = b"".join(field(1, value.encode()) for value in inputs)
    body += b"".join(field(2, value.encode()) for value in outputs)
    return body + field(3, name.encode()) + field(4, op_type.encode()) + attributes


def model(graph: bytes, extra: bytes = b"") -> bytes:
    opset = field(1, b"") + number_field(2, 21)
    return number_field(1, 10) + field(8, opset) + extra + field(7, graph)


def dequantize_then_gather(
    *,
    scale_dims: list[int] | None = None,
    second_consumer: bool = False,
    extra: bytes = b"",
    outer: bytes = b"",
    gather_first: bool = False,
    node_extra: bytes = b"",
    export_table: bool = False,
    collide: bool = False,
    no_outputs: bool = False,
    gather_axis: bool = False,
    gather_ref_attr: bool = False,
    collide_input: bool = False,
    collide_value_info: bool = False,
    indices_via_node: bool = False,
) -> bytes:
    """The shipped pattern in miniature: an int8 table dequantized whole, then indexed."""
    table = bytes(range(12))
    scale = struct.pack("<f", 0.5) * (1 if scale_dims is None else max(1, scale_dims[0]))
    axis = field(5, field(1, b"axis") + number_field(3, 0) + number_field(20, 2))
    if gather_ref_attr:
        # `ref_attr_name` makes the value come from the enclosing function at
        # instantiation, so the `i` written here is not what the node will run with.
        axis = field(
            5,
            field(1, b"axis") + number_field(3, 0) + number_field(20, 2) + field(21, b"axis_ref"),
        )
    nodes = [
        field(
            1,
            node(
                "DequantizeLinear",
                ["table", "scale", "zero"],
                [] if no_outputs else ["table_f"],
                "dq",
                node_extra,
            ),
        ),
        field(
            1,
            node(
                "Gather",
                ["table_f", "idx" if indices_via_node else "ids"],
                ["rows"],
                "gather",
                axis if gather_axis or gather_ref_attr else b"",
            ),
        ),
    ]
    if indices_via_node:
        # The indices are computed *between* the two nodes, so moving the Gather up to
        # where the DequantizeLinear sits would read them before anything writes them.
        nodes.insert(1, field(1, node("Identity", ["ids"], ["idx"], "indices")))
    if gather_first:
        nodes.reverse()
    outputs = [field(12, value_info("rows", FLOAT, [2, 3]))]
    if export_table:
        outputs.append(field(12, value_info("table_f", FLOAT, [4, 3])))
    if collide:
        nodes.append(field(1, node("Identity", ["ids"], ["table_f_rows"], "collision")))
        outputs.append(field(12, value_info("table_f_rows", INT64, [2])))
    if second_consumer:
        nodes.append(field(1, node("Identity", ["table_f"], ["copy"], "identity")))
        outputs.append(field(12, value_info("copy", FLOAT, [4, 3])))
    graph = b"".join(nodes)
    graph += field(2, b"tiny")
    graph += field(5, tensor("table", INT8, [4, 3], table))
    graph += field(5, tensor("scale", FLOAT, scale_dims or [1], scale))
    graph += field(5, tensor("zero", INT8, [1], b"\x02"))
    graph += field(11, value_info("ids", INT64, [2]))
    if collide_input:
        graph += field(11, value_info("table_f_rows", INT64, [2]))
    graph += b"".join(outputs)
    graph += field(13, value_info("table_f", FLOAT, [4, 3]))
    if collide_value_info:
        graph += field(13, value_info("table_f_rows", FLOAT, [2, 3]))
    graph += extra
    return model(graph, outer)


def run(graph: bytes) -> dict[str, np.ndarray]:
    import onnxruntime

    options = onnxruntime.SessionOptions()
    session = onnxruntime.InferenceSession(graph, options, providers=["CPUExecutionProvider"])
    names = [output.name for output in session.get_outputs()]
    fed = {"ids": np.array([0, 3], dtype=np.int64)}
    return dict(zip(names, session.run(names, fed), strict=True))


def test_the_rewritten_graph_computes_exactly_what_the_original_did() -> None:
    """Dequantizing is (q - zero_point) * scale, value by value, and gathering picks rows:

    with one scale for the whole tensor, doing them in the other order cannot change a
    number. What it does change is that the full table is never materialised in float32.
    """
    original = dequantize_then_gather()
    rewritten = gather_before_dequantize(original)
    assert rewritten is not None
    assert np.array_equal(run(original)["rows"], run(rewritten)["rows"])


def test_the_full_table_is_no_longer_dequantized() -> None:
    rewritten = gather_before_dequantize(dequantize_then_gather())
    assert rewritten is not None
    # The DequantizeLinear now reads what the Gather wrote, not the initializer.
    assert b"table_f_rows" in rewritten
    assert rewritten.index(b"Gather") < rewritten.index(b"DequantizeLinear")


def test_a_dangling_value_info_for_the_dequantized_table_is_dropped() -> None:
    rewritten = gather_before_dequantize(dequantize_then_gather())
    assert rewritten is not None
    assert rewritten.count(b"table_f") == 2  # the Gather's output, and the DQ's input


@pytest.mark.parametrize(
    ("description", "graph"),
    [
        ("the pattern is not there", model(field(2, b"empty"))),
        ("the scale is per axis", dequantize_then_gather(scale_dims=[4])),
        (
            "something else reads the dequantized table",
            dequantize_then_gather(second_consumer=True),
        ),
        ("a group, which this does not model", dequantize_then_gather(extra=varint(99 << 3 | 3))),
        ("the gather comes first", dequantize_then_gather(gather_first=True)),
        ("the node names another domain", dequantize_then_gather(node_extra=field(7, b"com.ms"))),
        (
            "a doc string the rebuild would drop",
            dequantize_then_gather(node_extra=field(6, b"why")),
        ),
        ("the graph exports the dequantized table", dequantize_then_gather(export_table=True)),
        ("the name the rewrite introduces is taken", dequantize_then_gather(collide=True)),
        (
            "the name the rewrite introduces is a graph input",
            dequantize_then_gather(collide_input=True),
        ),
        (
            "the name the rewrite introduces is declared in value_info",
            dequantize_then_gather(collide_value_info=True),
        ),
        (
            "the gather's axis is taken from somewhere else",
            dequantize_then_gather(gather_ref_attr=True),
        ),
        ("field number zero", dequantize_then_gather(extra=varint(0 << 3 | 0) + b"\x01")),
        ("a node with no outputs at all", dequantize_then_gather(no_outputs=True)),
        ("the indices are not available that early", dequantize_then_gather(indices_via_node=True)),
    ],
)
def test_anything_but_the_expected_pattern_is_refused(description: str, graph: bytes) -> None:
    """A refusal costs the memory saving. Rewriting a graph this does not understand

    would cost correctness, which is not a trade worth making for 1 GB.
    """
    assert gather_before_dequantize(graph) is None, description


def test_a_gather_that_spells_out_axis_zero_is_still_rewritten() -> None:
    """`axis=0` written out means what leaving it out means, and refusing it would cost

    the memory saving over a difference that is only notation.
    """
    original = dequantize_then_gather(gather_axis=True)
    rewritten = gather_before_dequantize(original)
    assert rewritten is not None
    assert np.array_equal(run(original)["rows"], run(rewritten)["rows"])


def test_a_graph_too_large_to_be_this_one_is_refused() -> None:
    assert gather_before_dequantize(b"\x00" * (MAX_GRAPH_BYTES + 1)) is None


def test_everything_it_does_not_own_survives_byte_for_byte() -> None:
    """Producer name, doc strings and metadata_props are copied as their original spans."""
    producer = field(2, b"markdown-memory-test")
    metadata = field(14, field(1, b"key") + field(2, b"value"))
    original = dequantize_then_gather(extra=field(10, b"a doc string"), outer=producer + metadata)
    rewritten = gather_before_dequantize(original)
    assert rewritten is not None
    for kept in (producer, metadata, field(10, b"a doc string")):
        assert original.count(kept) == 1
        assert rewritten.count(kept) == 1, "a preserved field was dropped or duplicated"
    # Order too: spans are copied where they were, so relative order cannot drift.
    assert rewritten.index(producer) < rewritten.index(metadata)
    # Everything the rewrite does not own is still there, byte for byte: the only
    # difference is the two nodes and the value_info that described the dequantized table.
    initializers = (
        field(5, tensor("table", INT8, [4, 3], bytes(range(12)))),
        field(5, tensor("scale", FLOAT, [1], struct.pack("<f", 0.5))),
        field(5, tensor("zero", INT8, [1], b"\x02")),
    )
    for initializer in initializers:
        assert original.count(initializer) == 1
        assert rewritten.count(initializer) == 1, "an initializer was dropped or duplicated"
    assert np.array_equal(run(original)["rows"], run(rewritten)["rows"])


# --- Against the real model -------------------------------------------------------------

_PEAK_MEGABYTES = 900
_MEASURE_PEAK = """
import numpy as np, onnxruntime, sys
from pathlib import Path
graph = Path(sys.argv[1])
options = onnxruntime.SessionOptions()
options.add_session_config_entry("session.intra_op.allow_spinning", "0")
session = onnxruntime.InferenceSession(str(graph), options, providers=["CPUExecutionProvider"])
session.run(
    ["sentence_embedding"],
    {
        "input_ids": np.arange(512, dtype=np.int64).reshape(1, 512),
        "attention_mask": np.ones((1, 512), dtype=np.int64),
    },
)
peak = [line for line in open("/proc/self/status") if line.startswith("VmHWM")][0]
print(int(peak.split()[1]) // 1024)
"""


@pytest.fixture
def model_dir() -> Path:
    from markdown_memory.indexer import GEMMA_MODEL_FILE, gemma_model_dir
    from markdown_memory.server import ServerConfig

    cache_dir = ServerConfig.from_env().model_cache_dir
    directory = gemma_model_dir(cache_dir)
    if not (directory / GEMMA_MODEL_FILE).is_file():
        pytest.skip(f"the embedding model is not downloaded into {cache_dir}")
    return directory


@pytest.mark.embedding
def test_the_rewrite_of_the_real_graph_matches_its_pinned_checksum(model_dir: Path) -> None:
    """The pin is the rewrite's version: change the rewriter and this is what says so."""
    from markdown_memory.indexer import DERIVED_GRAPH_SHA256, GEMMA_MODEL_FILE

    rewritten = gather_before_dequantize((model_dir / GEMMA_MODEL_FILE).read_bytes())
    assert rewritten is not None
    assert hashlib.sha256(rewritten).hexdigest() == DERIVED_GRAPH_SHA256


@pytest.mark.embedding
def test_the_real_model_embeds_identically_through_both_graphs(model_dir: Path) -> None:
    """Bit-identical, not merely close. onnxruntime treats a one-element scale as per

    tensor (its own kernel rule rather than the ONNX spec's), and this is the check that
    would notice an upgrade changing that.
    """
    import onnxruntime
    from tokenizers import Tokenizer

    from markdown_memory.indexer import DERIVED_GRAPH_FILE, GEMMA_MAX_TOKENS, GEMMA_MODEL_FILE

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=GEMMA_MAX_TOKENS)
    tokenizer.enable_padding()
    encodings = tokenizer.encode_batch(
        [
            "task: search result | query: ENOSPC",
            "title: none | text: " + "a long passage " * 200,
            "title: none | text: 日本語 mixed with مرحبا and an emoji 🚀",
        ]
    )
    feed = {
        "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
    }

    def embed(name: str) -> np.ndarray:
        options = onnxruntime.SessionOptions()
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        session = onnxruntime.InferenceSession(
            str(model_dir / name), options, providers=["CPUExecutionProvider"]
        )
        result: np.ndarray = session.run(["sentence_embedding"], feed)[0]
        return result

    derived = model_dir / DERIVED_GRAPH_FILE
    if not derived.is_file():
        pytest.skip("the derived graph has not been generated yet")
    assert np.array_equal(embed(GEMMA_MODEL_FILE), embed(DERIVED_GRAPH_FILE))


@pytest.mark.embedding
@pytest.mark.skipif(not Path("/proc/self/status").exists(), reason="VmHWM is Linux-only")
def test_one_run_of_the_real_model_no_longer_peaks_over_a_gigabyte(model_dir: Path) -> None:
    """In its own process, because the peak is what is being measured and a shared one

    carries every other test's allocations. As published, one 512-token run peaks at
    ~1.45 GB: the whole 262144x768 vocabulary table in float32, to keep 512 rows.
    """
    from markdown_memory.indexer import DERIVED_GRAPH_FILE

    graph = model_dir / DERIVED_GRAPH_FILE
    if not graph.is_file():
        pytest.skip("the derived graph has not been generated yet")
    finished = subprocess.run(
        [sys.executable, "-c", _MEASURE_PEAK, str(graph)],
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )
    assert int(finished.stdout.strip()) < _PEAK_MEGABYTES
