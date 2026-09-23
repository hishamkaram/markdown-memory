"""Rewrite EmbeddingGemma's graph to gather vocabulary rows before dequantizing them.

The published graph stores the vocabulary table as `int8 [262144, 768]` - 201M of the
model's 300M parameters - and runs `DequantizeLinear` on it and then `Gather`. So every
run materialises the whole table in float32 (805 MB) to keep the ~20 rows its tokens
need (~60 KB), and onnxruntime's arena holds on to the buffer for reuse. Measured: 1.5 GB
resident after one query, against ~0.5 GB with the two nodes the other way round.

Swapping them is value-identical here, not merely close: the table's quantization is per
tensor (one scale, one zero point, no axis), dequantizing is `(q - zp) * scale` value by
value, and gathering only picks rows. Vectors from the rewritten graph are bit-identical
to the original's, which is asserted against the real model in the tests.

No `onnx` dependency: the wheel is ~17.6 MB and pulls `ml_dtypes` behind it, to edit two
nodes of one pinned graph. This reads the protobuf wire format directly, and copies every
field it does not replace as its original bytes, in its original order. It runs *only* on
the graph whose sha256 is pinned in `indexer.py`, and the sha256 of what it produces is
pinned too, so an accidental change to either end fails verification rather than quietly
shipping a different model. Anything that does not match the expected pattern exactly -
including wire types this does not model - is refused, and the caller falls back to the
original graph.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Refuse anything larger. The pinned graph is ~568 KB; the weights live beside it in an
#: external data file, which this never reads.
MAX_GRAPH_BYTES = 64 * 1024 * 1024

_VARINT, _FIXED64, _LENGTH, _START_GROUP, _END_GROUP, _FIXED32 = range(6)

# ModelProto.graph
_MODEL_GRAPH = 7
# GraphProto.node, .initializer, .input, .output, .value_info
_GRAPH_NODE, _GRAPH_INITIALIZER, _GRAPH_INPUT, _GRAPH_OUTPUT, _GRAPH_VALUE_INFO = 1, 5, 11, 12, 13
# NodeProto.input, .output, .name, .op_type, .attribute
_NODE_INPUT, _NODE_OUTPUT, _NODE_NAME, _NODE_OP_TYPE, _NODE_ATTRIBUTE = 1, 2, 3, 4, 5
_NODE_DOC_STRING, _NODE_DOMAIN = 6, 7
# TensorProto.dims, .data_type, .name
_TENSOR_DIMS, _TENSOR_DATA_TYPE, _TENSOR_NAME = 1, 2, 8
# AttributeProto.name, .i, .type
_ATTRIBUTE_NAME, _ATTRIBUTE_INT, _ATTRIBUTE_TYPE = 1, 3, 20
#: The only fields an `axis = 0` attribute may carry. Anything else - `ref_attr_name`
#: above all, which makes the value come from elsewhere at instantiation - means the
#: attribute does not say what it appears to say.
_AXIS_FIELDS = frozenset({_ATTRIBUTE_NAME, _ATTRIBUTE_INT, _ATTRIBUTE_TYPE})
# ValueInfoProto.name
_VALUE_INFO_NAME = 1
_TENSOR_INT8 = 3


class _RefusedError(Exception):
    """Something in the graph is not what this rewrite models. Nothing is changed."""


@dataclass(frozen=True, slots=True)
class _Field:
    """One wire-format field: where the whole thing is, and where its payload is."""

    number: int
    wire: int
    start: int  # of the tag
    end: int  # one past the field
    value_start: int
    value_end: int


def _read_varint(data: bytes, position: int, end: int) -> tuple[int, int]:
    result = shift = 0
    while position < end:
        byte = data[position]
        position += 1
        if shift == 63 and byte > 1:
            raise _RefusedError("varint does not fit in 64 bits")
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, position
        shift += 7
        if shift > 63:
            raise _RefusedError("varint longer than 64 bits")
    raise _RefusedError("varint runs off the end of the message")


def _fields(data: bytes, start: int, end: int) -> list[_Field]:
    """Every field of one message, in order, as spans. Groups are refused."""
    found: list[_Field] = []
    position = start
    while position < end:
        tag_start = position
        tag, position = _read_varint(data, position, end)
        number, wire = tag >> 3, tag & 7
        if number == 0:
            raise _RefusedError("field number 0 does not exist")
        if wire == _LENGTH:
            length, position = _read_varint(data, position, end)
            value_start, position = position, position + length
        elif wire == _VARINT:
            value_start = position
            _, position = _read_varint(data, position, end)
        elif wire == _FIXED64:
            value_start, position = position, position + 8
        elif wire == _FIXED32:
            value_start, position = position, position + 4
        else:
            raise _RefusedError(f"wire type {wire} is not modelled")
        if position > end:
            raise _RefusedError("field runs past the end of its message")
        found.append(_Field(number, wire, tag_start, position, value_start, position))
    return found


def _only(fields: list[_Field], number: int) -> _Field | None:
    matches = [field for field in fields if field.number == number]
    if len(matches) > 1:
        raise _RefusedError(f"field {number} appears {len(matches)} times")
    return matches[0] if matches else None


def _text(data: bytes, field: _Field) -> str:
    return data[field.value_start : field.value_end].decode("utf-8", "replace")


def _strings(data: bytes, fields: list[_Field], number: int) -> list[str]:
    return [_text(data, field) for field in fields if field.number == number]


def _write_varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | 0x80 if value else byte)
        if not value:
            return bytes(out)


def _delimited(number: int, payload: bytes) -> bytes:
    return _write_varint(number << 3 | _LENGTH) + _write_varint(len(payload)) + payload


def _node_bytes(op_type: str, inputs: list[str], outputs: list[str], name: str) -> bytes:
    """A NodeProto with no attributes and the default domain, which is all this builds."""
    payload = b"".join(_delimited(_NODE_INPUT, value.encode("utf-8")) for value in inputs)
    payload += b"".join(_delimited(_NODE_OUTPUT, value.encode("utf-8")) for value in outputs)
    payload += _delimited(_NODE_NAME, name.encode("utf-8"))
    payload += _delimited(_NODE_OP_TYPE, op_type.encode("utf-8"))
    return _delimited(_GRAPH_NODE, payload)


@dataclass(frozen=True, slots=True)
class _Node:
    field: _Field
    op_type: str
    name: str
    inputs: list[str]
    outputs: list[str]
    attributes: list[_Field]
    #: Anything beyond input, output, name and op_type. The replacements are rebuilt from
    #: those four, so a node carrying more than them is one this must not rewrite.
    extras: list[int]


def _nodes(data: bytes, graph_fields: list[_Field]) -> list[_Node]:
    parsed: list[_Node] = []
    for field in graph_fields:
        if field.number != _GRAPH_NODE:
            continue
        inner = _fields(data, field.value_start, field.value_end)
        op_type = _only(inner, _NODE_OP_TYPE)
        name = _only(inner, _NODE_NAME)
        parsed.append(
            _Node(
                field=field,
                op_type=_text(data, op_type) if op_type else "",
                name=_text(data, name) if name else "",
                inputs=_strings(data, inner, _NODE_INPUT),
                outputs=_strings(data, inner, _NODE_OUTPUT),
                attributes=[entry for entry in inner if entry.number == _NODE_ATTRIBUTE],
                extras=[
                    entry.number
                    for entry in inner
                    if entry.number
                    not in {
                        _NODE_INPUT,
                        _NODE_OUTPUT,
                        _NODE_NAME,
                        _NODE_OP_TYPE,
                        _NODE_ATTRIBUTE,
                    }
                ],
            )
        )
    return parsed


def _initializers(data: bytes, graph_fields: list[_Field]) -> dict[str, tuple[int, list[int]]]:
    """Each initializer's element type and shape, by name."""
    found: dict[str, tuple[int, list[int]]] = {}
    for field in graph_fields:
        if field.number != _GRAPH_INITIALIZER:
            continue
        inner = _fields(data, field.value_start, field.value_end)
        name = _only(inner, _TENSOR_NAME)
        data_type = _only(inner, _TENSOR_DATA_TYPE)
        if name is None or data_type is None:
            continue
        dims: list[int] = []
        for entry in inner:
            if entry.number != _TENSOR_DIMS:
                continue
            if entry.wire == _VARINT:
                dims.append(_read_varint(data, entry.value_start, entry.value_end)[0])
            else:  # packed dims
                position = entry.value_start
                while position < entry.value_end:
                    dimension, position = _read_varint(data, position, entry.value_end)
                    dims.append(dimension)
        found[_text(data, name)] = (
            _read_varint(data, data_type.value_start, data_type.end)[0],
            dims,
        )
    return found


def _elements(shape: list[int]) -> int:
    total = 1
    for dimension in shape:
        total *= dimension
    return total


def _axis_is_zero(data: bytes, node: _Node) -> bool:
    for attribute in node.attributes:
        inner = _fields(data, attribute.value_start, attribute.value_end)
        if any(field.number not in _AXIS_FIELDS for field in inner):
            return False  # an attribute carrying more than a plain integer value
        name = _only(inner, _ATTRIBUTE_NAME)
        if name is None or _text(data, name) != "axis":
            return False  # an attribute this does not model
        value = _only(inner, _ATTRIBUTE_INT)
        if value is None or _read_varint(data, value.value_start, value.end)[0] != 0:
            return False
    return True


def gather_before_dequantize(model: bytes) -> bytes | None:
    """The graph with the vocabulary gathered first, or None when it is not that graph.

    None is a refusal, never a failure: the caller keeps the original graph, which is
    correct and only costs the memory this saves.
    """
    try:
        return _rewrite(model)
    except _RefusedError as refusal:
        logger.warning(
            "Not rewriting the embedding graph (%s); it will run as published, which "
            "costs about 1 GB more memory per query",
            refusal,
        )
        return None


def _rewrite(model: bytes) -> bytes:
    if len(model) > MAX_GRAPH_BYTES:
        raise _RefusedError(f"the graph is {len(model)} bytes, over the {MAX_GRAPH_BYTES} limit")
    top = _fields(model, 0, len(model))
    graph_field = _only(top, _MODEL_GRAPH)
    if graph_field is None:
        raise _RefusedError("the model has no graph")
    graph = _fields(model, graph_field.value_start, graph_field.value_end)
    nodes = _nodes(model, graph)
    initializers = _initializers(model, graph)

    consumers: dict[str, int] = {}
    for node in nodes:
        for name in node.inputs:
            consumers[name] = consumers.get(name, 0) + 1

    produced_by_graph = {
        _value_info_name(model, field) for field in graph if field.number == _GRAPH_OUTPUT
    }
    written_by = {name: node.field.start for node in nodes for name in node.outputs}
    # Every name already spoken for, so the one the rewrite introduces cannot shadow one.
    # Graph inputs and `value_info` entries name tensors no node writes, so a name can be
    # declared there and still be free of every other set here.
    declared = {
        _value_info_name(model, field)
        for field in graph
        if field.number in (_GRAPH_INPUT, _GRAPH_VALUE_INFO)
    }
    taken = set(written_by) | set(consumers) | set(initializers) | produced_by_graph | declared
    candidates = [
        (dequantize, gather)
        for dequantize in nodes
        if dequantize.op_type == "DequantizeLinear"
        for gather in _matching_gather(
            model, dequantize, nodes, initializers, consumers, produced_by_graph, taken, written_by
        )
    ]
    if len(candidates) != 1:
        raise _RefusedError(f"{len(candidates)} nodes match the pattern, expected exactly one")
    dequantize, gather = candidates[0]

    table, scale, zero_point = (dequantize.inputs + ["", ""])[:3]
    rows = f"{dequantize.outputs[0]}_rows"
    # Both nodes keep their positions, so the graph stays topologically sorted: the new
    # Gather reads a graph input and an initializer, and the new DequantizeLinear reads
    # what that Gather writes.
    new_gather = _node_bytes("Gather", [table, gather.inputs[1]], [rows], gather.name)
    new_dequantize = _node_bytes(
        "DequantizeLinear",
        [rows, scale, zero_point] if zero_point else [rows, scale],
        [gather.outputs[0]],
        dequantize.name,
    )

    dangling = dequantize.outputs[0]
    out = bytearray()
    for field in graph:
        if field.start == dequantize.field.start:
            out += new_gather
        elif field.start == gather.field.start:
            out += new_dequantize
        elif field.number == _GRAPH_VALUE_INFO and _value_info_name(model, field) == dangling:
            continue  # nothing produces the dequantized table any more
        else:
            out += model[field.start : field.end]

    rewritten = bytearray()
    for field in top:
        if field.start == graph_field.start:
            rewritten += _delimited(_MODEL_GRAPH, bytes(out))
        else:
            rewritten += model[field.start : field.end]
    return bytes(rewritten)


def _value_info_name(model: bytes, field: _Field) -> str:
    name = _only(_fields(model, field.value_start, field.value_end), _VALUE_INFO_NAME)
    return _text(model, name) if name else ""


def _matching_gather(
    model: bytes,
    dequantize: _Node,
    nodes: list[_Node],
    initializers: dict[str, tuple[int, list[int]]],
    consumers: dict[str, int],
    outputs: set[str],
    taken: set[str],
    written_by: dict[str, int],
) -> list[_Node]:
    """The one Gather this DequantizeLinear may swap with, if everything lines up."""
    if dequantize.attributes or not 2 <= len(dequantize.inputs) <= 3:
        return []  # `axis` or `block_size` would make the swap change the values
    if dequantize.extras:
        # A domain puts the operator in another namespace, and a doc string would be
        # dropped by the rebuild. Either way this is not the node this knows how to move.
        return []
    table = initializers.get(dequantize.inputs[0])
    if table is None or table[0] != _TENSOR_INT8 or len(table[1]) != 2:
        return []
    for name in dequantize.inputs[1:]:
        parameter = initializers.get(name)
        # Per tensor, which is what makes gathering first give identical values. One
        # element counts as per tensor by onnxruntime's rule, not by the ONNX spec's.
        if parameter is None or _elements(parameter[1]) != 1:
            return []
    if len(dequantize.outputs) != 1:
        return []
    produced = dequantize.outputs[0]
    if consumers.get(produced) != 1:
        return []
    if produced in outputs:
        # The graph hands the dequantized table out; removing its producer would leave an
        # output nothing writes.
        return []
    if f"{produced}_rows" in taken:
        return []  # the name the rewrite introduces belongs to something already
    return [
        node
        for node in nodes
        if node.op_type == "Gather"
        and node.inputs[:1] == [produced]
        and len(node.inputs) == 2
        and len(node.outputs) == 1
        and not node.extras
        and _axis_is_zero(model, node)
        # Positions are kept, so swapping a Gather that comes first would put the
        # DequantizeLinear before the rows it reads and leave the graph unsorted.
        and node.field.start > dequantize.field.start
        # The indices move up to the DequantizeLinear's position, so whatever produces
        # them has to be there already: a graph input, an initializer, or a node above it.
        and written_by.get(node.inputs[1], -1) < dequantize.field.start
    ]
