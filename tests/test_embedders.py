"""What the embedding model costs to run, measured rather than assumed.

The accuracy of what it returns is the retrieval gate's job (`scripts/eval_retrieval.py`).
This file is about the resource the gate cannot see: a server that sits in an editor all
day is killed by memory long before it is killed by being slightly wrong.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

import pytest

from markdown_memory.config import ServerConfig
from markdown_memory.embedders import GEMMA_QUERY_PROMPT
from markdown_memory.model_cache import GEMMA_MODEL_FILE, GEMMA_REVISION, gemma_model_dir

#: One 512-token run measured 448-449 MB across three runs on the reference machine. The
#: ceiling is that with room for an onnxruntime release to move it, and far below the
#: ~1.45 GB the *unquantized* vocabulary table alone used to cost: this catches a graph
#: that stops quantizing the 262144x768 table, which is the whole reason q4 is the pin.
_PEAK_MEGABYTES = 600

_MEASURE_PEAK = """
import numpy as np, onnxruntime, sys
from pathlib import Path

options = onnxruntime.SessionOptions()
options.add_session_config_entry("session.intra_op.allow_spinning", "0")
session = onnxruntime.InferenceSession(
    sys.argv[1], options, providers=["CPUExecutionProvider"]
)
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
def graph() -> Path:
    path = gemma_model_dir(ServerConfig.from_env().model_cache_dir) / GEMMA_MODEL_FILE
    if not path.is_file():
        pytest.skip(f"the embedding model is not cached at {path}")
    return path


@pytest.mark.embedding
def test_one_run_of_the_real_model_stays_under_its_memory_ceiling(graph: Path) -> None:
    """Measured in its own process: VmHWM is a high-water mark and never comes back down.

    In-process this would read whatever the rest of the suite had already allocated, and
    the number it is here to defend is the model's, not pytest's.
    """
    if not Path("/proc/self/status").exists():
        pytest.skip("/proc is not available on this platform")
    finished = subprocess.run(
        [sys.executable, "-c", _MEASURE_PEAK, str(graph)],
        capture_output=True,
        text=True,
        check=True,
    )
    peak = int(finished.stdout.strip())
    assert peak < _PEAK_MEGABYTES, (
        f"one 512-token run peaked at {peak} MB, over the {_PEAK_MEGABYTES} MB ceiling"
    )


def test_two_graphs_at_one_revision_report_different_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The weights identity names the graph, because the revision does not distinguish it.

    `onnx-community/embeddinggemma-300m-ONNX` publishes several graphs at one commit, and
    they disagree: swapping int8 for 4-bit moves a query's vector by about 0.03 cosine,
    far more than the distance search ranks on. Were this the bare revision, an index
    built by one graph and searched by the other would pass `_refuse_foreign_vectors` -
    the query embedded by one model, ranked against another's vectors, with no error and
    a green `index_status`. Indexing catches the swap through `model_name`, but only once
    someone re-indexes; the documented workflow searches first.
    """
    from markdown_memory import model_cache
    from markdown_memory.embedders import EmbeddingGemmaEmbedder

    monkeypatch.setattr(model_cache, "GEMMA_MODEL_FILE", "onnx/model_quantized.onnx")
    int8 = EmbeddingGemmaEmbedder().weights_revision
    monkeypatch.setattr(model_cache, "GEMMA_MODEL_FILE", "onnx/model_q4.onnx")
    q4 = EmbeddingGemmaEmbedder().weights_revision

    assert int8 != q4, "two graphs at one revision must not claim the same weights"
    assert int8 is not None and q4 is not None
    assert model_cache.GEMMA_REVISION in int8 and model_cache.GEMMA_REVISION in q4


def test_a_mismatch_message_distinguishes_two_graphs_at_one_revision() -> None:
    """The short form has to keep what differs, or the message contradicts itself.

    Twelve characters of the revision were enough while one revision meant one graph.
    They are not now: both sides of "the weights changed from X to Y" would print the
    same twelve characters, and an agent reading it would have no way to see what moved.
    """
    from markdown_memory.embedders import short_weights

    revision = "5090578d9565bb06545b4552f76e6bc2c93e4a66"
    int8 = short_weights(f"{revision}/onnx/model_quantized.onnx")
    q4 = short_weights(f"{revision}/onnx/model_q4.onnx")

    assert int8 != q4, f"both graphs shorten to {int8!r}"
    assert "model_quantized" in int8 and "model_q4" in q4
    # fastembed reports a bare snapshot with no graph, and still shortens.
    assert short_weights(revision) == revision[:12]
    assert short_weights(None) == "no readable revision"


def test_the_graph_an_upgrade_left_behind_is_reported_not_hidden(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Upgrading the graph orphans 310 MB *inside* the current folder, not beside it.

    The cache directory is keyed on the revision, and both graphs share one, so the int8
    files an upgrade replaces stay where they are. Only looking at *other* revisions'
    folders would walk straight past them and report nothing.
    """
    import logging

    from markdown_memory import model_cache
    from markdown_memory.embedders import EmbeddingGemmaEmbedder

    model_dir = model_cache.gemma_model_dir(tmp_path)
    (model_dir / "onnx").mkdir(parents=True)
    for name in model_cache.GEMMA_FILES:
        (model_dir / name).write_bytes(b"x")
    (model_dir / model_cache._VERIFIED_STAMP).write_text("{}", encoding="utf-8")
    orphan = model_dir / "onnx" / "model_quantized.onnx_data"
    orphan.write_bytes(b"y" * 4096)

    with caplog.at_level(logging.INFO, logger="markdown_memory.embedders"):
        EmbeddingGemmaEmbedder(cache_dir=tmp_path)._report_other_versions()

    assert str(orphan) in caplog.text, "the orphaned graph was never mentioned"
    assert model_cache._VERIFIED_STAMP not in caplog.text, "the stamp is not wasted space"
    for name in model_cache.GEMMA_FILES:
        assert str(model_dir / name) not in caplog.text, f"{name} is in use, not an orphan"


#: How far a vector may sit from the committed reference and still be the same compute path.
#:
#: Both ends measured by `scripts/compare_compute_paths.py` on the reference machine
#: (x86_64, onnxruntime 1.30.0). Running the same graph twice, and again at a different
#: thread count, moves a vector by **~1e-7**, with every coordinate bit-identical. Forcing
#: the fp32 compute path moves it by **2.1e-4 to 6.1e-4**. This ceiling sits four times
#: below the smaller of those and several hundred times above the noise.
_COMPUTE_PATH_TOLERANCE = 5e-5

#: Past here it is not a compute path, it is a different model. Swapping the int8 graph for
#: the 4-bit one at the same revision moves a query vector by **~0.03** cosine (#20), which
#: is what this has to stay well under while leaving the whole fp32-fallback band below it.
#: Between the two the test skips rather than passes, and CI fails on that skip (the gate
#: runs no CPU that should land there) so the band cannot go quietly green.
_DIFFERENT_MODEL = 1e-3

_REFERENCE = Path(__file__).parent / "fixtures" / "gemma_q4_reference.json"


@pytest.mark.embedding
def test_the_real_model_returns_the_vector_it_was_baselined_on(real_embedder: object) -> None:
    """One fixed string, one committed vector: the only test that says the numbers agree.

    Everything else about this model is checked by proxy - that it loads, that it returns
    768 dimensions, that the manifest matches. None of that notices arithmetic. onnxruntime
    picks its `MatMulNBits` kernel from what the CPU offers, so the same graph can compute
    differently on another machine with no error and no log, and the provenance key names
    the revision and the graph but not the compute path.

    Cosine rather than elementwise: it is the unit search ranks in, and an `allclose` over
    768 coordinates either flakes on one near-zero coordinate or admits exactly the coherent
    shift a compute path produces.
    """
    import json
    import math

    reference = json.loads(_REFERENCE.read_text(encoding="utf-8"))
    # The static identity only. A runtime component - were the provenance key ever widened
    # to name the compute path - would differ on precisely the hosts this test is here to
    # measure, and would fail here before reaching the comparison that carries the finding.
    assert reference["revision"] == GEMMA_REVISION, "regenerate with --update-reference"
    assert reference["graph"] == GEMMA_MODEL_FILE, "regenerate with --update-reference"
    assert reference["prompt"] == GEMMA_QUERY_PROMPT, "the prompt moved; re-baseline it"

    actual = real_embedder.embed_query(reference["text"])  # type: ignore[attr-defined]
    assert len(actual) == len(reference["vector"])
    # Normalisation is what makes the dot product below a cosine, and cosine cannot see a
    # vector that stopped being a unit vector.
    assert abs(math.sqrt(sum(value * value for value in actual)) - 1.0) < 1e-5

    distance = 1.0 - sum(a * b for a, b in zip(reference["vector"], actual, strict=True))
    where = f"{platform.machine()} / {platform.system()}, reference from {reference['produced_on']}"
    # Three outcomes, not two, because "different numbers" and "different model" are not
    # the same finding. onnxruntime asks the CPU for the int8 `MatMulNBits` kernel and
    # quietly computes in fp32 where there is none - measured at 2.1e-4 to 6.1e-4 here,
    # and measured *not* to move a top-1 result on any of the 86 labelled eval queries.
    # Failing a machine for that would be failing it for working correctly; saying nothing
    # would waste the one place that can notice.
    assert distance < _DIFFERENT_MODEL, (
        f"this vector is {distance:.3e} from the reference ({where}), past the "
        f"{_DIFFERENT_MODEL:.0e} that separates a compute path from a different model. "
        "Either the pin moved without the fixture, or these weights are not ours."
    )
    if distance >= _COMPUTE_PATH_TOLERANCE:
        pytest.skip(
            f"this CPU takes a different MatMulNBits path: {distance:.3e} from the reference "
            f"({where}), inside the band measured for the fp32 fallback. What that costs "
            "retrieval has NOT been measured on this host - on the reference machine it left the "
            "labelled section in the top five on both splits, moving the fifth result on one "
            "query in 44. "
            "Run scripts/compare_compute_paths.py here before relying on the numbers."
        )
