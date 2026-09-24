"""What the embedding model costs to run, measured rather than assumed.

The accuracy of what it returns is the retrieval gate's job (`scripts/eval_retrieval.py`).
This file is about the resource the gate cannot see: a server that sits in an editor all
day is killed by memory long before it is killed by being slightly wrong.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from markdown_memory.config import ServerConfig
from markdown_memory.model_cache import GEMMA_MODEL_FILE, gemma_model_dir

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
