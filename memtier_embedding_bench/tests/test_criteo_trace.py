from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_trace import analyze_trace_file
from criteo_trace_dataset import parse_criteo_line, stable_hash64
from preprocess_criteo_trace import preprocess
from trace_reader import EmbeddingTrace


def _fake_row(label: int, row_id: int) -> str:
    dense = [str((row_id + i) % 7) for i in range(13)]
    cats = [f"{row_id:02x}_{i:02x}" for i in range(26)]
    cats[3] = ""
    return "\t".join([str(label), *dense, *cats])


def test_criteo_preprocess_reader_and_analyze(tmp_path: Path) -> None:
    raw = tmp_path / "train.txt"
    raw.write_text("\n".join(_fake_row(i % 2, i) for i in range(5)) + "\n", encoding="utf-8")

    label, cats = parse_criteo_line(raw.read_text(encoding="utf-8").splitlines()[0])
    assert label == 0
    assert len(cats) == 26
    assert stable_hash64("abc") == stable_hash64("abc")
    assert stable_hash64("abc") != stable_hash64("abd")

    out = tmp_path / "trace.npz"
    args = argparse.Namespace(
        input=raw,
        input_glob=None,
        output=out,
        max_samples=None,
        num_fields=26,
        batch_size=2,
        bag_size=26,
        hash_size="1024",
        global_id_mode="per_field_offset",
        rows_per_block=16,
        drop_empty=False,
        save_ids=True,
        save_blocks=True,
        dtype="uint64",
        seed=0,
        topk=10,
        stats_dir=tmp_path / "stats",
        block_bytes=16 * 4,
        progress_interval=0,
    )
    preprocess(args)

    data = np.load(out, allow_pickle=False)
    assert data["ids"].shape == (5, 26)
    assert np.array_equal(data["block_ids"], data["ids"] // np.uint64(16))

    trace = EmbeddingTrace(out, rows_per_block=16)
    batches = list(trace.iter_batches())
    assert trace.num_batches == 3
    assert batches[0].shape == (2 * 26,)
    assert batches[-1].shape == (1 * 26,)
    assert np.array_equal(trace.get_batch_block_ids(0), np.unique(batches[0] // np.uint64(16)))

    stats = analyze_trace_file(out, rows_per_block=16, topk=5, block_bytes=64)
    for key in [
        "total_lookups",
        "unique_ids",
        "unique_blocks",
        "block_frequency",
        "avg_unique_blocks_per_batch",
        "p95_unique_blocks_per_batch",
        "cache_simulation",
        "multi_gpu_overlap",
    ]:
        assert key in stats
