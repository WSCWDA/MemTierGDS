#!/usr/bin/env python3
"""Generate reproducible DLRM-like embedding ID traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def generate_ids(
    *,
    num_embeddings: int,
    batch_size: int,
    bag_size: int,
    num_batches: int,
    dist: str = "uniform",
    zipf_alpha: float = 1.0,
    overlap_ratio: float = 0.0,
    seed: int = 1234,
) -> np.ndarray:
    """Return trace with shape [num_batches, batch_size * bag_size].

    overlap_ratio controls how many IDs in batch N are copied from batch N-1,
    modeling multi-GPU or pipeline overlap/reuse without needing multiple GPUs.
    """

    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError("overlap_ratio must be in [0, 1)")
    total_per_batch = batch_size * bag_size
    rng = np.random.default_rng(seed)
    trace = np.empty((num_batches, total_per_batch), dtype=np.uint64)
    overlap = int(total_per_batch * overlap_ratio)

    for b in range(num_batches):
        fresh_count = total_per_batch - (overlap if b > 0 else 0)
        if dist == "uniform":
            fresh = rng.integers(0, num_embeddings, size=fresh_count, dtype=np.uint64)
        elif dist == "zipf":
            # Bounded Zipf-like ranks over [0, num_embeddings).  We use an
            # inverse-CDF approximation instead of numpy.random.zipf because the
            # paper sweeps alpha=0.8, and NumPy requires alpha > 1.
            u = rng.random(fresh_count)
            n = float(num_embeddings)
            if abs(zipf_alpha - 1.0) < 1e-8:
                ranks = np.floor(np.exp(u * np.log(n)))
            else:
                exponent = 1.0 - zipf_alpha
                ranks = np.floor((u * (n**exponent - 1.0) + 1.0) ** (1.0 / exponent))
            fresh = np.minimum(ranks.astype(np.uint64), np.uint64(num_embeddings - 1))
        else:
            raise ValueError(f"unsupported dist: {dist}")

        if b > 0 and overlap > 0:
            trace[b, :overlap] = trace[b - 1, :overlap]
            trace[b, overlap:] = fresh
        else:
            trace[b, :] = fresh
    return trace


def ids_to_blocks(ids: np.ndarray, rows_per_block: int) -> np.ndarray:
    return np.unique(ids.astype(np.uint64) // np.uint64(rows_per_block))


def estimate_block_reuse(trace: np.ndarray, rows_per_block: int) -> dict:
    previous: set[int] = set()
    reuse_rates = []
    unique_counts = []
    for batch in trace:
        blocks = set(int(x) for x in ids_to_blocks(batch, rows_per_block))
        unique_counts.append(len(blocks))
        if previous:
            reuse_rates.append(len(blocks & previous) / max(len(blocks), 1))
        previous = blocks
    return {
        "avg_unique_blocks_per_batch": float(np.mean(unique_counts)) if unique_counts else 0.0,
        "avg_inter_batch_block_reuse": float(np.mean(reuse_rates)) if reuse_rates else 0.0,
        "num_batches": int(trace.shape[0]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-source", choices=["synthetic", "criteo", "dlrm"], default="synthetic")
    parser.add_argument("--trace-file", type=Path, help="Existing Criteo/DLRM trace file to inspect instead of generating synthetic IDs")
    parser.add_argument("--criteo-input", type=Path, help="Raw Criteo input; run scripts/preprocess_criteo_trace.py first")
    parser.add_argument("--criteo-input-glob", help="Raw Criteo day_* glob; run scripts/preprocess_criteo_trace.py first")
    parser.add_argument("--dlrm-input", type=Path, help="DLRM sparse input; run scripts/convert_dlrm_trace.py first")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dist", choices=["uniform", "zipf"], default="uniform")
    parser.add_argument("--zipf-alpha", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--bag-size", type=int, required=True)
    parser.add_argument("--num-batches", type=int, required=True)
    parser.add_argument("--num-embeddings", type=int, required=True)
    parser.add_argument("--overlap-ratio", type=float, default=0.0)
    parser.add_argument("--rows-per-block", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trace_source != "synthetic":
        if args.trace_file:
            try:
                from trace_reader import EmbeddingTrace
            except ImportError:  # pragma: no cover
                from .trace_reader import EmbeddingTrace
            trace_obj = EmbeddingTrace(args.trace_file, rows_per_block=args.rows_per_block)
            meta = {"trace_source": args.trace_source, "trace_file": str(args.trace_file), "num_batches": trace_obj.num_batches, "metadata": trace_obj.metadata}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
            print(json.dumps(meta, indent=2, default=str))
            return
        tool = "scripts/preprocess_criteo_trace.py" if args.trace_source == "criteo" else "scripts/convert_dlrm_trace.py"
        raise SystemExit(f"trace-source={args.trace_source} requires a converted --trace-file. Run {tool} first.")

    trace = generate_ids(
        num_embeddings=args.num_embeddings,
        batch_size=args.batch_size,
        bag_size=args.bag_size,
        num_batches=args.num_batches,
        dist=args.dist,
        zipf_alpha=args.zipf_alpha,
        overlap_ratio=args.overlap_ratio,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, trace)
    summary = estimate_block_reuse(trace, args.rows_per_block)
    meta = {**vars(args), **summary, "trace_shape": list(trace.shape), "dtype": str(trace.dtype)}
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()
