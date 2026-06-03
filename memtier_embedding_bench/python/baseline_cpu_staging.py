#!/usr/bin/env python3
"""CPU staging baseline: SSD/POSIX reads per batch, optional torch CUDA copy."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from embedding_workload import generate_ids, ids_to_blocks
    from trace_reader import EmbeddingTrace
except ImportError:  # pragma: no cover
    from .embedding_workload import generate_ids, ids_to_blocks
    from .trace_reader import EmbeddingTrace


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    array = sorted(values)
    idx = min(len(array) - 1, max(0, int(round((pct / 100.0) * (len(array) - 1)))))
    return array[idx]


def _dtype_size(args: argparse.Namespace) -> int:
    if getattr(args, "dtype", None) == "fp16":
        return 2
    if getattr(args, "dtype", None) == "fp32":
        return 4
    return int(args.dtype_size)


def run(args: argparse.Namespace) -> dict:
    trace_name = None
    if args.trace_file or args.trace:
        trace_path = args.trace_file or args.trace
        trace_reader = EmbeddingTrace(trace_path, rows_per_block=args.rows_per_block)
        batches = trace_reader.iter_batches()
        num_batches = trace_reader.num_batches
        trace_name = trace_reader.metadata.get("trace_name", Path(trace_path).stem)
    else:
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
        batches = iter(trace)
        num_batches = int(trace.shape[0])
        trace_name = f"synthetic_{args.dist}"
    dtype_size = _dtype_size(args)
    block_bytes = args.rows_per_block * args.embedding_dim * dtype_size
    fd = os.open(args.embedding_file, os.O_RDONLY)
    latencies: list[float] = []
    bytes_read = 0
    dram_to_gpu = 0
    posix_reads = 0
    use_torch_copy = args.cuda_copy and torch is not None and torch.cuda.is_available()

    try:
        previous_blocks: set[int] = set()
        unique_ids_per_batch: list[int] = []
        unique_blocks_per_batch: list[int] = []
        block_reuse_ratios: list[float] = []
        all_unique_ids: set[int] = set()
        all_unique_blocks: set[int] = set()
        total_lookups = 0
        for batch in batches:
            batch = np.asarray(batch, dtype=np.uint64).reshape(-1)
            total_lookups += int(batch.size)
            blocks = ids_to_blocks(batch, args.rows_per_block)
            block_set = set(int(x) for x in blocks)
            id_set = set(int(x) for x in batch)
            all_unique_ids.update(id_set)
            all_unique_blocks.update(block_set)
            unique_ids_per_batch.append(len(id_set))
            unique_blocks_per_batch.append(len(block_set))
            if previous_blocks:
                block_reuse_ratios.append(len(block_set & previous_blocks) / max(len(block_set), 1))
            previous_blocks = block_set
            start = time.perf_counter()
            for block_id in blocks:
                offset = int(block_id) * block_bytes
                data = os.pread(fd, block_bytes, offset)
                if len(data) != block_bytes:
                    raise RuntimeError(f"short read for block {int(block_id)}: {len(data)} of {block_bytes}")
                bytes_read += len(data)
                posix_reads += 1
                if use_torch_copy:
                    cpu = torch.empty(block_bytes, dtype=torch.uint8, pin_memory=True)
                    cpu.numpy()[:] = np.frombuffer(data, dtype=np.uint8)
                    gpu = cpu.to("cuda:0", non_blocking=True)
                    torch.cuda.synchronize(0)
                    dram_to_gpu += block_bytes
                    del gpu, cpu
            latencies.append((time.perf_counter() - start) * 1000.0)
    finally:
        os.close(fd)

    lookups = int(total_lookups)
    total_s = sum(latencies) / 1000.0
    result = {
        "system": "cpu_staging",
        "dist": args.dist,
        "zipf_alpha": args.zipf_alpha,
        "num_embeddings": args.num_embeddings,
        "embedding_dim": args.embedding_dim,
        "rows_per_block": args.rows_per_block,
        "batch_size": args.batch_size,
        "bag_size": args.bag_size,
        "num_batches": int(num_batches),
        "avg_latency_ms": float(statistics.mean(latencies)) if latencies else 0.0,
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "throughput_lookups_per_sec": float(lookups / max(total_s, 1e-12)),
        "hbm_hit_rate": 0.0,
        "dram_hit_rate": 0.0,
        "ssd_read_bytes": int(bytes_read),
        "dram_to_gpu_bytes": int(dram_to_gpu),
        "posix_reads": int(posix_reads),
        "trace_source": args.trace_source,
        "trace_name": trace_name,
        "num_unique_ids": len(all_unique_ids),
        "num_unique_blocks": len(all_unique_blocks),
        "avg_unique_ids_per_batch": float(statistics.mean(unique_ids_per_batch)) if unique_ids_per_batch else 0.0,
        "avg_unique_blocks_per_batch": float(statistics.mean(unique_blocks_per_batch)) if unique_blocks_per_batch else 0.0,
        "p95_unique_blocks_per_batch": percentile(unique_blocks_per_batch, 95),
        "block_reuse_ratio": float(statistics.mean(block_reuse_ratios)) if block_reuse_ratios else 0.0,
        "gds_reads": 0,
        "cuda_copy_enabled": bool(use_torch_copy),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-file", required=True, type=Path)
    parser.add_argument("--trace", type=Path, help="Backward-compatible synthetic .npy trace path")
    parser.add_argument("--trace-file", type=Path, help="Synthetic, Criteo, or DLRM trace file read through EmbeddingTrace")
    parser.add_argument("--trace-source", choices=["synthetic", "criteo", "dlrm"], default="synthetic")
    parser.add_argument("--output", type=Path, default=Path("results/baselines.jsonl"))
    parser.add_argument("--num-embeddings", type=int, default=0)
    parser.add_argument("--embedding-dim", type=int, required=True)
    parser.add_argument("--rows-per-block", type=int, default=2048)
    parser.add_argument("--dtype-size", type=int, choices=[2, 4], default=2)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], help="Embedding file dtype; overrides --dtype-size")
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--bag-size", type=int, default=32)
    parser.add_argument("--num-batches", type=int, default=1000)
    parser.add_argument("--dist", choices=["uniform", "zipf"], default="uniform")
    parser.add_argument("--zipf-alpha", type=float, default=1.0)
    parser.add_argument("--overlap-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cuda-copy", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
