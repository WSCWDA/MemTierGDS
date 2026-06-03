#!/usr/bin/env python3
"""Small-table torch.nn.EmbeddingBag upper-bound baseline."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

try:
    from embedding_workload import generate_ids
except ImportError:  # pragma: no cover
    from .embedding_workload import generate_ids


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    array = sorted(values)
    idx = min(len(array) - 1, max(0, int(round((pct / 100.0) * (len(array) - 1)))))
    return array[idx]


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("torch embedding baseline requires CUDA")
    trace = np.load(args.trace) if args.trace else generate_ids(
        num_embeddings=args.num_embeddings,
        batch_size=args.batch_size,
        bag_size=args.bag_size,
        num_batches=args.num_batches,
        dist=args.dist,
        zipf_alpha=args.zipf_alpha,
        overlap_ratio=args.overlap_ratio,
        seed=args.seed,
    )
    dtype = torch.float16 if args.dtype_size == 2 else torch.float32
    emb = torch.nn.EmbeddingBag(args.num_embeddings, args.embedding_dim, mode=args.mode, include_last_offset=True, dtype=dtype).cuda()
    offsets = torch.arange(0, args.batch_size * args.bag_size + 1, args.bag_size, device="cuda:0", dtype=torch.long)
    latencies: list[float] = []
    with torch.no_grad():
        for batch in trace:
            ids_np = (batch % np.uint64(args.num_embeddings)).astype(np.int64)
            ids = torch.as_tensor(ids_np, device="cuda:0")
            torch.cuda.synchronize(0)
            start = time.perf_counter()
            out = emb(ids, offsets)
            torch.cuda.synchronize(0)
            latencies.append((time.perf_counter() - start) * 1000.0)
            del out, ids
    total_s = sum(latencies) / 1000.0
    result = {
        "system": "torch_embedding_upper_bound",
        "dist": args.dist,
        "zipf_alpha": args.zipf_alpha,
        "num_embeddings": args.num_embeddings,
        "embedding_dim": args.embedding_dim,
        "rows_per_block": args.rows_per_block,
        "batch_size": args.batch_size,
        "bag_size": args.bag_size,
        "num_batches": int(trace.shape[0]),
        "avg_latency_ms": float(statistics.mean(latencies)) if latencies else 0.0,
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "throughput_lookups_per_sec": float(trace.size / max(total_s, 1e-12)),
        "hbm_hit_rate": 1.0,
        "dram_hit_rate": 0.0,
        "ssd_read_bytes": 0,
        "dram_to_gpu_bytes": 0,
        "posix_reads": 0,
        "gds_reads": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/baselines.jsonl"))
    parser.add_argument("--num-embeddings", type=int, required=True)
    parser.add_argument("--embedding-dim", type=int, required=True)
    parser.add_argument("--rows-per-block", type=int, default=2048)
    parser.add_argument("--dtype-size", type=int, choices=[2, 4], default=2)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--bag-size", type=int, default=32)
    parser.add_argument("--num-batches", type=int, default=1000)
    parser.add_argument("--dist", choices=["uniform", "zipf"], default="uniform")
    parser.add_argument("--zipf-alpha", type=float, default=1.0)
    parser.add_argument("--overlap-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mode", choices=["sum", "mean"], default="sum")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
