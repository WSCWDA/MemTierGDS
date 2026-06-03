#!/usr/bin/env python3
"""ctypes wrapper for the Phase 1/2 MemTier embedding runtime."""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import time
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from embedding_workload import generate_ids, ids_to_blocks
except ImportError:  # pragma: no cover
    from .embedding_workload import generate_ids, ids_to_blocks


class MemTierEmbeddingConfig(ctypes.Structure):
    _fields_ = [
        ("num_embeddings", ctypes.c_uint64),
        ("embedding_dim", ctypes.c_uint32),
        ("rows_per_block", ctypes.c_uint32),
        ("dtype_size", ctypes.c_uint32),
        ("dram_cache_bytes", ctypes.c_uint64),
        ("hbm_cache_bytes", ctypes.c_uint64),
        ("enable_gds", ctypes.c_int),
        ("enable_prefetch", ctypes.c_int),
    ]


def default_library_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "build" / "libmemtier_embedding.so",
        root / "build" / "Release" / "libmemtier_embedding.so",
        root / "libmemtier_embedding.so",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


class MemTierEmbeddingBag:
    """Lookup unique blocks through libmemtier_embedding.so and report metrics.

    Phase 1/2 focuses on the runtime skeleton, residency table, DRAM cache, SSD
    reads, and HBM copy accounting.  The embedding-bag CUDA launcher is exported
    by the library for future use, but this Python wrapper does not yet allocate
    output tensors or call it.
    """

    def __init__(
        self,
        embedding_file: str | Path,
        *,
        num_embeddings: int,
        embedding_dim: int,
        rows_per_block: int,
        dtype_size: int,
        dram_cache_bytes: int,
        hbm_cache_bytes: int,
        lib_path: str | Path | None = None,
        enable_gds: bool = False,
        enable_prefetch: bool = False,
    ) -> None:
        self.embedding_file = str(embedding_file)
        self.rows_per_block = int(rows_per_block)
        self.lib = ctypes.CDLL(str(lib_path or default_library_path()))
        self._bind_api()
        cfg = MemTierEmbeddingConfig(
            int(num_embeddings),
            int(embedding_dim),
            int(rows_per_block),
            int(dtype_size),
            int(dram_cache_bytes),
            int(hbm_cache_bytes),
            int(enable_gds),
            int(enable_prefetch),
        )
        self.handle = self.lib.memtier_init_embedding(self.embedding_file.encode(), ctypes.byref(cfg))
        if not self.handle:
            raise RuntimeError("memtier_init_embedding failed; check CUDA, file path, and cache sizes")

    def _bind_api(self) -> None:
        self.lib.memtier_init_embedding.argtypes = [ctypes.c_char_p, ctypes.POINTER(MemTierEmbeddingConfig)]
        self.lib.memtier_init_embedding.restype = ctypes.c_void_p
        self.lib.memtier_lookup_blocks.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.memtier_lookup_blocks.restype = ctypes.c_int
        self.lib.memtier_release_blocks.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        self.lib.memtier_release_blocks.restype = ctypes.c_int
        self.lib.memtier_get_metrics.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        self.lib.memtier_get_metrics.restype = ctypes.c_int
        self.lib.memtier_destroy.argtypes = [ctypes.c_void_p]
        self.lib.memtier_destroy.restype = None

    def lookup_blocks(self, block_ids: np.ndarray) -> tuple[list[int], list[int]]:
        blocks = np.ascontiguousarray(block_ids.astype(np.uint64, copy=False))
        n = int(blocks.size)
        ptrs = (ctypes.c_void_p * n)()
        paths = (ctypes.c_int * n)()
        rc = self.lib.memtier_lookup_blocks(
            self.handle,
            blocks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            n,
            ptrs,
            paths,
        )
        if rc != 0:
            raise RuntimeError("memtier_lookup_blocks failed")
        return [int(ptrs[i] or 0) for i in range(n)], [int(paths[i]) for i in range(n)]

    def forward(self, ids: np.ndarray, bag_size: int, mode: str = "sum") -> dict:
        if mode not in {"sum", "mean"}:
            raise ValueError("mode must be 'sum' or 'mean'")
        del bag_size  # kernel execution is TODO in Phase 1/2.
        block_ids = ids_to_blocks(ids, self.rows_per_block)
        start = time.perf_counter()
        ptrs, paths = self.lookup_blocks(block_ids)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {"num_blocks": int(len(block_ids)), "latency_ms": elapsed_ms, "paths": paths, "device_ptrs": ptrs}

    def metrics(self) -> dict:
        buf = ctypes.create_string_buffer(8192)
        rc = self.lib.memtier_get_metrics(self.handle, buf, len(buf))
        if rc != 0:
            raise RuntimeError("memtier_get_metrics failed")
        return json.loads(buf.value.decode())

    def close(self) -> None:
        if getattr(self, "handle", None):
            self.lib.memtier_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):  # pragma: no cover
        self.close()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    array = sorted(values)
    idx = min(len(array) - 1, max(0, int(round((pct / 100.0) * (len(array) - 1)))))
    return array[idx]


def run(args: argparse.Namespace) -> dict:
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
    latencies: list[float] = []
    with MemTierEmbeddingBag(
        args.embedding_file,
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        rows_per_block=args.rows_per_block,
        dtype_size=args.dtype_size,
        dram_cache_bytes=int(args.dram_cache_gb * (1024**3)),
        hbm_cache_bytes=int(args.hbm_cache_gb * (1024**3)),
        lib_path=args.lib,
        enable_gds=args.enable_gds,
        enable_prefetch=args.enable_prefetch,
    ) as mt:
        for batch in trace:
            out = mt.forward(batch, args.bag_size, args.mode)
            latencies.append(out["latency_ms"])
        metrics = mt.metrics()

    lookups = int(trace.size)
    total_s = sum(latencies) / 1000.0
    posix_reads = int(metrics.get("posix_reads", 0))
    hbm_hits = int(metrics.get("hbm_hits", 0))
    dram_hits = int(metrics.get("dram_hits", 0))
    denom = max(hbm_hits + dram_hits + posix_reads + int(metrics.get("gds_reads", 0)), 1)
    result = {
        "system": "memtier",
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
        "throughput_lookups_per_sec": float(lookups / max(total_s, 1e-12)),
        "hbm_hit_rate": hbm_hits / denom,
        "dram_hit_rate": dram_hits / denom,
        "ssd_read_bytes": int(metrics.get("bytes_read_from_ssd", 0)),
        "dram_to_gpu_bytes": int(metrics.get("bytes_copied_dram_to_gpu", 0)),
        "posix_reads": posix_reads,
        "gds_reads": int(metrics.get("gds_reads", 0)),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-file", required=True, type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--lib", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/memtier.jsonl"))
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
    parser.add_argument("--dram-cache-gb", type=float, default=32.0)
    parser.add_argument("--hbm-cache-gb", type=float, default=4.0)
    parser.add_argument("--mode", choices=["sum", "mean"], default="sum")
    parser.add_argument("--enable-gds", action="store_true")
    parser.add_argument("--enable-prefetch", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
