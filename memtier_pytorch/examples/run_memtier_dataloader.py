#!/usr/bin/env python3
"""Run the MemTier PyTorch UVM DataLoader benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from memtier.benchmark import format_results, run_all


def ensure_binary_file(path: Path, rows: int, dim: int) -> None:
    expected_bytes = rows * dim * np.dtype(np.float32).itemsize
    if path.exists() and path.stat().st_size == expected_bytes:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating {path} with shape [{rows}, {dim}] ({expected_bytes / (1024**3):.2f} GiB)")
    mmap = np.memmap(path, mode="w+", dtype=np.float32, shape=(rows, dim))
    chunk_rows = max(1, min(rows, 65536))
    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        values = np.arange(start * dim, end * dim, dtype=np.float32).reshape(end - start, dim)
        mmap[start:end] = values / max(dim, 1)
    mmap.flush()
    del mmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default="/tmp/emb.bin", help="float32 binary file path")
    parser.add_argument("--rows", type=int, default=1_000_000, help="number of rows in the file")
    parser.add_argument("--dim", type=int, default=768, help="row dimension / number of float32 columns")
    parser.add_argument("--batch-rows", type=int, default=4096, help="rows per DataLoader item")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for the baseline path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.path)
    ensure_binary_file(path, args.rows, args.dim)
    results = run_all(
        path,
        rows=args.rows,
        cols=args.dim,
        batch_rows=args.batch_rows,
        device=args.device,
        num_workers=args.num_workers,
    )
    print(format_results(results))
    for result in results:
        if result.extension_stats:
            print(f"\n{result.name} extension stats:")
            for key, value in result.extension_stats.items():
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
