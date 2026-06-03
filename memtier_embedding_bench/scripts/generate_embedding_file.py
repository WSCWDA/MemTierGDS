#!/usr/bin/env python3
"""Generate a row-major binary embedding table without materializing it in RAM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def dtype_from_arg(name: str) -> np.dtype:
    if name == "fp16":
        return np.dtype(np.float16)
    if name == "fp32":
        return np.dtype(np.float32)
    raise ValueError(f"unsupported dtype: {name}")


def generate_embedding_file(output: Path, num_embeddings: int, embedding_dim: int, dtype: str, seed: int, chunk_mb: int) -> dict:
    if num_embeddings <= 0 or embedding_dim <= 0 or chunk_mb <= 0:
        raise ValueError("num_embeddings, embedding_dim, and chunk_mb must be positive")
    np_dtype = dtype_from_arg(dtype)
    row_bytes = embedding_dim * np_dtype.itemsize
    rows_per_chunk = max(1, (chunk_mb * 1024 * 1024) // row_bytes)
    rng = np.random.default_rng(seed)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        for start in range(0, num_embeddings, rows_per_chunk):
            rows = min(rows_per_chunk, num_embeddings - start)
            # Generate in fp32 and cast so fp16/fp32 files are deterministic for the
            # same seed while still bounded by chunk_mb.
            chunk = rng.standard_normal((rows, embedding_dim), dtype=np.float32).astype(np_dtype, copy=False)
            f.write(chunk.tobytes(order="C"))

    metadata = {
        "num_embeddings": int(num_embeddings),
        "embedding_dim": int(embedding_dim),
        "dtype": dtype,
        "dtype_size": int(np_dtype.itemsize),
        "file_size_bytes": int(output.stat().st_size),
        "layout": "row_major",
        "seed": int(seed),
    }
    meta_path = output.with_suffix(output.suffix + ".metadata.json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "metadata": str(meta_path), **metadata}, indent=2))
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-embeddings", required=True, type=int)
    parser.add_argument("--embedding-dim", required=True, type=int)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--chunk-mb", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_embedding_file(
        args.output,
        args.num_embeddings,
        args.embedding_dim,
        args.dtype,
        args.seed,
        args.chunk_mb,
    )


if __name__ == "__main__":
    main()
