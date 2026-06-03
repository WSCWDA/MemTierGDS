#!/usr/bin/env python3
"""Convert DLRM/TorchRec sparse ID files into the MemTier trace NPZ format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from analyze_trace import analyze_trace_file, write_csv
from criteo_trace_dataset import field_offsets_from_hash_sizes, parse_hash_sizes

KEY_CANDIDATES = ("sparse_ids", "sparse", "X_cat", "cat", "lS_i", "indices")


def detect_format(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return "npy"
    if suffix == ".npz":
        return "npz"
    if suffix == ".jsonl":
        return "jsonl"
    raise ValueError(f"cannot auto-detect input format for {path}")


def load_sparse(path: Path, fmt: str, max_samples: int | None) -> np.ndarray:
    if fmt == "npy":
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        arr = np.asarray(arr)
    elif fmt == "npz":
        data = np.load(path, allow_pickle=False)
        key = next((k for k in KEY_CANDIDATES if k in data), None)
        if key is None:
            raise ValueError(f"no sparse key found in {path}; tried {KEY_CANDIDATES}")
        arr = np.asarray(data[key])
    elif fmt == "jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                row = json.loads(line)
                rows.append(row.get("sparse_ids") or row.get("ids") or row.get("sparse"))
                if max_samples and len(rows) >= max_samples:
                    break
        arr = np.asarray(rows, dtype=np.uint64)
    else:
        raise ValueError(f"unsupported input format: {fmt}")

    if arr.ndim == 3:
        arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError(f"unsupported sparse array shape: {arr.shape}")
    if max_samples:
        arr = arr[:max_samples]
    return np.asarray(arr, dtype=np.uint64)


def apply_id_mapping(ids: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, list[int], list[int], int]:
    num_fields = int(ids.shape[1])
    if args.hash_size:
        hash_sizes = parse_hash_sizes(args.hash_size, num_fields)
    elif args.num_embeddings:
        hash_sizes = [max(1, int(args.num_embeddings) // num_fields)] * num_fields
    else:
        max_id = int(ids.max()) if ids.size else 0
        hash_sizes = [max_id + 1] * num_fields
    offsets = field_offsets_from_hash_sizes(hash_sizes)
    mapped = np.empty_like(ids, dtype=np.uint64)
    if args.field_offset_mode == "per_field_offset":
        for field in range(num_fields):
            mapped[:, field] = np.uint64(offsets[field]) + (ids[:, field] % np.uint64(hash_sizes[field]))
    elif args.field_offset_mode == "shared_hash":
        total = int(args.num_embeddings or sum(hash_sizes))
        mapped[:, :] = ids % np.uint64(total)
    elif args.field_offset_mode == "direct":
        total = int(args.num_embeddings or (int(ids.max()) + 1 if ids.size else 0))
        mapped[:, :] = ids % np.uint64(max(total, 1))
    else:
        raise ValueError(f"unsupported field_offset_mode: {args.field_offset_mode}")
    return mapped, offsets, hash_sizes, int(args.num_embeddings or sum(hash_sizes))


def convert(args: argparse.Namespace) -> dict:
    fmt = detect_format(args.input, args.input_format)
    raw = load_sparse(args.input, fmt, args.max_samples)
    ids, offsets, hash_sizes, total_embeddings = apply_id_mapping(raw, args)
    block_ids = ids // np.uint64(args.rows_per_block)
    bag_size = int(ids.shape[1])
    embedding_offsets = np.arange(0, ids.shape[0] * bag_size + 1, bag_size, dtype=np.uint64)
    metadata = {
        "trace_source": "dlrm",
        "trace_name": args.output.stem,
        "input_file": str(args.input),
        "input_format": fmt,
        "num_samples": int(ids.shape[0]),
        "num_fields": bag_size,
        "batch_size": int(args.batch_size),
        "bag_size": bag_size,
        "rows_per_block": int(args.rows_per_block),
        "field_offset_mode": args.field_offset_mode,
        "hash_sizes": [int(x) for x in hash_sizes],
        "field_offsets": [int(x) for x in offsets],
        "total_num_embeddings": total_embeddings,
        "trace_layout": "samples_by_fields",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        ids=ids,
        block_ids=block_ids.astype(np.uint64),
        offsets=embedding_offsets,
        field_offsets=np.asarray(offsets, dtype=np.uint64),
        hash_sizes=np.asarray(hash_sizes, dtype=np.uint64),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    stats_dir = args.stats_dir or (ROOT / "results" / "trace_stats")
    stats_json = stats_dir / f"{args.output.stem}.json"
    stats_csv = stats_dir / f"{args.output.stem}.csv"
    stats = analyze_trace_file(args.output, rows_per_block=args.rows_per_block, topk=args.topk, block_bytes=args.block_bytes)
    stats_json.parent.mkdir(parents=True, exist_ok=True)
    stats_json.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    write_csv(stats, stats_csv)
    print(json.dumps({"trace": str(args.output), "stats_json": str(stats_json), **stats}, indent=2))
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--input-format", choices=["npy", "npz", "jsonl", "auto"], default="auto")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows-per-block", required=True, type=int)
    parser.add_argument("--num-embeddings", type=int)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--field-offset-mode", choices=["per_field_offset", "shared_hash", "direct"], default="direct")
    parser.add_argument("--hash-size")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--stats-dir", type=Path)
    parser.add_argument("--block-bytes", type=int, default=2048 * 128 * 2)
    return parser.parse_args()


def main() -> None:
    convert(parse_args())


if __name__ == "__main__":
    main()
