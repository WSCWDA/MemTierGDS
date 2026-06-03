#!/usr/bin/env python3
"""Convert raw Criteo Kaggle/Terabyte text rows into a MemTier trace NPZ."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from analyze_trace import analyze_trace_file, write_csv
from criteo_trace_dataset import (
    categorical_to_ids,
    field_offsets_from_hash_sizes,
    iter_criteo_rows,
    parse_hash_sizes,
)


def dtype_from_name(name: str):
    if name == "uint32":
        return np.uint32
    if name == "uint64":
        return np.uint64
    raise ValueError(f"unsupported dtype: {name}")


def input_paths(single: Path | None, pattern: str | None) -> list[Path]:
    paths: list[Path] = []
    if single:
        paths.append(single)
    if pattern:
        paths.extend(Path(p) for p in sorted(glob.glob(pattern)))
    if not paths:
        raise ValueError("one of --input or --input-glob is required")
    return paths


def preprocess(args: argparse.Namespace) -> dict:
    paths = input_paths(args.input, args.input_glob)
    hash_sizes = parse_hash_sizes(args.hash_size, args.num_fields)
    field_offsets = field_offsets_from_hash_sizes(hash_sizes)
    np_dtype = dtype_from_name(args.dtype)
    ids_rows: list[list[int]] = []
    labels: list[int] = []

    for idx, (label, cats) in enumerate(iter_criteo_rows(paths, num_fields=args.num_fields), start=1):
        row_ids = categorical_to_ids(
            cats,
            hash_sizes=hash_sizes,
            field_offsets=field_offsets,
            global_id_mode=args.global_id_mode,
            drop_empty=args.drop_empty,
            seed=args.seed,
        )
        if args.drop_empty and len(row_ids) < args.bag_size:
            row_ids.extend([0] * (args.bag_size - len(row_ids)))
        ids_rows.append(row_ids[: args.bag_size])
        labels.append(-1 if label is None else int(label))
        if args.max_samples and idx >= args.max_samples:
            break
        if args.progress_interval and idx % args.progress_interval == 0:
            print(f"processed {idx} Criteo rows", file=sys.stderr)

    if not ids_rows:
        raise RuntimeError("no Criteo rows were processed")

    ids = np.asarray(ids_rows, dtype=np_dtype)
    block_ids = (ids.astype(np.uint64) // np.uint64(args.rows_per_block)).astype(np_dtype)
    offsets = np.arange(0, ids.shape[0] * ids.shape[1] + 1, ids.shape[1], dtype=np.uint64)
    metadata = {
        "trace_source": "criteo",
        "trace_name": args.output.stem,
        "input_files": [str(p) for p in paths],
        "num_samples": int(ids.shape[0]),
        "num_fields": int(ids.shape[1]),
        "batch_size": int(args.batch_size),
        "bag_size": int(args.bag_size),
        "rows_per_block": int(args.rows_per_block),
        "hash_sizes": [int(x) for x in hash_sizes],
        "field_offsets": [int(x) for x in field_offsets],
        "global_id_mode": args.global_id_mode,
        "total_num_embeddings": int(sum(hash_sizes)),
        "drop_empty": bool(args.drop_empty),
        "dtype": args.dtype,
        "trace_layout": "samples_by_fields",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata_json": np.asarray(json.dumps(metadata))}
    if args.save_ids:
        payload["ids"] = ids
    if args.save_blocks:
        payload["block_ids"] = block_ids
    payload["offsets"] = offsets
    payload["labels"] = np.asarray(labels, dtype=np.int8)
    payload["field_offsets"] = np.asarray(field_offsets, dtype=np.uint64)
    payload["hash_sizes"] = np.asarray(hash_sizes, dtype=np.uint64)
    np.savez(args.output, **payload)

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
    parser.add_argument("--input", type=Path)
    parser.add_argument("--input-glob")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--num-fields", type=int, default=26)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--bag-size", type=int, default=26)
    parser.add_argument("--hash-size", required=True)
    parser.add_argument("--global-id-mode", choices=["per_field_offset", "shared_hash"], default="per_field_offset")
    parser.add_argument("--rows-per-block", type=int, required=True)
    parser.add_argument("--drop-empty", action="store_true")
    parser.add_argument("--save-ids", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-blocks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dtype", choices=["uint64", "uint32"], default="uint64")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--stats-dir", type=Path)
    parser.add_argument("--block-bytes", type=int, default=2048 * 128 * 2)
    parser.add_argument("--progress-interval", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> None:
    preprocess(parse_args())


if __name__ == "__main__":
    main()
