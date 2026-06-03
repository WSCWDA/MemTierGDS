#!/usr/bin/env python3
"""Analyze synthetic/Criteo/DLRM embedding traces and emit JSON/CSV stats."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import numpy as np
from trace_reader import EmbeddingTrace


def percentile(values: list[int | float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return float(ordered[idx])


def coverage(counter: Counter, fractions=(0.01, 0.05, 0.10), topk: int = 100) -> dict:
    total = sum(counter.values())
    n_unique = len(counter)
    most = counter.most_common()
    out = {"topk": [[int(k), int(v)] for k, v in most[:topk]]}
    for frac in fractions:
        n = max(1, int(n_unique * frac)) if n_unique else 0
        covered = sum(v for _, v in most[:n]) if n else 0
        out[f"top_{int(frac * 100)}pct_coverage"] = covered / max(total, 1)
    return out


def estimate_zipf_slope(counter: Counter, topn: int = 1000) -> float:
    freqs = np.asarray([v for _, v in counter.most_common(topn)], dtype=np.float64)
    if freqs.size < 2:
        return 0.0
    ranks = np.arange(1, freqs.size + 1, dtype=np.float64)
    x = np.log(ranks)
    y = np.log(freqs)
    slope, _ = np.polyfit(x, y, 1)
    return float(-slope)


def update_lru_sims(blocks: np.ndarray, sims: dict[int, OrderedDict], hits: dict[int, int]) -> int:
    total = 0
    for raw in blocks:
        block = int(raw)
        total += 1
        for cap, cache in sims.items():
            if block in cache:
                hits[cap] += 1
                cache.move_to_end(block)
            else:
                cache[block] = None
                if len(cache) > cap:
                    cache.popitem(last=False)
    return total


def batch_multi_gpu_overlap(blocks: np.ndarray, ranks: int = 4) -> tuple[float, int]:
    chunks = np.array_split(np.asarray(blocks), ranks)
    sets = [set(int(x) for x in chunk) for chunk in chunks]
    union = set().union(*sets) if sets else set()
    total_rank_blocks = sum(len(s) for s in sets)
    duplicate = total_rank_blocks - len(union)
    return duplicate / max(total_rank_blocks, 1), duplicate


def analyze_trace_file(
    trace_file: str | Path,
    *,
    rows_per_block: int,
    topk: int = 100,
    block_bytes: int = 2048 * 128 * 2,
) -> dict:
    trace = EmbeddingTrace(trace_file, rows_per_block=rows_per_block, mmap_mode="r")
    id_counter: Counter[int] = Counter()
    block_counter: Counter[int] = Counter()
    unique_blocks_per_batch: list[int] = []
    unique_ids_per_batch: list[int] = []
    reuse_ratios: list[float] = []
    reuse_distances: list[int] = []
    last_seen: dict[int, int] = {}
    previous_blocks: set[int] = set()
    total_lookups = 0
    dram_gb = [1, 4, 8, 16, 32]
    capacities = [max(1, int(gb * (1024**3) // max(block_bytes, 1))) for gb in dram_gb]
    sims = {cap: OrderedDict() for cap in capacities}
    sim_hits = {cap: 0 for cap in capacities}
    sim_total = 0
    overlap_ratios: list[float] = []
    duplicate_blocks: list[int] = []

    for batch_idx, ids in enumerate(trace.iter_batches()):
        ids = np.asarray(ids, dtype=np.uint64).reshape(-1)
        blocks = np.unique(ids // np.uint64(rows_per_block))
        sim_total += update_lru_sims(blocks, sims, sim_hits)
        overlap_ratio, duplicate_count = batch_multi_gpu_overlap(blocks)
        overlap_ratios.append(overlap_ratio)
        duplicate_blocks.append(duplicate_count)
        total_lookups += int(ids.size)
        id_counter.update(int(x) for x in ids)
        block_counter.update(int(x) for x in blocks)
        unique_ids_per_batch.append(len(set(int(x) for x in ids)))
        current_blocks = set(int(x) for x in blocks)
        unique_blocks_per_batch.append(len(current_blocks))
        if previous_blocks:
            reuse_ratios.append(len(current_blocks & previous_blocks) / max(len(current_blocks), 1))
        previous_blocks = current_blocks
        for block in current_blocks:
            if block in last_seen:
                reuse_distances.append(batch_idx - last_seen[block])
            last_seen[block] = batch_idx

    cache_hit_rates = {str(cap): sim_hits[cap] / max(sim_total, 1) for cap in capacities}

    stats = {
        "trace_file": str(trace_file),
        "trace_name": Path(trace_file).stem,
        "rows_per_block": int(rows_per_block),
        "num_batches": int(trace.num_batches),
        "total_lookups": int(total_lookups),
        "unique_ids": int(len(id_counter)),
        "unique_blocks": int(len(block_counter)),
        "id_frequency": coverage(id_counter, topk=topk),
        "block_frequency": coverage(block_counter, topk=topk),
        "estimated_zipf_alpha_ids": estimate_zipf_slope(id_counter),
        "estimated_zipf_alpha_blocks": estimate_zipf_slope(block_counter),
        "avg_unique_ids_per_batch": float(mean(unique_ids_per_batch)) if unique_ids_per_batch else 0.0,
        "avg_unique_blocks_per_batch": float(mean(unique_blocks_per_batch)) if unique_blocks_per_batch else 0.0,
        "p50_unique_blocks_per_batch": percentile(unique_blocks_per_batch, 50),
        "p95_unique_blocks_per_batch": percentile(unique_blocks_per_batch, 95),
        "block_reuse_rate": float(mean(reuse_ratios)) if reuse_ratios else 0.0,
        "reuse_distance_approx": {
            "avg_batches": float(mean(reuse_distances)) if reuse_distances else 0.0,
            "p50_batches": percentile(reuse_distances, 50),
            "p95_batches": percentile(reuse_distances, 95),
        },
        "cache_simulation": {
            f"{gb}GB": cache_hit_rates.get(str(cap), 0.0) for gb, cap in zip(dram_gb, capacities)
        },
        "multi_gpu_overlap": {
            "ranks": 4,
            "avg_overlap_ratio": float(mean(overlap_ratios)) if overlap_ratios else 0.0,
            "avg_duplicate_blocks": float(mean(duplicate_blocks)) if duplicate_blocks else 0.0,
        },
        "metadata": trace.metadata,
    }
    return stats


def write_csv(stats: dict, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind in ["id_frequency", "block_frequency"]:
        for rank, (key, count) in enumerate(stats[kind]["topk"], start=1):
            rows.append({"kind": kind, "rank": rank, "key": key, "count": count})
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["kind", "rank", "key", "count"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-file", required=True, type=Path)
    parser.add_argument("--rows-per-block", required=True, type=int)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--block-bytes", type=int, default=2048 * 128 * 2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = analyze_trace_file(args.trace_file, rows_per_block=args.rows_per_block, topk=args.topk, block_bytes=args.block_bytes)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    if args.output_csv:
        write_csv(stats, args.output_csv)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
