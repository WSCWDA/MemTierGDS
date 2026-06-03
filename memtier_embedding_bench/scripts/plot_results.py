#!/usr/bin/env python3
"""Plot or summarize JSONL benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = load_jsonl(args.input)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        for row in rows:
            print(json.dumps({k: row.get(k) for k in ["system", "dist", "zipf_alpha", "avg_latency_ms", "throughput_lookups_per_sec"]}))
        return

    labels = [f"{r.get('system')}:{r.get('dist','')}/{r.get('zipf_alpha','')}" for r in rows]
    lat = [r.get("avg_latency_ms", 0.0) for r in rows]
    plt.figure(figsize=(max(8, len(labels) * 1.5), 4))
    plt.bar(labels, lat)
    plt.ylabel("Average latency (ms)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
