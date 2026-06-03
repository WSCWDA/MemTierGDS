"""Utilities for parsing Criteo-style categorical feature rows.

The raw Kaggle/Terabyte text rows are parsed in a streaming fashion by the
preprocessing script.  Only the 26 categorical columns are used for embedding
lookup traces; label and dense columns are retained only for optional metadata.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Iterator

DEFAULT_NUM_DENSE = 13
DEFAULT_NUM_FIELDS = 26


def stable_hash64(value: str | bytes, *, seed: int = 0) -> int:
    """Return a deterministic 64-bit hash suitable for persistent traces."""

    if isinstance(value, str):
        data = value.encode("utf-8", errors="surrogatepass")
    else:
        data = value
    person = seed.to_bytes(8, "little", signed=False)
    digest = hashlib.blake2b(data, digest_size=8, person=person).digest()
    return int.from_bytes(digest, "little", signed=False)


def parse_criteo_line(line: str, *, num_fields: int = DEFAULT_NUM_FIELDS) -> tuple[int | None, list[str]]:
    """Parse one Criteo Kaggle/Terabyte line and return label plus categories.

    Criteo rows contain label + 13 dense integer features + 26 categorical
    features.  Empty categorical values are represented as empty strings when the
    source line contains adjacent separators.
    """

    line = line.rstrip("\n")
    parts = line.split("\t") if "\t" in line else line.split()
    if not parts:
        raise ValueError("empty Criteo row")

    label = None
    try:
        label = int(parts[0])
    except ValueError:
        label = None

    cat_start = 1 + DEFAULT_NUM_DENSE
    cats = parts[cat_start : cat_start + num_fields]
    if len(cats) < num_fields:
        cats.extend([""] * (num_fields - len(cats)))
    return label, cats[:num_fields]


def iter_criteo_rows(paths: Iterable[Path], *, num_fields: int = DEFAULT_NUM_FIELDS) -> Iterator[tuple[int | None, list[str]]]:
    """Yield parsed rows from one or more Criteo files without loading them."""

    for path in paths:
        with Path(path).open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip():
                    yield parse_criteo_line(line, num_fields=num_fields)


def parse_hash_sizes(value: str, num_fields: int) -> list[int]:
    """Parse either one hash size or a comma-separated per-field list."""

    pieces = [x.strip() for x in str(value).split(",") if x.strip()]
    if len(pieces) == 1:
        size = int(pieces[0])
        if size <= 0:
            raise ValueError("hash sizes must be positive")
        return [size] * num_fields
    if len(pieces) != num_fields:
        raise ValueError(f"expected one hash size or {num_fields} comma-separated sizes")
    sizes = [int(x) for x in pieces]
    if any(x <= 0 for x in sizes):
        raise ValueError("hash sizes must be positive")
    return sizes


def field_offsets_from_hash_sizes(hash_sizes: list[int]) -> list[int]:
    offsets: list[int] = []
    total = 0
    for size in hash_sizes:
        offsets.append(total)
        total += int(size)
    return offsets


def categorical_to_ids(
    cats: list[str],
    *,
    hash_sizes: list[int],
    field_offsets: list[int],
    global_id_mode: str,
    drop_empty: bool = False,
    seed: int = 0,
) -> list[int]:
    """Map categorical feature values to deterministic global embedding IDs."""

    ids: list[int] = []
    total_embeddings = sum(hash_sizes)
    for field, raw in enumerate(cats):
        value = raw if raw != "" else f"__EMPTY_FIELD_{field}__"
        if raw == "" and drop_empty:
            continue
        if global_id_mode == "per_field_offset":
            local = stable_hash64(f"{field}:{value}", seed=seed) % hash_sizes[field]
            ids.append(field_offsets[field] + local)
        elif global_id_mode == "shared_hash":
            ids.append(stable_hash64(f"{field}:{value}", seed=seed) % total_embeddings)
        else:
            raise ValueError(f"unsupported global_id_mode: {global_id_mode}")
    return ids
