"""Unified reader for synthetic, Criteo-derived, and DLRM-derived traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np


class EmbeddingTrace:
    def __init__(self, path: str | Path, *, rows_per_block: int | None = None, mmap_mode: str | None = "r"):
        self.path = Path(path)
        self.rows_per_block = rows_per_block
        self.metadata: dict = {}
        self._npz = None
        self._jsonl_offsets: list[int] | None = None
        self.ids = None
        self.block_ids = None

        if self.path.suffix == ".npz":
            self._npz = np.load(self.path, mmap_mode=mmap_mode, allow_pickle=False)
            if "metadata_json" in self._npz:
                meta_raw = self._npz["metadata_json"]
                self.metadata = json.loads(str(meta_raw.item() if getattr(meta_raw, "shape", None) == () else meta_raw))
            self.ids = self._npz["ids"] if "ids" in self._npz else None
            self.block_ids = self._npz["block_ids"] if "block_ids" in self._npz else None
        elif self.path.suffix == ".npy":
            self.ids = np.load(self.path, mmap_mode=mmap_mode, allow_pickle=False)
            meta = self.path.with_suffix(self.path.suffix + ".json")
            if meta.exists():
                self.metadata = json.loads(meta.read_text(encoding="utf-8"))
        elif self.path.suffix == ".jsonl":
            self._index_jsonl()
        else:
            raise ValueError(f"unsupported trace format: {self.path}")

        if self.rows_per_block is None:
            self.rows_per_block = int(self.metadata.get("rows_per_block", 0) or 0) or None

    @property
    def batch_size(self) -> int | None:
        value = self.metadata.get("batch_size")
        return int(value) if value is not None else None

    @property
    def bag_size(self) -> int | None:
        value = self.metadata.get("bag_size") or self.metadata.get("num_fields")
        return int(value) if value is not None else None

    @property
    def num_batches(self) -> int:
        if self._jsonl_offsets is not None:
            return len(self._jsonl_offsets)
        if self.ids is None:
            return 0
        if self.ids.ndim == 1:
            return 1
        if self.metadata.get("trace_layout") == "samples_by_fields" and self.batch_size:
            return int((self.ids.shape[0] + self.batch_size - 1) // self.batch_size)
        if self.ids.ndim == 3:
            return int(self.ids.shape[0])
        if self.ids.ndim == 2 and self.bag_size and self.ids.shape[1] == self.bag_size and self.batch_size:
            return int((self.ids.shape[0] + self.batch_size - 1) // self.batch_size)
        return int(self.ids.shape[0])

    def _index_jsonl(self) -> None:
        self._jsonl_offsets = []
        offset = 0
        with self.path.open("rb") as f:
            for line in f:
                if line.strip():
                    self._jsonl_offsets.append(offset)
                offset += len(line)

    def _batch_from_array(self, array, batch_idx: int):
        if array.ndim == 1:
            if batch_idx != 0:
                raise IndexError(batch_idx)
            return np.asarray(array).reshape(-1)
        if array.ndim == 3:
            return np.asarray(array[batch_idx]).reshape(-1)
        if array.ndim == 2 and self.bag_size and array.shape[1] == self.bag_size and self.batch_size:
            start = batch_idx * self.batch_size
            end = min(array.shape[0], start + self.batch_size)
            return np.asarray(array[start:end]).reshape(-1)
        return np.asarray(array[batch_idx]).reshape(-1)

    def get_batch_ids(self, batch_idx: int):
        if batch_idx < 0 or batch_idx >= self.num_batches:
            raise IndexError(batch_idx)
        if self._jsonl_offsets is not None:
            with self.path.open("r", encoding="utf-8") as f:
                f.seek(self._jsonl_offsets[batch_idx])
                row = json.loads(f.readline())
            return np.asarray(row.get("sparse_ids") or row.get("ids"), dtype=np.uint64).reshape(-1)
        if self.ids is None:
            raise ValueError("trace does not contain ids")
        return self._batch_from_array(self.ids, batch_idx).astype(np.uint64, copy=False)

    def get_batch_block_ids(self, batch_idx: int):
        if self.block_ids is not None:
            return np.unique(self._batch_from_array(self.block_ids, batch_idx).astype(np.uint64, copy=False))
        if not self.rows_per_block:
            raise ValueError("rows_per_block is required when trace has no block_ids")
        return np.unique(self.get_batch_ids(batch_idx) // np.uint64(self.rows_per_block))

    def iter_batches(self) -> Iterator[np.ndarray]:
        for idx in range(self.num_batches):
            yield self.get_batch_ids(idx)

    def iter_block_batches(self) -> Iterator[np.ndarray]:
        for idx in range(self.num_batches):
            yield self.get_batch_block_ids(idx)
