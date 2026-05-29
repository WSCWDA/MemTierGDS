"""Dataset adapters for the MemTier PyTorch UVM prototype.

The baseline dataset models the standard PyTorch path: file-backed CPU data is
returned to DataLoader, DataLoader may pin it, and benchmark code calls
``batch.to("cuda", non_blocking=True)``.  The MemTier dataset bypasses that copy
path by asking the extension to ``pread`` directly into CUDA managed memory and
return a CPU Tensor view whose data pointer is UVM-backed.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


_FLOAT32_BYTES = np.dtype(np.float32).itemsize


class MemTierFileDataset(Dataset):
    """Return one UVM-backed float32 tensor batch per item.

    Each ``__getitem__`` computes a byte offset into the flat binary file, reads
    ``batch_rows * cols`` float32 values through the C++/CUDA extension, and
    returns a ``[actual_rows, cols]`` CPU Tensor view over CUDA managed memory.
    """

    def __init__(self, path, rows: int, cols: int, batch_rows: int, device: int = 0, policy: str = "gpu"):
        self.path = str(Path(path))
        self.rows = int(rows)
        self.cols = int(cols)
        self.batch_rows = int(batch_rows)
        self.device = int(device)
        self.policy = policy
        if self.rows <= 0 or self.cols <= 0 or self.batch_rows <= 0:
            raise ValueError("rows, cols, and batch_rows must be positive")
        if policy not in {"none", "cpu", "gpu", "accessed_by"}:
            raise ValueError(f"unsupported MemTier policy: {policy}")

    def __len__(self) -> int:
        return math.ceil(self.rows / self.batch_rows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        from . import require_extension

        start_row = idx * self.batch_rows
        actual_rows = min(self.batch_rows, self.rows - start_row)
        offset = start_row * self.cols * _FLOAT32_BYTES
        ext = require_extension()
        return ext.uvm_tensor_from_file(
            self.path,
            actual_rows,
            self.cols,
            offset,
            self.device,
            self.policy,
        )


class BaselineNumpyDataset(Dataset):
    """Return one CPU tensor batch per item using NumPy memmap/fromfile.

    Benchmark code should run this with DataLoader ``pin_memory=True`` and then
    call ``batch.to("cuda", non_blocking=True)`` to exercise PyTorch's default
    CPU/pinned-memory-to-GPU copy path.
    """

    def __init__(self, path, rows: int, cols: int, batch_rows: int, use_memmap: bool = True):
        self.path = str(Path(path))
        self.rows = int(rows)
        self.cols = int(cols)
        self.batch_rows = int(batch_rows)
        self.use_memmap = bool(use_memmap)
        if self.rows <= 0 or self.cols <= 0 or self.batch_rows <= 0:
            raise ValueError("rows, cols, and batch_rows must be positive")
        self._memmap = None
        if self.use_memmap:
            self._memmap = np.memmap(self.path, mode="r", dtype=np.float32, shape=(self.rows, self.cols))

    def __len__(self) -> int:
        return math.ceil(self.rows / self.batch_rows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        start_row = idx * self.batch_rows
        actual_rows = min(self.batch_rows, self.rows - start_row)
        if self._memmap is not None:
            array = np.asarray(self._memmap[start_row : start_row + actual_rows])
        else:
            offset = start_row * self.cols * _FLOAT32_BYTES
            array = np.fromfile(
                self.path,
                dtype=np.float32,
                count=actual_rows * self.cols,
                offset=offset,
            ).reshape(actual_rows, self.cols)
        return torch.from_numpy(array)
