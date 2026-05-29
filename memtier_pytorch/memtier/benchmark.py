"""Benchmark helpers for the MemTier PyTorch prototype."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
from torch.utils.data import DataLoader

from . import require_extension
from .dataset import BaselineNumpyDataset, MemTierFileDataset


@dataclass
class BenchmarkResult:
    name: str
    num_batches: int
    total_bytes: int
    total_s: float
    end_to_end_batch_ms: float
    effective_gb_s: float
    file_read_ms: float
    prefetch_ms: float
    gpu_touch_ms: float
    extension_stats: Dict[str, int]

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "num_batches": self.num_batches,
            "total_bytes": self.total_bytes,
            "total_s": self.total_s,
            "end_to_end_batch_ms": self.end_to_end_batch_ms,
            "effective_gb_s": self.effective_gb_s,
            "file_read_ms": self.file_read_ms,
            "prefetch_ms": self.prefetch_ms,
            "gpu_touch_ms": self.gpu_touch_ms,
            "extension_stats": self.extension_stats,
        }


def _require_cuda(device: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("MemTier benchmark requires CUDA, but torch.cuda.is_available() is False")
    if device < 0 or device >= torch.cuda.device_count():
        raise RuntimeError(f"invalid CUDA device {device}; visible count is {torch.cuda.device_count()}")


def _result(name: str, num_batches: int, total_bytes: int, elapsed_s: float, stats: Dict[str, int]) -> BenchmarkResult:
    return BenchmarkResult(
        name=name,
        num_batches=num_batches,
        total_bytes=total_bytes,
        total_s=elapsed_s,
        end_to_end_batch_ms=(elapsed_s * 1000.0 / max(num_batches, 1)),
        effective_gb_s=(total_bytes / max(elapsed_s, 1e-12) / 1e9),
        file_read_ms=stats.get("total_file_read_us", 0) / 1000.0,
        prefetch_ms=stats.get("total_prefetch_us", 0) / 1000.0,
        gpu_touch_ms=stats.get("total_gpu_touch_us", 0) / 1000.0,
        extension_stats=stats,
    )


def run_baseline(path, rows: int, cols: int, batch_rows: int, device: int = 0, num_workers: int = 0) -> BenchmarkResult:
    """Path A: NumPy file data -> DataLoader pin_memory -> tensor.to(cuda).

    This intentionally exercises PyTorch's ordinary host/pinned-memory staging
    and Copy.cu-backed transfer path for comparison with managed-memory access.
    """

    _require_cuda(device)
    dataset = BaselineNumpyDataset(path, rows, cols, batch_rows, use_memmap=True)
    loader = DataLoader(dataset, batch_size=None, num_workers=num_workers, pin_memory=True)
    total_bytes = 0
    num_batches = 0

    torch.cuda.synchronize(device)
    begin = time.perf_counter()
    for batch in loader:
        gpu_batch = batch.to(f"cuda:{device}", non_blocking=True)
        # Force the async copy to complete and make the path comparable to the
        # MemTier touch kernel, which synchronizes after reading managed memory.
        torch.cuda.synchronize(device)
        total_bytes += batch.numel() * batch.element_size()
        num_batches += 1
        del gpu_batch, batch
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - begin
    return _result("baseline", num_batches, total_bytes, elapsed, {})


def run_uvm(path, rows: int, cols: int, batch_rows: int, device: int = 0, policy: str = "none", num_workers: int = 0) -> BenchmarkResult:
    """Path B/C: file -> cudaMallocManaged -> pread -> optional UVM policy -> GPU touch."""

    _require_cuda(device)
    ext = require_extension()
    gc.collect()
    ext.reset_stats()
    dataset = MemTierFileDataset(path, rows, cols, batch_rows, device=device, policy=policy)
    # UVM tensors carry process-local CUDA allocations, so keep workers at zero
    # in this first prototype and avoid DataLoader pin_memory copying them away.
    loader = DataLoader(dataset, batch_size=None, num_workers=num_workers, pin_memory=False)
    total_bytes = 0
    num_batches = 0

    torch.cuda.synchronize(device)
    begin = time.perf_counter()
    for batch in loader:
        touched = ext.touch_gpu(batch)
        torch.cuda.synchronize(device)
        total_bytes += batch.numel() * batch.element_size()
        num_batches += 1
        del touched, batch
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - begin
    gc.collect()
    stats = dict(ext.get_stats())
    name = "uvm_prefetch" if policy == "gpu" else f"uvm_{policy}"
    return _result(name, num_batches, total_bytes, elapsed, stats)


def run_all(path, rows: int, cols: int, batch_rows: int, device: int = 0, num_workers: int = 0) -> List[BenchmarkResult]:
    """Run baseline, UVM without prefetch, and UVM with GPU prefetch."""

    return [
        run_baseline(path, rows, cols, batch_rows, device=device, num_workers=num_workers),
        run_uvm(path, rows, cols, batch_rows, device=device, policy="none", num_workers=0),
        run_uvm(path, rows, cols, batch_rows, device=device, policy="gpu", num_workers=0),
    ]


def format_results(results: Iterable[BenchmarkResult]) -> str:
    """Return a compact Markdown-ish table for CLI output."""

    rows = list(results)
    header = (
        "| path | batches | total GiB | batch ms | eff GB/s | file ms | prefetch ms | gpu touch ms |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for item in rows:
        lines.append(
            "| {name} | {batches} | {gib:.3f} | {batch_ms:.3f} | {gb_s:.3f} | {file_ms:.3f} | {prefetch_ms:.3f} | {touch_ms:.3f} |".format(
                name=item.name,
                batches=item.num_batches,
                gib=item.total_bytes / (1024**3),
                batch_ms=item.end_to_end_batch_ms,
                gb_s=item.effective_gb_s,
                file_ms=item.file_read_ms,
                prefetch_ms=item.prefetch_ms,
                touch_ms=item.gpu_touch_ms,
            )
        )
    return "\n".join(lines)
