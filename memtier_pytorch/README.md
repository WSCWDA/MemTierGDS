# MemTier PyTorch UVM Prototype

This directory contains a minimal PyTorch C++/CUDA extension prototype for a
MemTier data-supply path:

```text
file data -> CUDA UVM managed memory -> torch::from_blob Tensor -> GPU kernel touch
          -> optional cudaMemAdvise/cudaMemPrefetchAsync -> timing/stat counters
```

The prototype does **not** modify PyTorch core.  It is intentionally built as an
out-of-tree extension so the data path can be validated before deeper Storage or
Allocator integration.

## Build

A CUDA toolkit and a CUDA-enabled PyTorch installation are required.  If CUDA is
not visible, `setup.py` raises a clear error before compiling.

```bash
pip install -e ./memtier_pytorch
```

## Run

The example creates a flat float32 binary file if it does not already exist and
then benchmarks three paths:

1. `baseline`: NumPy file data -> PyTorch DataLoader `pin_memory=True` ->
   `batch.to("cuda", non_blocking=True)`.
2. `uvm_none`: POSIX `pread` -> `cudaMallocManaged` buffer -> GPU kernel touch,
   relying on UVM page faults.
3. `uvm_prefetch`: POSIX `pread` -> `cudaMallocManaged` buffer ->
   `cudaMemAdvise`/`cudaMemPrefetchAsync` -> GPU kernel touch.

```bash
python memtier_pytorch/examples/run_memtier_dataloader.py \
  --path /tmp/emb.bin \
  --rows 1000000 \
  --dim 768 \
  --batch-rows 4096
```

For a smaller smoke run:

```bash
python memtier_pytorch/examples/run_memtier_dataloader.py \
  --path /tmp/emb-small.bin \
  --rows 65536 \
  --dim 128 \
  --batch-rows 2048
```

## Python API

```python
from memtier.dataset import MemTierFileDataset, BaselineNumpyDataset
from memtier.benchmark import run_all, format_results

results = run_all("/tmp/emb.bin", rows=1_000_000, cols=768, batch_rows=4096)
print(format_results(results))
```

`MemTierFileDataset` returns one UVM-backed tensor batch per `__getitem__` call.
Use `DataLoader(..., batch_size=None, num_workers=0, pin_memory=False)` for this
prototype so the managed pointer is not copied away by collation or pinning.

`BaselineNumpyDataset` returns CPU tensors and is intended to be benchmarked with
`DataLoader(..., pin_memory=True)` followed by
`batch.to("cuda", non_blocking=True)`.

## Extension API

The compiled module is `memtier.memtier_ext` and exposes:

- `allocate_managed(bytes, device=0) -> int`
- `free_managed(ptr)`
- `advise_preferred_cpu(ptr, bytes)`
- `advise_preferred_gpu(ptr, bytes, device=0)`
- `advise_accessed_by_gpu(ptr, bytes, device=0)`
- `prefetch_to_gpu(ptr, bytes, device=0)`
- `prefetch_to_cpu(ptr, bytes)`
- `load_file_to_managed(path, offset, rows, cols, device=0) -> torch.Tensor`
- `uvm_tensor_from_file(path, rows, cols, offset=0, device=0, prefetch_policy="none") -> torch.Tensor`
- `touch_gpu(tensor) -> torch.Tensor`
- `get_stats() -> dict`
- `reset_stats()`

The first version fixes dtype to `float32`.  `torch::from_blob` wraps the managed
pointer as a CPU tensor and installs a CUDA-freeing deleter to avoid leaking UVM
memory when the tensor is destroyed.

## Relationship to PyTorch internals

This prototype maps to the referenced PyTorch source areas as follows:

- `torch/utils/data/dataloader.py` and
  `torch/utils/data/_utils/pin_memory.py`: the benchmark baseline uses the
  standard DataLoader pin-memory path, where CPU batches are copied into pinned
  host memory before asynchronous transfer to CUDA.
- `aten/src/ATen/native/cuda/Copy.cu`: the baseline's
  `batch.to("cuda", non_blocking=True)` exercises PyTorch's default CPU-to-GPU
  tensor copy implementation.  MemTier does not replace this file in the first
  prototype.
- `c10/core/StorageImpl.h` and `c10/core/Allocator.h`: these are the future deep
  integration points for attaching MemTier-managed storage to PyTorch tensors.
  The current prototype bypasses that work and uses `torch::from_blob` around a
  CUDA UVM pointer.
- `c10/cuda/CUDACachingAllocator.cpp`: this allocator is not modified.  MemTier
  has an independent minimal allocator based on `cudaMallocManaged`/`cudaFree`.

## Design notes and limitations

- No PyTorch source files are modified.
- No GDS/cuFile dependency is used; file reads are POSIX `pread` only.
- The prototype targets a single machine with at least one CUDA GPU.
- UVM tensors should stay in the same process in this first version; keep
  `num_workers=0` for `MemTierFileDataset`.
- Statistics are extension-local counters for allocated bytes, file-read bytes,
  allocation count, prefetch count/time, file-read time, and GPU-touch time.
