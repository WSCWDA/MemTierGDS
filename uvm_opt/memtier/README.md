# MemTier UVM/GDS Prototype

MemTier exposes file-backed GPU-accessible objects. It maps a file range to a runtime-managed GPU pointer, maintains `VA range -> file_id + offset + length -> residency`, and uses NVIDIA GPUDirect Storage (GDS/cuFile) as one possible SSD miss direct-fill path.

This code is a user-space wrapper API prototype (`memtierMapFile`, `memtierMallocManagedFromFile`-style semantics), not a CUDA Runtime, CUDA Driver, or NVIDIA UVM driver modification.

## What this prototype does

- Provides `memtierInit`, `memtierMapFile`, explicit `memtierPrefetchToGpu` / `memtierPrefetchToDram`, and `memtierUnmapFile`.
- Maintains ObjectTable metadata for GPU-accessible VA ranges and backing file ranges.
- Maintains ResidencyTable states: `SSD_ONLY`, `DRAM_RESIDENT`, `HBM_RESIDENT`, `DRAM_AND_HBM`.
- Selects among POSIX pinned-DRAM fill, DRAM-cache reuse, and optional GDS direct fill.
- Uses default device-buffer allocation (`cudaMalloc`) for the first GDS-oriented implementation.
- Supports experimental managed allocation with `MEMTIER_ALLOC_MODE=managed` for POSIX/UVM-like paths.

## What this prototype does not do

- This prototype does not modify `cudaMallocManaged` internally.
- This prototype does not modify NVIDIA UVM driver.
- This prototype does not intercept GPU page faults.
- This prototype supports read-only file-backed objects only.
- This prototype does not implement GPU write-back to files.
- It does not implement dirty tracking, `fsync`, write consistency, or transparent OS/driver-level file-backed UVM.

## Relationship to NVIDIA samples

The CUDA-side pattern intentionally follows CUDA sample-style explicit CUDA calls: `cudaMalloc`, `cudaMallocManaged`, `cudaMemcpyAsync`, `cudaMemPrefetchAsync`, `cudaGetDevice`, and `cudaDeviceSynchronize`/stream synchronization with checked return values.

The GDS backend is guarded by `MEMTIER_ENABLE_GDS` and uses the official cuFile call sequence used by NVIDIA MagnumIO GDS samples and cuFile documentation: `cuFileDriverOpen`, `cuFileHandleRegister`, `cuFileBufRegister`, `cuFileRead`, `cuFileBufDeregister`, `cuFileHandleDeregister`, and `cuFileDriverClose`. Only the read path is implemented. `cuFileWrite` is intentionally not implemented.

TODO: verify with local CUDA/GDS version whether a given managed pointer can be registered with `cuFileBufRegister`; the default path uses `cudaMalloc` device memory for GDS.

## Build

```bash
cd uvm_opt/memtier
mkdir -p build
cd build
cmake ..
make -j
```

or:

```bash
cd uvm_opt/memtier
./scripts/build.sh
```

If `libcufile.so` or `cufile.h` is unavailable, CMake builds a POSIX-only version with `MEMTIER_ENABLE_GDS=0`.

## Run

```bash
cd uvm_opt/memtier
./build/simple_map_file
./build/compare_paths --file /mnt/gds/test.bin --size-mb 1024 --mode auto --repeat 3
```

The simple example creates an input file if it does not exist, maps it, explicitly prefetches it before launching a kernel, and prints:

```text
selected_path
residency_before
residency_after
elapsed_time_ms
verification_passed
```

## GDS fallback behavior

If GDS is unavailable, MemTier falls back to POSIX `pread()` into pinned DRAM followed by `cudaMemcpyAsync` to the GPU-accessible allocation. If GDS is available but `cuFileHandleRegister`, `cuFileBufRegister`, or `cuFileRead` fails, the runtime prints diagnostics and falls back to the POSIX path.

## DRAM cache

`DramCache` is an exact-match pinned-memory cache. Configure capacity with:

```bash
MEMTIER_DRAM_CACHE_MB=4096
```

The cache is distinct from the OS page cache. Benchmarks should distinguish cold OS cache, warm OS cache, and MemTier DRAM-cache hits.

## Paper/prototype correspondence

- file-backed managed allocation abstraction: `memtierMapFile` returns a GPU-accessible pointer with file metadata.
- ObjectTable: maps VA range to file identity, offset, length, and object metadata.
- ResidencyTable: tracks SSD/DRAM/HBM residency.
- PathSelector: selects static first-version fill paths.
- GDS direct fill path: optional `cuFileRead` into device memory.
- DRAM reuse path: exact-match pinned-DRAM cache plus `cudaMemcpyAsync`.

## Current limitations

- Explicit prefetch is required before kernels; there is no GPU page-fault interception.
- Read-only file-backed objects only.
- No write-back or coherency protocol.
- GDS requires local CUDA/GDS support and may have alignment and filesystem constraints.
- The first policy is static; TODO: replace with access monitor and reuse estimator.
