# MemTier

MemTier is a user-level GPU-aware data supply runtime.

## Supported paths
- POSIX buffered path
- DRAM cache path (userspace cache)
- Optional GDS path (real backend if available, otherwise stub/fallback)
- CUDA GPU target if available

## Build

### CPU-only
```bash
mkdir -p build && cd build
cmake ..
make -j
ctest
```

### One-click rebuild
```bash
./scripts/rebuild.sh
# optional custom build dir and extra cmake args
./scripts/rebuild.sh ./build -DMEMTIER_ENABLE_CUDA=ON
```

### CUDA
```bash
cmake -DMEMTIER_ENABLE_CUDA=ON ..
```

### GDS
```bash
cmake -DMEMTIER_ENABLE_GDS=ON ..
```

## Run examples
```bash
./build/basic_read <file>
./build/cache_reuse
./build/path_policy_demo
./build/readv_small_io
./build/async_pipeline
./build/prefetch_demo
```

## Benchmarks
```bash
./build/bench_large_read --file=test.bin --size=1073741824 --mode=memtier
./build/bench_reuse test.bin
./build/bench_small_readv test.bin
```

## API quick usage
```c
memtier_ctx_t* ctx = NULL;
memtier_init(NULL, &ctx);
memtier_read(ctx, path, offset, size, dst, MEMTIER_TARGET_CPU);
memtier_prefetch(ctx, path, offset, size, NULL);
memtier_stats_t st;
memtier_stats(ctx, &st);
memtier_finalize(ctx);
```

## Limitations
- Does not modify Linux Page Cache or kernel behavior.
- DRAM cache is userspace cache only.
- UVM/HBM cache is not fully implemented.
- GDS is optional; missing GDS uses stub/fallback.
- Prefetch TO_HBM and async prefetch are currently unsupported.

## Troubleshooting
- No CUDA: GPU target returns `MEMTIER_ERR_UNSUPPORTED`.
- No GDS: build still works; runtime uses stub/fallback.
- O_DIRECT alignment error: ensure offset/size meet alignment requirements.
- Permission error: check file permissions and mount options.

## Prepare experiment files on NVMe
Use the helper module/executable to create experiment files under `/mnt/gds2/cwd_test`:
```bash
./build/prepare_experiment_files memtier_exp_1g.bin 1073741824
```
This creates `/mnt/gds2/cwd_test/memtier_exp_1g.bin` with deterministic content.

## Object-level API (v1)
- `memtier_load_tensor`: for model weights/safetensors/checkpoint tensor payload reads.
- `memtier_load_chunks`: for RAG chunks/KV blocks/index blocks.
- `MemTierDataset`: simple DataLoader-style indexed sample reader.
- These are lightweight wrappers that only translate object semantics to file-range reads; final path selection is still decided by MemTier runtime.

### Python wrapper status
- Current Python API is a lightweight wrapper (`python/memtier`) intended for quick integration.
- CPU-first behavior works without CUDA/GDS/PyTorch.
- If PyTorch is unavailable, APIs return `bytes`/NumPy-compatible data.
