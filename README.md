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
