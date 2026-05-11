# MemTier Runtime (P1)

MemTier is a user-level GPU-aware data supply runtime.

## Build
```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Runtime API
- `memtier_init(const memtier_config_t*)`: create runtime context and initialize DRAM cache directory, POSIX backend, optional GDS/CUDA stubs, policy, and stats.
- `memtier_finalize(memtier_ctx_t*)`: release runtime resources.
- `memtier_stats(memtier_ctx_t*, memtier_stats_t*)`: snapshot current runtime stats.

### Default config
- device_id=0
- dram_cache_size=1GB
- hbm_cache_size=0
- chunk_size=1MB
- enable_posix=true
- enable_gds=false
- enable_dram_cache=true
- enable_hbm_cache=false
- gds_min_size=1MB
- gds_alignment=4096
- num_workers=0
- policy="adaptive"

### Runtime info example
```bash
./build/runtime_info
```

## Notes
- Current stage focuses on runtime lifecycle management API.
- GDS/CUDA are optional backends (stub in CPU-only environment).
- DRAM cache is initialized; full read path is reserved for later stages.
