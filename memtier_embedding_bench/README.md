# W1: Out-of-core Embedding / DLRM-like Benchmark

This benchmark is the Phase 1/2 evaluation harness for a MemTier paper workload:
out-of-core embedding lookup when the embedding table is larger than GPU HBM.
It creates a reproducible DLRM-like ID trace, stores the embedding table as a
row-major binary file on SSD, and compares CPU staging with a MemTier runtime
skeleton that models HBM/DRAM/SSD residency, pinned DRAM cache reuse, and
adaptive I/O path selection hooks.

## Current Phase 1/2 scope

Implemented now:

- Streaming embedding-table generation; no full-table allocation is required.
- Uniform and Zipf ID trace generation, saved as `.npy` so all baselines reuse
  the same workload.
- CPU staging baseline using POSIX `pread` per unique block.
- `libmemtier_embedding.so` C API skeleton with residency table, pinned DRAM
  cache, HBM cache, POSIX loader, GDS stub, and JSON metrics.
- CUDA embedding-bag launcher is exported, but the Python wrapper currently uses
  the runtime for block lookup and metrics only.

TODO / future phases:

- Replace the GDS stub with real `cuFileRead` under `ENABLE_GDS`.
- Add a full native UVM baseline with `cudaMallocManaged` and UVM fault/prefetch
  accounting.
- Wire the exported embedding-bag CUDA launcher into Python output tensors or a
  Torch extension wrapper.
- Integrate TorchRec/DLRM model code and multi-GPU overlap execution.

## Directory layout

```text
memtier_embedding_bench/
├── CMakeLists.txt
├── README.md
├── scripts/
├── python/
├── include/
├── src/
├── cuda/
└── results/
```

## Build

Requirements: CMake >= 3.18, C++17 compiler, CUDA toolkit, NVIDIA driver, a
single CUDA GPU, and Python packages `numpy` plus optional `torch`/`matplotlib`
for the PyTorch upper-bound baseline and plotting.  Build the C API shared library:

```bash
cd memtier_embedding_bench
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This produces `build/libmemtier_embedding.so`.  GDS hooks can be enabled at build
time with `-DENABLE_GDS=ON`, but Phase 1/2 still falls back to POSIX reads unless
the stub is replaced with cuFile code.

## Generate embedding data

The table file is row-major: embedding ID `i` is stored at row `i`, and every row
contains `embedding_dim` contiguous fp16/fp32 values.  The generator writes in
chunks controlled by `--chunk-mb` to avoid memory blow-up.

```bash
python scripts/generate_embedding_file.py \
  --output results/embeddings.fp16.bin \
  --num-embeddings 200000000 \
  --embedding-dim 128 \
  --dtype fp16 \
  --seed 1234 \
  --chunk-mb 256
```

A metadata JSON file is written next to the binary and records
`num_embeddings`, `embedding_dim`, `dtype`, `dtype_size`, and
`file_size_bytes`.

For a quick smoke test, use much smaller values:

```bash
python scripts/generate_embedding_file.py \
  --output results/smoke.fp16.bin \
  --num-embeddings 65536 \
  --embedding-dim 64 \
  --dtype fp16 \
  --chunk-mb 16
```

## Generate a workload trace

```bash
python python/embedding_workload.py \
  --output results/trace_zipf_1.0.npy \
  --dist zipf \
  --zipf-alpha 1.0 \
  --batch-size 16384 \
  --bag-size 32 \
  --num-batches 1000 \
  --num-embeddings 200000000 \
  --rows-per-block 2048 \
  --overlap-ratio 0.0
```

Each batch has shape `[batch_size * bag_size]`.  The module also exposes:

- `generate_ids(...)`
- `ids_to_blocks(ids, rows_per_block)`
- `estimate_block_reuse(trace, rows_per_block)`


## Criteo/DLRM Trace Mode

Synthetic uniform/Zipf traces are useful for controlled sweeps, but W1 can also
run Criteo/DLRM-derived categorical traces to reflect real DLRM sparse feature
locality, hot IDs, and block reuse.

### Prepare Criteo data

This repository does **not** download Criteo datasets automatically.  Download
data yourself after accepting the appropriate terms:

- Kaggle Display Advertising Challenge: use the raw `train.txt` file.
- Criteo Terabyte: use `day_0` ... `day_23` files and pass them with
  `--input-glob`.

The converter ignores labels/dense features for lookup generation and hashes only
the 26 categorical fields.  Empty categorical values are mapped to deterministic
field-specific unknown IDs unless `--drop-empty` is set.

```bash
python scripts/preprocess_criteo_trace.py \
  --input /data/criteo/train.txt \
  --output results/traces/criteo_kaggle_10m.npz \
  --max-samples 10000000 \
  --batch-size 16384 \
  --hash-size 10000000 \
  --global-id-mode per_field_offset \
  --rows-per-block 2048
```

For Terabyte day files:

```bash
python scripts/preprocess_criteo_trace.py \
  --input-glob "/data/criteo/day_*" \
  --output results/traces/criteo_tb_10m.npz \
  --max-samples 10000000 \
  --batch-size 16384 \
  --hash-size 10000000 \
  --global-id-mode per_field_offset \
  --rows-per-block 2048
```

The output `.npz` contains `ids`, `block_ids`, `offsets`, optional `labels`,
`field_offsets`, `hash_sizes`, and `metadata_json`.  Stable `blake2b` hashing is
used instead of Python's process-randomized `hash()`.

### Convert DLRM/TorchRec preprocessed sparse traces

`convert_dlrm_trace.py` supports `.npy`, `.npz`, and `.jsonl` inputs.  Current
recognized `.npz` keys include `lS_i`, `sparse`, `sparse_ids`, `X_cat`, and
`cat`.  Supported array shapes are `[num_samples, 26]` and
`[num_batches, batch_size, 26]`; JSONL rows may contain `{"sparse_ids": [...]}`.

```bash
python scripts/convert_dlrm_trace.py \
  --input /data/dlrm/sparse.npy \
  --input-format auto \
  --output results/traces/dlrm_sparse.npz \
  --rows-per-block 2048 \
  --num-embeddings 260000000 \
  --batch-size 16384 \
  --field-offset-mode direct
```

### Analyze trace locality and hot blocks

```bash
python scripts/analyze_trace.py \
  --trace-file results/traces/criteo_kaggle_10m.npz \
  --rows-per-block 2048 \
  --output-json results/trace_stats/criteo_kaggle_10m.json \
  --output-csv results/trace_stats/criteo_kaggle_10m.csv
```

The analyzer emits JSON/CSV data for paper plots, including top 1%/5%/10% block
coverage, estimated Zipf-like skew, p50/p95 unique blocks per batch,
approximate reuse distance, LRU DRAM-cache hit-rate simulation, and multi-GPU
rank overlap estimates.

### Run baselines with a Criteo trace

```bash
python python/baseline_cpu_staging.py \
  --embedding-file /mnt/nvme/emb_50gb.bin \
  --trace-source criteo \
  --trace-file results/traces/criteo_kaggle_10m.npz \
  --embedding-dim 128 \
  --rows-per-block 2048 \
  --dtype fp16 \
  --output results/cpu_staging_criteo.jsonl
```

### Run MemTier with a Criteo trace

```bash
python python/memtier_embedding.py \
  --embedding-file /mnt/nvme/emb_50gb.bin \
  --trace-source criteo \
  --trace-file results/traces/criteo_kaggle_10m.npz \
  --embedding-dim 128 \
  --rows-per-block 2048 \
  --dram-cache-gb 32 \
  --hbm-cache-gb 4 \
  --output results/memtier_criteo.jsonl
```

Both CPU staging and MemTier use the same `EmbeddingTrace` reader, so synthetic
`.npy`, Criteo `.npz`, DLRM `.npz/.npy`, and supported JSONL traces can be
compared through a common JSONL schema with `trace_source` and `trace_name`.

## Run baselines

CPU staging baseline:

```bash
python python/baseline_cpu_staging.py \
  --embedding-file results/embeddings.fp16.bin \
  --trace results/trace_zipf_1.0.npy \
  --output results/baselines.jsonl \
  --num-embeddings 200000000 \
  --embedding-dim 128 \
  --rows-per-block 2048 \
  --dtype-size 2 \
  --batch-size 16384 \
  --bag-size 32 \
  --num-batches 1000
```

Small in-HBM PyTorch upper bound:

```bash
python python/baseline_torch_embedding.py \
  --trace results/trace_uniform.npy \
  --output results/baselines.jsonl \
  --num-embeddings 1000000 \
  --embedding-dim 128 \
  --dtype-size 2 \
  --batch-size 16384 \
  --bag-size 32 \
  --num-batches 100
```

Convenience script:

```bash
scripts/run_baselines.sh
```

The native UVM baseline currently writes a JSONL TODO record.  It is reserved for
future `cudaMallocManaged`/UVM fault experiments.

## Run MemTier

```bash
python python/memtier_embedding.py \
  --embedding-file results/embeddings.fp16.bin \
  --trace results/trace_zipf_1.0.npy \
  --lib build/libmemtier_embedding.so \
  --output results/memtier.jsonl \
  --num-embeddings 200000000 \
  --embedding-dim 128 \
  --rows-per-block 2048 \
  --dtype-size 2 \
  --batch-size 16384 \
  --bag-size 32 \
  --num-batches 1000 \
  --dist zipf \
  --zipf-alpha 1.0 \
  --dram-cache-gb 32 \
  --hbm-cache-gb 4
```

Convenience script for uniform and Zipf alpha 0.8/1.0/1.2:

```bash
scripts/run_memtier.sh
```

Default large-run parameters are aligned with the paper workload sketch:

```text
--num-embeddings 200000000
--embedding-dim 128
--rows-per-block 2048
--batch-size 16384
--bag-size 32
--num-batches 1000
--dram-cache-gb 32
--hbm-cache-gb 4
```

## JSONL result schema and metrics

Each run appends one JSON object per line to `results/*.jsonl`:

```json
{
  "system": "memtier",
  "dist": "zipf",
  "zipf_alpha": 1.0,
  "num_embeddings": 200000000,
  "embedding_dim": 128,
  "rows_per_block": 2048,
  "batch_size": 16384,
  "bag_size": 32,
  "num_batches": 1000,
  "avg_latency_ms": 0.0,
  "p50_latency_ms": 0.0,
  "p95_latency_ms": 0.0,
  "throughput_lookups_per_sec": 0.0,
  "hbm_hit_rate": 0.0,
  "dram_hit_rate": 0.0,
  "ssd_read_bytes": 0,
  "dram_to_gpu_bytes": 0,
  "posix_reads": 0,
  "gds_reads": 0
}
```

Metric meanings:

- `avg_latency_ms`, `p50_latency_ms`, `p95_latency_ms`: per-batch block lookup
  latency, including SSD read and DRAM-to-HBM copies for MemTier.
- `throughput_lookups_per_sec`: `batch_size * bag_size * num_batches` divided by
  measured elapsed lookup time.
- `hbm_hit_rate`: fraction of block requests served by HBM cache.
- `dram_hit_rate`: fraction served by pinned DRAM cache and copied to HBM.
- `ssd_read_bytes`: bytes read from SSD through POSIX/GDS path.
- `dram_to_gpu_bytes`: bytes copied from pinned DRAM cache to GPU HBM.
- `posix_reads`, `gds_reads`: block read counts by selected I/O path.

## Plot or summarize results

```bash
python scripts/plot_results.py --input results/memtier.jsonl --output results/memtier_latency.png
```

If matplotlib is unavailable, the script prints a compact JSON summary instead.

## Limitations

- Phase 1/2 does not yet compute final embedding-bag outputs from Python; it
  focuses on block residency, cache behavior, and metrics.
- The HBM cache is a simple LRU buffer.  Evicting an HBM block currently removes
  its residency entry and may force a future POSIX reload instead of preserving a
  DRAM-only residency marker.
- The GDS path is a compile-time stub and deliberately avoids a libcufile
  dependency.
- The UVM baseline is a TODO skeleton.
- The benchmark assumes a single process and single CUDA GPU.
