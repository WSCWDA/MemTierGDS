#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EMB_FILE="${EMB_FILE:-$ROOT/results/embeddings.fp16.bin}"
OUT="${OUT:-$ROOT/results/memtier.jsonl}"
LIB="${LIB:-$ROOT/build/libmemtier_embedding.so}"
NUM_EMBEDDINGS="${NUM_EMBEDDINGS:-200000000}"
EMBEDDING_DIM="${EMBEDDING_DIM:-128}"
ROWS_PER_BLOCK="${ROWS_PER_BLOCK:-2048}"
BATCH_SIZE="${BATCH_SIZE:-16384}"
BAG_SIZE="${BAG_SIZE:-32}"
NUM_BATCHES="${NUM_BATCHES:-1000}"
DRAM_CACHE_GB="${DRAM_CACHE_GB:-32}"
HBM_CACHE_GB="${HBM_CACHE_GB:-4}"
DTYPE_SIZE="${DTYPE_SIZE:-2}"

mkdir -p "$ROOT/results"

if [[ ! -f "$EMB_FILE" ]]; then
  "$PYTHON_BIN" "$ROOT/scripts/generate_embedding_file.py" \
    --output "$EMB_FILE" \
    --num-embeddings "$NUM_EMBEDDINGS" \
    --embedding-dim "$EMBEDDING_DIM" \
    --dtype fp16 \
    --chunk-mb 256
fi

run_one() {
  local dist="$1"
  local alpha="$2"
  local trace="$ROOT/results/trace_${dist}_${alpha}.npy"
  "$PYTHON_BIN" "$ROOT/python/embedding_workload.py" \
    --output "$trace" \
    --dist "$dist" \
    --zipf-alpha "$alpha" \
    --batch-size "$BATCH_SIZE" \
    --bag-size "$BAG_SIZE" \
    --num-batches "$NUM_BATCHES" \
    --num-embeddings "$NUM_EMBEDDINGS" \
    --rows-per-block "$ROWS_PER_BLOCK"
  "$PYTHON_BIN" "$ROOT/python/memtier_embedding.py" \
    --embedding-file "$EMB_FILE" \
    --trace "$trace" \
    --lib "$LIB" \
    --output "$OUT" \
    --num-embeddings "$NUM_EMBEDDINGS" \
    --embedding-dim "$EMBEDDING_DIM" \
    --rows-per-block "$ROWS_PER_BLOCK" \
    --dtype-size "$DTYPE_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --bag-size "$BAG_SIZE" \
    --num-batches "$NUM_BATCHES" \
    --dist "$dist" \
    --zipf-alpha "$alpha" \
    --dram-cache-gb "$DRAM_CACHE_GB" \
    --hbm-cache-gb "$HBM_CACHE_GB"
}

run_one uniform 1.0
run_one zipf 0.8
run_one zipf 1.0
run_one zipf 1.2
