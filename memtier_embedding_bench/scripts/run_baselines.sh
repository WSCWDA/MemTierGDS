#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EMB_FILE="${EMB_FILE:-$ROOT/results/embeddings.fp16.bin}"
TRACE="${TRACE:-$ROOT/results/trace_uniform.npy}"
OUT="${OUT:-$ROOT/results/baselines.jsonl}"
NUM_EMBEDDINGS="${NUM_EMBEDDINGS:-1000000}"
EMBEDDING_DIM="${EMBEDDING_DIM:-128}"
ROWS_PER_BLOCK="${ROWS_PER_BLOCK:-2048}"
BATCH_SIZE="${BATCH_SIZE:-16384}"
BAG_SIZE="${BAG_SIZE:-32}"
NUM_BATCHES="${NUM_BATCHES:-100}"
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

if [[ ! -f "$TRACE" ]]; then
  "$PYTHON_BIN" "$ROOT/python/embedding_workload.py" \
    --output "$TRACE" \
    --dist uniform \
    --batch-size "$BATCH_SIZE" \
    --bag-size "$BAG_SIZE" \
    --num-batches "$NUM_BATCHES" \
    --num-embeddings "$NUM_EMBEDDINGS" \
    --rows-per-block "$ROWS_PER_BLOCK"
fi

"$PYTHON_BIN" "$ROOT/python/baseline_cpu_staging.py" \
  --embedding-file "$EMB_FILE" \
  --trace "$TRACE" \
  --output "$OUT" \
  --num-embeddings "$NUM_EMBEDDINGS" \
  --embedding-dim "$EMBEDDING_DIM" \
  --rows-per-block "$ROWS_PER_BLOCK" \
  --dtype-size "$DTYPE_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --bag-size "$BAG_SIZE" \
  --num-batches "$NUM_BATCHES"

# Small-table GPU upper bound.  Use TORCH_NUM_EMBEDDINGS to keep it inside HBM.
TORCH_NUM_EMBEDDINGS="${TORCH_NUM_EMBEDDINGS:-1000000}"
"$PYTHON_BIN" "$ROOT/python/baseline_torch_embedding.py" \
  --trace "$TRACE" \
  --output "$OUT" \
  --num-embeddings "$TORCH_NUM_EMBEDDINGS" \
  --embedding-dim "$EMBEDDING_DIM" \
  --rows-per-block "$ROWS_PER_BLOCK" \
  --dtype-size "$DTYPE_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --bag-size "$BAG_SIZE" \
  --num-batches "$NUM_BATCHES" || true

"$PYTHON_BIN" "$ROOT/python/baseline_uvm.py" --output "$OUT"
