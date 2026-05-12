#include "memtier/object_api.h"

#include <vector>

int memtier_load_chunks(memtier_ctx_t* ctx, const memtier_chunk_desc_t* chunks, size_t nchunks,
                        void** dsts, const memtier_chunk_options_t* options) {
  if (!ctx || !chunks || !dsts || nchunks == 0) return MEMTIER_ERR_INVALID;
  for (size_t i = 0; i < nchunks; ++i) {
    if (!chunks[i].path || chunks[i].size == 0 || !dsts[i]) return MEMTIER_ERR_INVALID;
  }

  std::vector<memtier_range_t> ranges(nchunks);
  for (size_t i = 0; i < nchunks; ++i) {
    ranges[i] = memtier_range_t{chunks[i].path, chunks[i].offset, chunks[i].size};
  }

  memtier_read_options_t ro{};
  ro.target = options ? options->target : MEMTIER_TARGET_CPU;
  ro.hints = options ? options->hints
                     : (MEMTIER_HINT_RANDOM | MEMTIER_HINT_SMALL_IO | MEMTIER_HINT_REUSE);
  return memtier_readv(ctx, ranges.data(), dsts, nchunks, &ro);
}
