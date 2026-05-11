#ifndef MEMTIER_INTERNAL_H
#define MEMTIER_INTERNAL_H

#include <cstddef>
#include <cstdint>

#include "cache_directory.h"
#include "memtier/memtier.h"
#include <mutex>

struct memtier_ctx_s {
  memtier_options_t options;
  memtier::CacheDirectory cache;
  memtier_stats_t stats;
  std::mutex mu;

  explicit memtier_ctx_s(const memtier_options_t& opt) : options(opt), cache(opt.chunk_size, opt.dram_cache_size), stats{} {}
};

struct memtier_req_s;

namespace memtier {
int posix_pread_full(const char* path, uint64_t offset, size_t size, uint8_t* dst, size_t* bytes_read);
}

#endif
