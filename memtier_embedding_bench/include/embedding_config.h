#pragma once

#include "memtier_api.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace memtier {

inline uint64_t block_size_bytes(const memtier_embedding_config_t& cfg) {
  return static_cast<uint64_t>(cfg.rows_per_block) * cfg.embedding_dim * cfg.dtype_size;
}

struct DramCacheStats {
  uint64_t dram_hit = 0;
  uint64_t dram_miss = 0;
  uint64_t eviction_count = 0;
};

class DramCache {
 public:
  DramCache(uint64_t cache_bytes, uint64_t block_bytes);
  ~DramCache();
  DramCache(const DramCache&) = delete;
  DramCache& operator=(const DramCache&) = delete;

  void* get(uint64_t block_id);
  void* put(uint64_t block_id, const void* data, size_t bytes);
  void evict_if_needed();
  void* allocate_slot_for_load(uint64_t block_id);
  DramCacheStats stats() const;
  uint64_t capacity_blocks() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class PosixLoader {
 public:
  PosixLoader(const std::string& path, const memtier_embedding_config_t& config, bool use_odirect = false);
  ~PosixLoader();
  PosixLoader(const PosixLoader&) = delete;
  PosixLoader& operator=(const PosixLoader&) = delete;

  size_t load_block(uint64_t block_id, void* dst, size_t bytes);

 private:
  int fd_ = -1;
  memtier_embedding_config_t config_{};
  std::string path_;
};

size_t gds_load_block_or_posix(PosixLoader& fallback,
                               uint64_t block_id,
                               void* dst,
                               size_t bytes,
                               bool enable_gds);

}  // namespace memtier
