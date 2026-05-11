#ifndef MEMTIER_CONTEXT_H
#define MEMTIER_CONTEXT_H

#include <memory>
#include <string>

#include "cache_directory.h"
#include "cuda_backend.h"
#include "gds_backend.h"
#include "memtier/memtier.h"
#include "posix_backend.h"
#include "stats.h"

struct MemTierContext {
  memtier_config_t config{};
  std::unique_ptr<CacheDirectory> dram_cache;
  std::unique_ptr<PosixBackend> posix_backend;
  std::unique_ptr<GdsBackend> gds_backend;
  std::unique_ptr<CudaBackend> cuda_backend;
  Stats stats;
  bool initialized{false};
  std::string policy;
};

struct memtier_ctx { std::unique_ptr<MemTierContext> impl; };

memtier_config_t memtier_default_config();

#endif
