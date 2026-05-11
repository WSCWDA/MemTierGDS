#include "memtier/memtier.h"

#include <cstdio>

int main() {
  memtier_config_t cfg{};
  cfg.dram_cache_size = 64ull << 20;
  cfg.chunk_size = 1ull << 20;
  cfg.enable_gds = true;
  cfg.enable_dram_cache = true;
  cfg.policy = "adaptive";

  memtier_ctx_t* ctx = memtier_init(&cfg);
  if (!ctx) {
    std::printf("runtime init failed\n");
    return 1;
  }
  memtier_stats_t s{};
  if (memtier_stats(ctx, &s) != MEMTIER_OK) return 2;
  std::printf("runtime init ok\n");
  std::printf("chunk size=%zu dram cache size=%zu\n", cfg.chunk_size, cfg.dram_cache_size);
  std::printf("gds enabled=%d\n", cfg.enable_gds ? 1 : 0);
  std::printf("initial total_requests=%llu\n", (unsigned long long)s.total_requests);
  return memtier_finalize(ctx) == MEMTIER_OK ? 0 : 3;
}
