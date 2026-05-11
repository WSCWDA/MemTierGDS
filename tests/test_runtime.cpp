#include "memtier/memtier.h"

#include <cstring>
#include <string>

int main() {
  memtier_ctx_t* ctx = memtier_init(nullptr);
  if (!ctx) return 1;
  memtier_stats_t s{};
  if (memtier_stats(ctx, &s) != MEMTIER_OK) return 2;
  memtier_stats_t z{};
  if (std::memcmp(&s, &z, sizeof(s)) != 0) return 3;
  if (memtier_finalize(ctx) != MEMTIER_OK) return 4;

  memtier_config_t cfg{};
  cfg.chunk_size = 2ull << 20;
  cfg.dram_cache_size = 16ull << 20;
  cfg.enable_dram_cache = true;
  cfg.enable_gds = true;
  cfg.enable_hbm_cache = true;
  cfg.policy = "adaptive";
  ctx = memtier_init(&cfg);
  if (!ctx) return 5;
  if (memtier_finalize(ctx) != MEMTIER_OK) return 6;

  if (memtier_stats(nullptr, &s) != MEMTIER_ERR_INVALID_ARG) return 7;
  ctx = memtier_init(nullptr);
  if (!ctx) return 8;
  if (memtier_stats(ctx, nullptr) != MEMTIER_ERR_INVALID_ARG) return 9;
  if (memtier_finalize(nullptr) != MEMTIER_ERR_INVALID_ARG) return 10;
  if (!memtier_strerror(MEMTIER_OK) || !memtier_strerror(MEMTIER_ERR_INVALID_ARG)) return 11;
  if (std::string(memtier_strerror(-12345)) != "unknown error") return 12;
  return memtier_finalize(ctx) == MEMTIER_OK ? 0 : 13;
}
