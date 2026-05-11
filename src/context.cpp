#include "context.h"

memtier_config_t memtier_default_config() {
  memtier_config_t c{};
  c.device_id = 0;
  c.dram_cache_size = 1ull << 30;
  c.hbm_cache_size = 0;
  c.chunk_size = 1ull << 20;
  c.enable_posix = true;
  c.enable_gds = false;
  c.enable_dram_cache = true;
  c.enable_hbm_cache = false;
  c.gds_min_size = 1ull << 20;
  c.gds_alignment = 4096;
  c.num_workers = 0;
  c.policy = "adaptive";
  return c;
}
