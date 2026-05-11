#include "memtier/memtier.h"

#include <cstdio>
#include <fstream>
#include <vector>
#include "../src/file_prep.h"

int main() {
  const char* path = "/mnt/gds2/cwd_test/memtier_cache_reuse.bin";
  if (memtier_prepare_experiment_file("/mnt/gds2/cwd_test", "memtier_cache_reuse.bin", 1ull << 20) != 0) return 1;

  memtier_ctx_t* ctx = nullptr;
  memtier_options_t opt{};
  opt.enable_dram_cache = 1;
  opt.cache_admit = 1;
  if (memtier_init(&opt, &ctx) != MEMTIER_OK) return 1;

  std::vector<unsigned char> out(4096);
  memtier_read(ctx, path, 0, out.size(), out.data(), MEMTIER_TARGET_CPU);
  memtier_stats_t s1{}; memtier_stats(ctx, &s1);
  memtier_read(ctx, path, 0, out.size(), out.data(), MEMTIER_TARGET_CPU);
  memtier_stats_t s2{}; memtier_stats(ctx, &s2);

  std::printf("after first read: misses=%llu hits=%llu posix=%llu\n",
              (unsigned long long)s1.dram_misses, (unsigned long long)s1.dram_hits, (unsigned long long)s1.posix_reads);
  std::printf("after second read: misses=%llu hits=%llu posix=%llu\n",
              (unsigned long long)s2.dram_misses, (unsigned long long)s2.dram_hits, (unsigned long long)s2.posix_reads);

  memtier_finalize(ctx);
  std::remove(path);
  return 0;
}
