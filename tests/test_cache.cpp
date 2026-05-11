#include "memtier/memtier.h"

#include <cstdio>
#include <fstream>
#include <vector>

int main() {
  const char* path = "memtier_cache_test.bin";
  {
    std::ofstream ofs(path, std::ios::binary);
    std::vector<unsigned char> data(4 * 1024 * 1024);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<unsigned char>(i % 251);
    ofs.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
  }

  memtier_options_t opt{};
  opt.chunk_size = 1 << 20;
  opt.dram_cache_size = 2 << 20;
  opt.enable_dram_cache = 1;
  opt.cache_admit = 1;
  memtier_ctx_t* ctx = nullptr;
  if (memtier_init(&opt, &ctx) != MEMTIER_OK) return 1;

  std::vector<unsigned char> buf(4096);
  if (memtier_read(ctx, path, 0, buf.size(), buf.data(), MEMTIER_TARGET_CPU) != MEMTIER_OK) return 2;
  memtier_stats_t s1{};
  memtier_stats(ctx, &s1);
  if (s1.dram_misses == 0 || s1.posix_reads == 0) return 3;

  if (memtier_read(ctx, path, 0, buf.size(), buf.data(), MEMTIER_TARGET_CPU) != MEMTIER_OK) return 4;
  memtier_stats_t s2{};
  memtier_stats(ctx, &s2);
  if (s2.dram_hits == 0) return 5;
  if (s2.ssd_bytes_read != s1.ssd_bytes_read) return 6;

  std::vector<unsigned char> cross(2048);
  if (memtier_read(ctx, path, (1 << 20) - 512, cross.size(), cross.data(), MEMTIER_TARGET_CPU) != MEMTIER_OK) return 7;
  for (size_t i = 0; i < cross.size(); ++i) {
    if (cross[i] != static_cast<unsigned char>(((1 << 20) - 512 + i) % 251)) return 8;
  }

  std::vector<unsigned char> tmp(4096);
  memtier_read(ctx, path, 2 << 20, tmp.size(), tmp.data(), MEMTIER_TARGET_CPU);
  memtier_read(ctx, path, 3 << 20, tmp.size(), tmp.data(), MEMTIER_TARGET_CPU);
  memtier_stats_t s3{};
  memtier_stats(ctx, &s3);
  if (s3.dram_misses < 3) return 9;

  memtier_finalize(ctx);
  std::remove(path);
  return 0;
}
