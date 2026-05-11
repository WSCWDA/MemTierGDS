#include "memtier/memtier.h"

#include <cstdio>
#include <fstream>
#include <vector>

int main() {
  const char* path = "memtier_test.bin";
  {
    std::ofstream ofs(path, std::ios::binary);
    std::vector<unsigned char> data(2 * 1024 * 1024);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<unsigned char>(i % 251);
    ofs.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
  }

  memtier_ctx_t* ctx = nullptr;
  memtier_options_t opt{};
  opt.chunk_size = 1 << 20;
  opt.dram_cache_size = 4 << 20;
  opt.enable_dram_cache = 1;
  opt.cache_admit = 1;
  if (memtier_init(&opt, &ctx) != MEMTIER_OK) return 1;

  std::vector<unsigned char> out(4096);
  int rc = memtier_read(ctx, path, 1024, out.size(), out.data(), MEMTIER_TARGET_CPU);
  if (rc != MEMTIER_OK) return 2;
  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i] != static_cast<unsigned char>((1024 + i) % 251)) return 3;
  }

  rc = memtier_read(ctx, path, (2 * 1024 * 1024) - 512, 4096, out.data(), MEMTIER_TARGET_CPU);
  if (rc != MEMTIER_ERR_IO) return 4;

  memtier_finalize(ctx);
  std::remove(path);
  return 0;
}
