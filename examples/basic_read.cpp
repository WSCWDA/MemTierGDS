#include "memtier/memtier.h"

#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: %s <file>\n", argv[0]);
    return 1;
  }
  memtier_ctx_t* ctx = nullptr;
  if (memtier_init(nullptr, &ctx) != MEMTIER_OK) return 1;
  std::vector<unsigned char> buf(128);
  int rc = memtier_read(ctx, argv[1], 0, buf.size(), buf.data(), MEMTIER_TARGET_CPU);
  if (rc != MEMTIER_OK) {
    std::printf("read failed: %s\n", memtier_strerror(rc));
  } else {
    std::printf("read ok, first byte=%u\n", static_cast<unsigned>(buf[0]));
  }
  memtier_finalize(ctx);
  return rc == MEMTIER_OK ? 0 : 2;
}
