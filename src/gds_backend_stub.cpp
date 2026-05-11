#include "memtier_internal.h"

#include <vector>

namespace memtier {

int gds_stub_read(const memtier_options_t& opt, const char* path, uint64_t offset, size_t size, uint8_t* dst, size_t* out_bytes) {
  if (!opt.enable_gds) return MEMTIER_ERR_UNSUPPORTED;
  if (!opt.gds_stub_success) return MEMTIER_ERR_GDS;
  return posix_pread_full(path, offset, size, dst, out_bytes);
}

}
