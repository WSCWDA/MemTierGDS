#include "gds_backend.h"

#include "memtier_internal.h"

int memtier_gds_available() { return 0; }

int memtier_gds_read(const memtier_options_t& opt, const char* path, uint64_t offset, size_t size, void* dst, size_t* out_bytes) {
  if (!opt.enable_gds) return MEMTIER_ERR_UNSUPPORTED;
  if (!opt.gds_stub_success) return MEMTIER_ERR_GDS;
  return memtier::posix_pread_full(path, offset, size, static_cast<uint8_t*>(dst), out_bytes);
}
