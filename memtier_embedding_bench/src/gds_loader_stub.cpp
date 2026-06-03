#include "embedding_config.h"

namespace memtier {

size_t gds_load_block_or_posix(PosixLoader& fallback,
                               uint64_t block_id,
                               void* dst,
                               size_t bytes,
                               bool enable_gds) {
#ifdef ENABLE_GDS
  // TODO: replace this stub with cuFileRead once libcufile is an explicit build
  // dependency.  Phase 1/2 deliberately keeps POSIX as the always-available path.
  (void)enable_gds;
#else
  (void)enable_gds;
#endif
  return fallback.load_block(block_id, dst, bytes);
}

}  // namespace memtier
