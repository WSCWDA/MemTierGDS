#include "path_selector.h"

namespace memtier {

bool is_gds_available(const memtier_options_t& opt) { return opt.enable_gds != 0; }

bool is_request_gds_suitable(uint64_t offset, size_t size, const memtier_options_t& opt) {
  if (size < opt.gds_min_size) return false;
  if (opt.gds_alignment == 0) return false;
  if ((offset % opt.gds_alignment) != 0 || (size % opt.gds_alignment) != 0) return false;
  if (opt.hints & (MEMTIER_HINT_REUSE | MEMTIER_HINT_SMALL_IO | MEMTIER_HINT_RANDOM)) return false;
  return true;
}

memtier_path_t select_path(bool dram_hit_candidate, uint64_t offset, size_t size, const memtier_options_t& opt) {
  if (dram_hit_candidate && !opt.bypass_cache) return MEMTIER_PATH_DRAM_HIT;
  if (opt.force_posix) return MEMTIER_PATH_POSIX_READ_THEN_CACHE;

  bool gds_suitable = is_request_gds_suitable(offset, size, opt) ||
                      (opt.hints & (MEMTIER_HINT_STREAMING | MEMTIER_HINT_CHECKPOINT | MEMTIER_HINT_SEQUENTIAL));

  if (opt.force_gds) {
    if (is_gds_available(opt) && gds_suitable) return MEMTIER_PATH_GDS_READ;
    return MEMTIER_PATH_GDS_STUB_FALLBACK;
  }

  if (is_gds_available(opt) && gds_suitable) return MEMTIER_PATH_GDS_READ;
  if (opt.bypass_cache || !opt.enable_dram_cache) return MEMTIER_PATH_POSIX_READ;
  return MEMTIER_PATH_POSIX_READ_THEN_CACHE;
}

}
