#ifndef MEMTIER_PATH_SELECTOR_H
#define MEMTIER_PATH_SELECTOR_H

#include "memtier/types.h"

namespace memtier {

bool is_gds_available(const memtier_options_t& opt);
bool is_request_gds_suitable(uint64_t offset, size_t size, const memtier_options_t& opt);
memtier_path_t select_path(bool dram_hit_candidate, uint64_t offset, size_t size, const memtier_options_t& opt);

}

#endif
