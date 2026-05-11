#ifndef MEMTIER_GDS_BACKEND_H
#define MEMTIER_GDS_BACKEND_H

#include <cstddef>
#include <cstdint>

#include "memtier/types.h"

int memtier_gds_available();
int memtier_gds_read(const memtier_options_t& opt, const char* path, uint64_t offset, size_t size, void* dst, size_t* out_bytes);

#endif
