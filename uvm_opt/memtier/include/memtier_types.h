#pragma once

#include <cstddef>
#include <cstdint>

enum class MemTierResidency {
    SSD_ONLY,
    DRAM_RESIDENT,
    HBM_RESIDENT,
    DRAM_AND_HBM
};

enum class MemTierPath {
    POSIX_TO_DRAM,
    DRAM_TO_HBM,
    GDS_TO_HBM,
    HYBRID_GDS_AND_DRAM
};

enum MemTierFlags {
    MEMTIER_READ_ONLY        = 1 << 0,
    MEMTIER_PREFER_GDS       = 1 << 1,
    MEMTIER_PREFER_DRAM      = 1 << 2,
    MEMTIER_STREAMING        = 1 << 3,
    MEMTIER_REUSE_EXPECTED   = 1 << 4,
    MEMTIER_MULTI_GPU_SHARED = 1 << 5
};

struct MemTierHandle {
    void* ptr = nullptr;
    size_t length = 0;
    uint64_t file_offset = 0;
    uint64_t file_id = 0;
    int object_id = -1;
};

const char* memtierResidencyToString(MemTierResidency r);
const char* memtierPathToString(MemTierPath p);
