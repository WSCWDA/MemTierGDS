#include "internal/path_selector.h"
MemTierPath selectPath(const ObjectEntry& obj, const RuntimeCapability& cap) {
    // TODO: replace static policy with access monitor and reuse estimator.
    if (obj.residency == MemTierResidency::HBM_RESIDENT || obj.residency == MemTierResidency::DRAM_AND_HBM) return MemTierPath::DRAM_TO_HBM;
    if (obj.residency == MemTierResidency::DRAM_RESIDENT) return MemTierPath::DRAM_TO_HBM;
    if ((obj.flags & MEMTIER_STREAMING) && (obj.flags & MEMTIER_PREFER_GDS) && cap.gds_available) return MemTierPath::GDS_TO_HBM;
    if (obj.flags & MEMTIER_REUSE_EXPECTED) return MemTierPath::POSIX_TO_DRAM;
    if (cap.gds_available && obj.file_range.length >= (1u<<20)) return MemTierPath::GDS_TO_HBM;
    return MemTierPath::POSIX_TO_DRAM;
}
