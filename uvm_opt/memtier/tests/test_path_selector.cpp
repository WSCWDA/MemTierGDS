#include "internal/path_selector.h"
#include <cassert>
#include <cstdio>
int main(){ RuntimeCapability cap{true,true,0}; ObjectEntry o; o.file_range.length=2<<20; o.flags=MEMTIER_STREAMING|MEMTIER_PREFER_GDS; assert(selectPath(o,cap)==MemTierPath::GDS_TO_HBM); o.flags=MEMTIER_REUSE_EXPECTED; assert(selectPath(o,cap)==MemTierPath::POSIX_TO_DRAM); o.residency=MemTierResidency::DRAM_RESIDENT; assert(selectPath(o,cap)==MemTierPath::DRAM_TO_HBM); o.residency=MemTierResidency::SSD_ONLY; o.flags=0; cap.gds_available=false; assert(selectPath(o,cap)==MemTierPath::POSIX_TO_DRAM); cap.gds_available=true; o.file_range.length=4096; assert(selectPath(o,cap)==MemTierPath::POSIX_TO_DRAM); puts("test_path_selector passed"); }
