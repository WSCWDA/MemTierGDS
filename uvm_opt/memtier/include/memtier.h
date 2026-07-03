#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include "memtier_types.h"

#ifdef __cplusplus
extern "C" {
#endif

int memtierInit(int device_id);
int memtierFinalize();
int memtierMapFile(const char* path, uint64_t file_offset, size_t length, unsigned flags, MemTierHandle* out_handle);
int memtierPrefetchToGpu(MemTierHandle handle, cudaStream_t stream);
int memtierPrefetchToDram(MemTierHandle handle);
int memtierEnsureResident(MemTierHandle handle, int target_tier, cudaStream_t stream);
int memtierUnmapFile(MemTierHandle handle);
void memtierPrintState();
MemTierResidency memtierGetResidency(MemTierHandle handle);
MemTierPath memtierGetLastPath(MemTierHandle handle);

#ifdef __cplusplus
}
#endif
