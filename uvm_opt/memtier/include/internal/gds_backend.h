#pragma once
#include "internal/object_table.h"
#include <sys/types.h>
#if MEMTIER_ENABLE_GDS
#include <cufile.h>
#endif
class GdsBackend {
public:
    bool init(); void finalize(); bool available() const { return available_; }
    int openFile(ObjectEntry& obj); int registerFile(ObjectEntry& obj); int registerBuffer(ObjectEntry& obj, void* gpu_ptr, size_t length);
    ssize_t readToGpu(ObjectEntry& obj, void* gpu_ptr, size_t length, uint64_t file_offset);
    int deregisterBuffer(ObjectEntry& obj); int closeFile(ObjectEntry& obj);
private:
    bool available_ = false;
#if MEMTIER_ENABLE_GDS
    CUfileHandle_t cf_handle_{};
#endif
};
