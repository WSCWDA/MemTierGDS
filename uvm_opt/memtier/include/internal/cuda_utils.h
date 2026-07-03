#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s returned %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(_e)); \
        return -1; \
    } \
} while (0)

namespace memtier::cuda_utils {
int set_device(int device_id);
int get_current_device();
int allocate_device(void** ptr, size_t length);
int allocate_managed(void** ptr, size_t length);
int allocate_pinned(void** ptr, size_t length);
void free_device(void* ptr);
void free_pinned(void* ptr);
float elapsed_ms(cudaEvent_t start, cudaEvent_t stop);
}
