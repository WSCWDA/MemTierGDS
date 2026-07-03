#include "internal/cuda_utils.h"

namespace memtier::cuda_utils {
int set_device(int device_id) { CUDA_CHECK(cudaSetDevice(device_id)); return 0; }
int get_current_device() { int d = -1; if (cudaGetDevice(&d) != cudaSuccess) return -1; return d; }
int allocate_device(void** ptr, size_t length) { CUDA_CHECK(cudaMalloc(ptr, length)); return 0; }
int allocate_managed(void** ptr, size_t length) { CUDA_CHECK(cudaMallocManaged(ptr, length)); return 0; }
int allocate_pinned(void** ptr, size_t length) { CUDA_CHECK(cudaMallocHost(ptr, length)); return 0; }
void free_device(void* ptr) { if (ptr) (void)cudaFree(ptr); }
void free_pinned(void* ptr) { if (ptr) (void)cudaFreeHost(ptr); }
float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) { float ms = 0.0f; (void)cudaEventElapsedTime(&ms, start, stop); return ms; }
}
