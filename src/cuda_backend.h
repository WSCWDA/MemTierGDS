#ifndef MEMTIER_CUDA_BACKEND_H
#define MEMTIER_CUDA_BACKEND_H

#include <cstddef>

int memtier_cuda_copy_to_gpu(void* gpu_dst, const void* host_src, size_t size, void* stream);
int memtier_cuda_alloc(void** ptr, size_t size, int device_id);
int memtier_cuda_free(void* ptr);
int memtier_cuda_synchronize(void* stream);
int memtier_cuda_available();
int memtier_cuda_copy_to_host(void* host_dst, const void* gpu_src, size_t size);
int memtier_cuda_device_name(int device_id, char* out_name, size_t out_len);

#endif
