#include "cuda_backend.h"
#include "memtier/types.h"

#include <cuda_runtime.h>
#include <cstdio>

int memtier_cuda_copy_to_gpu(void* gpu_dst, const void* host_src, size_t size, void* stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  cudaError_t e = cudaMemcpyAsync(gpu_dst, host_src, size, cudaMemcpyHostToDevice, s);
  return e == cudaSuccess ? MEMTIER_OK : MEMTIER_ERR_CUDA;
}
int memtier_cuda_alloc(void** ptr, size_t size, int device_id) {
  if (cudaSetDevice(device_id) != cudaSuccess) return MEMTIER_ERR_CUDA;
  return cudaMalloc(ptr, size) == cudaSuccess ? MEMTIER_OK : MEMTIER_ERR_CUDA;
}
int memtier_cuda_free(void* ptr) { return cudaFree(ptr) == cudaSuccess ? MEMTIER_OK : MEMTIER_ERR_CUDA; }
int memtier_cuda_synchronize(void* stream) {
  return cudaStreamSynchronize(static_cast<cudaStream_t>(stream)) == cudaSuccess ? MEMTIER_OK : MEMTIER_ERR_CUDA;
}
int memtier_cuda_available() { int n = 0; return cudaGetDeviceCount(&n) == cudaSuccess && n > 0; }
int memtier_cuda_copy_to_host(void* host_dst, const void* gpu_src, size_t size) {
  return cudaMemcpy(host_dst, gpu_src, size, cudaMemcpyDeviceToHost) == cudaSuccess ? MEMTIER_OK : MEMTIER_ERR_CUDA;
}
int memtier_cuda_device_name(int device_id, char* out_name, size_t out_len) {
  if (!out_name || out_len == 0) return MEMTIER_ERR_INVALID;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) return MEMTIER_ERR_CUDA;
  snprintf(out_name, out_len, "%s", prop.name);
  return MEMTIER_OK;
}
