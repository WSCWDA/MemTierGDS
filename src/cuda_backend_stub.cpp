#include "cuda_backend.h"
#include "memtier/types.h"

int memtier_cuda_copy_to_gpu(void*, const void*, size_t, void*) { return MEMTIER_ERR_UNSUPPORTED; }
int memtier_cuda_alloc(void**, size_t, int) { return MEMTIER_ERR_UNSUPPORTED; }
int memtier_cuda_free(void*) { return MEMTIER_ERR_UNSUPPORTED; }
int memtier_cuda_synchronize(void*) { return MEMTIER_ERR_UNSUPPORTED; }
int memtier_cuda_available() { return 0; }
int memtier_cuda_copy_to_host(void*, const void*, size_t) { return MEMTIER_ERR_UNSUPPORTED; }
