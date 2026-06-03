#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

void cuda_check(cudaError_t err, const char* expr) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error in ") + expr + ": " +
                             cudaGetErrorString(err));
  }
}

__device__ int find_block(uint64_t block_id, const uint64_t* block_ids, int num_blocks) {
  for (int i = 0; i < num_blocks; ++i) {
    if (block_ids[i] == block_id) {
      return i;
    }
  }
  return -1;
}

__global__ void embedding_bag_fp32_kernel(const uint64_t* ids,
                                          int batch_size,
                                          int bag_size,
                                          void** block_ptrs,
                                          const uint64_t* block_ids,
                                          int num_blocks,
                                          int rows_per_block,
                                          int embedding_dim,
                                          int mean_mode,
                                          float* output) {
  const int bag = blockIdx.x;
  const int dim = threadIdx.x + blockIdx.y * blockDim.x;
  if (bag >= batch_size || dim >= embedding_dim) {
    return;
  }

  float acc = 0.0f;
  for (int j = 0; j < bag_size; ++j) {
    const uint64_t id = ids[static_cast<uint64_t>(bag) * bag_size + j];
    const uint64_t bid = id / rows_per_block;
    const int row = static_cast<int>(id % rows_per_block);
    const int block_index = find_block(bid, block_ids, num_blocks);
    if (block_index >= 0) {
      const float* block = static_cast<const float*>(block_ptrs[block_index]);
      acc += block[static_cast<uint64_t>(row) * embedding_dim + dim];
    }
  }
  if (mean_mode && bag_size > 0) {
    acc /= static_cast<float>(bag_size);
  }
  output[static_cast<uint64_t>(bag) * embedding_dim + dim] = acc;
}

__global__ void embedding_bag_fp16_kernel(const uint64_t* ids,
                                          int batch_size,
                                          int bag_size,
                                          void** block_ptrs,
                                          const uint64_t* block_ids,
                                          int num_blocks,
                                          int rows_per_block,
                                          int embedding_dim,
                                          int mean_mode,
                                          float* output) {
  const int bag = blockIdx.x;
  const int dim = threadIdx.x + blockIdx.y * blockDim.x;
  if (bag >= batch_size || dim >= embedding_dim) {
    return;
  }

  float acc = 0.0f;
  for (int j = 0; j < bag_size; ++j) {
    const uint64_t id = ids[static_cast<uint64_t>(bag) * bag_size + j];
    const uint64_t bid = id / rows_per_block;
    const int row = static_cast<int>(id % rows_per_block);
    const int block_index = find_block(bid, block_ids, num_blocks);
    if (block_index >= 0) {
      const __half* block = static_cast<const __half*>(block_ptrs[block_index]);
      acc += __half2float(block[static_cast<uint64_t>(row) * embedding_dim + dim]);
    }
  }
  if (mean_mode && bag_size > 0) {
    acc /= static_cast<float>(bag_size);
  }
  output[static_cast<uint64_t>(bag) * embedding_dim + dim] = acc;
}

}  // namespace

extern "C" int launch_embedding_bag(const uint64_t* device_ids,
                                     int batch_size,
                                     int bag_size,
                                     void** device_block_ptrs,
                                     const uint64_t* device_block_ids,
                                     int num_blocks,
                                     int rows_per_block,
                                     int embedding_dim,
                                     int dtype_size,
                                     int mean_mode,
                                     float* device_output,
                                     cudaStream_t stream) {
  try {
    if (device_ids == nullptr || device_block_ptrs == nullptr || device_block_ids == nullptr ||
        device_output == nullptr) {
      return -1;
    }
    constexpr int kThreads = 128;
    dim3 block(kThreads);
    dim3 grid(batch_size, (embedding_dim + kThreads - 1) / kThreads);
    if (dtype_size == 2) {
      embedding_bag_fp16_kernel<<<grid, block, 0, stream>>>(device_ids,
                                                            batch_size,
                                                            bag_size,
                                                            device_block_ptrs,
                                                            device_block_ids,
                                                            num_blocks,
                                                            rows_per_block,
                                                            embedding_dim,
                                                            mean_mode,
                                                            device_output);
    } else {
      embedding_bag_fp32_kernel<<<grid, block, 0, stream>>>(device_ids,
                                                            batch_size,
                                                            bag_size,
                                                            device_block_ptrs,
                                                            device_block_ids,
                                                            num_blocks,
                                                            rows_per_block,
                                                            embedding_dim,
                                                            mean_mode,
                                                            device_output);
    }
    cuda_check(cudaGetLastError(), "embedding_bag_kernel launch");
    return 0;
  } catch (...) {
    return -1;
  }
}
