#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <torch/extension.h>

// Metadata for one UVM-backed data buffer.  The first prototype returns tensors
// directly, but this struct mirrors the allocator/storage state that a future
// PyTorch StorageImpl integration would carry.
struct MemTierBuffer {
  void* ptr;
  size_t bytes;
  int device;
  bool is_loaded;
  std::string source_path;
  size_t file_offset;
};

uintptr_t allocate_managed(size_t bytes, int device);
void free_managed(uintptr_t ptr);
void advise_preferred_cpu(uintptr_t ptr, size_t bytes);
void advise_preferred_gpu(uintptr_t ptr, size_t bytes, int device);
void advise_accessed_by_gpu(uintptr_t ptr, size_t bytes, int device);
void prefetch_to_gpu(uintptr_t ptr, size_t bytes, int device);
void prefetch_to_cpu(uintptr_t ptr, size_t bytes);

torch::Tensor load_file_to_managed(const std::string& path,
                                   size_t offset,
                                   int64_t rows,
                                   int64_t cols,
                                   int device);
torch::Tensor uvm_tensor_from_file(const std::string& path,
                                   int64_t rows,
                                   int64_t cols,
                                   size_t offset,
                                   int device,
                                   const std::string& prefetch_policy);
torch::Tensor touch_gpu(torch::Tensor tensor);
pybind11::dict get_stats();
void reset_stats();
