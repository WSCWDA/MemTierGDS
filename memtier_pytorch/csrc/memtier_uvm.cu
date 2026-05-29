#include "memtier_uvm.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <pybind11/pybind11.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

#define MEMTIER_CUDA_CHECK(expr)                                              \
  do {                                                                        \
    cudaError_t _err = (expr);                                                \
    if (_err != cudaSuccess) {                                                \
      throw std::runtime_error(std::string("CUDA error at ") + __FILE__ +     \
                               ":" + std::to_string(__LINE__) + " in " +     \
                               #expr + ": " + cudaGetErrorString(_err));      \
    }                                                                         \
  } while (0)

struct Stats {
  std::atomic<size_t> allocated_bytes{0};
  std::atomic<size_t> file_read_bytes{0};
  std::atomic<size_t> num_allocations{0};
  std::atomic<size_t> num_prefetch_gpu{0};
  std::atomic<size_t> num_prefetch_cpu{0};
  std::atomic<uint64_t> total_file_read_us{0};
  std::atomic<uint64_t> total_prefetch_us{0};
  std::atomic<uint64_t> total_gpu_touch_us{0};
};

Stats g_stats;
std::mutex g_alloc_mutex;
std::unordered_map<void*, size_t> g_alloc_sizes;

uint64_t elapsed_us(std::chrono::steady_clock::time_point begin,
                    std::chrono::steady_clock::time_point end) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
          .count());
}

void ensure_cuda_device(int device) {
  int count = 0;
  MEMTIER_CUDA_CHECK(cudaGetDeviceCount(&count));
  if (count <= 0) {
    throw std::runtime_error("MemTier requires CUDA, but no CUDA device is visible");
  }
  if (device < 0 || device >= count) {
    throw std::runtime_error("Invalid CUDA device " + std::to_string(device) +
                             "; visible device count is " +
                             std::to_string(count));
  }
}

void record_allocation(void* ptr, size_t bytes) {
  std::lock_guard<std::mutex> lock(g_alloc_mutex);
  g_alloc_sizes[ptr] = bytes;
  g_stats.allocated_bytes.fetch_add(bytes, std::memory_order_relaxed);
  g_stats.num_allocations.fetch_add(1, std::memory_order_relaxed);
}

void record_free(void* ptr, size_t fallback_bytes = 0) {
  size_t bytes = fallback_bytes;
  {
    std::lock_guard<std::mutex> lock(g_alloc_mutex);
    auto it = g_alloc_sizes.find(ptr);
    if (it != g_alloc_sizes.end()) {
      bytes = it->second;
      g_alloc_sizes.erase(it);
    }
  }
  if (bytes != 0) {
    g_stats.allocated_bytes.fetch_sub(bytes, std::memory_order_relaxed);
  }
}

// POSIX pread path: file data is copied directly into the UVM allocation.  This
// intentionally bypasses PyTorch pin_memory/DataLoader staging so the prototype
// can compare UVM page migration against the default CPU->GPU copy path.
void pread_exact(const std::string& path, size_t offset, void* dst, size_t bytes) {
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    throw std::runtime_error("open(" + path + ") failed: " + std::strerror(errno));
  }

  char* out = static_cast<char*>(dst);
  size_t total = 0;
  while (total < bytes) {
    ssize_t got = ::pread(fd, out + total, bytes - total,
                          static_cast<off_t>(offset + total));
    if (got < 0) {
      if (errno == EINTR) {
        continue;
      }
      int saved = errno;
      ::close(fd);
      throw std::runtime_error("pread(" + path + ") failed: " +
                               std::strerror(saved));
    }
    if (got == 0) {
      ::close(fd);
      throw std::runtime_error("pread(" + path + ") reached EOF after " +
                               std::to_string(total) + " of " +
                               std::to_string(bytes) + " bytes");
    }
    total += static_cast<size_t>(got);
  }

  if (::close(fd) != 0) {
    throw std::runtime_error("close(" + path + ") failed: " + std::strerror(errno));
  }
}

__global__ void touch_kernel(const float* __restrict__ x,
                             float* __restrict__ y,
                             int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = x[i] * 1.0f;
  }
}

void cuda_free_noexcept(void* ptr, size_t bytes) {
  if (ptr == nullptr) {
    return;
  }
  cudaError_t err = cudaFree(ptr);
  if (err == cudaSuccess) {
    record_free(ptr, bytes);
  }
}

}  // namespace

uintptr_t allocate_managed(size_t bytes, int device) {
  ensure_cuda_device(device);
  c10::cuda::CUDAGuard guard(device);
  void* ptr = nullptr;
  MEMTIER_CUDA_CHECK(cudaMallocManaged(&ptr, bytes, cudaMemAttachGlobal));
  record_allocation(ptr, bytes);
  return reinterpret_cast<uintptr_t>(ptr);
}

void free_managed(uintptr_t addr) {
  void* ptr = reinterpret_cast<void*>(addr);
  if (ptr == nullptr) {
    return;
  }
  MEMTIER_CUDA_CHECK(cudaFree(ptr));
  record_free(ptr);
}

void advise_preferred_cpu(uintptr_t addr, size_t bytes) {
  void* ptr = reinterpret_cast<void*>(addr);
  MEMTIER_CUDA_CHECK(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation,
                                   cudaCpuDeviceId));
}

void advise_preferred_gpu(uintptr_t addr, size_t bytes, int device) {
  ensure_cuda_device(device);
  void* ptr = reinterpret_cast<void*>(addr);
  MEMTIER_CUDA_CHECK(
      cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, device));
}

void advise_accessed_by_gpu(uintptr_t addr, size_t bytes, int device) {
  ensure_cuda_device(device);
  void* ptr = reinterpret_cast<void*>(addr);
  MEMTIER_CUDA_CHECK(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetAccessedBy, device));
}

void prefetch_to_gpu(uintptr_t addr, size_t bytes, int device) {
  ensure_cuda_device(device);
  c10::cuda::CUDAGuard guard(device);
  void* ptr = reinterpret_cast<void*>(addr);
  auto begin = std::chrono::steady_clock::now();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device);
  MEMTIER_CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, device, stream));
  MEMTIER_CUDA_CHECK(cudaStreamSynchronize(stream));
  auto end = std::chrono::steady_clock::now();
  g_stats.num_prefetch_gpu.fetch_add(1, std::memory_order_relaxed);
  g_stats.total_prefetch_us.fetch_add(elapsed_us(begin, end),
                                      std::memory_order_relaxed);
}

void prefetch_to_cpu(uintptr_t addr, size_t bytes) {
  void* ptr = reinterpret_cast<void*>(addr);
  auto begin = std::chrono::steady_clock::now();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  MEMTIER_CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, cudaCpuDeviceId, stream));
  MEMTIER_CUDA_CHECK(cudaStreamSynchronize(stream));
  auto end = std::chrono::steady_clock::now();
  g_stats.num_prefetch_cpu.fetch_add(1, std::memory_order_relaxed);
  g_stats.total_prefetch_us.fetch_add(elapsed_us(begin, end),
                                      std::memory_order_relaxed);
}

torch::Tensor load_file_to_managed(const std::string& path,
                                   size_t offset,
                                   int64_t rows,
                                   int64_t cols,
                                   int device) {
  if (rows <= 0 || cols <= 0) {
    throw std::invalid_argument("rows and cols must be positive");
  }
  const size_t bytes = static_cast<size_t>(rows) * static_cast<size_t>(cols) *
                       sizeof(float);
  uintptr_t addr = allocate_managed(bytes, device);
  void* ptr = reinterpret_cast<void*>(addr);

  try {
    auto begin = std::chrono::steady_clock::now();
    pread_exact(path, offset, ptr, bytes);
    auto end = std::chrono::steady_clock::now();
    g_stats.file_read_bytes.fetch_add(bytes, std::memory_order_relaxed);
    g_stats.total_file_read_us.fetch_add(elapsed_us(begin, end),
                                         std::memory_order_relaxed);
  } catch (...) {
    cuda_free_noexcept(ptr, bytes);
    throw;
  }

  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto deleter = [bytes](void* p) { cuda_free_noexcept(p, bytes); };
  return torch::from_blob(ptr, {rows, cols}, deleter, options);
}

torch::Tensor uvm_tensor_from_file(const std::string& path,
                                   int64_t rows,
                                   int64_t cols,
                                   size_t offset,
                                   int device,
                                   const std::string& prefetch_policy) {
  torch::Tensor tensor = load_file_to_managed(path, offset, rows, cols, device);
  const size_t bytes = static_cast<size_t>(rows) * static_cast<size_t>(cols) *
                       sizeof(float);
  uintptr_t addr = reinterpret_cast<uintptr_t>(tensor.data_ptr<float>());

  // Policy hook for UVM migration experiments: "none" lets the GPU fault pages
  // on first touch, while "gpu" uses cudaMemAdvise + cudaMemPrefetchAsync before
  // the CUDA kernel consumes the managed pointer.
  if (prefetch_policy == "cpu") {
    advise_preferred_cpu(addr, bytes);
  } else if (prefetch_policy == "gpu") {
    advise_preferred_gpu(addr, bytes, device);
    prefetch_to_gpu(addr, bytes, device);
  } else if (prefetch_policy == "accessed_by") {
    advise_accessed_by_gpu(addr, bytes, device);
  } else if (prefetch_policy == "none") {
    // Intentionally no-op.
  } else {
    throw std::invalid_argument("Unknown MemTier prefetch policy: " +
                                prefetch_policy);
  }
  return tensor;
}

torch::Tensor touch_gpu(torch::Tensor tensor) {
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
              "touch_gpu currently supports float32 tensors only");
  TORCH_CHECK(tensor.is_contiguous(), "touch_gpu expects a contiguous tensor");
  TORCH_CHECK(!tensor.is_cuda(),
              "touch_gpu expects the UVM pointer wrapped as a CPU tensor");
  TORCH_CHECK(tensor.numel() > 0, "touch_gpu expects a non-empty tensor");

  int device = 0;
  MEMTIER_CUDA_CHECK(cudaGetDevice(&device));
  c10::cuda::CUDAGuard guard(device);

  auto out = torch::empty(tensor.sizes(),
                          tensor.options().device(torch::kCUDA, device));
  const int threads = 256;
  const int64_t n = tensor.numel();
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device);

  auto begin = std::chrono::steady_clock::now();
  touch_kernel<<<blocks, threads, 0, stream>>>(tensor.data_ptr<float>(),
                                               out.data_ptr<float>(), n);
  MEMTIER_CUDA_CHECK(cudaGetLastError());
  MEMTIER_CUDA_CHECK(cudaStreamSynchronize(stream));
  auto end = std::chrono::steady_clock::now();
  g_stats.total_gpu_touch_us.fetch_add(elapsed_us(begin, end),
                                       std::memory_order_relaxed);
  return out;
}

pybind11::dict get_stats() {
  pybind11::dict d;
  d["allocated_bytes"] = g_stats.allocated_bytes.load(std::memory_order_relaxed);
  d["file_read_bytes"] = g_stats.file_read_bytes.load(std::memory_order_relaxed);
  d["num_allocations"] = g_stats.num_allocations.load(std::memory_order_relaxed);
  d["num_prefetch_gpu"] = g_stats.num_prefetch_gpu.load(std::memory_order_relaxed);
  d["num_prefetch_cpu"] = g_stats.num_prefetch_cpu.load(std::memory_order_relaxed);
  d["total_file_read_us"] =
      g_stats.total_file_read_us.load(std::memory_order_relaxed);
  d["total_prefetch_us"] =
      g_stats.total_prefetch_us.load(std::memory_order_relaxed);
  d["total_gpu_touch_us"] =
      g_stats.total_gpu_touch_us.load(std::memory_order_relaxed);
  return d;
}

void reset_stats() {
  g_stats.allocated_bytes.store(0, std::memory_order_relaxed);
  g_stats.file_read_bytes.store(0, std::memory_order_relaxed);
  g_stats.num_allocations.store(0, std::memory_order_relaxed);
  g_stats.num_prefetch_gpu.store(0, std::memory_order_relaxed);
  g_stats.num_prefetch_cpu.store(0, std::memory_order_relaxed);
  g_stats.total_file_read_us.store(0, std::memory_order_relaxed);
  g_stats.total_prefetch_us.store(0, std::memory_order_relaxed);
  g_stats.total_gpu_touch_us.store(0, std::memory_order_relaxed);
}
