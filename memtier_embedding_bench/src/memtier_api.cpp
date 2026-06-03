#include "memtier_api.h"

#include "embedding_config.h"
#include "metrics.h"
#include "residency_table.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

void cuda_check(cudaError_t err, const char* expr) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error in ") + expr + ": " +
                             cudaGetErrorString(err));
  }
}

class HbmCache {
 public:
  HbmCache(uint64_t cache_bytes, uint64_t block_bytes)
      : block_bytes_(block_bytes), capacity_(std::max<uint64_t>(1, cache_bytes / block_bytes)) {
    const uint64_t bytes = capacity_ * block_bytes_;
    cuda_check(cudaMalloc(&base_, bytes), "cudaMalloc HBM cache");
    free_slots_.reserve(capacity_);
    for (uint64_t i = 0; i < capacity_; ++i) {
      free_slots_.push_back(capacity_ - 1 - i);
    }
  }

  ~HbmCache() {
    if (base_ != nullptr) {
      cudaFree(base_);
    }
  }

  HbmCache(const HbmCache&) = delete;
  HbmCache& operator=(const HbmCache&) = delete;

  void* get(uint64_t block_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(block_id);
    if (it == entries_.end()) {
      return nullptr;
    }
    lru_.erase(it->second.lru_it);
    lru_.push_front(block_id);
    it->second.lru_it = lru_.begin();
    return it->second.ptr;
  }

  std::pair<void*, uint64_t> allocate(uint64_t block_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto existing = entries_.find(block_id);
    if (existing != entries_.end()) {
      lru_.erase(existing->second.lru_it);
      lru_.push_front(block_id);
      existing->second.lru_it = lru_.begin();
      return {existing->second.ptr, UINT64_MAX};
    }

    uint64_t evicted = UINT64_MAX;
    if (free_slots_.empty()) {
      evicted = lru_.back();
      auto victim = entries_.find(evicted);
      if (victim != entries_.end()) {
        free_slots_.push_back(victim->second.slot);
        entries_.erase(victim);
      }
      lru_.pop_back();
    }

    const uint64_t slot = free_slots_.back();
    free_slots_.pop_back();
    void* ptr = static_cast<char*>(base_) + slot * block_bytes_;
    lru_.push_front(block_id);
    entries_[block_id] = Entry{ptr, slot, lru_.begin()};
    return {ptr, evicted};
  }

 private:
  struct Entry {
    void* ptr = nullptr;
    uint64_t slot = 0;
    std::list<uint64_t>::iterator lru_it;
  };

  void* base_ = nullptr;
  uint64_t block_bytes_ = 0;
  uint64_t capacity_ = 0;
  std::vector<uint64_t> free_slots_;
  std::list<uint64_t> lru_;
  std::unordered_map<uint64_t, Entry> entries_;
  std::mutex mutex_;
};

}  // namespace

struct memtier_handle {
  std::string embedding_file;
  memtier_embedding_config_t config{};
  uint64_t block_bytes = 0;
  std::unique_ptr<memtier::ResidencyTable> residency;
  std::unique_ptr<memtier::DramCache> dram_cache;
  std::unique_ptr<memtier::PosixLoader> posix_loader;
  std::unique_ptr<HbmCache> hbm_cache;
  memtier::Metrics metrics;
  std::mutex mutex;
};

extern "C" memtier_handle_t* memtier_init_embedding(
    const char* embedding_file,
    const memtier_embedding_config_t* config) {
  try {
    if (embedding_file == nullptr || config == nullptr) {
      return nullptr;
    }
    if (config->embedding_dim == 0 || config->rows_per_block == 0 || config->dtype_size == 0) {
      return nullptr;
    }
    auto* handle = new memtier_handle();
    handle->embedding_file = embedding_file;
    handle->config = *config;
    handle->block_bytes = memtier::block_size_bytes(*config);
    handle->residency = std::make_unique<memtier::ResidencyTable>();
    handle->dram_cache = std::make_unique<memtier::DramCache>(config->dram_cache_bytes, handle->block_bytes);
    handle->posix_loader = std::make_unique<memtier::PosixLoader>(handle->embedding_file, *config, false);
    handle->hbm_cache = std::make_unique<HbmCache>(config->hbm_cache_bytes, handle->block_bytes);
    return handle;
  } catch (...) {
    return nullptr;
  }
}

extern "C" int memtier_lookup_blocks(memtier_handle_t* handle,
                                      const uint64_t* block_ids,
                                      size_t num_blocks,
                                      void** device_block_ptrs,
                                      memtier_path_t* selected_paths) {
  if (handle == nullptr || block_ids == nullptr || device_block_ptrs == nullptr ||
      selected_paths == nullptr) {
    return -1;
  }

  try {
    std::lock_guard<std::mutex> lock(handle->mutex);
    const auto begin = std::chrono::steady_clock::now();
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");

    for (size_t i = 0; i < num_blocks; ++i) {
      const uint64_t block_id = block_ids[i];
      auto entry = handle->residency->lookup(block_id);

      if (entry.state == memtier::ResidencyState::IN_HBM && entry.device_ptr != nullptr) {
        device_block_ptrs[i] = entry.device_ptr;
        selected_paths[i] = MEMTIER_PATH_HBM_HIT;
        handle->metrics.hbm_hits++;
        continue;
      }

      void* dram_ptr = nullptr;
      memtier_path_t path = MEMTIER_PATH_POSIX_READ;
      if (entry.state == memtier::ResidencyState::IN_DRAM && entry.dram_ptr != nullptr) {
        dram_ptr = entry.dram_ptr;
        path = MEMTIER_PATH_DRAM_HIT;
        handle->metrics.dram_hits++;
      } else {
        dram_ptr = handle->dram_cache->get(block_id);
        if (dram_ptr != nullptr) {
          path = MEMTIER_PATH_DRAM_HIT;
          handle->metrics.dram_hits++;
        } else {
          handle->metrics.dram_misses++;
          handle->residency->mark_loading(block_id);
          dram_ptr = handle->dram_cache->allocate_slot_for_load(block_id);
          const size_t read = memtier::gds_load_block_or_posix(*handle->posix_loader,
                                                               block_id,
                                                               dram_ptr,
                                                               handle->block_bytes,
                                                               handle->config.enable_gds != 0);
          handle->metrics.posix_reads++;
          handle->metrics.bytes_read_from_ssd += read;
          path = MEMTIER_PATH_POSIX_READ;
        }
        handle->residency->mark_dram(block_id, dram_ptr);
      }

      auto [device_ptr, evicted] = handle->hbm_cache->allocate(block_id);
      if (evicted != UINT64_MAX) {
        handle->residency->erase(evicted);
      }
      cuda_check(cudaMemcpyAsync(device_ptr,
                                 dram_ptr,
                                 handle->block_bytes,
                                 cudaMemcpyHostToDevice,
                                 stream),
                 "cudaMemcpyAsync DRAM->HBM");
      handle->metrics.bytes_copied_dram_to_gpu += handle->block_bytes;
      handle->residency->mark_hbm(block_id, device_ptr);
      device_block_ptrs[i] = device_ptr;
      selected_paths[i] = path;
    }

    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize lookup");
    cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    const auto end = std::chrono::steady_clock::now();
    handle->metrics.total_lookup_time_ms +=
        std::chrono::duration<double, std::milli>(end - begin).count();
    const auto dram_stats = handle->dram_cache->stats();
    handle->metrics.eviction_count = dram_stats.eviction_count;
    return 0;
  } catch (...) {
    return -1;
  }
}

extern "C" int memtier_prefetch_blocks(memtier_handle_t* handle,
                                        const uint64_t* block_ids,
                                        size_t num_blocks) {
  if (handle == nullptr || block_ids == nullptr) {
    return -1;
  }
  std::vector<void*> ptrs(num_blocks, nullptr);
  std::vector<memtier_path_t> paths(num_blocks, MEMTIER_PATH_POSIX_READ);
  return memtier_lookup_blocks(handle, block_ids, num_blocks, ptrs.data(), paths.data());
}

extern "C" int memtier_release_blocks(memtier_handle_t* handle,
                                       const uint64_t* block_ids,
                                       size_t num_blocks) {
  if (handle == nullptr || block_ids == nullptr) {
    return -1;
  }
  for (size_t i = 0; i < num_blocks; ++i) {
    handle->residency->release(block_ids[i]);
  }
  return 0;
}

extern "C" int memtier_get_metrics(memtier_handle_t* handle,
                                    char* json_buffer,
                                    size_t buffer_size) {
  if (handle == nullptr || json_buffer == nullptr || buffer_size == 0) {
    return -1;
  }
  const std::string json = handle->metrics.to_json();
  if (json.size() + 1 > buffer_size) {
    return -1;
  }
  std::memcpy(json_buffer, json.c_str(), json.size() + 1);
  return 0;
}

extern "C" void memtier_destroy(memtier_handle_t* handle) { delete handle; }
