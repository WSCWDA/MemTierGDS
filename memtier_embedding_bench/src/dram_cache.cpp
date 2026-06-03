#include "embedding_config.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <list>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace memtier {
namespace {

void cuda_check(cudaError_t err, const char* expr) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error in ") + expr + ": " +
                             cudaGetErrorString(err));
  }
}

}  // namespace

struct DramCache::Impl {
  uint64_t cache_bytes = 0;
  uint64_t block_bytes = 0;
  uint64_t capacity = 0;
  void* base = nullptr;
  std::vector<uint64_t> free_slots;
  std::list<uint64_t> lru;

  struct Entry {
    void* ptr = nullptr;
    uint64_t slot = 0;
    std::list<uint64_t>::iterator lru_it;
  };

  std::unordered_map<uint64_t, Entry> entries;
  DramCacheStats stats;
  mutable std::mutex mutex;
};

DramCache::DramCache(uint64_t cache_bytes, uint64_t block_bytes)
    : impl_(std::make_unique<Impl>()) {
  if (block_bytes == 0) {
    throw std::invalid_argument("block_bytes must be non-zero");
  }
  impl_->block_bytes = block_bytes;
  impl_->capacity = std::max<uint64_t>(1, cache_bytes / block_bytes);
  impl_->cache_bytes = impl_->capacity * block_bytes;
  cuda_check(cudaHostAlloc(&impl_->base, impl_->cache_bytes, cudaHostAllocPortable), "cudaHostAlloc");
  impl_->free_slots.reserve(impl_->capacity);
  for (uint64_t i = 0; i < impl_->capacity; ++i) {
    impl_->free_slots.push_back(impl_->capacity - 1 - i);
  }
}

DramCache::~DramCache() {
  if (impl_ && impl_->base != nullptr) {
    cudaFreeHost(impl_->base);
    impl_->base = nullptr;
  }
}

void* DramCache::get(uint64_t block_id) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto it = impl_->entries.find(block_id);
  if (it == impl_->entries.end()) {
    impl_->stats.dram_miss++;
    return nullptr;
  }
  impl_->stats.dram_hit++;
  impl_->lru.erase(it->second.lru_it);
  impl_->lru.push_front(block_id);
  it->second.lru_it = impl_->lru.begin();
  return it->second.ptr;
}

void DramCache::evict_if_needed() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->free_slots.empty() || impl_->entries.empty()) {
    return;
  }
  const uint64_t victim = impl_->lru.back();
  auto it = impl_->entries.find(victim);
  if (it != impl_->entries.end()) {
    impl_->free_slots.push_back(it->second.slot);
    impl_->entries.erase(it);
  }
  impl_->lru.pop_back();
  impl_->stats.eviction_count++;
}

void* DramCache::allocate_slot_for_load(uint64_t block_id) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto existing = impl_->entries.find(block_id);
  if (existing != impl_->entries.end()) {
    impl_->lru.erase(existing->second.lru_it);
    impl_->lru.push_front(block_id);
    existing->second.lru_it = impl_->lru.begin();
    return existing->second.ptr;
  }

  if (impl_->free_slots.empty()) {
    const uint64_t victim = impl_->lru.back();
    auto victim_it = impl_->entries.find(victim);
    if (victim_it != impl_->entries.end()) {
      impl_->free_slots.push_back(victim_it->second.slot);
      impl_->entries.erase(victim_it);
    }
    impl_->lru.pop_back();
    impl_->stats.eviction_count++;
  }

  const uint64_t slot = impl_->free_slots.back();
  impl_->free_slots.pop_back();
  auto* ptr = static_cast<char*>(impl_->base) + slot * impl_->block_bytes;
  impl_->lru.push_front(block_id);
  impl_->entries[block_id] = Impl::Entry{ptr, slot, impl_->lru.begin()};
  return ptr;
}

void* DramCache::put(uint64_t block_id, const void* data, size_t bytes) {
  if (bytes > impl_->block_bytes) {
    throw std::invalid_argument("put size exceeds DRAM cache block size");
  }
  void* dst = allocate_slot_for_load(block_id);
  std::memcpy(dst, data, bytes);
  return dst;
}

DramCacheStats DramCache::stats() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->stats;
}

uint64_t DramCache::capacity_blocks() const { return impl_->capacity; }

}  // namespace memtier
