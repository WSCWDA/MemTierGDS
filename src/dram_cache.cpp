#include "memtier_internal.h"

namespace memtier {

DramCache::DramCache(size_t chunk_size, size_t capacity_bytes)
    : chunk_size_(chunk_size), capacity_bytes_(capacity_bytes), used_bytes_(0), evictions_(0) {}

bool DramCache::lookup(const CacheKey& key, std::vector<uint8_t>* out) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = map_.find(key);
  if (it == map_.end()) {
    return false;
  }
  lru_.erase(it->second.lru_it);
  lru_.push_front(key);
  it->second.lru_it = lru_.begin();
  *out = it->second.data;
  return true;
}

bool DramCache::insert(const CacheKey& key, const uint8_t* data, size_t size) {
  if (size > chunk_size_) return false;
  std::lock_guard<std::mutex> lock(mu_);
  if (capacity_bytes_ == 0) return false;
  while (used_bytes_ + size > capacity_bytes_ && !lru_.empty()) {
    auto victim = lru_.back();
    auto it = map_.find(victim);
    if (it != map_.end()) {
      used_bytes_ -= it->second.data.size();
      map_.erase(it);
      evictions_++;
    }
    lru_.pop_back();
  }
  Entry e;
  e.data.assign(data, data + size);
  lru_.push_front(key);
  e.lru_it = lru_.begin();
  used_bytes_ += e.data.size();
  map_[key] = std::move(e);
  return true;
}

uint64_t DramCache::evictions() const {
  std::lock_guard<std::mutex> lock(mu_);
  return evictions_;
}

}  // namespace memtier
