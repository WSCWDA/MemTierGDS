#include "cache_directory.h"

namespace memtier {

CacheDirectory::CacheDirectory(size_t chunk_size, size_t cache_size_bytes)
    : chunk_size_(chunk_size), cache_size_bytes_(cache_size_bytes), used_bytes_(0), ticks_(0), evictions_(0) {}

void CacheDirectory::touch(Node& node, const CacheKey& key) {
  lru_.erase(node.lru_it);
  lru_.push_front(key);
  node.lru_it = lru_.begin();
  node.entry.last_access = ++ticks_;
  node.entry.access_count++;
}

bool CacheDirectory::lookup(const CacheKey& key, CacheEntry* out_entry) {
  std::lock_guard<std::mutex> g(mu_);
  auto it = table_.find(key);
  if (it == table_.end()) return false;
  touch(it->second, key);
  if (out_entry) *out_entry = it->second.entry;
  return true;
}

bool CacheDirectory::insert(const CacheKey& key, const uint8_t* data, size_t valid_size) {
  if (!data || valid_size == 0 || valid_size > chunk_size_) return false;
  std::lock_guard<std::mutex> g(mu_);
  if (chunk_size_ > cache_size_bytes_ || cache_size_bytes_ == 0) return false;
  while (used_bytes_ + chunk_size_ > cache_size_bytes_ && !lru_.empty()) {
    CacheKey victim = lru_.back();
    lru_.pop_back();
    auto it = table_.find(victim);
    if (it != table_.end()) {
      used_bytes_ -= it->second.entry.data.size();
      table_.erase(it);
      evictions_++;
    }
  }
  Node n;
  n.entry.data.assign(data, data + chunk_size_);
  n.entry.valid_size = valid_size;
  n.entry.last_access = ++ticks_;
  n.entry.access_count = 1;
  n.entry.pinned = false;
  lru_.push_front(key);
  n.lru_it = lru_.begin();
  used_bytes_ += n.entry.data.size();
  table_[key] = std::move(n);
  return true;
}


bool CacheDirectory::contains(const CacheKey& key) {
  std::lock_guard<std::mutex> g(mu_);
  return table_.find(key) != table_.end();
}

uint64_t CacheDirectory::evictions() const {
  std::lock_guard<std::mutex> g(mu_);
  return evictions_;
}

} // namespace memtier
