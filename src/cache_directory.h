#ifndef MEMTIER_CACHE_DIRECTORY_H
#define MEMTIER_CACHE_DIRECTORY_H

#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace memtier {

struct CacheKey {
  uint64_t file_id;
  uint64_t chunk_id;
  bool operator==(const CacheKey& o) const { return file_id == o.file_id && chunk_id == o.chunk_id; }
};

struct CacheEntry {
  std::vector<uint8_t> data;
  size_t valid_size;
  uint64_t last_access;
  uint64_t access_count;
  bool pinned;
};

class CacheDirectory {
 public:
  CacheDirectory(size_t chunk_size, size_t cache_size_bytes);
  bool lookup(const CacheKey& key, CacheEntry* out_entry);
  bool insert(const CacheKey& key, const uint8_t* data, size_t valid_size);
  size_t chunk_size() const { return chunk_size_; }
  uint64_t evictions() const;

 private:
  struct Node {
    CacheEntry entry;
    std::list<CacheKey>::iterator lru_it;
  };
  struct Hash { size_t operator()(const CacheKey& k) const noexcept { return std::hash<uint64_t>{}(k.file_id) ^ (k.chunk_id << 1); } };
  void touch(Node& node, const CacheKey& key);

  size_t chunk_size_;
  size_t cache_size_bytes_;
  size_t used_bytes_;
  uint64_t ticks_;
  uint64_t evictions_;
  std::list<CacheKey> lru_;
  std::unordered_map<CacheKey, Node, Hash> table_;
  mutable std::mutex mu_;
};

} // namespace memtier

#endif
