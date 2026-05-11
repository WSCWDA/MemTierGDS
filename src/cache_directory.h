#ifndef MEMTIER_CACHE_DIRECTORY_H
#define MEMTIER_CACHE_DIRECTORY_H

#include <cstddef>

class CacheDirectory {
 public:
  CacheDirectory(size_t chunk_size, size_t capacity_bytes)
      : chunk_size_(chunk_size), capacity_bytes_(capacity_bytes), current_bytes_(0) {}
  size_t chunk_size() const { return chunk_size_; }
  size_t capacity_bytes() const { return capacity_bytes_; }
  size_t current_bytes() const { return current_bytes_; }
 private:
  size_t chunk_size_;
  size_t capacity_bytes_;
  size_t current_bytes_;
};

#endif
