#pragma once
#include "internal/object_table.h"
#include <deque>
#include <unordered_map>
struct FileRangeHash { size_t operator()(const FileRange& r) const { return (size_t)(r.file_id ^ (r.offset<<1) ^ r.length); } };
class DramCache { public: DramCache(); ~DramCache(); bool contains(FileRange range) const; void* get(FileRange range); int put(FileRange range, void* pinned_ptr, size_t length); void evictIfNeeded(); private: size_t max_bytes_=0, used_bytes_=0; std::unordered_map<FileRange,std::pair<void*,size_t>,FileRangeHash> map_; std::deque<FileRange> fifo_; };
