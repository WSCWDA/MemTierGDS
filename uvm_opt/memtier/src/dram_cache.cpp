#include "internal/dram_cache.h"
#include "internal/cuda_utils.h"
#include <cstdlib>
DramCache::DramCache(){ const char* e=getenv("MEMTIER_DRAM_CACHE_MB"); max_bytes_=(e?strtoull(e,nullptr,10):512ull)*1024ull*1024ull; }
DramCache::~DramCache(){ for(auto& kv:map_) memtier::cuda_utils::free_pinned(kv.second.first); }
bool DramCache::contains(FileRange r) const { return map_.find(r)!=map_.end(); }
void* DramCache::get(FileRange r){ auto it=map_.find(r); return it==map_.end()?nullptr:it->second.first; }
int DramCache::put(FileRange r, void* p, size_t len){ if(contains(r)) return 0; map_[r]={p,len}; fifo_.push_back(r); used_bytes_+=len; evictIfNeeded(); return 0; }
void DramCache::evictIfNeeded(){ while(used_bytes_>max_bytes_ && !fifo_.empty()){ auto r=fifo_.front(); fifo_.pop_front(); auto it=map_.find(r); if(it!=map_.end()){ used_bytes_-=it->second.second; memtier::cuda_utils::free_pinned(it->second.first); map_.erase(it);} } /* TODO: range coalescing, multi-GPU sharing, reuse-aware eviction. */ }
