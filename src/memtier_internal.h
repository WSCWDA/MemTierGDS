#ifndef MEMTIER_INTERNAL_H
#define MEMTIER_INTERNAL_H

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>
#include <string>

#include "cache_directory.h"
#include "memtier/memtier.h"

struct memtier_req_s {
  memtier_ctx_t* ctx;
  std::string path;
  uint64_t offset;
  size_t size;
  void* dst;
  memtier_target_t target;
  int result;
  bool done;
  std::mutex mu;
  std::condition_variable cv;
};

struct memtier_ctx_s {
  memtier_options_t options;
  memtier::CacheDirectory cache;
  memtier_stats_t stats;
  std::mutex mu;

  std::mutex q_mu;
  std::condition_variable q_cv;
  std::deque<memtier_req_t*> queue;
  std::vector<std::thread> workers;
  bool stop;

  explicit memtier_ctx_s(const memtier_options_t& opt) : options(opt), cache(opt.chunk_size, opt.dram_cache_size), stats{}, stop(false) {}
};

namespace memtier {
int posix_pread_full(const char* path, uint64_t offset, size_t size, uint8_t* dst, size_t* bytes_read);
}

#endif
