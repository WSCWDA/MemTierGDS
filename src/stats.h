#ifndef MEMTIER_STATS_H
#define MEMTIER_STATS_H

#include <atomic>
#include "memtier/types.h"

class Stats {
 public:
  void reset();
  memtier_stats_t snapshot() const;
 private:
  std::atomic<uint64_t> fields_[16]{};
};

#endif
