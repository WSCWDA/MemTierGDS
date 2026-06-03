#pragma once

#include <cstdint>
#include <string>

namespace memtier {

struct MetricsSnapshot {
  uint64_t hbm_hits = 0;
  uint64_t dram_hits = 0;
  uint64_t dram_misses = 0;
  uint64_t posix_reads = 0;
  uint64_t gds_reads = 0;
  uint64_t eviction_count = 0;
  uint64_t bytes_read_from_ssd = 0;
  uint64_t bytes_copied_dram_to_gpu = 0;
  double total_lookup_time_ms = 0.0;
};

struct Metrics {
  uint64_t hbm_hits = 0;
  uint64_t dram_hits = 0;
  uint64_t dram_misses = 0;
  uint64_t posix_reads = 0;
  uint64_t gds_reads = 0;
  uint64_t eviction_count = 0;
  uint64_t bytes_read_from_ssd = 0;
  uint64_t bytes_copied_dram_to_gpu = 0;
  double total_lookup_time_ms = 0.0;

  MetricsSnapshot snapshot() const;
  std::string to_json() const;
};

std::string metrics_to_json(const MetricsSnapshot& snapshot);

}  // namespace memtier
