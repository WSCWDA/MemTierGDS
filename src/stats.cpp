#include "stats.h"

void Stats::reset() { for (auto& f : fields_) f.store(0); }
memtier_stats_t Stats::snapshot() const {
  memtier_stats_t s{};
  s.total_requests = fields_[0].load();
  s.dram_hits = fields_[1].load();
  s.dram_misses = fields_[2].load();
  s.posix_reads = fields_[3].load();
  s.gds_reads = fields_[4].load();
  s.gds_fallbacks = fields_[5].load();
  s.ssd_bytes_read = fields_[6].load();
  s.dram_bytes_served = fields_[7].load();
  s.cache_evictions = fields_[8].load();
  s.prefetch_requests = fields_[9].load();
  s.prefetch_bytes = fields_[10].load();
  s.readv_requests = fields_[11].load();
  s.original_ranges = fields_[12].load();
  s.coalesced_ranges = fields_[13].load();
  s.coalesced_bytes = fields_[14].load();
  s.total_latency_us = fields_[15].load();
  return s;
}
