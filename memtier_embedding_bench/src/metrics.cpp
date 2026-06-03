#include "metrics.h"

#include <sstream>

namespace memtier {

MetricsSnapshot Metrics::snapshot() const {
  return MetricsSnapshot{hbm_hits,
                         dram_hits,
                         dram_misses,
                         posix_reads,
                         gds_reads,
                         eviction_count,
                         bytes_read_from_ssd,
                         bytes_copied_dram_to_gpu,
                         total_lookup_time_ms};
}

std::string metrics_to_json(const MetricsSnapshot& s) {
  std::ostringstream out;
  out << "{";
  out << "\"hbm_hits\":" << s.hbm_hits << ",";
  out << "\"dram_hits\":" << s.dram_hits << ",";
  out << "\"dram_misses\":" << s.dram_misses << ",";
  out << "\"posix_reads\":" << s.posix_reads << ",";
  out << "\"gds_reads\":" << s.gds_reads << ",";
  out << "\"eviction_count\":" << s.eviction_count << ",";
  out << "\"bytes_read_from_ssd\":" << s.bytes_read_from_ssd << ",";
  out << "\"bytes_copied_dram_to_gpu\":" << s.bytes_copied_dram_to_gpu << ",";
  out << "\"total_lookup_time_ms\":" << s.total_lookup_time_ms;
  out << "}";
  return out.str();
}

std::string Metrics::to_json() const { return metrics_to_json(snapshot()); }

}  // namespace memtier
