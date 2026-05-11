#ifndef MEMTIER_TYPES_H
#define MEMTIER_TYPES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  MEMTIER_OK = 0,
  MEMTIER_ERR_INVALID_ARG = -1,
  MEMTIER_ERR_IO = -2,
  MEMTIER_ERR_GDS = -3,
  MEMTIER_ERR_CUDA = -4,
  MEMTIER_ERR_NOMEM = -5,
  MEMTIER_ERR_UNSUPPORTED = -6,
  MEMTIER_ERR_INTERNAL = -7
} memtier_error_t;

typedef enum { MEMTIER_TARGET_CPU = 0, MEMTIER_TARGET_GPU = 1 } memtier_target_t;

typedef enum {
  MEMTIER_HINT_NONE = 0,
  MEMTIER_HINT_SEQUENTIAL = 1 << 0,
  MEMTIER_HINT_RANDOM = 1 << 1,
  MEMTIER_HINT_REUSE = 1 << 2,
  MEMTIER_HINT_STREAMING = 1 << 3,
  MEMTIER_HINT_CHECKPOINT = 1 << 4,
  MEMTIER_HINT_SMALL_IO = 1 << 5,
  MEMTIER_HINT_LATENCY_SENSITIVE = 1 << 6
} memtier_hint_t;

typedef struct {
  int device_id;
  size_t dram_cache_size;
  size_t hbm_cache_size;
  size_t chunk_size;
  bool enable_posix;
  bool enable_gds;
  bool enable_dram_cache;
  bool enable_hbm_cache;
  size_t gds_min_size;
  size_t gds_alignment;
  int num_workers;
  const char* policy;
} memtier_config_t;

typedef struct {
  uint64_t total_requests;
  uint64_t dram_hits;
  uint64_t dram_misses;
  uint64_t posix_reads;
  uint64_t gds_reads;
  uint64_t gds_fallbacks;
  uint64_t ssd_bytes_read;
  uint64_t dram_bytes_served;
  uint64_t cache_evictions;
  uint64_t prefetch_requests;
  uint64_t prefetch_bytes;
  uint64_t readv_requests;
  uint64_t original_ranges;
  uint64_t coalesced_ranges;
  uint64_t coalesced_bytes;
  uint64_t total_latency_us;
} memtier_stats_t;

#ifdef __cplusplus
}
#endif

#endif
