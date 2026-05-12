#ifndef MEMTIER_TYPES_H
#define MEMTIER_TYPES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum memtier_target_e { MEMTIER_TARGET_CPU = 0, MEMTIER_TARGET_GPU = 1 } memtier_target_t;

typedef enum memtier_err_e {
  MEMTIER_OK = 0,
  MEMTIER_ERR_INVALID = -1,
  MEMTIER_ERR_IO = -2,
  MEMTIER_ERR_NOMEM = -3,
  MEMTIER_ERR_UNSUPPORTED = -4,
  MEMTIER_ERR_INTERNAL = -5,
  MEMTIER_ERR_GDS = -6,
  MEMTIER_ERR_CUDA = -7,
} memtier_err_t;

typedef enum memtier_path_e {
  MEMTIER_PATH_DRAM_HIT = 0,
  MEMTIER_PATH_POSIX_READ = 1,
  MEMTIER_PATH_POSIX_READ_THEN_CACHE = 2,
  MEMTIER_PATH_GDS_READ = 3,
  MEMTIER_PATH_GDS_STUB_FALLBACK = 4,
} memtier_path_t;

enum {
  MEMTIER_HINT_NONE = 0,
  MEMTIER_HINT_REUSE = 1 << 0,
  MEMTIER_HINT_SMALL_IO = 1 << 1,
  MEMTIER_HINT_RANDOM = 1 << 2,
  MEMTIER_HINT_STREAMING = 1 << 3,
  MEMTIER_HINT_CHECKPOINT = 1 << 4,
  MEMTIER_HINT_SEQUENTIAL = 1 << 5,
  MEMTIER_HINT_LATENCY_SENSITIVE = 1 << 6,
};


typedef enum memtier_prefetch_target_e {
  MEMTIER_PREFETCH_AUTO = 0,
  MEMTIER_PREFETCH_TO_DRAM = 1,
  MEMTIER_PREFETCH_TO_HBM = 2,
} memtier_prefetch_target_t;


typedef struct memtier_range_s {
  const char* path;
  uint64_t offset;
  size_t size;
} memtier_range_t;

typedef struct memtier_read_options_s {
  memtier_target_t target;
  uint32_t hints;
} memtier_read_options_t;

typedef struct memtier_prefetch_options_s {
  int device_id;
  memtier_prefetch_target_t target;
  uint32_t hints;
  int async;
  int allow_gds;
  int allow_posix;
} memtier_prefetch_options_t;

typedef struct memtier_stats_s {
  uint64_t total_requests;
  uint64_t dram_hits;
  uint64_t dram_misses;
  uint64_t posix_reads;
  uint64_t gds_reads;
  uint64_t gds_fallbacks;
  uint64_t ssd_bytes_read;
  uint64_t dram_bytes_served;
  uint64_t selected_gds;
  uint64_t selected_posix;
  uint64_t selected_cache;
  int last_selected_path;
  uint64_t prefetch_requests;
  uint64_t prefetch_bytes;
  uint64_t readv_requests;
  uint64_t original_ranges;
  uint64_t coalesced_ranges;
  uint64_t coalesced_bytes;
} memtier_stats_t;

typedef struct memtier_options_s {
  size_t chunk_size;
  size_t dram_cache_size;
  int enable_dram_cache;
  int cache_admit;
  int bypass_cache;
  int force_gds;
  int force_posix;
  int enable_gds;
  int gds_stub_success;
  size_t gds_min_size;
  size_t gds_alignment;
  uint32_t hints;
  int async_mode;
  void* stream;
  int num_workers;
  size_t coalesce_gap;
  size_t max_coalesce_size;
  int preferred_device_id;
} memtier_options_t;

#ifdef __cplusplus
}
#endif

#endif
