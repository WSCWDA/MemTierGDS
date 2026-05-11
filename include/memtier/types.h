#ifndef MEMTIER_TYPES_H
#define MEMTIER_TYPES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum memtier_target_e {
  MEMTIER_TARGET_CPU = 0,
  MEMTIER_TARGET_GPU = 1,
} memtier_target_t;

typedef enum memtier_err_e {
  MEMTIER_OK = 0,
  MEMTIER_ERR_INVALID = -1,
  MEMTIER_ERR_IO = -2,
  MEMTIER_ERR_NOMEM = -3,
  MEMTIER_ERR_UNSUPPORTED = -4,
  MEMTIER_ERR_INTERNAL = -5,
} memtier_err_t;

typedef struct memtier_stats_s {
  uint64_t total_requests;
  uint64_t dram_hits;
  uint64_t dram_misses;
  uint64_t posix_reads;
  uint64_t ssd_bytes_read;
  uint64_t dram_bytes_served;
} memtier_stats_t;

typedef struct memtier_options_s {
  size_t chunk_size;
  size_t dram_cache_size;
  int enable_dram_cache;
  int cache_admit;
  int bypass_cache;
  int force_gds;
  int force_posix;
} memtier_options_t;

#ifdef __cplusplus
}
#endif

#endif
