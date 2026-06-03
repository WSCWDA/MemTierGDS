#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MEMTIER_PATH_HBM_HIT = 0,
    MEMTIER_PATH_DRAM_HIT = 1,
    MEMTIER_PATH_POSIX_READ = 2,
    MEMTIER_PATH_GDS_READ = 3,
    MEMTIER_PATH_UVM_FAULT = 4
} memtier_path_t;

typedef struct {
    uint64_t num_embeddings;
    uint32_t embedding_dim;
    uint32_t rows_per_block;
    uint32_t dtype_size;
    uint64_t dram_cache_bytes;
    uint64_t hbm_cache_bytes;
    int enable_gds;
    int enable_prefetch;
} memtier_embedding_config_t;

typedef struct memtier_handle memtier_handle_t;

memtier_handle_t* memtier_init_embedding(
    const char* embedding_file,
    const memtier_embedding_config_t* config
);

int memtier_lookup_blocks(
    memtier_handle_t* handle,
    const uint64_t* block_ids,
    size_t num_blocks,
    void** device_block_ptrs,
    memtier_path_t* selected_paths
);

int memtier_prefetch_blocks(
    memtier_handle_t* handle,
    const uint64_t* block_ids,
    size_t num_blocks
);

int memtier_release_blocks(
    memtier_handle_t* handle,
    const uint64_t* block_ids,
    size_t num_blocks
);

int memtier_get_metrics(
    memtier_handle_t* handle,
    char* json_buffer,
    size_t buffer_size
);

void memtier_destroy(memtier_handle_t* handle);

#ifdef __cplusplus
}
#endif
