#ifndef MEMTIER_MEMTIER_H
#define MEMTIER_MEMTIER_H

#include "memtier/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct memtier_ctx memtier_ctx_t;

memtier_ctx_t* memtier_init(const memtier_config_t* config);
int memtier_finalize(memtier_ctx_t* ctx);
int memtier_stats(memtier_ctx_t* ctx, memtier_stats_t* stats);
const char* memtier_strerror(int err);

#ifdef __cplusplus
}
#endif

#endif
