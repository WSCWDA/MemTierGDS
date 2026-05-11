#ifndef MEMTIER_MEMTIER_H
#define MEMTIER_MEMTIER_H

#include <stddef.h>
#include <stdint.h>

#include "memtier/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct memtier_ctx_s memtier_ctx_t;
typedef struct memtier_req_s memtier_req_t;

int memtier_init(const memtier_options_t* options, memtier_ctx_t** out_ctx);
int memtier_finalize(memtier_ctx_t* ctx);
int memtier_read(memtier_ctx_t* ctx, const char* path, uint64_t offset, size_t size, void* dst,
                 memtier_target_t target);
int memtier_read_async(memtier_ctx_t* ctx, const char* path, uint64_t offset, size_t size, void* dst,
                       memtier_target_t target, memtier_req_t** out_req);
int memtier_wait(memtier_req_t* req);
int memtier_test(memtier_req_t* req, int* done);
int memtier_request_status(memtier_req_t* req, int* out_status);
void memtier_request_free(memtier_req_t* req);
int memtier_prefetch(memtier_ctx_t* ctx, const char* path, uint64_t offset, size_t size,
                    const memtier_prefetch_options_t* options);
int memtier_stats(memtier_ctx_t* ctx, memtier_stats_t* out_stats);
const char* memtier_strerror(int err);

#ifdef __cplusplus
}
#endif

#endif
