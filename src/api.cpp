#include "context.h"
#include "log.h"

#include <new>

memtier_ctx_t* memtier_init(const memtier_config_t* config) {
  try {
    auto ctx = new (std::nothrow) memtier_ctx_t();
    if (!ctx) return nullptr;
    ctx->impl = std::make_unique<MemTierContext>();
    if (!ctx->impl) { delete ctx; return nullptr; }

    memtier_config_t c = config ? *config : memtier_default_config();
    if (c.chunk_size == 0) c.chunk_size = 1ull << 20;
    if (c.num_workers < 0) c.num_workers = 0;
    if (!c.policy) c.policy = "adaptive";
    if (c.enable_hbm_cache) {
      memtier_log(LogLevel::WARN, "HBM cache requested but CUDA backend is stub; disable HBM cache.");
      c.enable_hbm_cache = false;
    }

    ctx->impl->config = c;
    ctx->impl->policy = c.policy;
    if (c.enable_posix) ctx->impl->posix_backend = std::make_unique<PosixBackend>();
    if (c.enable_dram_cache) ctx->impl->dram_cache = std::make_unique<CacheDirectory>(c.chunk_size, c.dram_cache_size);
    ctx->impl->gds_backend = std::make_unique<GdsBackend>(c.enable_gds);
    ctx->impl->cuda_backend = std::make_unique<CudaBackend>(c.enable_hbm_cache, c.device_id);
    ctx->impl->stats.reset();
    ctx->impl->initialized = true;
    return ctx;
  } catch (...) {
    return nullptr;
  }
}

int memtier_finalize(memtier_ctx_t* ctx) {
  if (!ctx) return MEMTIER_ERR_INVALID_ARG;
  try {
    ctx->impl.reset();
    delete ctx;
    return MEMTIER_OK;
  } catch (...) {
    return MEMTIER_ERR_INTERNAL;
  }
}

int memtier_stats(memtier_ctx_t* ctx, memtier_stats_t* stats) {
  if (!ctx || !stats) return MEMTIER_ERR_INVALID_ARG;
  *stats = ctx->impl->stats.snapshot();
  return MEMTIER_OK;
}

const char* memtier_strerror(int err) {
  switch (err) {
    case MEMTIER_OK: return "ok";
    case MEMTIER_ERR_INVALID_ARG: return "invalid argument";
    case MEMTIER_ERR_IO: return "io error";
    case MEMTIER_ERR_GDS: return "gds error";
    case MEMTIER_ERR_CUDA: return "cuda error";
    case MEMTIER_ERR_NOMEM: return "out of memory";
    case MEMTIER_ERR_UNSUPPORTED: return "unsupported";
    case MEMTIER_ERR_INTERNAL: return "internal error";
    default: return "unknown error";
  }
}
