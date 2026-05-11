#include "memtier_internal.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <thread>
#include <vector>

struct memtier_req_s {
  int rc;
  std::thread worker;
};

static uint64_t file_id_from_path(const char* path) {
  std::error_code ec;
  auto canon = std::filesystem::weakly_canonical(path, ec).string();
  return std::hash<std::string>{}(canon.empty() ? std::string(path) : canon);
}

int memtier_init(const memtier_options_t* options, memtier_ctx_t** out_ctx) {
  if (!out_ctx) return MEMTIER_ERR_INVALID;
  memtier_options_t opt{};
  opt.chunk_size = 1u << 20;
  opt.dram_cache_size = 64u << 20;
  opt.enable_dram_cache = 1;
  opt.cache_admit = 1;
  if (options) opt = *options;
  if (opt.chunk_size == 0) opt.chunk_size = 1u << 20;
  if (opt.dram_cache_size == 0) opt.dram_cache_size = 64u << 20;
  *out_ctx = new (std::nothrow) memtier_ctx_s(opt);
  return *out_ctx ? MEMTIER_OK : MEMTIER_ERR_NOMEM;
}

int memtier_finalize(memtier_ctx_t* ctx) { delete ctx; return MEMTIER_OK; }

int memtier_read(memtier_ctx_t* ctx, const char* path, uint64_t offset, size_t size, void* dst, memtier_target_t target) {
  if (!ctx || !path || !dst || size == 0) return MEMTIER_ERR_INVALID;
  uint8_t* out = static_cast<uint8_t*>(dst);
  if (target == MEMTIER_TARGET_GPU) {
    // CPU-only mock mode: dst treated as CPU pointer.
  }
  if (ctx->options.bypass_cache || !ctx->options.enable_dram_cache) {
    size_t n = 0;
    int rc = memtier::posix_pread_full(path, offset, size, out, &n);
    if (rc != MEMTIER_OK) return rc;
    std::lock_guard<std::mutex> lg(ctx->mu);
    ctx->stats.total_requests++;
    ctx->stats.posix_reads++;
    ctx->stats.ssd_bytes_read += n;
    return MEMTIER_OK;
  }

  const uint64_t fid = file_id_from_path(path);
  size_t copied = 0;
  const size_t chunk_size = ctx->options.chunk_size;
  while (copied < size) {
    const uint64_t cur_off = offset + copied;
    const uint64_t chunk_id = cur_off / chunk_size;
    const size_t in_chunk = static_cast<size_t>(cur_off % chunk_size);
    const size_t need = std::min(chunk_size - in_chunk, size - copied);
    memtier::CacheKey key{fid, chunk_id};
    memtier::CacheEntry e;
    if (ctx->cache.lookup(key, &e) && in_chunk + need <= e.valid_size) {
      std::memcpy(out + copied, e.data.data() + in_chunk, need);
      std::lock_guard<std::mutex> lg(ctx->mu);
      ctx->stats.dram_hits++;
      ctx->stats.dram_bytes_served += need;
    } else {
      std::vector<uint8_t> chunk_buf(chunk_size, 0);
      size_t read_bytes = 0;
      int rc = memtier::posix_pread_full(path, chunk_id * chunk_size, chunk_size, chunk_buf.data(), &read_bytes);
      if (rc != MEMTIER_OK) return rc;
      std::memcpy(out + copied, chunk_buf.data() + in_chunk, need);
      if (ctx->options.cache_admit) ctx->cache.insert(key, chunk_buf.data(), read_bytes);
      std::lock_guard<std::mutex> lg(ctx->mu);
      ctx->stats.dram_misses++;
      ctx->stats.posix_reads++;
      ctx->stats.ssd_bytes_read += read_bytes;
    }
    copied += need;
  }
  std::lock_guard<std::mutex> lg(ctx->mu);
  ctx->stats.total_requests++;
  return MEMTIER_OK;
}

int memtier_read_async(memtier_ctx_t* ctx, const char* path, uint64_t offset, size_t size, void* dst, memtier_target_t target, memtier_req_t** out_req) {
  if (!out_req) return MEMTIER_ERR_INVALID;
  auto* req = new (std::nothrow) memtier_req_s();
  if (!req) return MEMTIER_ERR_NOMEM;
  req->worker = std::thread([=]() { req->rc = memtier_read(ctx, path, offset, size, dst, target); });
  *out_req = req;
  return MEMTIER_OK;
}

int memtier_wait(memtier_ctx_t*, memtier_req_t* req) {
  if (!req) return MEMTIER_ERR_INVALID;
  if (req->worker.joinable()) req->worker.join();
  int rc = req->rc;
  delete req;
  return rc;
}

int memtier_prefetch(memtier_ctx_t* ctx, const char* path, uint64_t offset, size_t size) {
  std::vector<uint8_t> tmp(size);
  return memtier_read(ctx, path, offset, size, tmp.data(), MEMTIER_TARGET_CPU);
}

int memtier_stats(memtier_ctx_t* ctx, memtier_stats_t* out_stats) {
  if (!ctx || !out_stats) return MEMTIER_ERR_INVALID;
  std::lock_guard<std::mutex> lg(ctx->mu);
  *out_stats = ctx->stats;
  return MEMTIER_OK;
}

const char* memtier_strerror(int err) {
  switch (err) {
    case MEMTIER_OK: return "ok";
    case MEMTIER_ERR_INVALID: return "invalid argument";
    case MEMTIER_ERR_IO: return "io error";
    case MEMTIER_ERR_NOMEM: return "out of memory";
    case MEMTIER_ERR_UNSUPPORTED: return "unsupported";
    default: return "internal error";
  }
}
