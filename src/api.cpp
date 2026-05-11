#include "memtier_internal.h"

#include "path_selector.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <thread>
#include <vector>

struct memtier_req_s { int rc; std::thread worker; };

static uint64_t file_id_from_path(const char* path) {
  std::error_code ec;
  auto canon = std::filesystem::weakly_canonical(path, ec).string();
  return std::hash<std::string>{}(canon.empty() ? std::string(path) : canon);
}

static void on_selected(memtier_ctx_t* ctx, memtier_path_t p) {
  ctx->stats.last_selected_path = static_cast<int>(p);
  if (p == MEMTIER_PATH_GDS_READ || p == MEMTIER_PATH_GDS_STUB_FALLBACK) ctx->stats.selected_gds++;
  else if (p == MEMTIER_PATH_DRAM_HIT || p == MEMTIER_PATH_POSIX_READ_THEN_CACHE) ctx->stats.selected_cache++;
  else ctx->stats.selected_posix++;
}

int memtier_init(const memtier_options_t* options, memtier_ctx_t** out_ctx) {
  if (!out_ctx) return MEMTIER_ERR_INVALID;
  memtier_options_t opt{};
  opt.chunk_size = 1u << 20; opt.dram_cache_size = 64u << 20; opt.enable_dram_cache = 1; opt.cache_admit = 1;
  opt.gds_min_size = 1u << 20; opt.gds_alignment = 4096; opt.gds_stub_success = 1;
  if (options) opt = *options;
  if (opt.chunk_size == 0) opt.chunk_size = 1u << 20;
  if (opt.dram_cache_size == 0) opt.dram_cache_size = 64u << 20;
  if (opt.gds_min_size == 0) opt.gds_min_size = 1u << 20;
  if (opt.gds_alignment == 0) opt.gds_alignment = 4096;
  *out_ctx = new (std::nothrow) memtier_ctx_s(opt);
  return *out_ctx ? MEMTIER_OK : MEMTIER_ERR_NOMEM;
}
int memtier_finalize(memtier_ctx_t* ctx) { delete ctx; return MEMTIER_OK; }

int memtier_read(memtier_ctx_t* ctx, const char* path, uint64_t offset, size_t size, void* dst, memtier_target_t target) {
  if (!ctx || !path || !dst || size == 0) return MEMTIER_ERR_INVALID;
  (void)target;
  uint8_t* out = static_cast<uint8_t*>(dst);
  uint64_t fid = file_id_from_path(path); size_t copied = 0, chunk_size = ctx->options.chunk_size;
  while (copied < size) {
    uint64_t cur_off = offset + copied, chunk_id = cur_off / chunk_size;
    size_t in_chunk = static_cast<size_t>(cur_off % chunk_size), need = std::min(chunk_size - in_chunk, size - copied);
    memtier::CacheKey key{fid, chunk_id}; memtier::CacheEntry ce;
    bool hit = ctx->cache.lookup(key, &ce) && (in_chunk + need <= ce.valid_size);
    memtier_path_t p = memtier::select_path(hit, cur_off, need, ctx->options);

    {
      std::lock_guard<std::mutex> lg(ctx->mu);
      on_selected(ctx, p);
    }

    if (p == MEMTIER_PATH_DRAM_HIT) {
      std::memcpy(out + copied, ce.data.data() + in_chunk, need);
      std::lock_guard<std::mutex> lg(ctx->mu); ctx->stats.dram_hits++; ctx->stats.dram_bytes_served += need;
    } else if (p == MEMTIER_PATH_GDS_READ || p == MEMTIER_PATH_GDS_STUB_FALLBACK) {
      size_t n = 0; int rc = memtier::gds_stub_read(ctx->options, path, cur_off, need, out + copied, &n);
      if (rc == MEMTIER_OK) { std::lock_guard<std::mutex> lg(ctx->mu); ctx->stats.gds_reads++; }
      else {
        std::vector<uint8_t> tmp(need); size_t pn = 0; int prc = memtier::posix_pread_full(path, cur_off, need, tmp.data(), &pn); if (prc != MEMTIER_OK) return prc;
        std::memcpy(out + copied, tmp.data(), need);
        std::lock_guard<std::mutex> lg(ctx->mu); ctx->stats.gds_fallbacks++; ctx->stats.posix_reads++; ctx->stats.ssd_bytes_read += pn;
      }
    } else {
      std::vector<uint8_t> chunk_buf((p == MEMTIER_PATH_POSIX_READ_THEN_CACHE) ? chunk_size : need);
      size_t n = 0; uint64_t read_off = (p == MEMTIER_PATH_POSIX_READ_THEN_CACHE) ? chunk_id * chunk_size : cur_off;
      size_t read_sz = (p == MEMTIER_PATH_POSIX_READ_THEN_CACHE) ? chunk_size : need;
      int rc = memtier::posix_pread_full(path, read_off, read_sz, chunk_buf.data(), &n); if (rc != MEMTIER_OK) return rc;
      size_t src_off = (p == MEMTIER_PATH_POSIX_READ_THEN_CACHE) ? in_chunk : 0;
      std::memcpy(out + copied, chunk_buf.data() + src_off, need);
      if (p == MEMTIER_PATH_POSIX_READ_THEN_CACHE && ctx->options.cache_admit) ctx->cache.insert(key, chunk_buf.data(), n);
      std::lock_guard<std::mutex> lg(ctx->mu); ctx->stats.dram_misses++; ctx->stats.posix_reads++; ctx->stats.ssd_bytes_read += n;
    }
    copied += need;
  }
  std::lock_guard<std::mutex> lg(ctx->mu); ctx->stats.total_requests++;
  return MEMTIER_OK;
}

int memtier_read_async(memtier_ctx_t* c,const char* p,uint64_t o,size_t s,void* d,memtier_target_t t,memtier_req_t** r){if(!r)return MEMTIER_ERR_INVALID;auto*req=new(std::nothrow)memtier_req_s();if(!req)return MEMTIER_ERR_NOMEM;req->worker=std::thread([=](){req->rc=memtier_read(c,p,o,s,d,t);});*r=req;return MEMTIER_OK;}
int memtier_wait(memtier_ctx_t*, memtier_req_t* req){if(!req)return MEMTIER_ERR_INVALID; if(req->worker.joinable())req->worker.join();int rc=req->rc;delete req;return rc;}
int memtier_prefetch(memtier_ctx_t* c,const char* p,uint64_t o,size_t s){std::vector<uint8_t> t(s);return memtier_read(c,p,o,s,t.data(),MEMTIER_TARGET_CPU);} 
int memtier_stats(memtier_ctx_t* c, memtier_stats_t* o){if(!c||!o)return MEMTIER_ERR_INVALID;std::lock_guard<std::mutex> lg(c->mu);*o=c->stats;return MEMTIER_OK;}
const char* memtier_strerror(int e){switch(e){case MEMTIER_OK:return"ok";case MEMTIER_ERR_INVALID:return"invalid";case MEMTIER_ERR_IO:return"io";case MEMTIER_ERR_NOMEM:return"nomem";case MEMTIER_ERR_UNSUPPORTED:return"unsupported";case MEMTIER_ERR_GDS:return"gds error";default:return"internal";}}
