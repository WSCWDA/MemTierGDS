#include "memtier_internal.h"

#include "cuda_backend.h"
#include "path_selector.h"
#include "gds_backend.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <vector>

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

static int do_read_impl(memtier_ctx_t* ctx, const char* path, uint64_t offset, size_t size, void* dst, memtier_target_t target) {
  if (!ctx || !path || !dst || size == 0) return MEMTIER_ERR_INVALID;
  bool gpu_target = target == MEMTIER_TARGET_GPU;
  if (gpu_target && !memtier_cuda_available()) return MEMTIER_ERR_UNSUPPORTED;
  uint8_t* out = static_cast<uint8_t*>(dst);
  uint64_t fid = file_id_from_path(path);
  size_t copied = 0, chunk_size = ctx->options.chunk_size;
  while (copied < size) {
    uint64_t cur_off = offset + copied, chunk_id = cur_off / chunk_size;
    size_t in_chunk = static_cast<size_t>(cur_off % chunk_size), need = std::min(chunk_size - in_chunk, size - copied);
    memtier::CacheKey key{fid, chunk_id};
    memtier::CacheEntry ce;
    bool hit = ctx->cache.lookup(key, &ce) && (in_chunk + need <= ce.valid_size);
    memtier_path_t p = memtier::select_path(hit, cur_off, need, ctx->options);
    { std::lock_guard<std::mutex> lg(ctx->mu); on_selected(ctx, p); }

    auto copy_target = [&](const uint8_t* src) -> int {
      if (!gpu_target) { std::memcpy(out + copied, src, need); return MEMTIER_OK; }
      int rc = memtier_cuda_copy_to_gpu(out + copied, src, need, ctx->options.stream);
      if (rc != MEMTIER_OK) return MEMTIER_ERR_CUDA;
      if (!ctx->options.async_mode && memtier_cuda_synchronize(ctx->options.stream) != MEMTIER_OK) return MEMTIER_ERR_CUDA;
      return MEMTIER_OK;
    };

    if (p == MEMTIER_PATH_DRAM_HIT) {
      int rc = copy_target(ce.data.data() + in_chunk); if (rc != MEMTIER_OK) return rc;
      std::lock_guard<std::mutex> lg(ctx->mu); ctx->stats.dram_hits++; ctx->stats.dram_bytes_served += need;
    } else if (p == MEMTIER_PATH_GDS_READ || p == MEMTIER_PATH_GDS_STUB_FALLBACK) {
      std::vector<uint8_t> buf(need); size_t n = 0; int rc = memtier_gds_read(ctx->options, path, cur_off, need, buf.data(), &n);
      if (rc == MEMTIER_OK) { rc = copy_target(buf.data()); if (rc != MEMTIER_OK) return rc; std::lock_guard<std::mutex> lg(ctx->mu); ctx->stats.gds_reads++; }
      else { size_t pn=0; rc = memtier::posix_pread_full(path, cur_off, need, buf.data(), &pn); if (rc != MEMTIER_OK) return rc; rc = copy_target(buf.data()); if (rc != MEMTIER_OK) return rc; std::lock_guard<std::mutex> lg(ctx->mu); ctx->stats.gds_fallbacks++; ctx->stats.posix_reads++; ctx->stats.ssd_bytes_read += pn; }
    } else {
      std::vector<uint8_t> buf((p == MEMTIER_PATH_POSIX_READ_THEN_CACHE) ? chunk_size : need);
      uint64_t roff = (p == MEMTIER_PATH_POSIX_READ_THEN_CACHE) ? chunk_id * chunk_size : cur_off;
      size_t rsz = (p == MEMTIER_PATH_POSIX_READ_THEN_CACHE) ? chunk_size : need, n = 0;
      int rc = memtier::posix_pread_full(path, roff, rsz, buf.data(), &n); if (rc != MEMTIER_OK) return rc;
      size_t src_off = (p == MEMTIER_PATH_POSIX_READ_THEN_CACHE) ? in_chunk : 0;
      rc = copy_target(buf.data() + src_off); if (rc != MEMTIER_OK) return rc;
      if (p == MEMTIER_PATH_POSIX_READ_THEN_CACHE && ctx->options.cache_admit) ctx->cache.insert(key, buf.data(), n);
      std::lock_guard<std::mutex> lg(ctx->mu); ctx->stats.dram_misses++; ctx->stats.posix_reads++; ctx->stats.ssd_bytes_read += n;
    }
    copied += need;
  }
  std::lock_guard<std::mutex> lg(ctx->mu); ctx->stats.total_requests++; return MEMTIER_OK;
}


int memtier_readv(memtier_ctx_t* ctx, const memtier_range_t* ranges, void** dsts, size_t nranges, const memtier_read_options_t* options){
  if(!ctx||!ranges||!dsts||nranges==0) return MEMTIER_ERR_INVALID;
  memtier_target_t target = options ? options->target : MEMTIER_TARGET_CPU;
  struct Item{std::string path;uint64_t off;size_t sz;void* dst;};
  std::vector<Item> items; items.reserve(nranges);
  for(size_t i=0;i<nranges;++i){ if(!ranges[i].path||!dsts[i]||ranges[i].size==0) return MEMTIER_ERR_INVALID; items.push_back({ranges[i].path,ranges[i].offset,ranges[i].size,dsts[i]}); }
  std::sort(items.begin(), items.end(), [](const Item&a,const Item&b){ if(a.path!=b.path) return a.path<b.path; return a.off<b.off;});
  size_t i=0, coalesced=0; uint64_t cbytes=0;
  while(i<items.size()){ size_t j=i+1; uint64_t base=items[i].off; size_t total=items[i].sz; std::string pth=items[i].path;
    while(j<items.size() && items[j].path==pth){ uint64_t prev_end=base+total; if(items[j].off>=prev_end && items[j].off-prev_end<=ctx->options.coalesce_gap && (items[j].off+items[j].sz-base)<=ctx->options.max_coalesce_size){ total=(size_t)(items[j].off+items[j].sz-base); ++j;} else break; }
    std::vector<uint8_t> tmp(total); int rc=do_read_impl(ctx,pth.c_str(),base,total,tmp.data(),MEMTIER_TARGET_CPU); if(rc!=MEMTIER_OK) return rc;
    for(size_t k=i;k<j;++k){ size_t off=(size_t)(items[k].off-base); if(target==MEMTIER_TARGET_CPU){ std::memcpy(items[k].dst,tmp.data()+off,items[k].sz);} else { int crc=memtier_cuda_copy_to_gpu(items[k].dst,tmp.data()+off,items[k].sz,ctx->options.stream); if(crc!=MEMTIER_OK) return MEMTIER_ERR_CUDA; if(!ctx->options.async_mode&&memtier_cuda_synchronize(ctx->options.stream)!=MEMTIER_OK) return MEMTIER_ERR_CUDA; } }
    coalesced++; cbytes += total; i=j;
  }
  {std::lock_guard<std::mutex> lg(ctx->mu); ctx->stats.readv_requests++; ctx->stats.original_ranges += nranges; ctx->stats.coalesced_ranges += coalesced; ctx->stats.coalesced_bytes += cbytes;}
  return MEMTIER_OK;
}

int memtier_init(const memtier_options_t* options, memtier_ctx_t** out_ctx) {
  if (!out_ctx) return MEMTIER_ERR_INVALID;
  memtier_options_t opt{}; opt.chunk_size=1u<<20; opt.dram_cache_size=64u<<20; opt.enable_dram_cache=1; opt.cache_admit=1; opt.gds_min_size=1u<<20; opt.gds_alignment=4096; opt.gds_stub_success=1; opt.num_workers=4; opt.coalesce_gap=64u<<10; opt.max_coalesce_size=1u<<20;
  if (options) opt = *options; if(opt.coalesce_gap==0) opt.coalesce_gap=64u<<10; if(opt.max_coalesce_size==0) opt.max_coalesce_size=1u<<20; if (opt.chunk_size==0) opt.chunk_size=1u<<20; if(opt.dram_cache_size==0) opt.dram_cache_size=64u<<20; if(opt.gds_min_size==0) opt.gds_min_size=1u<<20; if(opt.gds_alignment==0) opt.gds_alignment=4096; if(opt.num_workers<=0) opt.num_workers=4;
  auto* ctx = new (std::nothrow) memtier_ctx_s(opt); if(!ctx) return MEMTIER_ERR_NOMEM;
  for(int i=0;i<opt.num_workers;++i){ctx->workers.emplace_back([ctx](){for(;;){memtier_req_t* r=nullptr;{std::unique_lock<std::mutex> lk(ctx->q_mu);ctx->q_cv.wait(lk,[&](){return ctx->stop||!ctx->queue.empty();}); if(ctx->stop&&ctx->queue.empty()) return; r=ctx->queue.front(); ctx->queue.pop_front();} int rc=do_read_impl(ctx,r->path.c_str(),r->offset,r->size,r->dst,r->target); {std::lock_guard<std::mutex> lk(r->mu); r->result=rc; r->done=true;} r->cv.notify_all(); }});}
  *out_ctx = ctx; return MEMTIER_OK;
}
int memtier_finalize(memtier_ctx_t* ctx){ if(!ctx) return MEMTIER_OK; {std::lock_guard<std::mutex> lk(ctx->q_mu); ctx->stop=true;} ctx->q_cv.notify_all(); for(auto& w:ctx->workers){if(w.joinable()) w.join();} delete ctx; return MEMTIER_OK; }
int memtier_read(memtier_ctx_t* ctx,const char* p,uint64_t o,size_t s,void* d,memtier_target_t t){ return do_read_impl(ctx,p,o,s,d,t);} 
int memtier_read_async(memtier_ctx_t* ctx,const char* p,uint64_t o,size_t s,void* d,memtier_target_t t,memtier_req_t** out){ if(!ctx||!p||!d||!out) return MEMTIER_ERR_INVALID; auto* r=new(std::nothrow) memtier_req_s(); if(!r) return MEMTIER_ERR_NOMEM; r->ctx=ctx; r->path=p; r->offset=o; r->size=s; r->dst=d; r->target=t; r->result=MEMTIER_ERR_INTERNAL; r->done=false; {std::lock_guard<std::mutex> lk(ctx->q_mu); ctx->queue.push_back(r);} ctx->q_cv.notify_one(); *out=r; return MEMTIER_OK; }
int memtier_wait(memtier_req_t* r){ if(!r) return MEMTIER_ERR_INVALID; std::unique_lock<std::mutex> lk(r->mu); r->cv.wait(lk,[&](){return r->done;}); return r->result; }
int memtier_test(memtier_req_t* r,int* done){ if(!r||!done) return MEMTIER_ERR_INVALID; std::lock_guard<std::mutex> lk(r->mu); *done=r->done?1:0; return MEMTIER_OK; }
int memtier_request_status(memtier_req_t* r,int* out){ if(!r||!out) return MEMTIER_ERR_INVALID; std::lock_guard<std::mutex> lk(r->mu); *out=r->result; return MEMTIER_OK; }
void memtier_request_free(memtier_req_t* r){ delete r; }
int memtier_prefetch(memtier_ctx_t* c,const char* p,uint64_t o,size_t s,const memtier_prefetch_options_t* opt){
  if(!c||!p||s==0) return MEMTIER_ERR_INVALID;
  memtier_prefetch_options_t def{}; def.target=MEMTIER_PREFETCH_AUTO; def.allow_posix=1;
  const memtier_prefetch_options_t* po = opt ? opt : &def;
  if(po->target==MEMTIER_PREFETCH_TO_HBM) return MEMTIER_ERR_UNSUPPORTED;
  if(po->async) return MEMTIER_ERR_UNSUPPORTED;
  size_t chunk=c->options.chunk_size; uint64_t fid=file_id_from_path(p); size_t done=0;
  while(done<s){ uint64_t cur=o+done; uint64_t cid=cur/chunk; memtier::CacheKey key{fid,cid}; if(!c->cache.contains(key)){ std::vector<uint8_t> buf(chunk,0); size_t n=0; int rc=memtier::posix_pread_full(p,cid*chunk,chunk,buf.data(),&n); if(rc!=MEMTIER_OK) return rc; c->cache.insert(key,buf.data(),n); std::lock_guard<std::mutex> lg(c->mu); c->stats.prefetch_bytes += n; c->stats.posix_reads++; c->stats.ssd_bytes_read += n; } done += std::min(chunk - (size_t)(cur%chunk), s-done); }
  {std::lock_guard<std::mutex> lg(c->mu); c->stats.prefetch_requests++;}
  return MEMTIER_OK;
}
int memtier_stats(memtier_ctx_t* c, memtier_stats_t* o){if(!c||!o)return MEMTIER_ERR_INVALID;std::lock_guard<std::mutex> lg(c->mu);*o=c->stats;return MEMTIER_OK;}
const char* memtier_strerror(int e){switch(e){case MEMTIER_OK:return"ok";case MEMTIER_ERR_INVALID:return"invalid";case MEMTIER_ERR_IO:return"io";case MEMTIER_ERR_NOMEM:return"nomem";case MEMTIER_ERR_UNSUPPORTED:return"unsupported";case MEMTIER_ERR_GDS:return"gds error";case MEMTIER_ERR_CUDA:return"cuda error";default:return"internal";}}
