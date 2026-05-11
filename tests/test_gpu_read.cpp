#include "memtier/memtier.h"
#include "../src/cuda_backend.h"

#include <cstdio>
#include <fstream>
#include <vector>

int main() {
  if (!memtier_cuda_available()) {
    std::printf("SKIP: CUDA unavailable\n");
    return 0;
  }
  const char* path = "memtier_gpu_test.bin";
  { std::ofstream ofs(path, std::ios::binary); std::vector<unsigned char> d(1<<20); for(size_t i=0;i<d.size();++i)d[i]=i%251; ofs.write((const char*)d.data(), d.size()); }

  memtier_options_t o{}; o.enable_dram_cache=1; o.cache_admit=1; o.chunk_size=1<<20; o.dram_cache_size=2<<20;
  memtier_ctx_t* ctx=nullptr; if(memtier_init(&o,&ctx)!=MEMTIER_OK) return 1;
  void* gpu=nullptr; if(memtier_cuda_alloc(&gpu,4096,0)!=MEMTIER_OK) return 2;
  if(memtier_read(ctx,path,0,4096,gpu,MEMTIER_TARGET_GPU)!=MEMTIER_OK) return 3;
  std::vector<unsigned char> h(4096); if(memtier_cuda_copy_to_host(h.data(),gpu,h.size())!=MEMTIER_OK) return 4;
  for(size_t i=0;i<h.size();++i) if(h[i]!=(unsigned char)(i%251)) return 5;
  if(memtier_read(ctx,path,0,4096,gpu,MEMTIER_TARGET_GPU)!=MEMTIER_OK) return 6;
  memtier_stats_t s{}; memtier_stats(ctx,&s); if(s.dram_hits==0) return 7;
  memtier_cuda_free(gpu); memtier_finalize(ctx); std::remove(path); return 0;
}
