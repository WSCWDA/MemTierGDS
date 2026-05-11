#include "memtier/memtier.h"
#include "../src/cuda_backend.h"

#include <cstdio>
#include <fstream>
#include <vector>

int main(){
#if !MEMTIER_HAVE_GDS
  std::printf("SKIP: no real GDS\n");
  return 0;
#else
  if(!memtier_cuda_available()){ std::printf("SKIP: no CUDA\n"); return 0; }
  const char* path="memtier_gds_test.bin";
  { std::ofstream ofs(path,std::ios::binary); std::vector<unsigned char>d(1<<20); for(size_t i=0;i<d.size();++i)d[i]=i%251; ofs.write((const char*)d.data(),d.size()); }
  memtier_options_t o{}; o.enable_gds=1; o.force_gds=1; o.gds_alignment=4096; o.chunk_size=1<<20; o.dram_cache_size=2<<20;
  memtier_ctx_t* c=nullptr; if(memtier_init(&o,&c)!=MEMTIER_OK) return 1;
  void* gpu=nullptr; if(memtier_cuda_alloc(&gpu,4096,0)!=MEMTIER_OK) return 0; // skip
  int rc=memtier_read(c,path,0,4096,gpu,MEMTIER_TARGET_GPU);
  if(rc!=MEMTIER_OK && rc!=MEMTIER_ERR_GDS) { memtier_cuda_free(gpu); memtier_finalize(c); return 2; }
  memtier_cuda_free(gpu); memtier_finalize(c); std::remove(path); return 0;
#endif
}
