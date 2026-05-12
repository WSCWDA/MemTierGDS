#include "memtier/object_api.h"
#include <cstdio>
#include <fstream>
#include <vector>

int main(){
  const char* path="memtier_obj_chunks.bin";
  {std::ofstream ofs(path,std::ios::binary); std::vector<unsigned char> d(1<<20); for(size_t i=0;i<d.size();++i)d[i]=i%251; ofs.write((char*)d.data(), d.size());}
  memtier_ctx_t* ctx=nullptr; memtier_options_t o{}; o.chunk_size=1<<20; o.dram_cache_size=4<<20; o.enable_dram_cache=1; o.cache_admit=1; if(memtier_init(&o,&ctx)!=MEMTIER_OK) return 1;
  const size_t n=10; std::vector<memtier_chunk_desc_t> ch(n); std::vector<std::vector<unsigned char>> bufs(n,std::vector<unsigned char>(4096)); std::vector<void*> dsts(n);
  for(size_t i=0;i<n;++i){ ch[i]={path,(uint64_t)(i*4096),4096,(uint64_t)i}; dsts[i]=bufs[i].data(); }
  if(memtier_load_chunks(ctx,ch.data(),n,dsts.data(),nullptr)!=MEMTIER_OK) return 2;
  for(size_t i=0;i<n;++i) for(size_t j=0;j<4096;++j) if(bufs[i][j]!=(unsigned char)((i*4096+j)%251)) return 3;
  memtier_stats_t s{}; memtier_stats(ctx,&s); if(s.readv_requests==0 && s.coalesced_ranges==0) return 4;
  memtier_finalize(ctx); std::remove(path); return 0;
}
