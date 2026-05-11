#include "memtier/memtier.h"

#include <cstdio>
#include <fstream>
#include <vector>

int main(){
  const char* path="memtier_async_test.bin";
  {std::ofstream ofs(path,std::ios::binary);std::vector<unsigned char>d(16<<20);for(size_t i=0;i<d.size();++i)d[i]=i%251;ofs.write((const char*)d.data(),d.size());}
  memtier_options_t o{};o.chunk_size=1<<20;o.dram_cache_size=8<<20;o.enable_dram_cache=1;o.cache_admit=1;o.num_workers=4;
  memtier_ctx_t* c=nullptr; if(memtier_init(&o,&c)!=MEMTIER_OK) return 1;
  std::vector<std::vector<unsigned char>> bufs(4,std::vector<unsigned char>(1<<20));
  std::vector<memtier_req_t*> reqs(4,nullptr);
  for(int i=0;i<4;++i) if(memtier_read_async(c,path,(uint64_t)i*(1<<20),1<<20,bufs[i].data(),MEMTIER_TARGET_CPU,&reqs[i])!=MEMTIER_OK) return 2;
  int done=0; memtier_test(reqs[0],&done);
  for(int i=0;i<4;++i){int rc=memtier_wait(reqs[i]); if(rc!=MEMTIER_OK) return 3; memtier_request_free(reqs[i]);}
  for(int i=0;i<4;++i){for(size_t j=0;j<bufs[i].size();++j){ if(bufs[i][j]!=(unsigned char)(((size_t)i*(1<<20)+j)%251)) return 4; }}
  for(int i=0;i<4;++i) if(memtier_read_async(c,path,(uint64_t)i*(1<<20),1<<20,bufs[i].data(),MEMTIER_TARGET_CPU,&reqs[i])!=MEMTIER_OK) return 5;
  for(int i=0;i<4;++i){int rc=memtier_wait(reqs[i]); if(rc!=MEMTIER_OK) return 6; memtier_request_free(reqs[i]);}
  memtier_stats_t s{}; memtier_stats(c,&s); if(s.dram_hits==0) return 7;
  memtier_finalize(c); std::remove(path); return 0;
}
