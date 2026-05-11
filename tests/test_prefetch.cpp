#include "memtier/memtier.h"

#include <cstdio>
#include <fstream>
#include <vector>

int main(){
  const char* path="memtier_prefetch_test.bin";
  {std::ofstream ofs(path,std::ios::binary);std::vector<unsigned char>d(4<<20);for(size_t i=0;i<d.size();++i)d[i]=i%251;ofs.write((const char*)d.data(),d.size());}
  memtier_options_t o{};o.chunk_size=1<<20;o.dram_cache_size=2<<20;o.enable_dram_cache=1;o.cache_admit=1;
  memtier_ctx_t* c=nullptr; if(memtier_init(&o,&c)!=MEMTIER_OK) return 1;
  memtier_prefetch_options_t po{}; po.target=MEMTIER_PREFETCH_TO_DRAM; po.allow_posix=1;
  if(memtier_prefetch(c,path,0,2<<20,&po)!=MEMTIER_OK) return 2;
  memtier_stats_t s1{}; memtier_stats(c,&s1);
  std::vector<unsigned char> b(4096); if(memtier_read(c,path,0,b.size(),b.data(),MEMTIER_TARGET_CPU)!=MEMTIER_OK) return 3;
  memtier_stats_t s2{}; memtier_stats(c,&s2); if(s2.dram_hits<=s1.dram_hits) return 4; if(s2.ssd_bytes_read!=s1.ssd_bytes_read) return 5;
  if(memtier_prefetch(c,path,0,2<<20,&po)!=MEMTIER_OK) return 6; memtier_stats_t s3{}; memtier_stats(c,&s3); if(s3.ssd_bytes_read!=s2.ssd_bytes_read) return 7;
  if(memtier_prefetch(c,path,0,4<<20,&po)!=MEMTIER_OK) return 8;
  memtier_finalize(c); std::remove(path); return 0;
}
