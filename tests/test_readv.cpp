#include "memtier/memtier.h"

#include <cstdio>
#include <fstream>
#include <vector>

int main(){
  const char* path="memtier_readv_test.bin";
  {std::ofstream ofs(path,std::ios::binary);std::vector<unsigned char>d(4<<20);for(size_t i=0;i<d.size();++i)d[i]=i%251;ofs.write((const char*)d.data(),d.size());}
  memtier_options_t o{};o.chunk_size=1<<20;o.dram_cache_size=2<<20;o.enable_dram_cache=1;o.cache_admit=1;o.coalesce_gap=64<<10;o.max_coalesce_size=1<<20;
  memtier_ctx_t* c=nullptr; if(memtier_init(&o,&c)!=MEMTIER_OK) return 1;
  const size_t n=100; std::vector<memtier_range_t> rs(n); std::vector<std::vector<unsigned char>> bufs(n,std::vector<unsigned char>(4096)); std::vector<void*> dsts(n);
  for(size_t i=0;i<n;++i){ rs[i]={path,(uint64_t)(i*8192),4096}; dsts[i]=bufs[i].data(); }
  memtier_read_options_t ro{}; ro.target=MEMTIER_TARGET_CPU;
  if(memtier_readv(c,rs.data(),dsts.data(),n,&ro)!=MEMTIER_OK) return 2;
  for(size_t i=0;i<n;++i){ for(size_t j=0;j<4096;++j){ if(bufs[i][j]!=(unsigned char)((i*8192+j)%251)) return 3; } }
  memtier_stats_t s1{}; memtier_stats(c,&s1); if(!(s1.coalesced_ranges < s1.original_ranges)) return 4;
  if(memtier_readv(c,rs.data(),dsts.data(),n,&ro)!=MEMTIER_OK) return 5; memtier_stats_t s2{}; memtier_stats(c,&s2); if(s2.dram_hits<=s1.dram_hits) return 6;
  memtier_finalize(c); std::remove(path); return 0;
}
