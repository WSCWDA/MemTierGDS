#include "memtier/object_api.h"
#include <cstdio>
#include <fstream>
#include <vector>

int main(){
  const char* data="memtier_dataset.bin"; const char* idx="memtier_dataset.idx";
  {std::ofstream ofs(data,std::ios::binary); std::vector<unsigned char> d(8192); for(size_t i=0;i<d.size();++i)d[i]=i%251; ofs.write((char*)d.data(), d.size());}
  {std::ofstream o(idx); o<<"0 "<<data<<" 0 4096 3\n"; o<<"1 "<<data<<" 4096 4096 7\n";}
  memtier_options_t opt{}; opt.chunk_size=1<<20; opt.dram_cache_size=4<<20; opt.enable_dram_cache=1; opt.cache_admit=1;
  memtier_ctx_t* ctx=nullptr; if(memtier_init(&opt,&ctx)!=MEMTIER_OK) return 1;
  MemTierDataset ds(ctx, idx); if(ds.size()!=2) return 2;
  std::vector<unsigned char> out0(4096), out1(4096);
  if(ds.read_sample(0,out0.data(),nullptr)!=MEMTIER_OK) return 3;
  if(ds.read_sample(1,out1.data(),nullptr)!=MEMTIER_OK) return 4;
  for(size_t i=0;i<4096;++i){ if(out0[i]!=(unsigned char)(i%251)) return 5; if(out1[i]!=(unsigned char)((4096+i)%251)) return 6; }
  memtier_stats_t s1{}; memtier_stats(ctx,&s1);
  if(ds.read_sample(0,out0.data(),nullptr)!=MEMTIER_OK) return 7;
  memtier_stats_t s2{}; memtier_stats(ctx,&s2); if(s2.dram_hits<=s1.dram_hits) return 8;
  if(ds.sample(0).label!=3 || ds.sample(1).label!=7) return 9;
  memtier_finalize(ctx); std::remove(data); std::remove(idx); return 0;
}
