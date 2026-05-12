#include "memtier/object_api.h"
#include <cstdio>
#include <fstream>
#include <vector>

int main(){
  const char* path="memtier_obj_tensor.bin";
  {std::ofstream ofs(path,std::ios::binary); std::vector<unsigned char> d(4<<20); for(size_t i=0;i<d.size();++i)d[i]=i%251; ofs.write((char*)d.data(), d.size());}
  memtier_ctx_t* ctx=nullptr; memtier_options_t o{}; o.chunk_size=1<<20; o.dram_cache_size=8<<20; o.enable_dram_cache=1; o.cache_admit=1; if(memtier_init(&o,&ctx)!=MEMTIER_OK) return 1;
  std::vector<unsigned char> out(1<<20); int64_t shape[1]={static_cast<int64_t>(1<<20)};
  memtier_tensor_desc_t desc{}; desc.path=path; desc.data_offset=1<<20; desc.nbytes=1<<20; desc.dtype=MEMTIER_DTYPE_UINT8; desc.shape=shape; desc.ndim=1;
  if(memtier_load_tensor(ctx,&desc,out.data(),nullptr)!=MEMTIER_OK) return 2;
  for(size_t i=0;i<out.size();++i) if(out[i]!=(unsigned char)(((1<<20)+i)%251)) return 3;
  memtier_stats_t s1{}; memtier_stats(ctx,&s1);
  if(memtier_load_tensor(ctx,&desc,out.data(),nullptr)!=MEMTIER_OK) return 4;
  memtier_stats_t s2{}; memtier_stats(ctx,&s2);
  if(!(s2.dram_hits>s1.dram_hits || s2.ssd_bytes_read==s1.ssd_bytes_read)) return 5;
  memtier_finalize(ctx); std::remove(path); return 0;
}
