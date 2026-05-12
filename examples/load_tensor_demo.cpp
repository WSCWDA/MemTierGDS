#include "memtier/object_api.h"
#include <fstream>
#include <iostream>
#include <vector>
int main(){
  const char* p="demo_tensor.bin";
  {std::ofstream o(p,std::ios::binary);std::vector<unsigned char>d(2<<20);for(size_t i=0;i<d.size();++i)d[i]=i%251;o.write((char*)d.data(),d.size());}
  memtier_ctx_t* c=nullptr; memtier_options_t opt{}; opt.chunk_size=1<<20; opt.dram_cache_size=4<<20; opt.enable_dram_cache=1; opt.cache_admit=1; memtier_init(&opt,&c);
  std::vector<unsigned char> out(1<<20); int64_t shape[1]={static_cast<int64_t>(1<<20)}; memtier_tensor_desc_t desc{}; desc.path=p;desc.data_offset=0;desc.nbytes=out.size();desc.shape=shape;desc.ndim=1;desc.dtype=MEMTIER_DTYPE_UINT8;
  int rc=memtier_load_tensor(c,&desc,out.data(),nullptr);
  for(size_t i=0;i<out.size();++i){ if(out[i]!=(unsigned char)(i%251)){ std::cerr<<"verify fail\n"; return 2; } }
  memtier_stats_t s{};memtier_stats(c,&s);std::cout<<"rc="<<rc<<" dram_hits="<<s.dram_hits<<" ssd_bytes="<<s.ssd_bytes_read<<"\n";
  memtier_finalize(c); return rc==MEMTIER_OK?0:1;
}
