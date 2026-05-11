#include "memtier/object_api.h"
#include <cstdio>
#include <fstream>
#include <vector>

int main(){
  const char* data="memtier_dataset.bin"; const char* idx="memtier_dataset.idx";
  {std::ofstream ofs(data,std::ios::binary); std::vector<unsigned char> d(8192); for(size_t i=0;i<d.size();++i)d[i]=i%251; ofs.write((char*)d.data(), d.size());}
  {std::ofstream o(idx); o<<"0 "<<data<<" 0 4096\n"; o<<"1 "<<data<<" 4096 4096\n";}
  MemTierDataset ds(idx); if(ds.size()!=2) return 1;
  memtier_ctx_t* ctx=nullptr; if(memtier_init(nullptr,&ctx)!=MEMTIER_OK) return 2;
  std::vector<unsigned char> out(4096); if(ds.read_sample(ctx,1,out.data(),nullptr)!=MEMTIER_OK) return 3;
  for(size_t i=0;i<out.size();++i) if(out[i]!=(unsigned char)((4096+i)%251)) return 4;
  memtier_finalize(ctx); std::remove(data); std::remove(idx); return 0;
}
