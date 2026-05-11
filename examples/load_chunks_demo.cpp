#include "memtier/object_api.h"
#include <fstream>
#include <iostream>
#include <vector>
int main(){const char* p="demo_chunks.bin";{std::ofstream o(p,std::ios::binary);std::vector<unsigned char>d(1<<20);for(size_t i=0;i<d.size();++i)d[i]=i%251;o.write((char*)d.data(),d.size());}
memtier_ctx_t* c=nullptr;memtier_init(nullptr,&c);std::vector<memtier_chunk_desc_t> ch(8);std::vector<std::vector<unsigned char>> bufs(8,std::vector<unsigned char>(4096));std::vector<void*> dsts(8);
for(size_t i=0;i<8;++i){ch[i]={p,(uint64_t)(i*4096),4096,(uint64_t)i};dsts[i]=bufs[i].data();}
int rc=memtier_load_chunks(c,ch.data(),ch.size(),dsts.data(),nullptr);memtier_stats_t s{};memtier_stats(c,&s);std::cout<<"rc="<<rc<<" readv="<<s.readv_requests<<" coalesced="<<s.coalesced_ranges<<" dram_hits="<<s.dram_hits<<"\n";memtier_finalize(c);return rc==MEMTIER_OK?0:1;}
