#include "memtier/object_api.h"
#include <fstream>
#include <iostream>
#include <vector>
int main(){const char* p="demo_tensor.bin";{std::ofstream o(p,std::ios::binary);std::vector<unsigned char>d(2<<20);for(size_t i=0;i<d.size();++i)d[i]=i%251;o.write((char*)d.data(),d.size());}
memtier_ctx_t* c=nullptr;memtier_init(nullptr,&c);std::vector<unsigned char> out(1<<20);memtier_tensor_desc_t desc{};desc.path=p;desc.data_offset=0;desc.nbytes=out.size();int rc=memtier_load_tensor(c,&desc,out.data(),nullptr);memtier_stats_t s{};memtier_stats(c,&s);std::cout<<"rc="<<rc<<" last_path="<<s.last_selected_path<<" total="<<s.total_requests<<"\n";memtier_finalize(c);return rc==MEMTIER_OK?0:1;}
