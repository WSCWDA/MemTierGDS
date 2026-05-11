#include "memtier/memtier.h"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc,char**argv){std::string file="test.bin",mode="memtier";size_t size=1ull<<30;for(int i=1;i<argc;++i){if(std::strncmp(argv[i],"--file=",7)==0)file=argv[i]+7;else if(std::strncmp(argv[i],"--size=",7)==0)size=std::stoull(argv[i]+7);else if(std::strncmp(argv[i],"--mode=",7)==0)mode=argv[i]+7;}
std::vector<unsigned char> buf(1<<20); memtier_options_t o{};o.enable_dram_cache=1;o.cache_admit=1;o.chunk_size=1<<20;o.dram_cache_size=64<<20; if(mode=="posix")o.force_posix=1; if(mode=="gds")o.force_gds=1,o.enable_gds=1;
memtier_ctx_t* c=nullptr; memtier_init(&o,&c); auto st=std::chrono::steady_clock::now(); size_t done=0; while(done<size){size_t n=std::min(buf.size(),size-done); if(memtier_read(c,file.c_str(),done,n,buf.data(),MEMTIER_TARGET_CPU)!=MEMTIER_OK){std::cerr<<"read failed\n";break;} done+=n;} auto us=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-st).count(); memtier_stats_t s{}; memtier_stats(c,&s); double gbps=(double)done/(double)us; std::cout<<"mode="<<mode<<" bytes="<<done<<" time_us="<<us<<" throughput_MBps="<<gbps<<" selected_path="<<s.last_selected_path<<"\n"; memtier_finalize(c); }
