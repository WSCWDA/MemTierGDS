#include "memtier/memtier.h"
#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>

int run(const char* file,const std::string& mode){memtier_options_t o{};o.enable_dram_cache=1;o.cache_admit=1;o.chunk_size=1<<20;o.dram_cache_size=8<<20; if(mode=="posix")o.force_posix=1; if(mode=="gds")o.force_gds=1,o.enable_gds=1; memtier_ctx_t* c=nullptr; memtier_init(&o,&c); std::vector<unsigned char> b(4096); auto st=std::chrono::steady_clock::now(); for(int r=0;r<5000;++r){memtier_read(c,file,(r%256)*4096,b.size(),b.data(),MEMTIER_TARGET_CPU);} auto us=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-st).count(); memtier_stats_t s{}; memtier_stats(c,&s); double hit_ratio=s.total_requests? (double)s.dram_hits/s.total_requests:0; std::cout<<"mode="<<mode<<" us="<<us<<" ssd_bytes="<<s.ssd_bytes_read<<" hit_ratio="<<hit_ratio<<"\n"; memtier_finalize(c); return 0;}
int main(int argc,char**argv){const char* file="test.bin"; if(argc>1) file=argv[1]; run(file,"posix"); run(file,"gds"); run(file,"memtier"); }
