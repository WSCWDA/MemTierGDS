#include "memtier/memtier.h"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <vector>
#include "../src/file_prep.h"

int main(){const char* path="/mnt/gds2/cwd_test/memtier_prefetch_demo.bin";if(memtier_prepare_experiment_file("/mnt/gds2/cwd_test", "memtier_prefetch_demo.bin", 64ull<<20)!=0){return 1;}
memtier_options_t o{};o.chunk_size=1<<20;o.dram_cache_size=16<<20;o.enable_dram_cache=1;o.cache_admit=1;memtier_ctx_t* c=nullptr;memtier_init(&o,&c);std::vector<unsigned char>b(1<<20);
auto t1=std::chrono::steady_clock::now();memtier_read(c,path,0,b.size(),b.data(),MEMTIER_TARGET_CPU);auto t2=std::chrono::steady_clock::now();
memtier_prefetch_options_t po{};po.target=MEMTIER_PREFETCH_TO_DRAM;memtier_prefetch(c,path,0,8<<20,&po);
auto t3=std::chrono::steady_clock::now();memtier_read(c,path,0,b.size(),b.data(),MEMTIER_TARGET_CPU);auto t4=std::chrono::steady_clock::now();
memtier_stats_t s{};memtier_stats(c,&s);printf("before prefetch read us=%lld\n",(long long)std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count());printf("after prefetch read us=%lld, hits=%llu prefetch_bytes=%llu\n",(long long)std::chrono::duration_cast<std::chrono::microseconds>(t4-t3).count(),(unsigned long long)s.dram_hits,(unsigned long long)s.prefetch_bytes);
memtier_finalize(c);std::remove(path);}
