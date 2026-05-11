#include "memtier/memtier.h"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <vector>
#include "../src/file_prep.h"

int main(){const char* path="/mnt/gds2/cwd_test/memtier_readv_demo.bin";if(memtier_prepare_experiment_file("/mnt/gds2/cwd_test", "memtier_policy_demo.bin", 4ull<<20)!=0){return 1;}
memtier_options_t o{};o.chunk_size=1<<20;o.dram_cache_size=2<<20;o.enable_dram_cache=1;o.cache_admit=1; memtier_ctx_t* c=nullptr; memtier_init(&o,&c);
const size_t n=100; std::vector<memtier_range_t> rs(n); std::vector<std::vector<unsigned char>> bufs(n,std::vector<unsigned char>(4096)); std::vector<void*> dsts(n);
for(size_t i=0;i<n;++i){rs[i]={path,(uint64_t)(i*8192),4096};dsts[i]=bufs[i].data();}
auto t1=std::chrono::steady_clock::now(); for(size_t i=0;i<n;++i) memtier_read(c,path,rs[i].offset,rs[i].size,dsts[i],MEMTIER_TARGET_CPU); auto t2=std::chrono::steady_clock::now();
memtier_read_options_t ro{}; ro.target=MEMTIER_TARGET_CPU; auto t3=std::chrono::steady_clock::now(); memtier_readv(c,rs.data(),dsts.data(),n,&ro); auto t4=std::chrono::steady_clock::now();
memtier_stats_t s{}; memtier_stats(c,&s); printf("loop read us=%lld, readv us=%lld, original=%llu coalesced=%llu\n",(long long)std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count(),(long long)std::chrono::duration_cast<std::chrono::microseconds>(t4-t3).count(),(unsigned long long)s.original_ranges,(unsigned long long)s.coalesced_ranges);
memtier_finalize(c); std::remove(path); }
