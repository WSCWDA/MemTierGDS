#include "memtier/memtier.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <filesystem>
#include "../src/file_prep.h"

int main(int argc,char**argv){const char* file=argc>1?argv[1]:"/mnt/gds2/cwd_test/test.bin"; memtier_options_t o{};o.enable_dram_cache=1;o.cache_admit=1;o.chunk_size=1<<20;o.dram_cache_size=8<<20; memtier_ctx_t* c=nullptr; memtier_init(&o,&c);
if(!std::filesystem::exists(file)){ if(memtier_prepare_experiment_file("/mnt/gds2/cwd_test", std::filesystem::path(file).filename().c_str(), 64ull<<20)!=0){ std::cerr<<"prepare file failed\n"; return 1; }}
const size_t n=2000; std::vector<memtier_range_t> rs(n); std::vector<std::vector<unsigned char>> bufs(n,std::vector<unsigned char>(4096)); std::vector<void*> dsts(n);
for(size_t i=0;i<n;++i){rs[i]={file,(uint64_t)(i*4096),4096};dsts[i]=bufs[i].data();}
auto t1=std::chrono::steady_clock::now(); for(size_t i=0;i<n;++i) memtier_read(c,file,rs[i].offset,rs[i].size,dsts[i],MEMTIER_TARGET_CPU); auto t2=std::chrono::steady_clock::now();
memtier_read_options_t ro{}; ro.target=MEMTIER_TARGET_CPU; auto t3=std::chrono::steady_clock::now(); memtier_readv(c,rs.data(),dsts.data(),n,&ro); auto t4=std::chrono::steady_clock::now();
memtier_stats_t s{}; memtier_stats(c,&s); std::cout<<"loop_us="<<std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()<<" readv_us="<<std::chrono::duration_cast<std::chrono::microseconds>(t4-t3).count()<<" original="<<s.original_ranges<<" coalesced="<<s.coalesced_ranges<<"\n"; memtier_finalize(c); }
