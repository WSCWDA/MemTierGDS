#include "memtier/memtier.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <thread>
#include <vector>
#include "../src/file_prep.h"

int main(){
  const char* path="/mnt/gds2/cwd_test/memtier_async_demo.bin";
  if(memtier_prepare_experiment_file("/mnt/gds2/cwd_test", "memtier_async_demo.bin", 8ull<<20)!=0){return 1;}
  memtier_options_t o{};o.chunk_size=1<<20;o.dram_cache_size=4<<20;o.enable_dram_cache=1;o.cache_admit=1;o.num_workers=4;
  memtier_ctx_t* c=nullptr; memtier_init(&o,&c);
  std::vector<std::vector<unsigned char>> bufs(8,std::vector<unsigned char>(1<<20)); std::vector<memtier_req_t*> reqs(8,nullptr);
  auto st=std::chrono::steady_clock::now();
  for(int i=0;i<8;++i) memtier_read_async(c,path,(uint64_t)i*(1<<20),1<<20,bufs[i].data(),MEMTIER_TARGET_CPU,&reqs[i]);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  for(auto* r:reqs){memtier_wait(r); memtier_request_free(r);} 
  auto us=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-st).count();
  memtier_stats_t s{}; memtier_stats(c,&s);
  std::printf("async done in %lld us, requests=%llu hits=%llu misses=%llu\n", (long long)us, (unsigned long long)s.total_requests, (unsigned long long)s.dram_hits, (unsigned long long)s.dram_misses);
  memtier_finalize(c); std::remove(path); return 0;
}
