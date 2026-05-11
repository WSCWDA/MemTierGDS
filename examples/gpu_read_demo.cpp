#include "memtier/memtier.h"
#include "../src/cuda_backend.h"
#include <cstdio>
#include <fstream>
#include <vector>
#include "../src/file_prep.h"

int main(){ if(!memtier_cuda_available()){ std::printf("CUDA unavailable, skip demo\n"); return 0; }
const char* path="/mnt/gds2/cwd_test/memtier_gpu_demo.bin"; if(memtier_prepare_experiment_file("/mnt/gds2/cwd_test", "memtier_gpu_demo.bin", 1ull<<20)!=0){return 1;}
memtier_options_t o{};o.enable_dram_cache=1;o.cache_admit=1;o.chunk_size=1<<20;o.dram_cache_size=2<<20; memtier_ctx_t* c=nullptr; memtier_init(&o,&c);
void* gpu=nullptr; memtier_cuda_alloc(&gpu,4096,0); memtier_read(c,path,0,4096,gpu,MEMTIER_TARGET_GPU); std::vector<unsigned char>h(4096); memtier_cuda_copy_to_host(h.data(),gpu,h.size());
bool ok=true; for(size_t i=0;i<h.size();++i){if(h[i]!=(unsigned char)(i%251)){ok=false;break;}}
memtier_stats_t s{}; memtier_read(c,path,0,4096,gpu,MEMTIER_TARGET_GPU); memtier_stats(c,&s);
std::printf("gpu read ok=%d, dram_hits=%llu\n", ok?1:0, (unsigned long long)s.dram_hits);
memtier_cuda_free(gpu); memtier_finalize(c); std::remove(path); }
