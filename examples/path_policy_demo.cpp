#include "memtier/memtier.h"
#include <cstdio>
#include <fstream>
#include <vector>

static const char* pname(int p){switch(p){case MEMTIER_PATH_DRAM_HIT:return"DRAM_HIT";case MEMTIER_PATH_POSIX_READ:return"POSIX";case MEMTIER_PATH_POSIX_READ_THEN_CACHE:return"POSIX_THEN_CACHE";case MEMTIER_PATH_GDS_READ:return"GDS";case MEMTIER_PATH_GDS_STUB_FALLBACK:return"GDS_FALLBACK";default:return"UNK";}}
int main(){const char* path="memtier_policy_demo.bin";{std::ofstream ofs(path,std::ios::binary);std::vector<unsigned char>d(4<<20,1);ofs.write((const char*)d.data(),d.size());}
memtier_options_t o{};o.enable_dram_cache=1;o.cache_admit=1;o.enable_gds=1;o.gds_stub_success=1;o.gds_min_size=1<<20;o.gds_alignment=4096;o.chunk_size=1<<20;o.dram_cache_size=2<<20;
memtier_ctx_t* c=nullptr;std::vector<unsigned char>a(1<<20),b(4096);
memtier_init(&o,&c);memtier_read(c,path,0,a.size(),a.data(),MEMTIER_TARGET_CPU);memtier_stats_t s{};memtier_stats(c,&s);std::printf("large sequential path=%s\n",pname(s.last_selected_path));memtier_finalize(c);

o.enable_gds=0;memtier_init(&o,&c);memtier_read(c,path,12345,b.size(),b.data(),MEMTIER_TARGET_CPU);memtier_stats(c,&s);std::printf("small random path=%s\n",pname(s.last_selected_path));memtier_finalize(c);

o.enable_gds=0;memtier_init(&o,&c);memtier_read(c,path,0,b.size(),b.data(),MEMTIER_TARGET_CPU);memtier_read(c,path,0,b.size(),b.data(),MEMTIER_TARGET_CPU);memtier_stats(c,&s);std::printf("repeated read path=%s hits=%llu misses=%llu\n",pname(s.last_selected_path),(unsigned long long)s.dram_hits,(unsigned long long)s.dram_misses);memtier_finalize(c);
std::remove(path);}
