#include "memtier/memtier.h"

#include <cstdio>
#include <fstream>
#include <vector>

int main() {
  const char* path = "memtier_path_test.bin";
  {
    std::ofstream ofs(path, std::ios::binary);
    std::vector<unsigned char> data(4 * 1024 * 1024);
    for (size_t i=0;i<data.size();++i) data[i]= static_cast<unsigned char>(i%251);
    ofs.write((const char*)data.data(), data.size());
  }
  memtier_options_t opt{};
  opt.chunk_size = 1<<20; opt.dram_cache_size = 2<<20; opt.enable_dram_cache=1; opt.cache_admit=1;
  opt.enable_gds = 1; opt.gds_stub_success = 1; opt.gds_min_size = 1<<20; opt.gds_alignment = 4096; opt.hints = MEMTIER_HINT_SEQUENTIAL;
  memtier_ctx_t* ctx=nullptr; if(memtier_init(&opt,&ctx)!=MEMTIER_OK) return 1;

  std::vector<unsigned char> big(1<<20); if(memtier_read(ctx,path,0,big.size(),big.data(),MEMTIER_TARGET_CPU)!=MEMTIER_OK) return 2;
  memtier_stats_t s{}; memtier_stats(ctx,&s); if(s.selected_gds==0) return 3;

  opt.force_posix=1; memtier_finalize(ctx); if(memtier_init(&opt,&ctx)!=MEMTIER_OK) return 4;
  std::vector<unsigned char> small(4096); memtier_read(ctx,path,0,small.size(),small.data(),MEMTIER_TARGET_CPU); memtier_stats(ctx,&s); if(s.selected_posix==0 && s.selected_cache==0) return 5;

  opt.force_posix=0; opt.force_gds=1; opt.gds_stub_success=0; memtier_finalize(ctx); memtier_init(&opt,&ctx);
  memtier_read(ctx,path,0,big.size(),big.data(),MEMTIER_TARGET_CPU); memtier_stats(ctx,&s); if(s.gds_fallbacks==0) return 6;

  opt.force_gds=0; opt.gds_stub_success=1; opt.hints=MEMTIER_HINT_REUSE; memtier_finalize(ctx); memtier_init(&opt,&ctx);
  memtier_read(ctx,path,0,4096,small.data(),MEMTIER_TARGET_CPU); memtier_read(ctx,path,0,4096,small.data(),MEMTIER_TARGET_CPU); memtier_stats(ctx,&s); if(s.dram_hits==0) return 7;

  memtier_finalize(ctx); std::remove(path); return 0;
}
