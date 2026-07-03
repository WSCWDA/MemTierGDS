#include "memtier.h"
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cstdlib>
static void make_file(const char* p, size_t bytes){ int fd=open(p,O_CREAT|O_TRUNC|O_WRONLY,0644); std::vector<char> buf(1<<20,1); for(size_t w=0;w<bytes;w+=buf.size()) write(fd,buf.data(), std::min(buf.size(),bytes-w)); close(fd); }
int main(int argc,char** argv){ const char* file="compare.bin"; size_t mb=64; const char* mode="auto"; int rep=1; for(int i=1;i<argc;i++){ if(!strcmp(argv[i],"--file")) file=argv[++i]; else if(!strcmp(argv[i],"--size-mb")) mb=strtoull(argv[++i],nullptr,10); else if(!strcmp(argv[i],"--mode")) mode=argv[++i]; else if(!strcmp(argv[i],"--repeat")) rep=atoi(argv[++i]); } size_t bytes=mb*1024ull*1024ull; make_file(file,bytes); memtierInit(0); printf("mode,size_mb,repeat,path,time_ms,throughput_gbps,residency\n"); for(int r=0;r<rep;r++){ unsigned flags=MEMTIER_READ_ONLY; if(!strcmp(mode,"gds")) flags|=MEMTIER_PREFER_GDS|MEMTIER_STREAMING; if(!strcmp(mode,"dram")) flags|=MEMTIER_REUSE_EXPECTED|MEMTIER_PREFER_DRAM; MemTierHandle h; if(memtierMapFile(file,0,bytes,flags,&h)!=0) return 1; cudaStream_t s; cudaStreamCreate(&s); if(!strcmp(mode,"dram")) memtierPrefetchToDram(h); cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a,s); int rc=memtierPrefetchToGpu(h,s); cudaEventRecord(b,s); cudaEventSynchronize(b); float ms=0; cudaEventElapsedTime(&ms,a,b); printf("%s,%zu,%d,%s,%.3f,%.3f,%s\n",mode,mb,r,rc==0?memtierPathToString(memtierGetLastPath(h)):"skipped",ms, ms>0?(double)bytes*8.0/(ms*1e6):0.0, memtierResidencyToString(memtierGetResidency(h))); memtierUnmapFile(h); cudaStreamDestroy(s); } memtierFinalize(); }
