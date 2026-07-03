#include "memtier.h"
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <vector>
#include <sys/stat.h>
__global__ void check_kernel(const float* in, int* ok){ int i=threadIdx.x; if(i<16 && in[i] != float(i)) *ok=0; }
static void make_file(const char* p, size_t n){ struct stat st{}; if(stat(p,&st)==0 && (size_t)st.st_size>=n*sizeof(float)) return; int fd=open(p,O_CREAT|O_TRUNC|O_WRONLY,0644); std::vector<float> v(n); for(size_t i=0;i<n;i++) v[i]=float(i); write(fd,v.data(),v.size()*sizeof(float)); close(fd); }
int main(int argc,char** argv){ const char* path=argc>1?argv[1]:"input.bin"; size_t n=1<<20; make_file(path,n); memtierInit(0); cudaStream_t s; cudaStreamCreate(&s); MemTierHandle h; if(memtierMapFile(path,0,n*sizeof(float),MEMTIER_READ_ONLY|MEMTIER_PREFER_GDS,&h)!=0) return 1; printf("residency_before=%s\n",memtierResidencyToString(memtierGetResidency(h))); cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a,s); if(memtierPrefetchToGpu(h,s)!=0) return 2; cudaEventRecord(b,s); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms,a,b); int* ok; cudaMallocManaged(&ok,sizeof(int)); *ok=1; check_kernel<<<1,32,0,s>>>((const float*)h.ptr,ok); cudaStreamSynchronize(s); printf("selected_path=%s\nresidency_after=%s\nelapsed_time_ms=%.3f\nverification_passed=%s\n",memtierPathToString(memtierGetLastPath(h)),memtierResidencyToString(memtierGetResidency(h)),ms,*ok?"true":"false"); int passed=*ok; cudaFree(ok); memtierUnmapFile(h); cudaStreamDestroy(s); memtierFinalize(); return passed?0:3; }
