#include "internal/gds_backend.h"
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#if MEMTIER_ENABLE_GDS
static bool ok(CUfileError_t s, const char* what){ if(s.err != CU_FILE_SUCCESS){ std::fprintf(stderr,"MemTier GDS %s failed: err=%d cu_err=%d\n",what,s.err,s.cu_err); return false;} return true; }
bool GdsBackend::init(){ auto st=cuFileDriverOpen(); available_=ok(st,"cuFileDriverOpen"); return available_; }
void GdsBackend::finalize(){ if(available_) (void)cuFileDriverClose(); available_=false; }
int GdsBackend::openFile(ObjectEntry& obj){ if(obj.fd>=0) return 0; obj.fd=open(obj.path.c_str(), O_RDONLY | O_DIRECT); if(obj.fd<0){ perror("MemTier GDS open O_DIRECT"); return -1;} return 0; }
int GdsBackend::registerFile(ObjectEntry& obj){ CUfileDescr_t desc{}; desc.handle.fd=obj.fd; desc.type=CU_FILE_HANDLE_TYPE_OPAQUE_FD; auto st=cuFileHandleRegister(&cf_handle_, &desc); if(!ok(st,"cuFileHandleRegister")) return -1; obj.file_registered=true; return 0; }
int GdsBackend::registerBuffer(ObjectEntry& obj, void* gpu_ptr, size_t length){ auto st=cuFileBufRegister(gpu_ptr, length, 0); if(!ok(st,"cuFileBufRegister")) return -1; obj.gds_registered=true; return 0; }
ssize_t GdsBackend::readToGpu(ObjectEntry&, void* gpu_ptr, size_t length, uint64_t file_offset){ ssize_t n=cuFileRead(cf_handle_, gpu_ptr, length, file_offset, 0); if(n<0) std::fprintf(stderr,"MemTier GDS cuFileRead failed: %zd\n", n); return n; }
int GdsBackend::deregisterBuffer(ObjectEntry& obj){ if(obj.gds_registered){ (void)cuFileBufDeregister(obj.device_ptr); obj.gds_registered=false;} return 0; }
int GdsBackend::closeFile(ObjectEntry& obj){ if(obj.file_registered){ (void)cuFileHandleDeregister(cf_handle_); obj.file_registered=false;} if(obj.fd>=0){ close(obj.fd); obj.fd=-1;} return 0; }
#else
bool GdsBackend::init(){ available_=false; return false; }
void GdsBackend::finalize(){ available_=false; }
int GdsBackend::openFile(ObjectEntry&){ return -1; } int GdsBackend::registerFile(ObjectEntry&){ return -1; } int GdsBackend::registerBuffer(ObjectEntry&, void*, size_t){ return -1; } ssize_t GdsBackend::readToGpu(ObjectEntry&, void*, size_t, uint64_t){ return -1; } int GdsBackend::deregisterBuffer(ObjectEntry&){ return 0; } int GdsBackend::closeFile(ObjectEntry&){ return 0; }
#endif
