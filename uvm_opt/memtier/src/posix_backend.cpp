#include "internal/posix_backend.h"
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
int PosixBackend::openFile(ObjectEntry& obj){ if(obj.fd>=0) return 0; obj.fd=open(obj.path.c_str(), O_RDONLY); if(obj.fd<0){ std::perror("MemTier POSIX open"); return -1;} return 0; }
ssize_t PosixBackend::readToPinnedHost(ObjectEntry& obj, void* host_ptr, size_t length, uint64_t off){ size_t done=0; while(done<length){ ssize_t n=pread(obj.fd,(char*)host_ptr+done,length-done,(off_t)(off+done)); if(n<0){ if(errno==EINTR) continue; std::perror("MemTier POSIX pread"); return -1;} if(n==0) break; done+=size_t(n);} return (ssize_t)done; }
int PosixBackend::closeFile(ObjectEntry& obj){ if(obj.fd>=0){ if(close(obj.fd)!=0){ std::perror("MemTier POSIX close"); return -1;} obj.fd=-1;} return 0; }
