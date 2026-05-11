#ifndef MEMTIER_CUDA_BACKEND_H
#define MEMTIER_CUDA_BACKEND_H
class CudaBackend { public: CudaBackend(bool e,int d):en_(e),dev_(d){} bool available() const { return false; } bool enabled() const { return en_; } const char* backend_name() const { return "cuda_stub"; } private: bool en_; int dev_; };
#endif
