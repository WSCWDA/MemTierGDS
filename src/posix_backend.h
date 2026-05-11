#ifndef MEMTIER_POSIX_BACKEND_H
#define MEMTIER_POSIX_BACKEND_H
class PosixBackend { public: PosixBackend() = default; bool available() const { return true; } };
#endif
