#include "embedding_config.h"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

namespace memtier {

PosixLoader::PosixLoader(const std::string& path,
                         const memtier_embedding_config_t& config,
                         bool use_odirect)
    : config_(config), path_(path) {
  int flags = O_RDONLY | O_CLOEXEC;
#ifdef O_DIRECT
  if (use_odirect) {
    flags |= O_DIRECT;
  }
#else
  (void)use_odirect;
#endif
  fd_ = ::open(path.c_str(), flags);
  if (fd_ < 0) {
    throw std::runtime_error("open(" + path + ") failed: " + std::strerror(errno));
  }
}

PosixLoader::~PosixLoader() {
  if (fd_ >= 0) {
    ::close(fd_);
  }
}

size_t PosixLoader::load_block(uint64_t block_id, void* dst, size_t bytes) {
  const uint64_t offset = block_id * block_size_bytes(config_);
  char* out = static_cast<char*>(dst);
  size_t total = 0;
  while (total < bytes) {
    ssize_t got = ::pread(fd_, out + total, bytes - total, static_cast<off_t>(offset + total));
    if (got < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::runtime_error("pread block " + std::to_string(block_id) + " failed: " +
                               std::strerror(errno));
    }
    if (got == 0) {
      throw std::runtime_error("short read for block " + std::to_string(block_id) +
                               ": got " + std::to_string(total) + " of " +
                               std::to_string(bytes) + " bytes");
    }
    total += static_cast<size_t>(got);
  }
  return total;
}

}  // namespace memtier
