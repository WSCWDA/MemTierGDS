#include "memtier_internal.h"

#include <cerrno>
#include <fcntl.h>
#include <unistd.h>

namespace memtier {

int posix_pread_full(const char* path, uint64_t offset, size_t size, uint8_t* dst, size_t* bytes_read) {
  if (!path || !dst || !bytes_read) return MEMTIER_ERR_INVALID;
  *bytes_read = 0;
  int fd = open(path, O_RDONLY);
  if (fd < 0) return MEMTIER_ERR_IO;

  size_t total = 0;
  while (total < size) {
    ssize_t n = pread(fd, dst + total, size - total, static_cast<off_t>(offset + total));
    if (n == 0) {
      close(fd);
      return MEMTIER_ERR_IO;
    }
    if (n < 0) {
      if (errno == EINTR) continue;
      close(fd);
      return MEMTIER_ERR_IO;
    }
    total += static_cast<size_t>(n);
  }
  close(fd);
  *bytes_read = total;
  return MEMTIER_OK;
}

}  // namespace memtier
