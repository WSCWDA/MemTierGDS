#include "file_prep.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <string>
#include <vector>

int memtier_prepare_experiment_file(const char* dir, const char* name, size_t size_bytes) {
  if (!dir || !name || size_bytes == 0) return -1;
  if (mkdir(dir, 0755) != 0 && errno != EEXIST) return -2;
  std::string path = std::string(dir) + "/" + name;
  int fd = open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
  if (fd < 0) return -3;

  const size_t block = 1 << 20;
  std::vector<unsigned char> buf(block);
  for (size_t i = 0; i < block; ++i) buf[i] = static_cast<unsigned char>(i % 251);

  size_t written = 0;
  while (written < size_bytes) {
    size_t n = std::min(block, size_bytes - written);
    ssize_t w = write(fd, buf.data(), n);
    if (w < 0) {
      close(fd);
      return -4;
    }
    written += static_cast<size_t>(w);
  }
  fsync(fd);
  close(fd);
  return 0;
}
