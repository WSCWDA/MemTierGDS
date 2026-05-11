#include "../src/file_prep.h"

#include <sys/stat.h>
#include <unistd.h>

#include <string>

int main() {
  const char* dir = "/tmp/memtier_prep_test";
  const char* name = "a.bin";
  size_t size = 2 * 1024 * 1024 + 123;
  int rc = memtier_prepare_experiment_file(dir, name, size);
  if (rc != 0) return 1;
  std::string p = std::string(dir) + "/" + name;
  struct stat st {};
  if (stat(p.c_str(), &st) != 0) return 2;
  if (static_cast<size_t>(st.st_size) != size) return 3;
  unlink(p.c_str());
  rmdir(dir);
  return 0;
}
