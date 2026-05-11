#include "../src/file_prep.h"

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
  const char* dir = "/mnt/gds2/cwd_test";
  size_t size = 1ull << 30;
  const char* name = "memtier_exp_1g.bin";
  if (argc > 1) name = argv[1];
  if (argc > 2) size = std::strtoull(argv[2], nullptr, 10);

  int rc = memtier_prepare_experiment_file(dir, name, size);
  if (rc != 0) {
    std::printf("prepare failed rc=%d\n", rc);
    return 1;
  }
  std::printf("prepared %s/%s size=%zu\n", dir, name, size);
  return 0;
}
