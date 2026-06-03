#include <cuda_runtime.h>

#include <cstdint>
#include <cstddef>

// Placeholder for future per-row lookup kernels.  Phase 1/2 uses the embedding
// bag launcher in embedding_bag_kernel.cu after memtier_lookup_blocks has made
// all referenced blocks resident in the HBM cache.
extern "C" int launch_embedding_lookup_noop(const uint64_t* ids, size_t num_ids) {
  (void)ids;
  (void)num_ids;
  return 0;
}
