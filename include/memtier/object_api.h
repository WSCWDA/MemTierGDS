#ifndef MEMTIER_OBJECT_API_H
#define MEMTIER_OBJECT_API_H

#include "memtier/memtier.h"
#include "memtier/object_types.h"

#ifdef __cplusplus
extern "C" {
#endif

int memtier_load_tensor(memtier_ctx_t* ctx, const memtier_tensor_desc_t* desc, void* dst,
                        const memtier_tensor_options_t* options);

int memtier_load_chunks(memtier_ctx_t* ctx, const memtier_chunk_desc_t* chunks, size_t nchunks,
                        void** dsts, const memtier_chunk_options_t* options);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <string>
#include <vector>

class MemTierDataset {
 public:
  struct SampleRef {
    uint64_t sample_id;
    std::string path;
    uint64_t offset;
    size_t size;
  };

  explicit MemTierDataset(const std::string& index_file);
  size_t size() const;
  int read_sample(memtier_ctx_t* ctx, size_t idx, void* dst,
                  const memtier_read_options_t* options = nullptr) const;
  const SampleRef& sample(size_t idx) const;

 private:
  std::vector<SampleRef> samples_;
};
#endif

#endif
