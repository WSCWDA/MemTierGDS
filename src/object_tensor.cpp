#include "memtier/object_api.h"

int memtier_load_tensor(memtier_ctx_t* ctx, const memtier_tensor_desc_t* desc, void* dst,
                        const memtier_tensor_options_t* options) {
  if (!ctx || !desc || !desc->path || desc->nbytes == 0 || !dst) return MEMTIER_ERR_INVALID;

  uint32_t hints = MEMTIER_HINT_SEQUENTIAL;
  if (options) hints = options->hints;
  if (desc->nbytes >= (1u << 20)) hints |= MEMTIER_HINT_SEQUENTIAL;

  memtier_target_t target = options ? options->target : MEMTIER_TARGET_CPU;
  (void)hints;  // semantic hints prepared for runtime policy extensions.
  return memtier_read(ctx, desc->path, desc->data_offset, desc->nbytes, dst, target);
}
