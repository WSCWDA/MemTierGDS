#include "memtier/object_api.h"

int memtier_load_tensor(memtier_ctx_t* ctx, const memtier_tensor_desc_t* desc, void* dst,
                        const memtier_tensor_options_t* options) {
  if (!ctx || !desc || !desc->path || desc->nbytes == 0 || !dst) return MEMTIER_ERR_INVALID;
  if (desc->ndim < 0) return MEMTIER_ERR_INVALID;
  if (desc->ndim > 0 && desc->shape == nullptr) return MEMTIER_ERR_INVALID;

  memtier_tensor_options_t default_opt{};
  default_opt.device_id = 0;
  default_opt.target = MEMTIER_TARGET_CPU;
  default_opt.async = false;
  default_opt.stream = nullptr;
  default_opt.cache_admit = true;
  default_opt.hints = MEMTIER_HINT_SEQUENTIAL;
  const auto* opt = options ? options : &default_opt;

  uint32_t hints = opt->hints;
  if (desc->nbytes >= (1u << 20)) hints |= MEMTIER_HINT_SEQUENTIAL;
  (void)hints;
  return memtier_read(ctx, desc->path, desc->data_offset, desc->nbytes, dst, opt->target);
}
