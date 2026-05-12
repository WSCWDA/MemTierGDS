#ifndef MEMTIER_OBJECT_TYPES_H
#define MEMTIER_OBJECT_TYPES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "memtier/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  MEMTIER_DTYPE_UINT8 = 0,
  MEMTIER_DTYPE_INT32 = 1,
  MEMTIER_DTYPE_FLOAT16 = 2,
  MEMTIER_DTYPE_FLOAT32 = 3
} memtier_dtype_t;

typedef struct {
  const char* path;
  const char* tensor_name;
  uint64_t data_offset;
  size_t nbytes;
  const int64_t* shape;
  int ndim;
  memtier_dtype_t dtype;
} memtier_tensor_desc_t;

typedef struct {
  int device_id;
  memtier_target_t target;
  bool async;
  void* stream;
  bool cache_admit;
  uint32_t hints;
} memtier_tensor_options_t;

typedef struct {
  const char* path;
  uint64_t offset;
  size_t size;
  uint64_t chunk_id;
} memtier_chunk_desc_t;

typedef struct {
  int device_id;
  memtier_target_t target;
  bool async;
  void* stream;
  bool pack_output;
  bool cache_admit;
  uint32_t hints;
} memtier_chunk_options_t;

#ifdef __cplusplus
}
#endif

#endif
