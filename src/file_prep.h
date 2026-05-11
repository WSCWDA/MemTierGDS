#ifndef MEMTIER_FILE_PREP_H
#define MEMTIER_FILE_PREP_H

#include <cstddef>

int memtier_prepare_experiment_file(const char* dir, const char* name, size_t size_bytes);

#endif
