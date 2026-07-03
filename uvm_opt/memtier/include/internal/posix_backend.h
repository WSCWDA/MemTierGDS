#pragma once
#include "internal/object_table.h"
#include <sys/types.h>
class PosixBackend { public: int openFile(ObjectEntry& obj); ssize_t readToPinnedHost(ObjectEntry& obj, void* host_ptr, size_t length, uint64_t file_offset); int closeFile(ObjectEntry& obj); };
