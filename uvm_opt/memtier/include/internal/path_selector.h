#pragma once
#include "internal/object_table.h"
struct RuntimeCapability { bool gds_available=false; bool posix_available=true; int device_id=0; };
MemTierPath selectPath(const ObjectEntry& obj, const RuntimeCapability& cap);
