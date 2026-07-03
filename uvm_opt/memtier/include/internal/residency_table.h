#pragma once
#include "memtier_types.h"
#include <unordered_map>
class ResidencyTable { public: void setResidency(int object_id, MemTierResidency r); MemTierResidency getResidency(int object_id) const; bool isGpuResident(int object_id) const; bool isDramResident(int object_id) const; void remove(int object_id); private: std::unordered_map<int,MemTierResidency> r_; };
