#include "internal/residency_table.h"
void ResidencyTable::setResidency(int id, MemTierResidency r){ r_[id]=r; }
MemTierResidency ResidencyTable::getResidency(int id) const { auto it=r_.find(id); return it==r_.end()?MemTierResidency::SSD_ONLY:it->second; }
bool ResidencyTable::isGpuResident(int id) const { auto r=getResidency(id); return r==MemTierResidency::HBM_RESIDENT || r==MemTierResidency::DRAM_AND_HBM; }
bool ResidencyTable::isDramResident(int id) const { auto r=getResidency(id); return r==MemTierResidency::DRAM_RESIDENT || r==MemTierResidency::DRAM_AND_HBM; }
void ResidencyTable::remove(int id){ r_.erase(id); }
