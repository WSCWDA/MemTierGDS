#pragma once

#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace memtier {

enum class ResidencyState : uint8_t {
  NOT_RESIDENT = 0,
  IN_DRAM = 1,
  IN_HBM = 2,
  LOADING = 3,
};

struct ResidencyEntry {
  uint64_t block_id = 0;
  ResidencyState state = ResidencyState::NOT_RESIDENT;
  void* dram_ptr = nullptr;
  void* device_ptr = nullptr;
  uint64_t last_access_step = 0;
  uint64_t access_count = 0;
  uint32_t ref_count = 0;
};

class ResidencyTable {
 public:
  ResidencyTable() = default;

  ResidencyEntry lookup(uint64_t block_id);
  ResidencyEntry mark_loading(uint64_t block_id);
  ResidencyEntry mark_dram(uint64_t block_id, void* ptr);
  ResidencyEntry mark_hbm(uint64_t block_id, void* ptr);
  ResidencyEntry release(uint64_t block_id);
  void erase(uint64_t block_id);
  void clear();
  size_t size() const;

 private:
  mutable std::mutex mutex_;
  std::unordered_map<uint64_t, ResidencyEntry> entries_;
  uint64_t step_ = 0;
};

}  // namespace memtier
