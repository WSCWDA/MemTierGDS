#include "residency_table.h"

namespace memtier {

ResidencyEntry ResidencyTable::lookup(uint64_t block_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto& entry = entries_[block_id];
  if (entry.state == ResidencyState::NOT_RESIDENT) {
    entry.block_id = block_id;
  }
  entry.last_access_step = ++step_;
  entry.access_count++;
  entry.ref_count++;
  return entry;
}

ResidencyEntry ResidencyTable::mark_loading(uint64_t block_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto& entry = entries_[block_id];
  entry.block_id = block_id;
  entry.state = ResidencyState::LOADING;
  entry.last_access_step = ++step_;
  return entry;
}

ResidencyEntry ResidencyTable::mark_dram(uint64_t block_id, void* ptr) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto& entry = entries_[block_id];
  entry.block_id = block_id;
  entry.state = ResidencyState::IN_DRAM;
  entry.dram_ptr = ptr;
  entry.last_access_step = ++step_;
  return entry;
}

ResidencyEntry ResidencyTable::mark_hbm(uint64_t block_id, void* ptr) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto& entry = entries_[block_id];
  entry.block_id = block_id;
  entry.state = ResidencyState::IN_HBM;
  entry.device_ptr = ptr;
  entry.last_access_step = ++step_;
  return entry;
}

ResidencyEntry ResidencyTable::release(uint64_t block_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = entries_.find(block_id);
  if (it == entries_.end()) {
    ResidencyEntry entry;
    entry.block_id = block_id;
    return entry;
  }
  if (it->second.ref_count > 0) {
    it->second.ref_count--;
  }
  it->second.last_access_step = ++step_;
  return it->second;
}

void ResidencyTable::erase(uint64_t block_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  entries_.erase(block_id);
}

void ResidencyTable::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  entries_.clear();
}

size_t ResidencyTable::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return entries_.size();
}

}  // namespace memtier
