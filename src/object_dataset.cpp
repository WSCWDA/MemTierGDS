#include "memtier/object_api.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

MemTierDataset::MemTierDataset(const std::string& index_file) {
  std::ifstream in(index_file);
  if (!in) throw std::runtime_error("failed to open index file: " + index_file);

  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::istringstream iss(line);
    SampleRef s{};
    if (!(iss >> s.sample_id >> s.path >> s.offset >> s.size)) {
      throw std::runtime_error("invalid index line: " + line);
    }
    samples_.push_back(std::move(s));
  }
}

size_t MemTierDataset::size() const { return samples_.size(); }

const MemTierDataset::SampleRef& MemTierDataset::sample(size_t idx) const {
  if (idx >= samples_.size()) throw std::out_of_range("dataset index out of range");
  return samples_[idx];
}

int MemTierDataset::read_sample(memtier_ctx_t* ctx, size_t idx, void* dst,
                                const memtier_read_options_t* options) const {
  if (!ctx || !dst || idx >= samples_.size()) return MEMTIER_ERR_INVALID;
  const auto& s = samples_[idx];
  memtier_target_t target = options ? options->target : MEMTIER_TARGET_CPU;
  return memtier_read(ctx, s.path.c_str(), s.offset, s.size, dst, target);
}
