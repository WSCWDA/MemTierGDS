#include "memtier/object_api.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

MemTierDataset::MemTierDataset(memtier_ctx_t* ctx, const std::string& index_file) : ctx_(ctx) {
  if (!ctx_) throw std::runtime_error("null memtier ctx");
  std::ifstream in(index_file);
  if (!in) throw std::runtime_error("failed to open index file: " + index_file);

  std::string line;
  size_t lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    if (line.empty()) continue;
    std::istringstream iss(line);
    SampleRef s{};
    s.label = -1;
    if (!(iss >> s.sample_id >> s.path >> s.offset >> s.size)) {
      throw std::runtime_error("invalid index line " + std::to_string(lineno));
    }
    iss >> s.label;
    samples_.push_back(std::move(s));
  }
}

size_t MemTierDataset::size() const { return samples_.size(); }

const MemTierDataset::SampleRef& MemTierDataset::sample(size_t idx) const {
  if (idx >= samples_.size()) throw std::out_of_range("dataset index out of range");
  return samples_[idx];
}

int MemTierDataset::read_sample(size_t idx, void* dst, const memtier_read_options_t* options) const {
  if (!ctx_ || !dst || idx >= samples_.size()) return MEMTIER_ERR_INVALID;
  const auto& s = samples_[idx];
  memtier_target_t target = options ? options->target : MEMTIER_TARGET_CPU;
  return memtier_read(ctx_, s.path.c_str(), s.offset, s.size, dst, target);
}
