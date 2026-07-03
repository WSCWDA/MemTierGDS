#pragma once
#include "memtier_types.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct FileRange { uint64_t file_id = 0; uint64_t offset = 0; size_t length = 0; bool operator==(const FileRange& o) const { return file_id==o.file_id && offset==o.offset && length==o.length; } };
struct VaRange { void* start = nullptr; size_t length = 0; };
struct ObjectEntry {
    int object_id = -1; VaRange va; FileRange file_range; std::string path; int fd = -1; unsigned flags = 0;
    void* managed_ptr = nullptr; void* device_ptr = nullptr; void* host_pinned_ptr = nullptr;
    MemTierResidency residency = MemTierResidency::SSD_ONLY; MemTierPath last_path = MemTierPath::POSIX_TO_DRAM;
    uint64_t reuse_count = 0; uint64_t fill_count = 0; uint64_t last_access_epoch = 0;
    bool gds_registered = false; bool file_registered = false;
};
class ObjectTable {
public:
    int insert(ObjectEntry entry); ObjectEntry* lookupById(int object_id); const ObjectEntry* lookupById(int object_id) const;
    ObjectEntry* lookupByPointer(const void* ptr); bool remove(int object_id); void print() const;
    static uint64_t fileIdForPath(const std::string& path);
private:
    int next_id_ = 1; std::unordered_map<int,ObjectEntry> objects_;
};
