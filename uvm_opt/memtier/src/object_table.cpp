#include "internal/object_table.h"
#include <sys/stat.h>
#include <cstdio>
#include <functional>

int ObjectTable::insert(ObjectEntry entry) { entry.object_id = next_id_++; objects_[entry.object_id] = std::move(entry); return next_id_-1; }
ObjectEntry* ObjectTable::lookupById(int id) { auto it=objects_.find(id); return it==objects_.end()?nullptr:&it->second; }
const ObjectEntry* ObjectTable::lookupById(int id) const { auto it=objects_.find(id); return it==objects_.end()?nullptr:&it->second; }
ObjectEntry* ObjectTable::lookupByPointer(const void* ptr) { auto p=(const char*)ptr; for (auto& kv:objects_) { auto s=(const char*)kv.second.va.start; if (s && p>=s && p<s+kv.second.va.length) return &kv.second; } return nullptr; }
bool ObjectTable::remove(int id) { return objects_.erase(id)>0; }
uint64_t ObjectTable::fileIdForPath(const std::string& path) { struct stat st{}; if (stat(path.c_str(), &st)==0) return (uint64_t(st.st_dev)<<32) ^ uint64_t(st.st_ino); std::fprintf(stderr,"MemTier: stat failed for %s; falling back to path hash for file_id\n", path.c_str()); return std::hash<std::string>{}(path); }
void ObjectTable::print() const { for (const auto& kv:objects_) { const auto& o=kv.second; std::printf("object_id=%d path=%s offset=%llu length=%zu residency=%s last_path=%s fill_count=%llu\n", o.object_id,o.path.c_str(),(unsigned long long)o.file_range.offset,o.file_range.length,memtierResidencyToString(o.residency),memtierPathToString(o.last_path),(unsigned long long)o.fill_count); } }
