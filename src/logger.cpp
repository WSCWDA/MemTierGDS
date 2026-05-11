#include "memtier_internal.h"

#include <cstdlib>
#include <iostream>

namespace memtier {

void log_message(const char* level, const std::string& msg) {
  static int threshold = []() {
    const char* env = std::getenv("MEMTIER_LOG_LEVEL");
    if (!env) return 1;
    std::string v(env);
    if (v == "ERROR") return 0;
    if (v == "WARN") return 1;
    if (v == "INFO") return 2;
    if (v == "DEBUG") return 3;
    return 1;
  }();
  int lvl = 1;
  std::string l(level);
  if (l == "ERROR") lvl = 0;
  if (l == "INFO") lvl = 2;
  if (l == "DEBUG") lvl = 3;
  if (lvl <= threshold) {
    std::cerr << "[memtier][" << level << "] " << msg << "\n";
  }
}

}  // namespace memtier
