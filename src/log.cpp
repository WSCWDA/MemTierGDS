#include "log.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>

static int threshold() {
  const char* e = std::getenv("MEMTIER_LOG_LEVEL");
  if (!e) return 1;
  std::string v(e);
  if (v == "ERROR") return 0;
  if (v == "WARN") return 1;
  if (v == "INFO") return 2;
  if (v == "DEBUG") return 3;
  return 1;
}

void memtier_log(LogLevel level, const char* fmt, ...) {
  if ((int)level > threshold()) return;
  va_list ap;
  va_start(ap, fmt);
  std::vfprintf(stderr, fmt, ap);
  std::fprintf(stderr, "\n");
  va_end(ap);
}
