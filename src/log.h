#ifndef MEMTIER_LOG_H
#define MEMTIER_LOG_H

enum class LogLevel { ERROR = 0, WARN = 1, INFO = 2, DEBUG = 3 };
void memtier_log(LogLevel level, const char* fmt, ...);

#endif
