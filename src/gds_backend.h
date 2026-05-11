#ifndef MEMTIER_GDS_BACKEND_H
#define MEMTIER_GDS_BACKEND_H
class GdsBackend { public: explicit GdsBackend(bool e):en_(e){} bool available() const { return false; } bool enabled() const { return en_; } const char* backend_name() const { return "gds_stub"; } private: bool en_; };
#endif
