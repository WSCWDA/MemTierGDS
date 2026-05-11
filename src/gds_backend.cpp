#include "gds_backend.h"

#include <fcntl.h>
#include <unistd.h>

#include <mutex>
#include <string>
#include <unordered_map>

#include <cufile.h>

namespace {
std::mutex g_mu;
bool g_inited = false;
bool g_ok = false;
struct HandleEntry { int fd; CUfileHandle_t h; };
std::unordered_map<std::string, HandleEntry> g_handles;

bool init_driver() {
  std::lock_guard<std::mutex> lk(g_mu);
  if (g_inited) return g_ok;
  CUfileError_t st = cuFileDriverOpen();
  g_ok = (st.err == CU_FILE_SUCCESS);
  g_inited = true;
  return g_ok;
}

bool get_handle(const char* path, HandleEntry* out) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto it = g_handles.find(path);
  if (it != g_handles.end()) { *out = it->second; return true; }
  int fd = open(path, O_RDONLY | O_DIRECT);
  if (fd < 0) return false;
  CUfileDescr_t desc{};
  desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  desc.handle.fd = fd;
  CUfileHandle_t h;
  CUfileError_t st = cuFileHandleRegister(&h, &desc);
  if (st.err != CU_FILE_SUCCESS) { close(fd); return false; }
  HandleEntry e{fd, h};
  g_handles[path] = e;
  *out = e;
  return true;
}
}

int memtier_gds_available() { return init_driver() ? 1 : 0; }

int memtier_gds_read(const memtier_options_t& opt, const char* path, uint64_t offset, size_t size, void* dst, size_t* out_bytes) {
  if (!opt.enable_gds) return MEMTIER_ERR_UNSUPPORTED;
  if ((offset % opt.gds_alignment) != 0 || (size % opt.gds_alignment) != 0) return MEMTIER_ERR_GDS;
  if (!init_driver()) return MEMTIER_ERR_GDS;
  HandleEntry he{};
  if (!get_handle(path, &he)) return MEMTIER_ERR_GDS;
  CUfileError_t bst = cuFileBufRegister(dst, size, 0);
  if (bst.err != CU_FILE_SUCCESS) return MEMTIER_ERR_GDS;
  ssize_t n = cuFileRead(he.h, dst, size, offset, 0);
  cuFileBufDeregister(dst);
  if (n < 0 || static_cast<size_t>(n) != size) return MEMTIER_ERR_GDS;
  if (out_bytes) *out_bytes = static_cast<size_t>(n);
  return MEMTIER_OK;
}
