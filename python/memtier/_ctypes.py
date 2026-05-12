import ctypes
import os

MEMTIER_OK = 0
MEMTIER_TARGET_CPU = 0
MEMTIER_TARGET_GPU = 1
MEMTIER_ERR_UNSUPPORTED = -4

HINTS = {
    "reuse": 1 << 0,
    "small_io": 1 << 1,
    "random": 1 << 2,
    "streaming": 1 << 3,
    "checkpoint": 1 << 4,
    "sequential": 1 << 5,
    "latency_sensitive": 1 << 6,
}

class _Lib:
    def __init__(self):
        p = os.environ.get("MEMTIER_LIBRARY_PATH") or os.environ.get("MEMTIER_LIB") or "libmemtier.so"
        try:
            self.lib = ctypes.CDLL(p)
        except OSError as e:
            raise RuntimeError(f"Cannot load libmemtier from '{p}'. Set MEMTIER_LIBRARY_PATH or LD_LIBRARY_PATH. {e}")
        self._bind()

    def _bind(self):
        L = self.lib
        L.memtier_init.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        L.memtier_init.restype = ctypes.c_int
        L.memtier_finalize.argtypes = [ctypes.c_void_p]
        L.memtier_finalize.restype = ctypes.c_int
        L.memtier_read.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int]
        L.memtier_read.restype = ctypes.c_int
        L.memtier_strerror.argtypes = [ctypes.c_int]
        L.memtier_strerror.restype = ctypes.c_char_p

    def strerror(self, code):
        return self.lib.memtier_strerror(code).decode("utf-8")

_lib = None
_ctx = None

def get_runtime():
    global _lib, _ctx
    if _lib is None:
        _lib = _Lib()
    if _ctx is None:
        out = ctypes.c_void_p()
        rc = _lib.lib.memtier_init(None, ctypes.byref(out))
        if rc != MEMTIER_OK:
            raise RuntimeError(f"memtier_init failed: {_lib.strerror(rc)}")
        _ctx = out
    return _lib, _ctx

def close_runtime():
    global _ctx
    if _ctx is not None:
        _lib.lib.memtier_finalize(_ctx)
        _ctx = None

def hints_to_mask(hints):
    if not hints:
        return 0
    m = 0
    for h in hints:
        k = h.lower()
        if k not in HINTS:
            raise ValueError(f"unknown hint: {h}")
        m |= HINTS[k]
    return m
