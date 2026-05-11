import ctypes
import os

try:
    import numpy as np
except Exception:
    np = None


def _load_lib():
    lib = os.environ.get("MEMTIER_LIB", "libmemtier.so")
    try:
        return ctypes.CDLL(lib)
    except OSError as e:
        raise RuntimeError(f"cannot load MemTier shared library '{lib}': {e}")


def load_tensor(path, offset, nbytes, shape=None, dtype="uint8", device="cpu", hints=None):
    if device.startswith("cuda"):
        raise RuntimeError("CUDA device requested but Python wrapper currently supports CPU-only reads")
    lib = _load_lib()
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read(nbytes)
    if len(data) != nbytes:
        raise RuntimeError("short read from tensor payload")
    if np is None:
        return data
    arr = np.frombuffer(data, dtype=dtype)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr
