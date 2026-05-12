import ctypes
import math

from ._ctypes import MEMTIER_ERR_UNSUPPORTED, MEMTIER_OK, MEMTIER_TARGET_CPU, MEMTIER_TARGET_GPU, get_runtime

try:
    import torch
except Exception:
    torch = None

try:
    import numpy as np
except Exception:
    np = None

_DTYPE_SIZE = {"uint8": 1, "int32": 4, "float16": 2, "float32": 4}
_TORCH_DTYPE = {"uint8": getattr(torch, "uint8", None), "int32": getattr(torch, "int32", None), "float16": getattr(torch, "float16", None), "float32": getattr(torch, "float32", None)}


def _numel(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return n


def read_into_tensor(path, offset, tensor, hints=None, cache_admit=True):
    if torch is None or not isinstance(tensor, torch.Tensor):
        raise TypeError("read_into_tensor expects a torch.Tensor")
    if not tensor.is_contiguous():
        raise ValueError("non-contiguous tensor is not supported")
    lib, ctx = get_runtime()
    target = MEMTIER_TARGET_GPU if tensor.is_cuda else MEMTIER_TARGET_CPU
    nbytes = tensor.numel() * tensor.element_size()
    rc = lib.lib.memtier_read(ctx, path.encode(), int(offset), int(nbytes), ctypes.c_void_p(int(tensor.data_ptr())), target)
    if rc != MEMTIER_OK:
        msg = lib.strerror(rc)
        if rc == MEMTIER_ERR_UNSUPPORTED and tensor.is_cuda:
            raise RuntimeError(f"GPU target unsupported: {msg}")
        raise RuntimeError(f"memtier_read failed: {msg}")


def load_tensor(path, offset, nbytes=None, shape=None, dtype="uint8", device="cpu", hints=None):
    if dtype not in _DTYPE_SIZE:
        raise ValueError("dtype must be one of uint8/int32/float16/float32")
    if nbytes is None:
        if shape is None:
            raise ValueError("nbytes is required when shape is None")
        nbytes = _numel(shape) * _DTYPE_SIZE[dtype]

    if torch is not None:
        t_dtype = _TORCH_DTYPE[dtype]
        out_shape = tuple(shape) if shape is not None else (int(nbytes),)
        if shape is None:
            t_dtype = torch.uint8
        t = torch.empty(out_shape, dtype=t_dtype, device=device)
        read_into_tensor(path, offset, t, hints=hints)
        return t

    if device != "cpu":
        raise RuntimeError("PyTorch is not available; only CPU fallback is supported")
    lib, ctx = get_runtime()
    buf = (ctypes.c_ubyte * int(nbytes))()
    rc = lib.lib.memtier_read(ctx, path.encode(), int(offset), int(nbytes), ctypes.cast(buf, ctypes.c_void_p), MEMTIER_TARGET_CPU)
    if rc != MEMTIER_OK:
        raise RuntimeError(f"memtier_read failed: {lib.strerror(rc)}")
    b = bytes(buf)
    if np is None:
        return b
    arr = np.frombuffer(b, dtype=dtype)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr
