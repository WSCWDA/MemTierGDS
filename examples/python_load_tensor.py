#!/usr/bin/env python3
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import memtier

with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "py_tensor.bin")
    data = bytes([i % 251 for i in range(1024 * 1024)])
    open(path, "wb").write(data)
    out = memtier.load_tensor(path, 256, nbytes=4096, device="cpu")
    shape = getattr(out, "shape", (len(out),))
    raw = out.tobytes() if hasattr(out, "tobytes") else bytes(out)
    print(type(out), shape, list(raw[:8]))
    assert raw == data[256:256+4096]
print("python_load_tensor ok")
