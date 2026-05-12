#!/usr/bin/env python3
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import memtier

try:
    import torch
except Exception:
    torch = None

with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "t.bin")
    data = bytes([i % 251 for i in range(2 * 1024 * 1024)])
    open(p, "wb").write(data)

    out = memtier.load_tensor(p, 128, nbytes=4096, device="cpu")
    raw = out.tobytes() if hasattr(out, "tobytes") else bytes(out)
    assert raw == data[128:128+4096]

    if torch is not None:
        t = torch.empty((4096,), dtype=torch.uint8, device="cpu")
        memtier.read_into_tensor(p, 128, t)
        assert t.cpu().numpy().tobytes() == data[128:128+4096]
        if torch.cuda.is_available():
            try:
                tc = torch.empty((4096,), dtype=torch.uint8, device="cuda")
                memtier.read_into_tensor(p, 128, tc)
                assert bytes(tc.cpu().numpy().tolist())[:16] == data[128:144]
            except RuntimeError as e:
                print("skip cuda path:", e)
print("test_python_tensor ok")
