#!/usr/bin/env python3
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import memtier

try:
    import torch
    from torch.utils.data import DataLoader
except Exception:
    torch = None

with tempfile.TemporaryDirectory() as td:
    data_file = os.path.join(td, "train.bin")
    idx_file = os.path.join(td, "train.idx")
    data = bytes([i % 251 for i in range(3 * 4096)])
    open(data_file, "wb").write(data)
    with open(idx_file, "w") as f:
        for i in range(3):
            f.write(f"{i} {i*4096} 4096 {i}\n")

    ds = memtier.MemTierDataset(index_file=idx_file, data_file=data_file, target="cpu")
    for i in range(3):
        s, l = ds[i]
        raw = s.tobytes() if hasattr(s, "tobytes") else bytes(s)
        print(i, l, list(raw[:4]))

    if torch is not None:
        dl = DataLoader(ds, batch_size=2)
        batch = next(iter(dl))
        print("dataloader batch ok", type(batch))
print("python_dataset_demo ok")
