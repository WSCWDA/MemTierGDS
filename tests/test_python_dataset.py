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
    dataf = os.path.join(td, "train.bin")
    idxf = os.path.join(td, "train.idx")
    data = bytes([i % 251 for i in range(8192)])
    open(dataf, "wb").write(data)
    with open(idxf, "w") as f:
        f.write("0 0 4096 3\n")
        f.write("1 4096 4096 7\n")

    ds = memtier.MemTierDataset(index_file=idxf, data_file=dataf, target="cpu", return_label=True)
    assert len(ds) == 2
    sample, label = ds[0]
    raw = sample.tobytes() if hasattr(sample, "tobytes") else bytes(sample)
    assert raw == data[:4096] and label == 3

    if torch is not None:
        loader = DataLoader(ds, batch_size=2)
        _ = next(iter(loader))
print("test_python_dataset ok")
