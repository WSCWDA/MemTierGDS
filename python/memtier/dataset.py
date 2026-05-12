from .tensor import load_tensor

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
except Exception:
    torch = None
    _TorchDataset = object


class MemTierDataset(_TorchDataset):
    def __init__(self, index_file, data_file=None, target="cpu", dtype="uint8", shape=None,
                 transform=None, cache_policy="adaptive", hints=None, return_label=True):
        self.index_file = index_file
        self.data_file = data_file
        self.target = target
        self.dtype = dtype
        self.shape = shape
        self.transform = transform
        self.cache_policy = cache_policy
        self.return_label = return_label
        self.hints = sorted(set((hints or []) + ["reuse", "random"]))
        self.samples = []
        self._parse_index()

    def _parse_index(self):
        with open(self.index_file, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 5:
                    sid, path, off, size, label = parts
                elif len(parts) == 4:
                    if not self.data_file:
                        raise ValueError(f"line {lineno}: format-B requires data_file")
                    sid, off, size, label = parts
                    path = self.data_file
                else:
                    raise ValueError(f"line {lineno}: invalid format")
                off, size = int(off), int(size)
                if off < 0 or size <= 0:
                    raise ValueError(f"line {lineno}: invalid offset/size")
                self.samples.append((int(sid), path, off, size, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, path, off, size, label = self.samples[idx]
        sample = load_tensor(path=path, offset=off, nbytes=size if self.shape is None else None,
                             shape=self.shape, dtype=self.dtype, device=self.target, hints=self.hints)
        if self.transform is not None:
            sample = self.transform(sample)
        return (sample, label) if self.return_label else sample
