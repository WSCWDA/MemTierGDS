class MemTierDataset:
    def __init__(self, index_file, device="cpu", cache_policy="adaptive", to_tensor=False):
        self.device = device
        self.cache_policy = cache_policy
        self.to_tensor = to_tensor
        self.samples = []
        with open(index_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sid, path, off, size = line.split()
                self.samples.append((int(sid), path, int(off), int(size)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, path, off, size = self.samples[idx]
        with open(path, "rb") as f:
            f.seek(off)
            b = f.read(size)
        return b
