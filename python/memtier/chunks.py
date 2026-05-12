def load_chunks(path, offsets, lengths, device="cpu", hints=None):
    if device.startswith("cuda"):
        raise RuntimeError("CUDA device requested but Python wrapper currently supports CPU-only reads")
    if len(offsets) != len(lengths):
        raise ValueError("offsets and lengths must have the same length")
    outs = []
    with open(path, "rb") as f:
        for off, ln in zip(offsets, lengths):
            f.seek(off)
            b = f.read(ln)
            if len(b) != ln:
                raise RuntimeError("short read while loading chunks")
            outs.append(b)
    return outs
