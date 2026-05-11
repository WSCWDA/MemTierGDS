from memtier import load_tensor

if __name__ == "__main__":
    path = "py_tensor.bin"
    data = bytes([i % 251 for i in range(1024 * 1024)])
    with open(path, "wb") as f:
        f.write(data)
    out = load_tensor(path, 128, 4096)
    raw = out.tobytes() if hasattr(out, "tobytes") else out
    assert raw == data[128:128+4096]
    print("ok")
