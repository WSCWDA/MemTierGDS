from memtier import load_chunks

if __name__ == "__main__":
    path = "py_chunks.bin"
    data = bytes([i % 251 for i in range(65536)])
    with open(path, "wb") as f:
        f.write(data)
    outs = load_chunks(path, [0, 4096, 8192], [1024, 1024, 1024])
    assert outs[1] == data[4096:5120]
    print("ok")
