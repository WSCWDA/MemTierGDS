from memtier import MemTierDataset

if __name__ == "__main__":
    data_file = "py_ds.bin"
    idx_file = "py_ds.idx"
    data = bytes([i % 251 for i in range(8192)])
    with open(data_file, "wb") as f:
        f.write(data)
    with open(idx_file, "w") as f:
        f.write(f"0 {data_file} 0 4096\n")
        f.write(f"1 {data_file} 4096 4096\n")
    ds = MemTierDataset(idx_file)
    assert ds[1] == data[4096:8192]
    print("ok")
