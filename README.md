# MemTier (prototype)

MemTier 是用户态 data supply runtime 原型，提供 file-range read 并支持 userspace DRAM cache。

- 不修改 Linux kernel / Page Cache。
- GDS backend 是可选项。
- 无 CUDA/GDS 可运行 CPU-only 构建与测试。

## Build

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

可选参数：

- `-DMEMTIER_ENABLE_CUDA=ON`
- `-DMEMTIER_ENABLE_GDS=ON`
- `-DMEMTIER_STRICT_GDS=ON`

## Example

```bash
./build/memtier_example <file>
```

## Limitations

- 当前仅实现 CPU target 的真实路径；GPU/GDS 为可选/预留。
- read_async 通过线程封装同步 read。
- 第一版只读，不提供写一致性协议。
- Prefetch TO_HBM and async prefetch are currently unsupported in this prototype.
