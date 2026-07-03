#!/usr/bin/env bash
set -e

echo "Checking CUDA..."
nvidia-smi || true
nvcc --version || true

echo "Checking GDS..."
which gdscheck || true
gdscheck -p || true

echo "Checking nvidia-fs..."
lsmod | grep nvidia_fs || true

echo "Checking libcufile..."
ldconfig -p | grep libcufile || true
