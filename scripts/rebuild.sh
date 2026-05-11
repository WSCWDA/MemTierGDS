#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-${ROOT_DIR}/build}"
CONFIG_ARGS=("${@:2}")

echo "[memtier] root: ${ROOT_DIR}"
echo "[memtier] build dir: ${BUILD_DIR}"

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" "${CONFIG_ARGS[@]}"
cmake --build "${BUILD_DIR}" -j
ctest --test-dir "${BUILD_DIR}" --output-on-failure

echo "[memtier] rebuild complete"
