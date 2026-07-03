#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
./build/simple_map_file "${1:-input.bin}"
