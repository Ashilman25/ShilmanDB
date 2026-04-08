#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD_TYPE="${1:-Debug}"
ENABLE_ML="${2:-OFF}"

cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DSHILMANDB_ENABLE_ML="$ENABLE_ML"

cmake --build build -j "$(sysctl -n hw.logicalcpu)"

echo "Build complete. Run tests with: cd build && ctest --output-on-failure"