#!/bin/bash
set -euo pipefail

LIBTORCH_DIR="$(cd "$(dirname "$0")/.." && pwd)/third_party/libtorch"

if [ -d "$LIBTORCH_DIR" ] && [ -f "$LIBTORCH_DIR/share/cmake/Torch/TorchConfig.cmake" ]; then
    echo "LibTorch already installed at $LIBTORCH_DIR"
    exit 0
fi

echo "Downloading LibTorch for macOS ARM..."
cd /tmp

curl -L -o libtorch.zip "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.2.0.zip"
unzip -o libtorch.zip

rm -rf "$LIBTORCH_DIR"
mv libtorch "$LIBTORCH_DIR"
rm libtorch.zip

echo "LibTorch installed to $LIBTORCH_DIR"