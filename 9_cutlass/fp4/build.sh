#!/bin/bash
set -e

echo "================================"
echo "Building nvFP4 GEMM (kernel 72a)"
echo "================================"

# Check if cutlass directory exists, if not clone it
if [ ! -d "cutlass" ]; then
    echo "Cloning CUTLASS repository..."
    git clone https://github.com/NVIDIA/cutlass.git
    cd cutlass
    # Checkout a stable version if needed
    # git checkout v3.x
    cd ..
else
    echo "CUTLASS directory already exists"
fi

# Create build directory for cutlass examples
echo "Building CUTLASS example 72a..."

# Set CUDA paths
export PATH=/usr/local/cuda-13.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.0

cd cutlass
mkdir -p build
cd build

# Configure with CMake for SM100 architecture (100a for B200)
cmake .. \
    -DCUTLASS_NVCC_ARCHS=100a \
    -DCUTLASS_ENABLE_EXAMPLES=ON \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=100a

# Build specifically the 72a example
cmake --build . --target 72a_blackwell_nvfp4_bf16_gemm -j$(nproc)

cd ../..

echo ""
echo "Build complete!"
echo "Binary location: cutlass/build/examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm"

