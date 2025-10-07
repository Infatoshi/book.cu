#!/bin/bash
set -e

echo "========================================"
echo "nvFP4 GEMM Benchmark"
echo "========================================"

# Set CUDA paths
export PATH=/usr/local/cuda-13.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.0

# Configuration
M=8192
N=8192
K=8192
BATCH=8
WARMUP=5
ITERS=20

# Check if CUTLASS is available
if [ ! -d "cutlass" ]; then
    echo "Error: cutlass directory not found"
    echo "Please run ./build.sh first"
    exit 1
fi

# Set CUTLASS path
CUTLASS_DIR="$(pwd)/cutlass"
CUTLASS_INCLUDE="${CUTLASS_DIR}/include"
CUTLASS_TOOLS="${CUTLASS_DIR}/tools/util/include"

echo ""
echo "Building benchmark..."

# Compile benchmark.cu with SM100 architecture
nvcc benchmark.cu \
    -o benchmark \
    -I${CUTLASS_INCLUDE} \
    -I${CUTLASS_TOOLS} \
    -arch=sm_100a \
    -std=c++17 \
    -O3 \
    -DCUTLASS_ARCH_MMA_SM100_SUPPORTED \
    --expt-relaxed-constexpr \
    -lineinfo

if [ $? -ne 0 ]; then
    echo "Error: Compilation failed"
    exit 1
fi

echo "Build successful!"
echo ""
echo "Running benchmark with configuration:"
echo "  Problem size: ${M} x ${N} x ${K}"
echo "  Batch size: ${BATCH}"
echo "  Warmup iterations: ${WARMUP}"
echo "  Timing iterations: ${ITERS}"
echo ""
echo "Note: Timing excludes data movement (H2D/D2H transfers)"
echo "      Only kernel execution time is measured"
echo ""

# Run the benchmark
./benchmark \
    --m=${M} \
    --n=${N} \
    --k=${K} \
    --batch=${BATCH} \
    --warmup=${WARMUP} \
    --iters=${ITERS}

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Benchmark completed successfully"
else
    echo ""
    echo "✗ Benchmark failed"
    exit 1
fi

