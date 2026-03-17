#!/bin/bash
#
# ==============================================================================
# Performance Benchmark Script for the Custom FP4 GEMM Kernel
#
# This script compiles and runs the custom `benchmark.cu` file, which contains
# a CUTLASS kernel specifically configured for FP4 GEMM on NVIDIA Blackwell
# architecture (SM100+).
#
# What it does:
# 1. Sets benchmark parameters (matrix dimensions, iterations).
# 2. Checks for the existence of the `cutlass` repository, which is a
#    prerequisite for compiling the kernel.
# 3. Compiles `benchmark.cu` using `nvcc`. It includes the necessary CUTLASS
#    header paths and sets the target architecture to `sm_100a`.
# 4. Executes the compiled binary with the specified problem dimensions.
# 5. The C++ application measures and reports the kernel's performance in TFLOPS.
#
# Prerequisites:
# - A CUDA Toolkit that supports the Blackwell architecture (e.g., 12.8+).
# - The `cutlass` repository must be present in the parent directory, which
#   can be set up by running the `build.sh` script.
# ==============================================================================

set -e

echo "======================================================"
echo "Compiling and Running Custom nvFP4 GEMM Benchmark"
echo "======================================================"

# --- 1. Benchmark Configuration ---
# Define the problem size and iteration counts for the benchmark.
M=8192
N=8192
K=8192
BATCH=8
WARMUP=5
ITERS=20

# --- 2. Check for CUTLASS Directory ---
# The compilation requires access to the CUTLASS header files.
if [ ! -d "cutlass" ]; then
    echo "❌ Error: 'cutlass' directory not found."
    echo "   Please run the main './build.sh' script first to clone the repository."
    exit 1
fi

# --- 3. Compile the Custom Benchmark Kernel ---
echo ""
echo "Building custom benchmark kernel (benchmark.cu)..."

# Define paths to the required CUTLASS include directories.
CUTLASS_DIR="$(pwd)/cutlass"
CUTLASS_INCLUDE="${CUTLASS_DIR}/include"
CUTLASS_TOOLS="${CUTLASS_DIR}/tools/util/include"
CUTLASS_EXAMPLES="${CUTLASS_DIR}/examples"

# Compile benchmark.cu using nvcc.
# -arch=sm_100a: Targets the NVIDIA Blackwell architecture.
# -DCUTLASS_ARCH_MMA_SM100_SUPPORTED: Enables SM100-specific MMA features.
nvcc benchmark.cu \
    -o benchmark \
    -I${CUTLASS_INCLUDE} \
    -I${CUTLASS_TOOLS} \
    -I${CUTLASS_EXAMPLES} \
    -arch=sm_100a \
    -std=c++17 \
    -O3 \
    -DCUTLASS_ARCH_MMA_SM100_SUPPORTED

if [ $? -ne 0 ]; then
    echo "❌ Error: Compilation failed."
    exit 1
fi

# --- 4. Run the Benchmark ---
echo "✅ Build successful!"
echo ""
echo "Running benchmark with configuration:"
echo "  Problem size: ${M} x ${N} x ${K}"
echo "  Batch size: ${BATCH}"
echo "  Warmup iterations: ${WARMUP}"
echo "  Timing iterations: ${ITERS}"
echo ""
echo "Note: Timing measures only kernel execution time, excluding data transfers."
echo ""

# Execute the compiled benchmark binary with the configured parameters.
./benchmark \
    --m=${M} \
    --n=${N} \
    --k=${K} \
    --batch=${BATCH} \
    --warmup=${WARMUP} \
    --iters=${ITERS}

# --- 5. Report Completion ---
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Benchmark completed successfully."
else
    echo ""
    echo "❌ Benchmark failed during execution."
    exit 1
fi

