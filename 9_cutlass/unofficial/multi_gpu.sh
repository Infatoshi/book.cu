#!/bin/bash
#
# ==============================================================================
# Build and Run Script for the "Unofficial" Multi-GPU GEMM Example
#
# This script automates the setup, compilation, and execution of the custom
# multi-GPU GEMM benchmark (`multi_gpu_gemm.cu`). This example demonstrates a
# simple data-parallel approach where each GPU runs an independent GEMM, as
# opposed to a truly distributed GEMM that solves a single problem.
#
# What it does:
# 1.  Detects GPU hardware using `nvidia-smi` to determine the correct SM
#     architecture flag for `nvcc`.
# 2.  Checks for the existence of the CUTLASS and NCCL libraries, which are
#     required dependencies. It will attempt to clone CUTLASS if not found.
# 3.  Compiles the `multi_gpu_gemm.cu` source file using `nvcc`, linking
#     against NCCL and including the necessary CUTLASS headers.
# 4.  Executes the compiled binary, which will run a GEMM benchmark on all
#     available GPUs and report the aggregate performance.
#
# Prerequisites:
# - `nvidia-smi` command must be available.
# - A compatible C++ compiler.
# - A CUDA Toolkit installation.
# - NCCL library (NVIDIA Collective Communications Library) must be installed.
# ==============================================================================

set -e

# --- Color Definitions for Output ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== CUTLASS Multi-GPU GEMM Scaling Test Build & Run ===${NC}\n"

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUTLASS_DIR="${CUTLASS_DIR:-$SCRIPT_DIR/cutlass}"
CUDA_DIR="${CUDA_DIR:-/usr/local/cuda}"
SOURCE_FILE="multi_gpu_gemm.cu"
OUTPUT_BINARY="multi_gpu_gemm"

# --- Step 1: Detect GPU Configuration ---
echo "Step 1: Detecting GPU configuration..."
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
    echo -e "${GREEN}✓${NC} Detected ${NUM_GPUS} GPU(s): ${GPU_NAME} (SM${COMPUTE_CAP})"

    if [ "$NUM_GPUS" -eq 1 ]; then
        echo -e "${YELLOW}⚠${NC}  Only 1 GPU detected. The benchmark will run on a single GPU."
    fi

    # Determine the correct architecture flag based on compute capability.
    if [[ "$COMPUTE_CAP" =~ ^8[069]$ ]]; then # Ampere/Ada
        ARCH_FLAG=${COMPUTE_CAP::1}0 # Use sm_80 for all 8.x
    elif [[ "$COMPUTE_CAP" =~ ^90 ]]; then # Hopper
        ARCH_FLAG="90a"
        ARCH_NAME="Hopper"
    else
        echo -e "${YELLOW}⚠${NC}  Unknown compute capability ${COMPUTE_CAP}, defaulting to sm_80 (Ampere)."
        ARCH_FLAG="80"
    fi
    echo -e "    Targeting architecture: sm_${ARCH_FLAG}\n"
else
    echo -e "${YELLOW}⚠${NC}  nvidia-smi not found. Defaulting to sm_80 (Ampere)."
    ARCH_FLAG="80"
fi

# --- Step 2: Check CUTLASS Installation ---
echo "Step 2: Checking for CUTLASS repository..."
if [ ! -d "$CUTLASS_DIR" ]; then
    echo -e "${YELLOW}⚠${NC}  CUTLASS not found at: $CUTLASS_DIR"
    echo "   Attempting to clone from GitHub..."
    if git clone --depth 1 https://github.com/NVIDIA/cutlass.git "$CUTLASS_DIR"; then
        echo -e "${GREEN}✓${NC} CUTLASS cloned successfully."
    else
        echo -e "${RED}✗${NC} Failed to clone CUTLASS."
        exit 1
    fi
elif [ ! -f "$CUTLASS_DIR/include/cutlass/cutlass.h" ]; then
    echo -e "${RED}✗${NC} CUTLASS directory found, but headers are missing. It may be corrupted."
    exit 1
else
    echo -e "${GREEN}✓${NC} CUTLASS found at: $CUTLASS_DIR"
fi
echo ""

# --- Step 3: Check CUDA Installation ---
echo "Step 3: Checking for CUDA Toolkit..."
if [ ! -d "$CUDA_DIR" ] || [ ! -f "$CUDA_DIR/bin/nvcc" ]; then
    echo -e "${RED}✗${NC} CUDA Toolkit not found at: $CUDA_DIR"
    echo "   Please set the CUDA_DIR environment variable or ensure CUDA is in a standard location."
    exit 1
fi
NVCC="$CUDA_DIR/bin/nvcc"
CUDA_VERSION=$($NVCC --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo -e "${GREEN}✓${NC} CUDA ${CUDA_VERSION} found.\n"

# --- Step 4: Check NCCL Installation ---
echo "Step 4: Checking for NCCL library..."
# Simple check for nccl.h in common locations. A robust build system would be better.
if ! echo '#include <nccl.h>' | gcc -E - >/dev/null 2>&1; then
    echo -e "${RED}✗${NC} NCCL header (nccl.h) not found in standard include paths."
    echo "   Please install NCCL or ensure it's in your compiler's include path."
    echo "   Common paths: /usr/include, /usr/local/cuda/include"
    exit 1
fi
echo -e "${GREEN}✓${NC} NCCL header found.\n"


# --- Step 5: Compile the Source ---
echo "Step 5: Compiling $SOURCE_FILE..."

# Construct the nvcc compilation command.
# -arch=sm_xx: Specifies the target architecture.
# -I...: Adds include paths for CUTLASS and CUDA headers.
# -lnccl: Links against the NCCL library.
COMPILE_CMD=(
    "$NVCC"
    "-O3"
    "-std=c++17"
    "-arch=sm_${ARCH_FLAG}"
    "--use_fast_math"
    "-I${CUTLASS_DIR}/include"
    "-I${CUTLASS_DIR}/tools/util/include"
    "-I${CUDA_DIR}/include"
    "-lnccl"
    "$SOURCE_FILE"
    "-o"
    "$OUTPUT_BINARY"
)

echo "Executing: ${COMPILE_CMD[*]}"
echo ""

# Execute the compilation command.
"${COMPILE_CMD[@]}"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Compilation successful!\n"
else
    echo -e "${RED}✗${NC} Compilation failed!"
    exit 1
fi

# --- Step 6: Run the Benchmark ---
echo "Step 6: Running multi-GPU GEMM scaling benchmark..."
echo "================================================="
./"$OUTPUT_BINARY"

exit_code=$?
echo "================================================="
if [ $exit_code -eq 0 ]; then
    echo -e "\n${GREEN}✓${NC} Execution completed successfully!"
else
    echo -e "\n${RED}✗${NC} Execution failed with exit code $exit_code."
fi

exit $exit_code

