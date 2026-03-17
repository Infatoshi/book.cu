#!/bin/bash
#
# ==============================================================================
# Build Script for the CUTLASS Blackwell FP4 GEMM Example
#
# This script automates the process of setting up the environment and compiling
# the specific CUTLASS example (72a) that demonstrates FP4 GEMM on the NVIDIA
# Blackwell architecture.
#
# What it does:
# 1. Clones the official NVIDIA CUTLASS repository from GitHub if it doesn't
#    already exist in the current directory.
# 2. Creates a build directory inside the `cutlass` repository.
# 3. Configures the build system using CMake. It specifically targets the
#    NVIDIA Blackwell architecture (SM100a) and enables the compilation
#    of the CUTLASS examples.
# 4. Compiles only the target example (`72a_blackwell_nvfp4_bf16_gemm`) to
#    save time, using all available CPU cores for a parallel build.
# 5. Reports the final location of the compiled binary.
#
# Prerequisites:
# - A compatible C++ compiler (e.g., g++)
# - CMake (version 3.18 or newer)
# - A CUDA Toolkit that supports the Blackwell architecture (e.g., 12.8+)
# ==============================================================================

set -e

echo "=========================================================="
echo "Building CUTLASS FP4 GEMM Example (72a for Blackwell)"
echo "=========================================================="

# --- 1. Clone CUTLASS Repository ---
# Check if the cutlass directory exists. If not, clone it from GitHub.
if [ ! -d "cutlass" ]; then
    echo "Cloning the official NVIDIA CUTLASS repository..."
    git clone https://github.com/NVIDIA/cutlass.git
    # Note: For production use, you might want to check out a specific
    # stable release tag instead of using the main branch.
    # cd cutlass && git checkout v3.6.0 && cd ..
else
    echo "CUTLASS repository already exists. Skipping clone."
fi

# --- 2. Configure and Build ---
echo "Configuring and building the CUTLASS example..."

# Create a build directory inside the cutlass folder.
mkdir -p cutlass/build
cd cutlass/build

# Configure the project with CMake.
# - DCUTLASS_NVCC_ARCHS=100a: Specifies the target GPU architecture. '100a' is
#   used for NVIDIA Blackwell GPUs like the B200. This must match your hardware.
# - DCUTLASS_ENABLE_EXAMPLES=ON: Ensures the example binaries are built.
# - CMAKE_CUDA_ARCHITECTURES=100a: An alternative way to set the target arch.
echo "Configuring CMake for Blackwell (sm_100a)..."
cmake .. \
    -DCUTLASS_NVCC_ARCHS=100a \
    -DCUTLASS_ENABLE_EXAMPLES=ON \
    -DCMAKE_CUDA_ARCHITECTURES="100a"

# Build only the specific example we care about.
# --target: Specifies the name of the build target.
# -j$(nproc): Uses all available processor cores to speed up compilation.
echo "Building example '72a_blackwell_nvfp4_bf16_gemm'..."
cmake --build . --target 72a_blackwell_nvfp4_bf16_gemm -j$(nproc)

# Return to the original directory.
cd ../..

# --- 3. Report Completion ---
echo ""
echo "✅ Build complete!"
echo "   Binary location: cutlass/build/examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm"
echo "   You can now run './verify.sh' or './benchmark.sh'."

