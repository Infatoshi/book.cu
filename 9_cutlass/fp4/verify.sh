#!/bin/bash
#
# ==============================================================================
# Verification Script for the Blackwell FP4 GEMM Example
#
# This script runs the pre-compiled CUTLASS example for FP4 GEMM on the NVIDIA
# Blackwell architecture (SM100). Its primary purpose is to verify the numerical
# correctness of the kernel.
#
# What it does:
# 1. Checks if the compiled binary for the CUTLASS example exists. If not, it
#    prompts the user to run the `build.sh` script first.
# 2. Executes the GEMM kernel with a small problem size (1024x1024x1024).
# 3. The underlying C++ example (`72a_blackwell_nvfp4_bf16_gemm`) contains a
#    built-in verification step that compares the GPU's output against a
#    CPU-based reference implementation.
# 4. The script checks the exit code of the binary to determine if the
#    verification passed or failed and reports the result to the user.
# ==============================================================================

set -e

echo "================================================================"
echo "Verifying Correctness of nvFP4 GEMM (CUTLASS Example 72a)"
echo "================================================================"

# Define the path to the compiled CUTLASS example binary.
# This assumes the CUTLASS repository is located at `cutlass/`.
BINARY="cutlass/build/examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm"

# Check if the binary exists. If not, provide guidance.
if [ ! -f "$BINARY" ]; then
    echo "❌ Error: Binary not found at $BINARY"
    echo "   Please ensure you have cloned the CUTLASS repository and run the ./build.sh script first."
    exit 1
fi

# Run the verification test.
# We use a standard problem size and only one iteration, as we are checking for
# correctness, not performance. The C++ binary will automatically compare its
# result against a CPU reference.
echo ""
echo "Running GEMM with M=1024, N=1024, K=1024..."
echo "The kernel's output will be compared against a CPU reference."
echo ""

# Execute the binary.
$BINARY --m=1024 --n=1024 --k=1024 --iterations=1

# Check the exit code of the last command. A non-zero exit code indicates an error
# or a verification failure from within the C++ application.
echo ""
if [ $? -eq 0 ]; then
    echo "✅ Verification PASSED: GPU and CPU results match within tolerance."
else
    echo "❌ Verification FAILED: GPU and CPU results do not match."
    exit 1
fi

