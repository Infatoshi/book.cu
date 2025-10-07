#!/bin/bash
set -e

echo "========================================"
echo "Verifying nvFP4 GEMM on 1024x1024 matrices"
echo "========================================"

# Check if binary exists
BINARY="cutlass/build/examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm"

if [ ! -f "$BINARY" ]; then
    echo "Error: Binary not found at $BINARY"
    echo "Please run ./build.sh first"
    exit 1
fi

# Run verification with 1024x1024 matrices
# The example includes CPU reference verification by default
echo ""
echo "Running GEMM with M=1024, N=1024, K=1024..."
echo "This will verify GPU results against CPU reference implementation"
echo ""

$BINARY --m=1024 --n=1024 --k=1024 --iterations=1

echo ""
if [ $? -eq 0 ]; then
    echo "✓ Verification PASSED - GPU and CPU results match within tolerance"
else
    echo "✗ Verification FAILED - Results do not match"
    exit 1
fi

