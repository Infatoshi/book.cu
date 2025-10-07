

set -e  

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' 

echo -e "${GREEN}=== CUTLASS Multi-GPU GEMM Build & Run ===${NC}\n"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUTLASS_DIR="${CUTLASS_DIR:-$SCRIPT_DIR/cutlass}"
CUDA_DIR="${CUDA_DIR:-/usr/local/cuda}"
SOURCE_FILE="multi_gpu_gemm.cu"
OUTPUT_BINARY="multi_gpu_gemm"

echo "Step 1: Detecting GPU configuration..."
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
    echo -e "${GREEN}✓${NC} Detected ${NUM_GPUS} GPU(s): ${GPU_NAME} (SM${COMPUTE_CAP})"
    
    if [ "$NUM_GPUS" -eq 1 ]; then
        echo -e "${YELLOW}⚠${NC}  Only 1 GPU detected. Multi-GPU scaling will be limited."
        echo "   The program will still run and test 1-GPU performance."
    fi
    
    
    if [ "$COMPUTE_CAP" == "80" ] || [ "$COMPUTE_CAP" == "86" ]; then
        ARCH_FLAG="80"
        ARCH_NAME="Ampere"
    elif [ "$COMPUTE_CAP" == "89" ]; then
        ARCH_FLAG="89"
        ARCH_NAME="Ada"
    elif [ "$COMPUTE_CAP" == "90" ]; then
        ARCH_FLAG="90"
        ARCH_NAME="Hopper"
    else
        echo -e "${YELLOW}⚠${NC}  Unknown compute capability ${COMPUTE_CAP}, defaulting to sm_80"
        ARCH_FLAG="80"
        ARCH_NAME="Unknown (using Ampere flags)"
    fi
    echo -e "    Architecture: ${ARCH_NAME} (using -arch=sm_${ARCH_FLAG})\n"
else
    echo -e "${YELLOW}⚠${NC}  nvidia-smi not found, defaulting to sm_80 (Ampere)"
    ARCH_FLAG="80"
    NUM_GPUS=1
fi

echo "Step 2: Checking CUTLASS installation..."
if [ ! -d "$CUTLASS_DIR" ]; then
    echo -e "${YELLOW}⚠${NC}  CUTLASS not found at: $CUTLASS_DIR"
    echo "   Cloning CUTLASS repository..."
    
    
    PARENT_DIR=$(dirname "$CUTLASS_DIR")
    CUTLASS_NAME=$(basename "$CUTLASS_DIR")
    
    
    if git clone --depth 1 https:
        echo -e "${GREEN}✓${NC} CUTLASS cloned successfully to: $CUTLASS_DIR"
    else
        echo -e "${RED}✗${NC} Failed to clone CUTLASS"
        echo "   You can manually clone it with:"
        echo "   git clone https:
        exit 1
    fi
elif [ ! -f "$CUTLASS_DIR/include/cutlass/cutlass.h" ]; then
    echo -e "${RED}✗${NC} CUTLASS directory exists but headers not found at: $CUTLASS_DIR/include"
    echo "   The directory may be corrupted. Please remove it and run this script again:"
    echo "   rm -rf $CUTLASS_DIR"
    exit 1
else
    echo -e "${GREEN}✓${NC} CUTLASS found at: $CUTLASS_DIR"
fi
echo ""

echo "Step 3: Checking CUDA installation..."
if [ ! -d "$CUDA_DIR" ]; then
    echo -e "${RED}✗${NC} CUDA not found at: $CUDA_DIR"
    echo "   Set CUDA_DIR environment variable"
    exit 1
fi
NVCC="$CUDA_DIR/bin/nvcc"
if [ ! -f "$NVCC" ]; then
    echo -e "${RED}✗${NC} nvcc not found at: $NVCC"
    exit 1
fi
CUDA_VERSION=$($NVCC --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo -e "${GREEN}✓${NC} CUDA ${CUDA_VERSION} found at: $CUDA_DIR\n"

echo "Step 4: Checking NCCL installation..."
NCCL_FOUND=0
NCCL_INCLUDE=""
NCCL_LIB=""

NCCL_SEARCH_PATHS=(
    "/usr/local/nccl2/include"
    "/usr/include"
    "/usr/local/cuda/include"
    "$CUDA_DIR/include"
    "/opt/nccl/include"
)

for path in "${NCCL_SEARCH_PATHS[@]}"; do
    if [ -f "$path/nccl.h" ]; then
        NCCL_INCLUDE="$path"
        NCCL_FOUND=1
        break
    fi
done

NCCL_LIB_SEARCH_PATHS=(
    "/usr/local/nccl2/lib"
    "/usr/lib/x86_64-linux-gnu"
    "/usr/local/cuda/lib64"
    "$CUDA_DIR/lib64"
    "/opt/nccl/lib"
)

for path in "${NCCL_LIB_SEARCH_PATHS[@]}"; do
    if [ -f "$path/libnccl.so" ]; then
        NCCL_LIB="$path"
        break
    fi
done

if [ $NCCL_FOUND -eq 1 ]; then
    echo -e "${GREEN}✓${NC} NCCL headers found at: $NCCL_INCLUDE"
    if [ -n "$NCCL_LIB" ]; then
        echo -e "${GREEN}✓${NC} NCCL library found at: $NCCL_LIB"
    else
        echo -e "${YELLOW}⚠${NC}  NCCL library path not found, relying on default linker paths"
    fi
else
    echo -e "${RED}✗${NC} NCCL not found!"
    echo "   NCCL is required for multi-GPU communication."
    echo "   Install NCCL:"
    echo "   Ubuntu/Debian: sudo apt install libnccl2 libnccl-dev"
    echo "   Or download from: https:
    echo ""
    echo "   Common NCCL locations checked:"
    for path in "${NCCL_SEARCH_PATHS[@]}"; do
        echo "   - $path"
    done
    exit 1
fi
echo ""

echo "Step 5: Compiling $SOURCE_FILE..."
echo "Command:"
echo "  $NVCC -O3 -std=c++17 -arch=sm_${ARCH_FLAG} \\"
echo "    -I$CUTLASS_DIR/include \\"
echo "    -I$CUTLASS_DIR/tools/util/include \\"
echo "    -I$CUDA_DIR/include \\"
if [ -n "$NCCL_INCLUDE" ]; then
    echo "    -I$NCCL_INCLUDE \\"
fi
if [ -n "$NCCL_LIB" ]; then
    echo "    -L$NCCL_LIB \\"
fi
echo "    -lnccl \\"
echo "    $SOURCE_FILE -o $OUTPUT_BINARY"
echo ""

if [ "$ARCH_FLAG" == "90" ]; then
    GENCODE_FLAGS="-gencode arch=compute_90a,code=sm_90a"
else
    GENCODE_FLAGS="-arch=sm_${ARCH_FLAG}"
fi

COMPILE_CMD="$NVCC -O3 -std=c++17 $GENCODE_FLAGS \
    --use_fast_math \
    -I$CUTLASS_DIR/include \
    -I$CUTLASS_DIR/tools/util/include \
    -I$CUDA_DIR/include"

if [ -n "$NCCL_INCLUDE" ]; then
    COMPILE_CMD="$COMPILE_CMD -I$NCCL_INCLUDE"
fi

if [ -n "$NCCL_LIB" ]; then
    COMPILE_CMD="$COMPILE_CMD -L$NCCL_LIB"
fi

COMPILE_CMD="$COMPILE_CMD -lnccl $SOURCE_FILE -o $OUTPUT_BINARY"

eval $COMPILE_CMD

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Compilation successful!\n"
else
    echo -e "${RED}✗${NC} Compilation failed!"
    exit 1
fi

echo "Step 6: Running multi-GPU GEMM scaling test..."
echo "========================================"
./"$OUTPUT_BINARY"

exit_code=$?
echo "========================================"
if [ $exit_code -eq 0 ]; then
    echo -e "\n${GREEN}✓${NC} Execution completed successfully!"
else
    echo -e "\n${RED}✗${NC} Execution failed with exit code $exit_code"
fi

exit $exit_code

