# ==============================================================================
# Makefile for the Standalone Distributed GEMM Example
#
# This Makefile automates the compilation of the `standalone_dist_gemm.cu`
# source file, which demonstrates a multi-GPU distributed GEMM using CUTLASS.
#
# Targets:
#   - `all`: (Default) Compiles the source code to create the executable binary.
#   - `clean`: Removes the compiled binary.
# ==============================================================================

# Compiler to use.
NVCC := nvcc

# Target GPU architecture.
# -gencode=arch=compute_90a,code=sm_90a specifies compilation for the NVIDIA
# Hopper architecture (Compute Capability 9.0a), which is required for the
# advanced features used in this example, such as GDS (GPU Direct Storage).
CUDA_ARCH := -gencode=arch=compute_90a,code=sm_90a

# Path to the CUTLASS repository. Assumes it's located two directories up.
CUTLASS_DIR := ../../cutlass

# Compiler flags for nvcc.
# -std=c++17: Use the C++17 standard.
# -O3: Enable high level of optimization.
# --use_fast_math: Enables faster, less precise math functions.
# --expt-relaxed-constexpr: Relaxes some C++11 constexpr restrictions.
# -I...: Include paths for CUTLASS library headers.
# -DCUTLASS_ENABLE_GDC_FOR_SM90=1: A specific flag required by CUTLASS
#   for enabling GPU Direct Communication features on SM90.
NVCC_FLAGS := -std=c++17 -O3 --use_fast_math $(CUDA_ARCH) \
              --expt-relaxed-constexpr \
              -I$(CUTLASS_DIR)/include \
              -I$(CUTLASS_DIR)/tools/util/include \
              -I$(CUTLASS_DIR)/examples \
              -DCUTLASS_ENABLE_GDC_FOR_SM90=1

# Define the target executable name and the source file.
TARGET := standalone_dist_gemm
SOURCE := standalone_dist_gemm.cu

# Default target: builds the executable.
all: $(TARGET)

# Rule for building the target executable from the source file.
$(TARGET): $(SOURCE)
	@echo "Compiling Standalone Distributed GEMM for Hopper (SM90a)..."
	$(NVCC) $(NVCC_FLAGS) $(SOURCE) -o $(TARGET)
	@echo "Build complete. You can run the benchmark with: ./$(TARGET)"

# Target to clean up build artifacts.
clean:
	rm -f $(TARGET)

# Declare 'all' and 'clean' as phony targets. This tells Make that these are
# not actual files to be built, but just names for commands to be executed.
.PHONY: all clean

