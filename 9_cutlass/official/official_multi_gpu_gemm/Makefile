# Makefile for Standalone Distributed GEMM

NVCC := nvcc
CUDA_ARCH := -gencode=arch=compute_90a,code=sm_90a
CUTLASS_DIR := ../../cutlass

NVCC_FLAGS := -std=c++17 -O3 --use_fast_math $(CUDA_ARCH) \
              --expt-relaxed-constexpr \
              -I$(CUTLASS_DIR)/include \
              -I$(CUTLASS_DIR)/tools/util/include \
              -I$(CUTLASS_DIR)/examples \
              -DCUTLASS_ENABLE_GDC_FOR_SM90=1

TARGET := standalone_dist_gemm
SOURCE := standalone_dist_gemm.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	@echo "Compiling $(TARGET)..."
	$(NVCC) $(NVCC_FLAGS) $(SOURCE) -o $(TARGET)
	@echo "Build complete: ./$(TARGET)"

clean:
	rm -f $(TARGET)

.PHONY: all clean

