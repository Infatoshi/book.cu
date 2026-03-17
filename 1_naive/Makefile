# CUDA Naive Neural Network Operations - Makefile
# Chapter 03: Building the Core Operations from Scratch

# Compiler
NVCC = nvcc

OUT_DIR = out
OUTPUT_FILE = -o $(OUT_DIR)/$@

# Compiler flags
NVCC_FLAGS = -O2 -I.

# All targets
TARGETS = vector_add matrix_add transpose gemm softmax conv1d conv2d maxpool2d

# Default target
all: pre-build $(TARGETS)

pre-build:
	@echo "pre build"
	mkdir -p $(OUT_DIR)

# Element-wise operations
vector_add: pre-build elementwise/vector_add.cu common.h
	$(NVCC) $(NVCC_FLAGS) elementwise/vector_add.cu $(OUTPUT_FILE)

matrix_add: pre-build elementwise/matrix_add.cu common.h
	$(NVCC) $(NVCC_FLAGS) elementwise/matrix_add.cu $(OUTPUT_FILE)

# Matrix operations
transpose: pre-build transpose/transpose.cu common.h
	$(NVCC) $(NVCC_FLAGS) transpose/transpose.cu $(OUTPUT_FILE)

gemm: pre-build gemm/gemm.cu common.h
	$(NVCC) $(NVCC_FLAGS) gemm/gemm.cu $(OUTPUT_FILE)

# Neural network operations
softmax: pre-build softmax/softmax.cu common.h
	$(NVCC) $(NVCC_FLAGS) softmax/softmax.cu $(OUTPUT_FILE)

conv1d: pre-build conv1d/conv1d.cu common.h
	$(NVCC) $(NVCC_FLAGS) conv1d/conv1d.cu $(OUTPUT_FILE)

conv2d: pre-build conv2d/conv2d.cu common.h
	$(NVCC) $(NVCC_FLAGS) conv2d/conv2d.cu $(OUTPUT_FILE)

maxpool2d: pre-build maxpool2d/maxpool2d.cu common.h
	$(NVCC) $(NVCC_FLAGS) maxpool2d/maxpool2d.cu $(OUTPUT_FILE)

# Build and run all examples
run: all
	@echo "=== Running All Naive CUDA Examples ==="
	@echo ""
	@echo "=== Vector Addition ==="
	./vector_add
	@echo ""
	@echo "=== Matrix Addition ==="
	./matrix_add
	@echo ""
	@echo "=== Matrix Transpose ==="
	./transpose
	@echo ""
	@echo "=== GEMM (Matrix Multiplication) ==="
	./gemm
	@echo ""
	@echo "=== Softmax ==="
	./softmax
	@echo ""
	@echo "=== 1D Convolution ==="
	./conv1d
	@echo ""
	@echo "=== 2D Convolution ==="
	./conv2d
	@echo ""
	@echo "=== 2D Max Pooling ==="
	./maxpool2d
	@echo ""
	@echo "=== All examples completed! ==="

# Clean build artifacts
clean:
	rm -f $(TARGETS)

# Individual run targets
run_vector_add: vector_add
	./vector_add

run_matrix_add: matrix_add
	./matrix_add

run_transpose: transpose
	./transpose

run_gemm: gemm
	./gemm

run_softmax: softmax
	./softmax

run_conv1d: conv1d
	./conv1d

run_conv2d: conv2d
	./conv2d

run_maxpool2d: maxpool2d
	./maxpool2d

# Show help
help:
	@echo "CUDA Naive Neural Network Operations - Chapter 03"
	@echo ""
	@echo "Available targets:"
	@echo "  all                     - Build all examples"
	@echo "  run                     - Build and run all examples"
	@echo "  clean                   - Remove all compiled executables"
	@echo ""
	@echo "Individual operations:"
	@echo "  vector_add              - 1D vector addition"
	@echo "  matrix_add              - 2D matrix addition"
	@echo "  transpose               - Matrix transposition"
	@echo "  gemm                    - General Matrix Multiplication"
	@echo "  softmax                 - Softmax activation"
	@echo "  conv1d                  - 1D convolution"
	@echo "  conv2d                  - 2D convolution"
	@echo "  maxpool2d               - 2D max pooling"
	@echo ""
	@echo "Run individual examples:"
	@echo "  run_vector_add, run_matrix_add, etc."
	@echo ""
	@echo "Usage: make [target]"
