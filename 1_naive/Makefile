# CUDA Naive Neural Network Operations - Makefile
# Chapter 03: Building the Core Operations from Scratch

# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -O2 -I.

TARGETS = elementwise/vector_add elementwise/matrix_add transpose/transpose \
          gemm/gemm softmax/softmax conv1d/conv1d conv2d/conv2d maxpool2d/maxpool2d

all: $(TARGETS)

$(TARGETS): %: %.cu common.h
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Keep the short names available as command-line targets.
vector_add: elementwise/vector_add
matrix_add: elementwise/matrix_add
transpose: transpose/transpose
gemm: gemm/gemm
softmax: softmax/softmax
conv1d: conv1d/conv1d
conv2d: conv2d/conv2d
maxpool2d: maxpool2d/maxpool2d

.PHONY: all run clean help vector_add matrix_add transpose gemm softmax \
        conv1d conv2d maxpool2d run_vector_add run_matrix_add run_transpose \
        run_gemm run_softmax run_conv1d run_conv2d run_maxpool2d

# Build and run all examples
run: all
	@echo "=== Running All Naive CUDA Examples ==="
	@echo ""
	@echo "=== Vector Addition ==="
	./elementwise/vector_add
	@echo ""
	@echo "=== Matrix Addition ==="
	./elementwise/matrix_add
	@echo ""
	@echo "=== Matrix Transpose ==="
	./transpose/transpose
	@echo ""
	@echo "=== GEMM (Matrix Multiplication) ==="
	./gemm/gemm
	@echo ""
	@echo "=== Softmax ==="
	./softmax/softmax
	@echo ""
	@echo "=== 1D Convolution ==="
	./conv1d/conv1d
	@echo ""
	@echo "=== 2D Convolution ==="
	./conv2d/conv2d
	@echo ""
	@echo "=== 2D Max Pooling ==="
	./maxpool2d/maxpool2d
	@echo ""
	@echo "=== All examples completed! ==="

# Clean build artifacts
clean:
	rm -f $(TARGETS)

# Individual run targets
run_vector_add: vector_add
	./elementwise/vector_add

run_matrix_add: matrix_add
	./elementwise/matrix_add

run_transpose: transpose
	./transpose/transpose

run_gemm: gemm
	./gemm/gemm

run_softmax: softmax
	./softmax/softmax

run_conv1d: conv1d
	./conv1d/conv1d

run_conv2d: conv2d
	./conv2d/conv2d

run_maxpool2d: maxpool2d
	./maxpool2d/maxpool2d

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
