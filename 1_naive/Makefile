# CUDA Naive Neural Network Operations - Makefile
# Chapter 03: Building the Core Operations from Scratch

# Compiler
NVCC = nvcc

# Executables go to bin/: several targets (transpose, gemm, softmax, ...) share their name
# with a source directory, so building them in place failed with "Is a directory".
BIN = bin

# Compiler flags
NVCC_FLAGS = -O2 -I.

# All targets
TARGETS = vector_add matrix_add transpose gemm softmax conv1d conv2d maxpool2d

.PHONY: all run clean help $(TARGETS) $(addprefix run_,$(TARGETS))

# Default target
all: $(TARGETS)

# `make gemm` etc. still work and build bin/gemm
$(TARGETS): %: $(BIN)/%

$(BIN):
	mkdir -p $(BIN)

# Element-wise operations
$(BIN)/vector_add: elementwise/vector_add.cu common.h | $(BIN)
	$(NVCC) $(NVCC_FLAGS) elementwise/vector_add.cu -o $@

$(BIN)/matrix_add: elementwise/matrix_add.cu common.h | $(BIN)
	$(NVCC) $(NVCC_FLAGS) elementwise/matrix_add.cu -o $@

# Matrix operations
$(BIN)/transpose: transpose/transpose.cu common.h | $(BIN)
	$(NVCC) $(NVCC_FLAGS) transpose/transpose.cu -o $@

$(BIN)/gemm: gemm/gemm.cu common.h | $(BIN)
	$(NVCC) $(NVCC_FLAGS) gemm/gemm.cu -o $@

# Neural network operations
$(BIN)/softmax: softmax/softmax.cu common.h | $(BIN)
	$(NVCC) $(NVCC_FLAGS) softmax/softmax.cu -o $@

$(BIN)/conv1d: conv1d/conv1d.cu common.h | $(BIN)
	$(NVCC) $(NVCC_FLAGS) conv1d/conv1d.cu -o $@

$(BIN)/conv2d: conv2d/conv2d.cu common.h | $(BIN)
	$(NVCC) $(NVCC_FLAGS) conv2d/conv2d.cu -o $@

$(BIN)/maxpool2d: maxpool2d/maxpool2d.cu common.h | $(BIN)
	$(NVCC) $(NVCC_FLAGS) maxpool2d/maxpool2d.cu -o $@

# Build and run all examples
run: all
	@echo "=== Running All Naive CUDA Examples ==="
	@echo ""
	@echo "=== Vector Addition ==="
	./$(BIN)/vector_add
	@echo ""
	@echo "=== Matrix Addition ==="
	./$(BIN)/matrix_add
	@echo ""
	@echo "=== Matrix Transpose ==="
	./$(BIN)/transpose
	@echo ""
	@echo "=== GEMM (Matrix Multiplication) ==="
	./$(BIN)/gemm
	@echo ""
	@echo "=== Softmax ==="
	./$(BIN)/softmax
	@echo ""
	@echo "=== 1D Convolution ==="
	./$(BIN)/conv1d
	@echo ""
	@echo "=== 2D Convolution ==="
	./$(BIN)/conv2d
	@echo ""
	@echo "=== 2D Max Pooling ==="
	./$(BIN)/maxpool2d
	@echo ""
	@echo "=== All examples completed! ==="

# Clean build artifacts
clean:
	rm -rf $(BIN)

# Individual run targets
run_vector_add: vector_add
	./$(BIN)/vector_add

run_matrix_add: matrix_add
	./$(BIN)/matrix_add

run_transpose: transpose
	./$(BIN)/transpose

run_gemm: gemm
	./$(BIN)/gemm

run_softmax: softmax
	./$(BIN)/softmax

run_conv1d: conv1d
	./$(BIN)/conv1d

run_conv2d: conv2d
	./$(BIN)/conv2d

run_maxpool2d: maxpool2d
	./$(BIN)/maxpool2d

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
