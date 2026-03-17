# CUDA Vector Addition Examples - Makefile
# Chapter 02: GPU Memory Management and Kernel Launch

# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -O2

# Targets
TARGETS = vecadd vecadd_scalable tensor_add_3d

# Default target
all: $(TARGETS)

# Basic vector addition example
vecadd: vecadd.cu
	$(NVCC) $(NVCC_FLAGS) vecadd.cu -o vecadd

# Scalable vector addition with boundary checks
vecadd_scalable: vecadd_scalable.cu
	$(NVCC) $(NVCC_FLAGS) vecadd_scalable.cu -o vecadd_scalable

# 3D tensor addition example
tensor_add_3d: tensor_add_3d.cu
	$(NVCC) $(NVCC_FLAGS) tensor_add_3d.cu -o tensor_add_3d

# Clean build artifacts
clean:
	rm -f $(TARGETS)

# Run all examples
run: all
	@echo "=== Running Basic Vector Addition ==="
	./vecadd
	@echo ""
	@echo "=== Running Scalable Vector Addition ==="
	./vecadd_scalable
	@echo ""
	@echo "=== Running 3D Tensor Addition ==="
	./tensor_add_3d

# Show help
help:
	@echo "CUDA Vector Addition Examples - Chapter 02"
	@echo ""
	@echo "Available targets:"
	@echo "  all             - Build all examples"
	@echo "  vecadd          - Basic 8-element vector addition"
	@echo "  vecadd_scalable - Scalable vector addition (1M elements)"
	@echo "  tensor_add_3d   - 3D tensor addition example"
	@echo "  clean           - Remove all compiled executables"
	@echo "  run             - Build and run all examples"
	@echo "  help            - Show this help message"
	@echo ""
	@echo "Usage: make [target]"
