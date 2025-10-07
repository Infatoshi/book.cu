# Makefile for Pipeline Parallel MLP Inference
# Unified implementation showing both naive and stream-based approaches

# Compiler and flags
NVCC = nvcc
ARCH ?= sm_90  # Default to H100, override with make ARCH=sm_80 for A100, etc.
CFLAGS = -O3 -arch=$(ARCH) -std=c++11
LDFLAGS = -lcublas

# Target
TARGET = pipeline

# Default target
all: $(TARGET)

# Build rule
pipeline: pipeline.cu
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)
	@echo "✓ Built $@ for architecture $(ARCH)"

# Clean build artifacts
clean:
	rm -f $(TARGET)
	@echo "✓ Cleaned build artifacts"

# Architecture-specific builds
h100: ARCH = sm_90
h100: all

a100: ARCH = sm_80
a100: all

v100: ARCH = sm_70
v100: all

# Help target
help:
	@echo "Pipeline Parallel MLP Inference - Build System"
	@echo ""
	@echo "Usage:"
	@echo "  make              - Build all targets (default sm_90 for H100)"
	@echo "  make ARCH=sm_80   - Build for specific architecture"
	@echo "  make h100         - Build for H100 (sm_90)"
	@echo "  make a100         - Build for A100 (sm_80)"
	@echo "  make v100         - Build for V100 (sm_70)"
	@echo "  make clean        - Remove build artifacts"
	@echo ""
	@echo "Targets:"
	@echo "  pipeline_naive    - Sequential baseline implementation"
	@echo "  pipeline_streams  - Optimized implementation with CUDA streams"

.PHONY: all clean help h100 a100 v100
