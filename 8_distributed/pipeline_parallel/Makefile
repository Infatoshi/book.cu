# Makefile for compiling the pipeline parallelism example.
# This example demonstrates a simple two-stage pipeline for MLP inference
# on two GPUs, showcasing how to overlap computation and communication.

# Compiler and flags
# NVCC: The NVIDIA CUDA compiler.
NVCC = nvcc
# ARCH: The target GPU architecture. Defaults to sm_90 (Hopper architecture, e.g., H100).
# This can be overridden from the command line, e.g., `make ARCH=sm_80` for Ampere.
ARCH ?= sm_90
# CFLAGS: Compiler flags.
# -O3: High level of optimization.
# -arch=$(ARCH): Specifies the target architecture.
# -std=c++11: Use the C++11 standard.
CFLAGS = -O3 -arch=$(ARCH) -std=c++11
# LDFLAGS: Linker flags.
# -lcublas: Link against the cuBLAS library for GPU-accelerated matrix multiplication.
LDFLAGS = -lcublas

# The name of the final executable binary.
TARGET = pipeline

# The default target executed when `make` is run without arguments.
all: $(TARGET)

# Rule for building the executable.
# This tells `make` that the `pipeline` target depends on `pipeline.cu`.
# If `pipeline.cu` has changed since the last build, this rule will be executed.
pipeline: pipeline.cu
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)
	@echo "✓ Built $@ for architecture $(ARCH)"

# Rule for cleaning up build artifacts.
# Removes the compiled executable.
clean:
	rm -f $(TARGET)
	@echo "✓ Cleaned build artifacts"

# Convenience targets for building for specific common GPU architectures.
h100: ARCH = sm_90
h100: all

a100: ARCH = sm_80
a100: all

v100: ARCH = sm_70
v100: all

# Help target to display usage information.
help:
	@echo "Pipeline Parallel MLP Inference - Build System"
	@echo ""
	@echo "Usage:"
	@echo "  make              - Build the project (defaults to sm_90 for H100 GPU)"
	@echo "  make ARCH=sm_80   - Build for a specific architecture (e.g., A100)"
	@echo "  make h100         - Build for H100 (sm_90)"
	@echo "  make a100         - Build for A100 (sm_80)"
	@echo "  make v100         - Build for V100 (sm_70)"
	@echo "  make clean        - Remove build artifacts"
	@echo ""
	@echo "Description:"
	@echo "  This Makefile compiles 'pipeline.cu' into an executable named 'pipeline'."
	@echo "  The program demonstrates a 2-GPU pipeline parallel setup."
	@echo "  Run the executable with './pipeline'."

# Declares targets that are not actual files. This prevents `make` from
# getting confused if a file with the same name as one of these targets exists.
.PHONY: all clean help h100 a100 v100
