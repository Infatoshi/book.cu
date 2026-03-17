# Makefile for compiling the tensor parallelism examples.
# This Makefile builds two separate executables:
# 1. 8gpu_single_node: For a single-node, 8-GPU tensor parallel benchmark.
# 2. 16gpu_multi_node: For a multi-node, 16-GPU tensor parallel benchmark.

# NVCC: The NVIDIA CUDA compiler.
NVCC = nvcc

# MPI paths: These point to the OpenMPI include and library directories.
# These paths might need to be adjusted depending on the system's OpenMPI installation.
MPI_INCLUDE = /usr/lib/x86_64-linux-gnu/openmpi/include
MPI_LIB = /usr/lib/x86_64-linux-gnu/openmpi/lib

# CFLAGS: Compiler and linker flags.
# -I$(MPI_INCLUDE): Adds the MPI include directory to the header search path.
# -L$(MPI_LIB): Adds the MPI library directory to the library search path.
# -lcublas: Links against the cuBLAS library for GPU-accelerated primitives.
# -lmpi, -lmpi_cxx: Links against the MPI C and C++ libraries.
# -std=c++11: Use the C++11 standard.
# -O3: High level of optimization.
# -arch=sm_90: Specifies the target GPU architecture (sm_90 for Hopper/H100).
#   This should be changed based on the target hardware (e.g., sm_80 for Ampere/A100).
CFLAGS = -I$(MPI_INCLUDE) -L$(MPI_LIB) -lcublas -lmpi -lmpi_cxx -std=c++11 -O3 -arch=sm_90

# The default target, which builds both executables.
all: 8gpu 16gpu

# Rule to build the single-node, 8-GPU executable.
8gpu: 8gpu_single_node.cu
	$(NVCC) -o 8gpu_single_node $< $(CFLAGS)

# Rule to build the multi-node, 16-GPU executable.
16gpu: 16gpu_multi_node.cu
	$(NVCC) -o 16gpu_multi_node $< $(CFLAGS)

# Rule to clean up build artifacts.
# Removes the compiled executables and any object files.
clean:
	rm -f 8gpu_single_node 16gpu_multi_node *.o

# Declares targets that are not actual files.
.PHONY: all clean

