# Makefile for the tensor parallelism example.
# Builds a single executable, tensor_parallel, that runs on any number of
# GPUs on one node (mpirun -np 8 ./tensor_parallel) or across nodes
# (mpirun -np 16 --hostfile hosts ./tensor_parallel).

# NVCC: The NVIDIA CUDA compiler.
NVCC = nvcc

# MPI paths: These point to the OpenMPI include and library directories.
# These paths might need to be adjusted depending on the system's OpenMPI
# installation (`mpicc -showme:incdirs` prints the right include path).
MPI_INCLUDE = /usr/lib/x86_64-linux-gnu/openmpi/include
MPI_LIB = /usr/lib/x86_64-linux-gnu/openmpi/lib

# ARCH: Target GPU architecture. `native` compiles for the GPUs present on
# the build machine. Override per machine, e.g.
#   make ARCH=sm_86  (RTX 3090)   make ARCH=sm_90  (H100)
ARCH ?= native

# CFLAGS: Compiler and linker flags.
# -lcublas: cuBLAS for the local GEMM.
# -lnccl:   NCCL for the AllReduce collective.
# -lmpi:    MPI for process management.
CFLAGS = -I$(MPI_INCLUDE) -L$(MPI_LIB) -lcublas -lnccl -lmpi -std=c++14 -O3 -arch=$(ARCH)

all: tensor_parallel

tensor_parallel: tensor_parallel.cu
	$(NVCC) -o tensor_parallel $< $(CFLAGS)

clean:
	rm -f tensor_parallel *.o

.PHONY: all clean
