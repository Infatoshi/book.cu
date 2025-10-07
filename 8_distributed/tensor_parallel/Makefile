NVCC = nvcc
MPI_INCLUDE = /usr/lib/x86_64-linux-gnu/openmpi/include
MPI_LIB = /usr/lib/x86_64-linux-gnu/openmpi/lib

CFLAGS = -I$(MPI_INCLUDE) -L$(MPI_LIB) -lcublas -lmpi -lmpi_cxx -std=c++11 -O3 -arch=sm_90

all: 8gpu 16gpu

8gpu: 8gpu_single_node.cu
	$(NVCC) -o 8gpu_single_node $< $(CFLAGS)

16gpu: 16gpu_multi_node.cu
	$(NVCC) -o 16gpu_multi_node $< $(CFLAGS)

clean:
	rm -f 8gpu_single_node 16gpu_multi_node *.o

.PHONY: all clean

