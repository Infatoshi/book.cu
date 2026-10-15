# Chapter 10: Multi-GPU computing

Code for chapter 10 of *CUDA for LLMs*. Both examples run on a single node.

## Contents

```
8_distributed/
├── tensor_parallel/
│   ├── tensor_parallel.cu   Row-parallel GEMM: each GPU multiplies its K-shard, NCCL AllReduce sums the partials
│   ├── Makefile
│   └── run_all.sh           Builds and runs on every local GPU
└── pipeline_parallel/
    ├── pipeline.cu          4-layer MLP split across GPUs: naive blocking pipeline vs. CUDA streams
    ├── Makefile
    ├── README.md
    └── ARCHITECTURE.md
```

## Requirements

- A node with 2 or more NVIDIA GPUs (the book's numbers come from 8x H100 SXM)
- CUDA Toolkit with cuBLAS, NCCL, and OpenMPI (appendix A of the book covers the install)

## Tensor parallelism

```bash
cd tensor_parallel
./run_all.sh                       # build and run on all local GPUs

# or by hand
make                               # ARCH=native by default; e.g. make ARCH=sm_90
mpirun -np 8 ./tensor_parallel     # default 4096 x 4096 x 4096
mpirun -np 8 ./tensor_parallel 8192 8192 8192
```

The program times the full GEMM on one GPU as a baseline, then reports the tensor-parallel step time with and without the AllReduce, the speedup, and the AllReduce share of the step. Rank 0 checks the result against an FP32 cuBLAS reference. If `mpi.h` is not found, set `MPI_INCLUDE` and `MPI_LIB` in the Makefile (`mpicc -showme:incdirs` prints the include path).

## Pipeline parallelism

```bash
cd pipeline_parallel
make
./pipeline 4                       # number of GPUs; default 2
```

Both versions run on the same GPUs, so the reported speedup is the gain from overlapping stages with streams, not a speedup over one GPU. Run `./pipeline 1` for the single-GPU comparison.
