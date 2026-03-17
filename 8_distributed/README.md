# Chapter 10: Distributed Computing - Code Examples

This directory contains all code examples for Chapter 10 of "CUDA for Deep Learning."

## Overview

This chapter teaches multi-GPU and multi-node parallelism through three progressively complex examples:

1. **Tensor Parallelism (8 GPUs, Single Node)** - Split large matrix across GPUs
2. **Tensor Parallelism (16 GPUs, Two Nodes)** - Scale to multiple nodes via InfiniBand
3. **Pipeline Parallelism (8 GPUs)** - Concurrent batch processing with CUDA streams

## Directory Structure

```
8_distributed/
├── README.md                    (this file)
├── setup_scripts/               (node configuration scripts)
│   ├── node_setup.sh           (run on both nodes)
│   └── node0_only.sh           (SSH setup, hostfile)
├── tensor_parallel/             (tensor parallelism examples)
│   ├── 8gpu_single_node.cu     (8 GPUs, single node)
│   ├── 16gpu_multi_node.cu     (16 GPUs, two nodes)
│   ├── Makefile
│   ├── run_all.sh              (automated benchmark runner)
│   ├── results_8gpu.txt        (benchmark results)
│   └── results_16gpu.txt       (benchmark results)
├── pipeline/                    (pipeline parallelism example)
│   ├── pipeline.cu             (naive vs async comparison)
│   ├── Makefile
│   └── run_benchmark.sh
├── local_sync.sh               (sync code to cluster)
└── local_retrieve.sh           (retrieve results from cluster)
```

## Prerequisites

### Hardware
- **Single-node examples:** 1 server with 8× H100 GPUs (or similar)
- **Multi-node examples:** 2 servers, each with 8× H100 GPUs, connected via InfiniBand or 100+ GbE

### Software
- Ubuntu 22.04 LTS
- CUDA Toolkit 12.4+ (`nvcc`)
- NVIDIA drivers (`nvidia-smi`)
- OpenMPI (`mpirun`)
- cuBLAS (included with CUDA)

### Installation

Run on **both nodes** (if multi-node):
```bash
cd setup_scripts
chmod +x node_setup.sh
./node_setup.sh
```

For multi-node setup, run on **Node 0 only**:
```bash
cd setup_scripts
chmod +x node0_only.sh
./node0_only.sh
# Follow prompts to configure SSH and hostfile
```

## Running the Examples

### Option 1: Tensor Parallelism (Automated)

```bash
cd tensor_parallel
chmod +x run_all.sh
./run_all.sh
```

This runs both 8-GPU and 16-GPU benchmarks and saves results.

### Option 2: Tensor Parallelism (Manual)

**8 GPUs (single node):**
```bash
cd tensor_parallel
make 8gpu
mpirun -np 8 --mca btl tcp,self ./8gpu_single_node
```

**16 GPUs (two nodes):**
```bash
cd tensor_parallel
make 16gpu

# Copy hostfile from setup (or create manually):
# cat > hosts << EOF
# <node0_ip> slots=8
# <node1_ip> slots=8
# EOF

mpirun -np 16 --hostfile ../hosts --mca btl tcp,self ./16gpu_multi_node
```

### Option 3: Pipeline Parallelism

```bash
cd pipeline
make
./run_benchmark.sh
```

## Expected Results

### Tensor Parallelism (8 GPUs, Single Node)
```
=== 8-GPU Single-Node Tensor Parallel Results ===
Rank 0: ~770000 GFLOPS, ~0.18 ms
...
TOTAL: ~6,160,000 GFLOPS
Avg per GPU: ~770,000 GFLOPS
Efficiency: ~100%
```

**Key Insight:** Near-perfect scaling on single node due to NVLink's high bandwidth (600 GB/s).

### Tensor Parallelism (16 GPUs, Two Nodes)
```
=== 16-GPU Multi-Node Results ===
Rank 0: ~554189 GFLOPS, ~0.031 ms
...
TOTAL: ~8,867,024 GFLOPS
Avg per GPU: ~554,189 GFLOPS
Efficiency: ~99.8%
```

**Key Insight:** Still excellent scaling across nodes via InfiniBand (200 Gb/s), but slightly lower per-GPU performance due to inter-node communication overhead.

### Pipeline Parallelism (8 GPUs)

**Naive (blocking synchronization):**
```
Throughput: ~110 batches/s
Speedup: 1.1x
Efficiency: 27%
```

**Optimized (async with streams):**
```
Throughput: ~396 batches/s
Speedup: 7.8x
Efficiency: 98%
```

**Key Insight:** CUDA streams enable concurrent execution across GPUs, achieving near-linear speedup.

## Understanding the Code

### Tensor Parallelism Strategy

**Problem:** Matrix multiplication too large for single GPU memory.

**Solution:** Split weight matrix by columns across N GPUs.

1. Each GPU gets: `input (M×K) @ weight_shard (K×N/num_gpus) = output_local (M×N/num_gpus)`
2. Broadcast input to all GPUs
3. Each GPU computes local GEMM (cuBLAS)
4. AllReduce sums partial results (NCCL)
5. Result: Full output matrix

**Code Highlights:**
- MPI for process management (`MPI_Init`, `MPI_Comm_rank`)
- CUDA for GPU assignment (`cudaSetDevice(rank % 8)`)
- cuBLAS for optimized GEMM (`cublasHgemm`)
- NCCL for efficient collectives (implicit in MPI reduction)

### Pipeline Parallelism Strategy

**Problem:** Sequential batch processing wastes GPU cycles.

**Solution:** Process multiple batches concurrently using streams.

1. Create one stream per batch
2. Create events to coordinate dependencies
3. Launch async operations (memcpy, kernels) on different streams
4. GPUs work on different batches simultaneously (staircase pattern)

**Code Highlights:**
- Naive version: `cudaDeviceSynchronize()` after each GPU → serialization
- Optimized version: `cudaStreamCreate()`, `cudaEventRecord()`, `cudaStreamWaitEvent()` → concurrency

## Hardware Topology

### Single Node (8 GPUs)

```
┌─────────────────────────────────────┐
│          Server (Node 0)            │
│  ┌───┐  ┌───┐  ┌───┐  ┌───┐       │
│  │GPU│══│GPU│══│GPU│══│GPU│       │ NVLink (600 GB/s)
│  │ 0 │  │ 1 │  │ 2 │  │ 3 │       │
│  └───┘  └───┘  └───┘  └───┘       │
│    ║      ║      ║      ║          │
│  ┌───┐  ┌───┐  ┌───┐  ┌───┐       │
│  │GPU│══│GPU│══│GPU│══│GPU│       │
│  │ 4 │  │ 5 │  │ 6 │  │ 7 │       │
│  └───┘  └───┘  └───┘  └───┘       │
└─────────────────────────────────────┘
```

**Bandwidth:** ~600 GB/s between GPUs (NVLink 4.0)

### Multi-Node (16 GPUs)

```
┌─────────────────────────────────────┐
│          Node 0 (8 GPUs)            │
│         [NVLink Mesh]               │
└──────────────┬──────────────────────┘
               │
               │ InfiniBand (200 Gb/s = 25 GB/s)
               │
┌──────────────┴──────────────────────┐
│          Node 1 (8 GPUs)            │
│         [NVLink Mesh]               │
└─────────────────────────────────────┘
```

**Intra-node bandwidth:** ~600 GB/s (NVLink)  
**Inter-node bandwidth:** ~25 GB/s (InfiniBand HDR)

## Troubleshooting

### MPI Errors

**Problem:** `mpirun` can't connect to Node 1

**Solution:**
```bash
# Verify SSH works without password
ssh ubuntu@<node1_ip> "echo success"

# Check hostfile format (no tabs, spaces only)
cat hosts

# Test localhost first
mpirun -np 8 --mca btl tcp,self ./mpi_test
```

### CUDA Errors

**Problem:** CUDA out of memory

**Solution:** Reduce problem size in source code (change `M`, `N`, `K` from 4096 to 2048).

**Problem:** `cudaErrorInvalidDevice`

**Solution:** Check GPU count matches MPI rank count:
```bash
nvidia-smi --list-gpus | wc -l  # Should be 8
```

### Compilation Errors

**Problem:** `fatal error: mpi.h: No such file or directory`

**Solution:**
```bash
# Install OpenMPI
sudo apt install -y openmpi-bin libopenmpi-dev

# Or update Makefile with correct MPI paths
ls /usr/lib/x86_64-linux-gnu/openmpi/include
```

## Performance Tips

1. **Use FP16 (half precision)** for Tensor Cores (~770 TFLOPS per H100)
2. **Enable peer access** for faster GPU-to-GPU transfers (automatic with NVLink)
3. **Tune problem size** to keep GPUs busy (larger M/N/K = better utilization)
4. **Profile with `nsys`** to identify bottlenecks:
   ```bash
   nsys profile --trace=cuda,nvtx mpirun -np 8 ./8gpu_single_node
   ```

## References

- **Chapter 10:** Distributed Computing (10.adoc)
- **Appendix D:** Multi-Node Setup Guide (13_appendix.adoc)
- **DUAL_NODE_COMPLETE_SETUP.md:** Complete setup guide from fresh Ubuntu installation
- **NCCL Documentation:** https://docs.nvidia.com/deeplearning/nccl/
- **OpenMPI Documentation:** https://www.open-mpi.org/doc/

## Contributing

Found a bug or improvement? Please submit an issue or PR to the book repository.

## License

All code examples are provided as-is for educational purposes as part of "CUDA for Deep Learning."

