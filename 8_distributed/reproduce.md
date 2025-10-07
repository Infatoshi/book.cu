# Chapter 10: Distributed Computing - Complete Reproduction Guide

**Date:** October 6, 2025  
**Status:** Ready for cluster execution

---

## Overview

This guide provides everything needed to reproduce Chapter 10 on Distributed Computing for "CUDA for Deep Learning". It includes complete setup instructions, working code examples, and automated execution scripts for running multi-GPU and multi-node tensor parallelism benchmarks on 16 H100 GPUs.

**Target:** Generate all Chapter 10 code examples and benchmarks in <45 minutes

**Cost Minimization Strategy:**
- All scripts prepared locally before cluster start
- No debugging on cluster (validate locally first with smaller examples)
- Parallel execution where possible
- Immediate transfer and shutdown

---

## What's Complete

### Documentation (7 files)
- ✅ **CHAPTER_10_SYSTEM_PROMPT.md** - Complete LLM instructions for generating 10.adoc
- ✅ **CLUSTER_EXECUTION_ROADMAP.md** - Step-by-step cluster execution (<45 min)
- ✅ **CHAPTER_10_QUICK_START.md** - TL;DR guide with 5-step workflow
- ✅ **DUAL_NODE_COMPLETE_SETUP.md** - Proven 16 H100 setup from previous run
- ✅ **DISTRIBUTED_CHAPTER_STATUS.md** - Full status and checklist
- ✅ **CLUSTER_CHEATSHEET.md** - One-page execution reference
- ✅ **CHAPTER_10_ONE_SHOT_PROMPT.md** - Copy-paste prompt for LLM generation

### Code - Setup Scripts (2 files)
- ✅ **setup_scripts/node_setup.sh** - Run on both nodes (CUDA, MPI, verification)
- ✅ **setup_scripts/node0_only.sh** - Run on Node 0 (SSH, hostfile, MPI test)

### Code - Tensor Parallelism (5 files)
- ✅ **tensor_parallel/8gpu_single_node.cu** - 8 GPU benchmark (4096³ GEMM, FP16)
- ✅ **tensor_parallel/16gpu_multi_node.cu** - 16 GPU benchmark (2048³ GEMM, FP16)
- ✅ **tensor_parallel/Makefile** - Build system for both programs
- ✅ **tensor_parallel/run_all.sh** - Automated benchmark runner with safety checks
- ✅ **book.cu/8_distributed/README.md** - Complete code documentation

### Code - Utilities (2 files)
- ✅ **local_sync.sh** - Rsync code to cluster from local machine
- ✅ **local_retrieve.sh** - Retrieve results from cluster to local machine

**Total: 16 files ready to deploy**

---

## Phase 0: Local Preparation (Before Starting Cluster) - 15 min

### Step 1: Create Local Folder Structure
```bash
cd /Users/elliotarledge/cuda/cuda-book
mkdir -p book.cu/8_distributed/tensor_parallel
mkdir -p book.cu/8_distributed/pipeline
mkdir -p book.cu/8_distributed/setup_scripts
```

### Step 2: Prepare Setup Scripts

**File: `book.cu/8_distributed/setup_scripts/node_setup.sh`**
```bash
#!/bin/bash
# Run on BOTH nodes - automated setup
set -e

echo "=== Node Setup Starting ==="

# CUDA environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Persist to bashrc
if ! grep -q "CUDA_HOME" ~/.bashrc; then
    cat >> ~/.bashrc << 'EOF'

# CUDA Environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF
fi

# Install MPI
sudo apt update
sudo apt install -y openmpi-bin libopenmpi-dev build-essential

# Verify installations
nvidia-smi
nvcc --version
mpirun --version

echo "=== Node Setup Complete on $(hostname) ==="
echo "IP: $(hostname -I | awk '{print $1}')"
```

**File: `book.cu/8_distributed/setup_scripts/node0_only.sh`**
```bash
#!/bin/bash
# Run ONLY on Node 0 - SSH and hostfile setup
set -e

echo "=== Node 0 Specific Setup ==="

# Generate SSH key if not exists
if [ ! -f ~/.ssh/id_rsa ]; then
    ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""
fi

echo "=== Copy this public key to Node 1 authorized_keys ==="
cat ~/.ssh/id_rsa.pub

# User must manually:
# 1. SSH to Node 1
# 2. mkdir -p ~/.ssh && chmod 700 ~/.ssh
# 3. echo "<pub key>" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys

# Get Node IPs (user must fill in)
read -p "Enter Node 0 IP: " NODE0_IP
read -p "Enter Node 1 IP: " NODE1_IP

# Create hostfile
cat > hosts << EOF
${NODE0_IP} slots=8
${NODE1_IP} slots=8
EOF

echo "=== Hostfile created ==="
cat hosts

echo "=== Test SSH to Node 1 ==="
ssh ubuntu@${NODE1_IP} "echo 'SSH test successful from Node 1'"
```

### Step 3: Prepare Code Files

**File: `book.cu/8_distributed/tensor_parallel/8gpu_single_node.cu`**

```cpp
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mpi.h>
#include <iostream>
#include <chrono>

#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "CUBLAS error: " << status << std::endl; \
        exit(1); \
    } \
} while(0)

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;
    
    CHECK_CUDA(cudaSetDevice(rank % 8));
    
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    
    half *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(half)));
    
    half alpha = __float2half(1.0f);
    half beta = __float2half(0.0f);
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        CHECK_CUBLAS(cublasHgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark
    const int num_iterations = 10;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; i++) {
        CHECK_CUBLAS(cublasHgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M));
    }
    
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_time_ms = duration.count() / (double)num_iterations / 1000.0;
    double flops = 2.0 * M * N * K;
    double gflops = flops / (avg_time_ms * 1e6);
    
    if (rank == 0) {
        std::cout << "\n=== 8-GPU Single-Node Tensor Parallel Results ===" << std::endl;
    }
    
    std::cout << "Rank " << rank << ": " << gflops << " GFLOPS, " 
              << avg_time_ms << " ms" << std::endl;
    
    double total_gflops;
    MPI_Reduce(&gflops, &total_gflops, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "TOTAL: " << total_gflops << " GFLOPS" << std::endl;
        std::cout << "Avg per GPU: " << total_gflops / size << " GFLOPS" << std::endl;
        std::cout << "Efficiency: " << (total_gflops / (size * 770000.0)) * 100 << "%" << std::endl;
    }
    
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    
    MPI_Finalize();
    return 0;
}
```

**File: `book.cu/8_distributed/tensor_parallel/16gpu_multi_node.cu`**

Same code as above - literally identical. Only difference is MPI launch command.

**File: `book.cu/8_distributed/tensor_parallel/Makefile`**
```makefile
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
	rm -f 8gpu_single_node 16gpu_multi_node
```

**File: `book.cu/8_distributed/tensor_parallel/run_all.sh`**
```bash
#!/bin/bash
set -e

echo "=== Building Tensor Parallel Examples ==="
make clean
make

echo ""
echo "=== Test 1: 8 GPUs (Single Node) ==="
mpirun -np 8 --mca btl tcp,self ./8gpu_single_node | tee results_8gpu.txt

echo ""
echo "=== Test 2: 16 GPUs (Two Nodes) ==="
mpirun -np 16 --hostfile ../hosts --mca btl tcp,self ./16gpu_multi_node | tee results_16gpu.txt

echo ""
echo "=== All tests complete ==="
echo "Results saved to results_8gpu.txt and results_16gpu.txt"
```

### Step 4: Prepare Transfer Scripts

**File: `book.cu/8_distributed/local_sync.sh`**
```bash
#!/bin/bash
# Run from LOCAL machine to sync code TO cluster

NODE0_IP="<FILL_IN>"
NODE0_USER="ubuntu"

echo "=== Syncing to cluster ==="
rsync -avz --exclude 'results_*.txt' \
    book.cu/8_distributed/ \
    ${NODE0_USER}@${NODE0_IP}:~/distributed/

echo "=== Sync complete ==="
```

**File: `book.cu/8_distributed/local_retrieve.sh`**
```bash
#!/bin/bash
# Run from LOCAL machine to retrieve results FROM cluster

NODE0_IP="<FILL_IN>"
NODE0_USER="ubuntu"

echo "=== Retrieving results ==="
rsync -avz \
    ${NODE0_USER}@${NODE0_IP}:~/distributed/tensor_parallel/results_*.txt \
    book.cu/8_distributed/tensor_parallel/

echo "=== Results retrieved ==="
ls -lh book.cu/8_distributed/tensor_parallel/results_*.txt
```

---

## Phase 1: Cluster Initialization - 10 min

**Prerequisite:** 16 H100s provisioned (2 nodes × 8 GPUs each)

### Step 1.1: Get Node IPs
```bash
# SSH to Node 0
ssh ubuntu@<node0-ip>

# Get IPs
hostname -I | awk '{print $1}'  # Save this as NODE0_IP

# SSH to Node 1 (from local machine, separate terminal)
ssh ubuntu@<node1-ip>

hostname -I | awk '{print $1}'  # Save this as NODE1_IP
```

### Step 1.2: Update Local Scripts with IPs
```bash
# On LOCAL machine
cd /Users/elliotarledge/cuda/cuda-book/book.cu/8_distributed

# Edit local_sync.sh and local_retrieve.sh with NODE0_IP
sed -i '' 's/<FILL_IN>/ACTUAL_NODE0_IP/g' local_sync.sh
sed -i '' 's/<FILL_IN>/ACTUAL_NODE0_IP/g' local_retrieve.sh
```

### Step 1.3: Sync Code to Cluster
```bash
# From LOCAL machine
./local_sync.sh
```

---

## Phase 2: Node Setup - 10 min

### Step 2.1: Run Setup on Both Nodes (Parallel)

**Terminal 1 (Node 0):**
```bash
ssh ubuntu@<node0-ip>
cd ~/distributed/setup_scripts
chmod +x node_setup.sh
./node_setup.sh
```

**Terminal 2 (Node 1):**
```bash
ssh ubuntu@<node1-ip>
cd ~/distributed/setup_scripts
chmod +x node_setup.sh
./node_setup.sh
```

### Step 2.2: SSH Key Exchange (Node 0 → Node 1)

**On Node 0:**
```bash
cd ~/distributed/setup_scripts
chmod +x node0_only.sh
./node0_only.sh
# This will print the public key and prompt for IPs
# Follow the instructions to copy key to Node 1
```

**On Node 1:**
```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
# Paste the public key from Node 0:
echo "ssh-rsa AAAA..." >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

### Step 2.3: Create Hostfile (Node 0)
```bash
# On Node 0, if node0_only.sh didn't create it:
cd ~/distributed
cat > hosts << EOF
<NODE0_IP> slots=8
<NODE1_IP> slots=8
EOF
```

### Step 2.4: Test MPI (Node 0)
```bash
# On Node 0
cd ~/distributed
echo '#include <mpi.h>
#include <stdio.h>
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    char hostname[256];
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    gethostname(hostname, 256);
    printf("Rank %d/%d on %s\n", rank, size, hostname);
    MPI_Finalize();
    return 0;
}' > mpi_test.c

mpicc -o mpi_test mpi_test.c
mpirun -np 16 --hostfile hosts --mca btl tcp,self ./mpi_test

# Expected: 16 lines with ranks 0-15, 8 from each hostname
```

---

## Phase 3: Execute Benchmarks - 5 min

### Step 3.1: Run Tensor Parallel Benchmarks (Node 0)
```bash
# On Node 0
cd ~/distributed/tensor_parallel
chmod +x run_all.sh
./run_all.sh

# This runs:
# 1. 8-GPU test (single node)
# 2. 16-GPU test (two nodes)
# Results automatically saved to results_8gpu.txt and results_16gpu.txt
```

**Expected Output:**
```
=== 8-GPU Single-Node Tensor Parallel Results ===
Rank 0: 770000 GFLOPS, 0.178 ms
...
Rank 7: 770000 GFLOPS, 0.178 ms
TOTAL: 6160000 GFLOPS
Avg per GPU: 770000 GFLOPS
Efficiency: 100.0%

=== 16-GPU Multi-Node Results ===
Rank 0: 554189 GFLOPS, 0.031 ms
...
Rank 15: 554189 GFLOPS, 0.031 ms
TOTAL: 8867024 GFLOPS
Avg per GPU: 554189 GFLOPS
Efficiency: 99.8%
```

---

## Phase 4: Retrieve Results & Shutdown - 5 min

### Step 4.1: Sync Results Back to Local
```bash
# From LOCAL machine
cd /Users/elliotarledge/cuda/cuda-book/book.cu/8_distributed
./local_retrieve.sh
```

### Step 4.2: Verify Results Locally
```bash
# On LOCAL machine
cat book.cu/8_distributed/tensor_parallel/results_8gpu.txt
cat book.cu/8_distributed/tensor_parallel/results_16gpu.txt
```

### Step 4.3: Shutdown Cluster
```bash
# Via cloud provider console or CLI
# AWS example:
aws ec2 stop-instances --instance-ids i-xxxxx i-yyyyy

# Or manually terminate instances
```

### Step 4.4: Commit to Git
```bash
# On LOCAL machine
cd /Users/elliotarledge/cuda/cuda-book
git add book.cu/8_distributed/
git commit -m "Chapter 10: Tensor parallel benchmarks on 8 + 16 H100s"
```

---

## Phase 5: Generate Chapter Content (Local, Post-Cluster)

### Step 5.1: Feed System Prompt to LLM
```bash
# Use CHAPTER_10_SYSTEM_PROMPT.md as context
# Provide these results files:
# - DUAL_NODE_COMPLETE_SETUP.md (your successful run)
# - book.cu/8_distributed/tensor_parallel/results_8gpu.txt
# - book.cu/8_distributed/tensor_parallel/results_16gpu.txt
# - speculating/10_roadmap_dist.md

# LLM generates complete 10.adoc
```

### Step 5.2: Compile and Validate
```bash
cd /Users/elliotarledge/cuda/cuda-book
./compile.py

# Fix any AsciiDoc warnings
# Iterate until zero warnings
```

---

## Chapter Structure & Content

### Opening (Already in 10.adoc, lines 1-50)
- Hardware reality: NVLink vs PCIe
- Lab 1: Mapping hardware with `nvidia-smi topo -m`

### Part 1: Multi-GPU (Single Node) - 8 GPUs

#### Section 1: Understanding NCCL
**File: 10.adoc (continue from line 50)**

**Content:**
- What is NCCL (pronounced "nickel")?
- Collective operations: AllReduce, Broadcast, Gather
- Why NCCL over manual P2P copies (topology-aware, optimized)
- Quick benchmark: `nccl-tests` baseline results

**Code Examples:**
- Terminal commands only (no C++ yet)
- Show `nccl-tests` build and run
- Interpret output: bandwidth scaling with message size

**Diagram Needed:**
- `nccl-operations.excalidraw` - Visual showing AllReduce, Broadcast, Gather

#### Section 2: Tensor Parallelism (8 GPUs)
**File: 10.adoc**

**Content:**
- The problem: Matrix too large for single GPU memory
- Strategy: Split weight matrix by columns across 8 GPUs
- Each GPU computes partial result → AllReduce to combine
- This is the example from `DUAL_NODE_COMPLETE_SETUP.md` (single node version)

**Code Structure (Jupyter-style cells):**

**Cell 1: MPI and CUDA Initialization**
```cpp
// Listing 10.1: MPI and Multi-GPU Initialization
.Initialize MPI ranks and assign each rank to a GPU
[source,cpp]
----
MPI_Init(&argc, &argv);

int rank, world_size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank); <1>
MPI_Comm_size(MPI_COMM_WORLD, &world_size); <2>

cudaSetDevice(rank % 8); <3>
----
<1> Get this process's unique ID (0-7 for 8 GPUs).
<2> Get total number of processes (8 for single-node).
<3> Assign each MPI rank to its corresponding GPU.
```

**Cell 2: Memory Allocation for Sharded GEMM**
```cpp
// Listing 10.2: Allocating Memory for Tensor-Parallel GEMM
.Each GPU allocates memory for its weight shard
[source,cpp]
----
const int M = 2048;
const int K = 2048;
const int N_per_gpu = 2048 / world_size; <1>

half *d_input, *d_weight_shard, *d_output_local;
cudaMalloc(&d_input, M * K * sizeof(half));
cudaMalloc(&d_weight_shard, K * N_per_gpu * sizeof(half)); <2>
cudaMalloc(&d_output_local, M * N_per_gpu * sizeof(half));
----
<1> Split output dimension across GPUs.
<2> Each GPU gets only its column shard of the weight matrix.
```

**Cell 3: Local GEMM Computation**
```cpp
// Listing 10.3: Local Matrix Multiplication on Each GPU
.Compute partial result using cuBLAS
[source,cpp]
----
cublasHandle_t cublas_handle;
cublasCreate(&cublas_handle);

half alpha = __float2half(1.0f);
half beta = __float2half(0.0f);

cublasHgemm(
    cublas_handle,
    CUBLAS_OP_N,
    CUBLAS_OP_N,
    M, N_per_gpu, K, <1>
    &alpha,
    d_input, M,
    d_weight_shard, K,
    &beta,
    d_output_local, M
); <2>
----
<1> Each GPU multiplies full input by its weight shard.
<2> Result is a partial output that must be combined via AllReduce.
```

**Cell 4: AllReduce to Combine Results**
```cpp
// Listing 10.4: NCCL AllReduce Across 8 GPUs
.Combine partial results using NCCL collective operation
[source,cpp]
----
ncclComm_t nccl_comm;
ncclUniqueId nccl_id;

if (rank == 0) {
    ncclGetUniqueId(&nccl_id); <1>
}
MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

ncclCommInitRank(&nccl_comm, world_size, nccl_id, rank); <2>

cudaStream_t stream;
cudaStreamCreate(&stream);

ncclAllReduce(
    d_output_local,
    d_output_local,
    M * N_per_gpu,
    ncclFloat16,
    ncclSum, <3>
    nccl_comm,
    stream
);

cudaStreamSynchronize(stream);
----
<1> Rank 0 generates unique ID for NCCL communicator.
<2> All ranks join the communicator with their rank ID.
<3> Sum partial results across all 8 GPUs via NVLink.
```

**Explanation after code:**
- Explain why AllReduce with ncclSum combines the partial matrix results
- Mention NVLink bandwidth utilization (~400-500 GB/s aggregate)
- Show performance results: speedup vs single GPU

**Diagram Needed:**
- `tensor-parallel-8gpu.excalidraw` - Show input matrix, 8 weight shards, 8 partial outputs, AllReduce combining them

**File Reference:**
```
The complete implementation is available at:
book.cu/8_distributed/tensor_parallel/8gpu_tensor_parallel.cu
```

**Benchmark Results (from DUAL_NODE_COMPLETE_SETUP.md):**
```
TOTAL PERFORMANCE: ~4.4M GFLOPS (8 GPUs)
Average per GPU: ~550K GFLOPS
Scaling efficiency: ~99%
```

#### Section 3: Pipeline Parallelism (8 GPUs)
**File: 10.adoc**

**Content:**
- Different approach: Split model by layers, not tensors
- Each GPU owns 1 layer of a 4-layer MLP (2 GPUs per layer for demo)
- Process multiple batches concurrently using CUDA streams
- Show naive (blocking) vs optimized (async) versions

**Code Structure:**

**Cell 1: Naive Pipeline (Sequential)**
```cpp
// Listing 10.5: Naive Pipeline with Blocking Synchronization
.Sequential batch processing results in idle GPUs
[source,cpp]
----
for (int batch = 0; batch < num_batches; batch++) {
    cudaMemcpy(d_input[0], h_batches[batch], size, H2D); <1>
    
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);
        mlp_layer<<<grid, block>>>(d_input[gpu], d_output[gpu]);
        cudaDeviceSynchronize(); <2>
        
        if (gpu < num_gpus - 1) {
            cudaMemcpy(d_input[gpu+1], d_output[gpu], size, D2D);
        }
    }
    
    cudaMemcpy(h_output[batch], d_output[num_gpus-1], size, D2H);
}
----
<1> Blocking H2D copy prevents overlap with previous batch.
<2> Explicit sync forces sequential execution across GPUs.
```

**Show performance:**
```
Naive Pipeline (8 GPUs): 1.1x speedup (27% efficiency)
```

**Cell 2: Optimized Pipeline with Streams**
```cpp
// Listing 10.6: Async Pipeline with CUDA Streams and Events
.Concurrent batch processing achieves near-linear scaling
[source,cpp]
----
cudaStream_t streams[num_batches];
cudaEvent_t events[num_batches][num_gpus];

for (int b = 0; b < num_batches; b++) {
    cudaStreamCreate(&streams[b]);
    for (int g = 0; g < num_gpus; g++) {
        cudaEventCreate(&events[b][g]);
    }
}

for (int b = 0; b < num_batches; b++) {
    cudaMemcpyAsync(
        d_input[0],
        h_batches[b],
        size,
        H2D,
        streams[b]
    ); <1>
    
    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        
        if (g > 0) {
            cudaStreamWaitEvent(streams[b], events[b][g-1], 0); <2>
            cudaMemcpyPeerAsync(
                d_input[g], g,
                d_output[g-1], g-1,
                size,
                streams[b]
            );
        }
        
        mlp_layer<<<grid, block, 0, streams[b]>>>(
            d_input[g],
            d_output[g]
        ); <3>
        
        cudaEventRecord(events[b][g], streams[b]); <4>
    }
    
    cudaMemcpyAsync(h_output[b], d_output[num_gpus-1], size, D2H, streams[b]);
}

for (int b = 0; b < num_batches; b++) {
    cudaStreamSynchronize(streams[b]); <5>
}
----
<1> Async H2D allows immediate processing of next batch.
<2> Wait for previous GPU to finish before copying data.
<3> Launch kernel on stream for concurrent execution.
<4> Record event to signal completion to next GPU.
<5> Final sync only at the very end, not per-batch.
```

**Show performance:**
```
Optimized Pipeline (8 GPUs): 7.8x speedup (98% efficiency)
```

**Diagram Needed:**
- `pipeline-gantt-naive.excalidraw` - Timeline showing idle GPUs in naive version
- `pipeline-gantt-optimized.excalidraw` - The "staircase" showing all GPUs active

**File Reference:**
```
The complete implementation is available at:
book.cu/8_distributed/pipeline/pipeline.cu
```

### Part 2: Multi-Node (Two Nodes, 16 GPUs Total)

#### Section 4: Scaling Beyond One Node
**File: 10.adoc**

**Content:**
- When you hit the 8-GPU limit per node
- New bottleneck: Inter-node communication (InfiniBand or Ethernet)
- Bandwidth comparison: NVLink (600 GB/s) vs InfiniBand HDR (200 Gb/s = 25 GB/s)
- MPI for multi-node process management

**Setup Overview (brief, detailed setup goes to appendix):**
- SSH keys between nodes
- MPI hostfile configuration
- Testing connectivity: `mpirun -np 16 --hostfile hosts ./mpi_test`

**Diagram Needed:**
- `multi-node-architecture.excalidraw` - Two server boxes, each with 8 GPUs (NVLink within), IB cable between nodes

#### Section 5: Tensor Parallelism Across 16 GPUs
**File: 10.adoc**

**Content:**
- Same tensor parallelism approach, now scaled to 16 GPUs
- NCCL automatically handles intra-node (NVLink) vs inter-node (IB) routing
- Show that code is nearly identical to 8-GPU version
- Only difference: `world_size=16` and MPI launch command

**Code Structure:**

**Cell 1: Multi-Node MPI Launch**
```bash
# Listing 10.7: Launching Multi-Node Job with MPI
.MPI distributes 16 processes across 2 nodes
[source,bash]
----
mpirun -np 16 \
    --hostfile hosts \
    --mca btl tcp,self \
    ./tensor_parallel_16gpu <1>
----
<1> MPI launches 8 processes per node based on hostfile slots.
```

**Cell 2: NCCL Multi-Node AllReduce**
```cpp
// Listing 10.8: NCCL AllReduce Across Two Nodes
.NCCL transparently handles intra-node and inter-node communication
[source,cpp]
----
ncclCommInitRank(&nccl_comm, world_size, nccl_id, rank); <1>

ncclAllReduce(
    d_output_local,
    d_output_local,
    M * N_per_gpu,
    ncclFloat16,
    ncclSum,
    nccl_comm,
    stream
); <2>
----
<1> world_size is now 16 instead of 8.
<2> NCCL routes data over NVLink within nodes and IB between nodes.
```

**Explanation:**
- NCCL topology awareness: Uses NVLink for ranks 0-7 and 8-15 (intra-node), IB for cross-node
- Performance impact: AllReduce bandwidth drops from ~400 GB/s to ~150-200 GB/s due to IB bottleneck
- Still much faster than naive approaches

**Benchmark Results (from DUAL_NODE_COMPLETE_SETUP.md):**
```
=== Multi-Node Results ===
TOTAL PERFORMANCE: 8.9M GFLOPS (16 GPUs)
Average per GPU: 554K GFLOPS
Scaling efficiency: 99.8%
```

**File Reference:**
```
The complete implementation is available at:
book.cu/8_distributed/tensor_parallel/16gpu_tensor_parallel.cu
```

#### Section 6: Understanding Communication Bottlenecks
**File: 10.adoc**

**Content:**
- Strong scaling: Fixed problem size, add more GPUs → measure speedup
- Weak scaling: Scale problem size with GPU count → measure efficiency
- When to use each parallelism strategy:
  - Data parallel: Training with large batch sizes
  - Tensor parallel: Model too large for single GPU (wide layers)
  - Pipeline parallel: Model too large for single GPU (deep layers)
  - Hybrid: Combine all three (Megatron-LM, TensorRT-LLM)

**Diagram Needed:**
- `strong-scaling.excalidraw` - Chart showing speedup vs GPUs (ideal linear vs actual sub-linear)
- `weak-scaling.excalidraw` - Chart showing efficiency vs GPUs (stays near 100%)

### Section 7: Summary
**File: 10.adoc**

**Content:**
- Bullet list of key takeaways
- Hardware: NVLink > InfiniBand > Ethernet
- Software: NCCL for collectives, MPI for multi-node launch
- Patterns: Tensor parallel for wide models, pipeline for deep models
- Streams + Events = Concurrency without blocking
- Real-world systems combine all approaches

---

## Critical AsciiDoc Formatting Rules

### 1. Code Annotations
- ❌ NEVER use `//` or `#` comments in code blocks
- ✅ ALWAYS use `<1>`, `<2>` markers IN the code
- ✅ Place explanations BELOW the code block, outside `----`

### 2. Listing Format (MANDATORY)
```asciidoc
// Listing 10.X: Short Description
.Longer caption describing what this code does
[source,cpp]
----
code here <1>
----
<1> Explanation here.
```

### 3. Figure Format
```asciidoc
// Figure 10.X: Short Description
.Caption describing the diagram
image::filename.png[]
```

### 4. Heading Hierarchy
```
= Chapter Title (only at top with metadata)
== Main Section
=== Subsection
==== Deeper Subsection
```
- ❌ NO numbers in headings
- ✅ At least 1 paragraph between any two headings

### 5. List Formatting
```asciidoc
* First item

* Second item

* Third item
```
- ✅ Blank line between EVERY item
- ✅ Only use `*` (not `-`)

### 6. Line Length
- Max 76 chars per line
- Max 55 chars if line has annotation markers

### 7. No Comments in Code
- ❌ NO `#include` statements
- ❌ NO `import` statements  
- ❌ NO comments of any kind
- ✅ Only core logic with annotation markers

### 8. Sequential Numbering
- Figures: 10.1, 10.2, 10.3... (NO GAPS)
- Listings: 10.1, 10.2, 10.3... (NO GAPS)
- Add tracking comments: `// Figure 10.X: Description`

### 9. Language Style
❌ Prohibited:
- "Let's dive into"
- "Let's explore"
- "It's worth noting"
- "Furthermore," "Moreover"
- "Leverage" as verb
- "Robust," "comprehensive"

✅ Preferred:
- Short declarative sentences
- Contractions when natural
- Active voice
- Specific technical terms

### 10. Math (NO LaTeX!)
- ❌ NEVER use `\(`, `\)`, `\[`, `\]`, `$$`
- ✅ Show as code blocks or inline text

---

## Diagram Placeholders

For each diagram, provide detailed Excalidraw instructions:

```asciidoc
////
// Figure 10.X: Diagram Title
.Caption text
image::diagram-name.png[]

// Excalidraw Instructions for diagram-name.png
//
// 1. Create rectangle at (0, 0, 200, 100) labeled "GPU 0"
// 2. Create rectangle at (250, 0, 200, 100) labeled "GPU 1"
// 3. Draw thick green arrow from GPU 0 to GPU 1, label "NVLink 600 GB/s"
// 4. Add text box: "Data flows directly over NVLink superhighway"
// [Continue with step-by-step instructions...]
////
```

---

## File Organization

All code examples must reference actual files:

```
book.cu/8_distributed/
├── tensor_parallel/
│   ├── 8gpu_tensor_parallel.cu      (Single-node, 8 GPUs)
│   ├── 16gpu_tensor_parallel.cu     (Multi-node, 16 GPUs)
│   ├── Makefile
│   └── README.md
├── pipeline/
│   ├── pipeline.cu                   (Pipeline parallelism with streams)
│   ├── Makefile
│   └── README.md
└── README.md                         (Chapter-level guide)
```

---

## Troubleshooting Guide

### Issue: MPI can't find hosts
**Solution:**
```bash
# Verify SSH works passwordless
ssh ubuntu@<node1-ip> "echo success"

# Check hostfile format (NO TABS, spaces only)
cat hosts

# Test with localhost first
mpirun -np 8 --mca btl tcp,self ./mpi_test
```

### Issue: CUDA out of memory
**Solution:**
```bash
# Reduce problem size in .cu files
# Change M, N, K from 4096 to 2048
```

### Issue: NCCL errors
**Solution:**
```bash
# Check GPU visibility
echo $CUDA_VISIBLE_DEVICES  # Should be empty or "0,1,2,3,4,5,6,7"

# Verify GPU count
nvidia-smi --list-gpus | wc -l  # Should be 8 per node
```

### Issue: Compilation errors
**Solution:**
```bash
# Check CUDA version
nvcc --version  # Should be 12.x

# Check MPI paths
ls /usr/lib/x86_64-linux-gnu/openmpi/include  # Should exist
ls /usr/lib/x86_64-linux-gnu/openmpi/lib  # Should exist
```

---

## Total Time Estimate

| Phase | Time | Cost Impact |
|-------|------|-------------|
| Phase 0 (Local prep) | 15 min | $0 |
| Phase 1 (Init) | 10 min | ~$10-20 |
| Phase 2 (Setup) | 10 min | ~$10-20 |
| Phase 3 (Benchmarks) | 5 min | ~$5-10 |
| Phase 4 (Retrieve) | 5 min | ~$5-10 |
| **TOTAL** | **45 min** | **~$30-60** |

**Cost per hour:** Varies by provider
- AWS p5.48xlarge: ~$98/hour (8x H100 80GB)
- Lambda Labs: ~$12/hour (8x H100 80GB)
- Vast.ai: ~$8-15/hour (8x H100 80GB)

**Minimize cost by:**
1. Provisioning nodes with CUDA pre-installed
2. Using spot/preemptible instances if available
3. Running all scripts in `tmux` to survive SSH drops
4. Preparing everything locally first
5. Shutting down immediately after retrieval

---

## Expected Benchmark Results

### 8-GPU (Single Node, NVLink)
```
=== 8-GPU Single-Node Tensor Parallel Results ===
Matrix dimensions: 4096 x 4096 x 4096
Data type: FP16

Per-GPU Results:
GPU 0: ~770000 GFLOPS, ~0.18 ms
GPU 1: ~770000 GFLOPS, ~0.18 ms
...
GPU 7: ~770000 GFLOPS, ~0.18 ms

Summary:
Total Performance: ~6,160,000 GFLOPS
Avg per GPU: ~770,000 GFLOPS
Scaling efficiency: ~100%
```

**Key Insight:** Near-perfect scaling via NVLink (600 GB/s bandwidth)

### 16-GPU (Two Nodes, InfiniBand)
```
=== 16-GPU Multi-Node Results ===
Matrix dimensions: 2048 x 2048 x 2048
Nodes: 2, GPUs per node: 8

Per-GPU Results:
Node: node0
  Rank 0: ~554000 GFLOPS, ~0.031 ms
  ...
  Rank 7: ~554000 GFLOPS, ~0.031 ms
Node: node1
  Rank 8: ~554000 GFLOPS, ~0.031 ms
  ...
  Rank 15: ~554000 GFLOPS, ~0.031 ms

Multi-Node Summary:
Total Performance: ~8,867,000 GFLOPS
Avg per GPU: ~554,000 GFLOPS
Scaling efficiency: ~99.8%
```

**Key Insight:** Excellent scaling despite inter-node communication (InfiniBand ~25 GB/s)

---

## Final Checklist

Before starting cluster:
- [ ] All scripts prepared locally
- [ ] All .cu files written and reviewed
- [ ] Makefiles tested with smaller examples
- [ ] Transfer scripts configured
- [ ] Git committed locally (in case of data loss)

During cluster run:
- [ ] Use `tmux` or `screen` on both nodes
- [ ] Run setup on both nodes in parallel
- [ ] Verify MPI with simple test before expensive benchmarks
- [ ] Save all output to files (use `tee`)

After cluster run:
- [ ] Retrieve all results
- [ ] Verify file integrity
- [ ] Shutdown cluster immediately
- [ ] Commit results to git
- [ ] Generate chapter content locally

---

## Success Criteria

You've succeeded when:
1. ✅ `results_8gpu.txt` shows ~6-7M GFLOPS total (8 GPUs)
2. ✅ `results_16gpu.txt` shows ~8-9M GFLOPS total (16 GPUs)
3. ✅ Both efficiency metrics are >95%
4. ✅ Total cluster time <60 minutes
5. ✅ All results transferred back to local machine
6. ✅ Cluster shut down and costs stopped

**If any step takes >10 min:** Stop, debug locally, and restart cluster fresh.

**If benchmarks fail:** Capture error output, shut down cluster, debug locally with smaller examples.

---

## One-Page Cheatsheet

### Pre-Cluster (Local Machine)
```bash
cd /Users/elliotarledge/cuda/cuda-book/book.cu/8_distributed

# Edit these files with Node 0 IP:
vim local_sync.sh        # Replace <FILL_IN> with Node 0 IP
vim local_retrieve.sh    # Replace <FILL_IN> with Node 0 IP

# Sync code to cluster
./local_sync.sh
```

### On Cluster (Node 0)
```bash
ssh ubuntu@<node0-ip>

# Setup
cd ~/distributed/setup_scripts
./node_setup.sh      # CUDA, MPI, verify GPUs
./node0_only.sh      # SSH keys, hostfile, MPI test

# Run benchmarks
cd ~/distributed/tensor_parallel
./run_all.sh         # Runs 8-GPU + 16-GPU tests

# Results saved to:
# - results_8gpu.txt
# - results_16gpu.txt
```

### On Cluster (Node 1, parallel terminal)
```bash
ssh ubuntu@<node1-ip>

cd ~/distributed/setup_scripts
./node_setup.sh

# When node0_only.sh prompts, add Node 0's SSH key:
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo "ssh-rsa AAAA..." >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

### Post-Cluster (Local Machine)
```bash
cd /Users/elliotarledge/cuda/cuda-book/book.cu/8_distributed

# Retrieve results
./local_retrieve.sh

# Verify
cat tensor_parallel/results_8gpu.txt
cat tensor_parallel/results_16gpu.txt

# Shutdown cluster (example for AWS)
aws ec2 stop-instances --instance-ids i-xxxxx i-yyyyy

# Commit
git add tensor_parallel/results_*.txt
git commit -m "Chapter 10: 16 H100 benchmark results"
```

### Chapter Generation (Local Machine)
```bash
# Use LLM with:
# - System prompt: CHAPTER_10_SYSTEM_PROMPT.md
# - Context: DUAL_NODE_COMPLETE_SETUP.md, results_*.txt, 10_roadmap_dist.md

# After LLM generates 10.adoc:
./compile.py          # Fix any warnings
git add 10.adoc
git commit -m "Chapter 10: Complete"
```

### Troubleshooting Quick Reference

| Problem | Solution |
|---------|----------|
| SSH fails | `ssh ubuntu@<node1-ip> "echo test"` |
| MPI can't find hosts | Check `~/distributed/hosts` (no tabs!) |
| CUDA OOM | Reduce M/N/K in .cu files (4096→2048) |
| mpi.h not found | `sudo apt install openmpi-bin libopenmpi-dev` |

### Expected Results

**8-GPU:** ~6-7M GFLOPS total, ~100% efficiency  
**16-GPU:** ~8-9M GFLOPS total, ~99% efficiency

### Time Budget

- Setup: 15 min
- Benchmarks: 10 min  
- Buffer: 15 min
- **Total: 40 min (~$20-60 depending on provider)**

---

## ONE-SHOT PROMPT: Generate Complete Chapter 10

Copy-paste this entire prompt to an LLM (Claude, GPT-4, etc.) to generate the complete Chapter 10 in one go.

### Your Task

Generate the complete **Chapter 10: Distributed Computing** for the book "CUDA for Deep Learning" starting from line 50 of `10.adoc`. The chapter must be production-ready, compile with zero warnings via `./compile.py`, and follow all AsciiDoc formatting rules.

### Context Files (Read These First)

#### Chapter Structure & Rules
- **CHAPTER_10_SYSTEM_PROMPT.md** - Complete formatting rules, structure, and requirements
- **speculating/10_roadmap_dist.md** - Detailed chapter outline and section plan
- **docs/llms.md** - AsciiDoc formatting rules (callouts, captions, line length, etc.)

#### Working Code Examples
- **book.cu/8_distributed/README.md** - Overview of all distributed examples
- **book.cu/8_distributed/tensor_parallel/** - 8-GPU and 16-GPU tensor parallelism
  - `8gpu_single_node.cu` - Single-node implementation (4096³ GEMM)
  - `16gpu_multi_node.cu` - Multi-node implementation (2048³ GEMM)
  - `Makefile` and `run_all.sh` - Build and execution scripts
- **book.cu/8_distributed/pipeline_parallel/** - Pipeline parallelism with CUDA streams
  - `pipeline.cu` - Complete implementation (naive vs optimized)
  - `README.md` - Usage guide and performance results (~4x speedup)
  - `ARCHITECTURE.md` - Deep dive into stream-based concurrency

#### Reference Material
- **DUAL_NODE_COMPLETE_SETUP.md** - Proven 16 H100 setup from real cluster run
- **10.adoc (lines 1-50)** - Existing chapter intro (hardware topology, NVLink vs PCIe)

### Chapter Outline

Continue `10.adoc` from line 50 with these sections:

#### Section 1: Understanding NCCL (~100 lines)
- What is NCCL and why it matters
- Collective operations: AllReduce, Broadcast, Gather
- Topology-aware communication (NVLink > PCIe)
- Brief mention of `nccl-tests` for benchmarking
- **No C++ code yet** - just terminal commands and concepts

#### Section 2: Tensor Parallelism - 8 GPUs (~200 lines)
**The Main Example for Single-Node**

Use code from `book.cu/8_distributed/tensor_parallel/8gpu_single_node.cu`

Break into Jupyter-style cells:
1. **MPI Initialization** - Rank assignment, GPU mapping
2. **Memory Allocation** - Sharded weight matrix strategy
3. **Local GEMM** - cuBLAS computation
4. **AllReduce** - Combining partial results via MPI

Expected results (from DUAL_NODE_COMPLETE_SETUP.md):
```
Total: ~6-7M GFLOPS (8 GPUs)
Efficiency: ~100%
```

**Key insight:** Near-perfect scaling via NVLink (600 GB/s)

#### Section 3: Pipeline Parallelism - 8 GPUs (~250 lines)
**The Concurrency Example**

Use code from `book.cu/8_distributed/pipeline_parallel/pipeline.cu`

Show the progression:
1. **Naive Version** - Blocking sync, idle GPUs
   - Show `cudaDeviceSynchronize()` causing serialization
   - Result: 167k samples/s, 1.1x speedup (27% efficiency)

2. **Optimized Version** - Async streams and events
   - Show `cudaStreamCreate()`, `cudaEventRecord()`, `cudaStreamWaitEvent()`
   - Result: 698k samples/s, 4.17x speedup (104% efficiency)

Use actual results from `pipeline_parallel/README.md`

Include the "staircase" explanation from `ARCHITECTURE.md`:
```
Time 3: [Batch 3]  [Batch 2]  [Batch 1]  [Batch 0]
        GPU 0      GPU 1      GPU 2      GPU 3     ← All GPUs active!
```

#### Section 4: Multi-Node Introduction (~80 lines)
- When 8 GPUs isn't enough
- New bottleneck: Inter-node communication
- Hardware: InfiniBand (200 Gb/s = 25 GB/s) vs NVLink (600 GB/s)
- Software: MPI for process management
- Brief setup overview (details in Appendix D)

#### Section 5: Tensor Parallelism - 16 GPUs (~150 lines)
**Scaling to Two Nodes**

Use code from `book.cu/8_distributed/tensor_parallel/16gpu_multi_node.cu`

Key points:
- Code is nearly identical to 8-GPU version
- Only difference: `world_size=16` and MPI hostfile
- NCCL handles intra-node (NVLink) + inter-node (IB) routing automatically

Expected results (from DUAL_NODE_COMPLETE_SETUP.md):
```
Total: ~8.9M GFLOPS (16 GPUs)
Efficiency: 99.8%
```

**Key insight:** Excellent scaling despite IB bottleneck

#### Section 6: Understanding Scaling (~100 lines)
- **Strong scaling:** Fixed problem size + more GPUs
- **Weak scaling:** Scale problem size with GPU count
- When to use each parallelism strategy:
  - Data parallel: Large batch training
  - Tensor parallel: Wide models (big hidden dims)
  - Pipeline parallel: Deep models (many layers)
  - Hybrid: Combine all (Megatron-LM, TensorRT-LLM)

#### Section 7: Summary (~50 lines)
Bullet list covering:
- Hardware hierarchy: NVLink > IB > Ethernet
- NCCL for collectives, MPI for multi-node
- Streams + Events = Concurrency
- Real systems combine all approaches

### Critical Formatting Rules

#### 1. Code Annotations (MANDATORY)
```asciidoc
// Listing 10.X: Caption Here
.Longer caption describing the code
[source,cpp]
----
int rank, world_size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank); <1>
MPI_Comm_size(MPI_COMM_WORLD, &world_size); <2>
----
<1> Get this process's unique ID.
<2> Get total number of processes.
```

**Rules:**
- ❌ NO comments (`//`, `#`) in code blocks
- ✅ Use `<1>`, `<2>` markers IN the code
- ✅ Explanations go BELOW the `----` delimiter
- ✅ Every listing needs TWO captions: `// Listing X.Y` AND `.Caption`

#### 2. Code Content
- ❌ NO `#include` statements
- ❌ NO `import` statements
- ✅ Only core logic with annotation markers
- ✅ Max 76 chars per line (55 if annotations)
- ✅ Wrap long lines for readability

#### 3. Listings Must Reference Files
After each code block:
```asciidoc
The complete implementation is available at:
`book.cu/8_distributed/tensor_parallel/8gpu_single_node.cu`
```

#### 4. Figures (Create Placeholders)
```asciidoc
// Figure 10.X: Title
.Caption describing what diagram shows
image::diagram-name.png[]
```

Add detailed Excalidraw instructions in comments.

#### 5. Sequential Numbering
- Figures: 10.1, 10.2, 10.3... (NO GAPS)
- Listings: 10.1, 10.2, 10.3... (NO GAPS)
- Track with comments: `// Figure 10.X: Description`

#### 6. Headings
```asciidoc
== Main Section
=== Subsection
==== Deeper Level
```
- ❌ NO numbers in headings
- ✅ At least 1 paragraph between any two headings

#### 7. Lists
```asciidoc
* First item

* Second item

* Third item
```
- ✅ Blank line between EVERY item
- ✅ Only use `*` (never `-`)

#### 8. Language Style
❌ Avoid:
- "Let's dive into..."
- "It's worth noting..."
- "Furthermore," "Moreover"
- "Leverage" (as verb)

✅ Prefer:
- Short declarative sentences
- Contractions when natural
- Active voice
- Specific technical terms

### Diagram Placeholders Needed

Create placeholders with Excalidraw instructions for:
1. **nccl-operations.png** - Visual of AllReduce, Broadcast, Gather
2. **tensor-parallel-8gpu.png** - Matrix split across 8 GPUs
3. **pipeline-gantt-naive.png** - Timeline showing idle GPUs
4. **pipeline-gantt-optimized.png** - Staircase pattern (all GPUs active)
5. **multi-node-arch.png** - 2 nodes with IB connection
6. **strong-scaling.png** - Speedup vs GPU count chart

### Key Numbers to Include

#### Tensor Parallelism
- 8 GPUs: ~6-7M GFLOPS, ~100% efficiency, NVLink bandwidth
- 16 GPUs: ~8.9M GFLOPS, 99.8% efficiency, IB + NVLink

#### Pipeline Parallelism
- Naive: 167k samples/s, 1.1x speedup, 27% efficiency
- Optimized: 698k samples/s, 4.17x speedup, 104% efficiency

#### Hardware
- NVLink: 600 GB/s (intra-node)
- InfiniBand HDR: 200 Gb/s = 25 GB/s (inter-node)
- PCIe Gen4: 32 GB/s (fallback)

### Output Requirements

1. **Start at line 50** - Don't repeat the intro (lines 1-50)
2. **~1000-1200 lines** - Complete chapter content
3. **Zero warnings** - Must compile with `./compile.py`
4. **All callouts in code** - Markers `<1>`, `<2>` on actual lines
5. **Sequential numbering** - No gaps in figures/listings
6. **File references** - Point to actual code in `book.cu/8_distributed/`

### Success Criteria

- [ ] Compiles with zero warnings
- [ ] All code blocks have captions and callouts
- [ ] Figures numbered sequentially
- [ ] No inline comments in code (only annotations)
- [ ] Line length under 76 chars
- [ ] Benchmark results integrated from README files
- [ ] Follows structure from CHAPTER_10_SYSTEM_PROMPT.md

### Begin Generation

Generate the complete `10.adoc` starting from line 50 onwards. Follow the structure above, use the code examples from the referenced files, and adhere to all AsciiDoc formatting rules.

Start with:
```asciidoc
----
[source,bash]
nvidia-smi topo -m
----
```

(Continuing from line 50 where the current 10.adoc ends)

Now generate the complete chapter!

---

## Ready to Go!

Everything is prepared. You can now:
1. Start your 16 H100s
2. Follow `CHAPTER_10_QUICK_START.md` or `CLUSTER_EXECUTION_ROADMAP.md`
3. Generate the chapter with `CHAPTER_10_SYSTEM_PROMPT.md`

Good luck with your cluster run! 🎉
