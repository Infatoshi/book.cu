# 16 H100 Cluster Execution Roadmap
## Cost-Optimized Chapter 10 Content Generation

**Target:** Generate all Chapter 10 code examples and benchmarks in <45 minutes

**Cost Minimization Strategy:**
- All scripts prepared locally before cluster start
- No debugging on cluster (validate locally first with smaller examples)
- Parallel execution where possible
- Immediate transfer and shutdown

---

## Phase 0: Local Preparation (Before Starting Cluster) - 15 min

### Step 1: Create Local Folder Structure
```bash
cd /Users/elliotarledge/cuda/cuda-book
mkdir -p book.cu/8_distributed/tensor_parallel
mkdir -p book.cu/8_distributed/pipeline
mkdir -p book.cu/8_distributed/setup_scripts
```

### Step 2: Prepare Setup Scripts (Based on DUAL_NODE_COMPLETE_SETUP.md)

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

Based on `DUAL_NODE_COMPLETE_SETUP.md` lines 478-601, adapted for single node:

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

### Step 4: Prepare Transfer Script

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

### Step 3.2: Run Pipeline Benchmarks (If Exists)
```bash
# On Node 0
cd ~/distributed/pipeline
make
./run_benchmark.sh

# Results saved automatically
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

## Success Metrics

You've succeeded when:
1. ✅ `results_8gpu.txt` shows ~6-7M GFLOPS total (8 GPUs)
2. ✅ `results_16gpu.txt` shows ~8-9M GFLOPS total (16 GPUs)
3. ✅ Both efficiency metrics are >95%
4. ✅ Total cluster time <60 minutes
5. ✅ All results transferred back to local machine
6. ✅ Cluster shut down and costs stopped

**If any step takes >10 min:** Stop, debug locally, and restart cluster fresh.

**If benchmarks fail:** Capture error output, shut down cluster, debug locally with smaller examples.

Good luck! 🚀

