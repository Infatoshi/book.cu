# System Prompt: Chapter 10 - Distributed Computing

## Mission

Generate a complete, production-ready Chapter 10 on Distributed Computing for "CUDA for Deep Learning" that teaches multi-GPU and multi-node parallelism through hands-on examples. The chapter must be compilable via `./compile.py` with zero warnings and follow all AsciiDoc formatting rules.

## Source Materials

You have access to:
1. **DUAL_NODE_COMPLETE_SETUP.md** - Complete working setup guide for 16x H100s (2 nodes × 8 GPUs)
2. **book.cu/8_distributed/pipeline/** - Pipeline parallelism example (multi-GPU, single node)
3. **speculating/10_dist.md** - Technical notes on NCCL, InfiniBand, multi-node setup
4. **speculating/10_roadmap_dist.md** - Detailed chapter structure plan
5. **10.adoc** - Current chapter file (lines 1-50, needs completion)

## Chapter Structure

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

## Code File Templates

### Template: 8gpu_tensor_parallel.cu

**Source from:** `DUAL_NODE_COMPLETE_SETUP.md` lines 174-279 (single GPU test) + lines 478-601 (multi-node test adapted for single node)

**Structure:**
1. Includes: `<cuda_runtime.h>`, `<cublas_v2.h>`, `<mpi.h>`, `<nccl.h>`
2. Error checking macros: `CHECK_CUDA`, `CHECK_CUBLAS`, `CHECK_NCCL`
3. `main()`:
   - Parse args: `--batch`, `--hidden`, `--iters`
   - MPI init, get rank/size
   - `cudaSetDevice(rank % 8)`
   - Allocate input/weight/output
   - Initialize NCCL communicator
   - Warmup iterations
   - Benchmark loop: GEMM + AllReduce
   - Print per-rank performance
   - MPI_Reduce to rank 0 for aggregate stats
   - Cleanup

**Key code sections for in-text display:**
- MPI init + GPU assignment
- Memory allocation (sharded)
- cuBLAS GEMM call
- NCCL AllReduce

### Template: 16gpu_tensor_parallel.cu

**Source from:** `DUAL_NODE_COMPLETE_SETUP.md` lines 478-601

**Differences from 8gpu version:**
- `world_size = 16` (MPI provides this automatically)
- Launch with: `mpirun -np 16 --hostfile hosts ...`
- NCCL automatically uses IB for inter-node communication
- Code is otherwise identical (NCCL abstracts topology)

### Template: pipeline.cu

**Source from:** `book.cu/8_distributed/pipeline/` (existing code)

**Structure:**
1. Two functions:
   - `process_batch_naive()` - Blocking sync version
   - `process_batch_async()` - Stream + event version
2. MLP kernel (simple GELU or ReLU layer)
3. `main()`:
   - Allocate buffers for all GPUs
   - Create streams and events
   - Benchmark both naive and async versions
   - Print throughput, speedup, efficiency

**Key code sections for in-text display:**
- Naive loop with `cudaDeviceSynchronize()`
- Async loop with streams and events
- Performance comparison table

## Appendix D: Multi-Node Setup (NEW FILE)

Create: `13_appendix.adoc` - Section on Distributed Setup

**Content from:**
- `DUAL_NODE_COMPLETE_SETUP.md` Part 0, Part 2
- `speculating/10_dist.md` lines 33-131 (setup commands)

**Structure:**
1. CUDA installation
2. MPI installation
3. SSH key setup
4. Hostfile configuration
5. InfiniBand setup (if available)
6. Network testing commands
7. NCCL installation and testing

**Purpose:** Keep chapter focused on concepts/code, move setup details to appendix

## Pre-Compilation Checklist

Before submitting 10.adoc:

- [ ] NO inline comments in code (only `<1>`, `<2>` markers)
- [ ] All listings have: `// Listing 10.X` comment + `.Caption` line
- [ ] All callout markers are IN code blocks
- [ ] All callout explanations are BELOW `----` delimiter
- [ ] Figures/listings numbered sequentially (10.1, 10.2, 10.3...)
- [ ] At least 1 paragraph between headings
- [ ] Blank lines between all list items
- [ ] Line length under 76 chars (55 with annotations)
- [ ] File references point to actual files in `book.cu/8_distributed/`
- [ ] No LLM slop phrases
- [ ] Run `./compile.py` - zero warnings

## Workflow

1. **Continue 10.adoc from line 50** (already has intro + Lab 1)
2. **Write Section 1** (NCCL intro) - terminal commands only
3. **Write Section 2** (8-GPU tensor parallel) - full Jupyter-style cells
4. **Write Section 3** (8-GPU pipeline) - naive vs async comparison
5. **Write Section 4** (Multi-node intro) - brief setup overview
6. **Write Section 5** (16-GPU tensor parallel) - emphasize code reuse
7. **Write Section 6** (Scaling analysis) - strong/weak scaling concepts
8. **Write Section 7** (Summary) - bullet list of key points
9. **Create diagram placeholders** with Excalidraw instructions
10. **Number everything** sequentially
11. **Run `./compile.py`** and fix warnings
12. **Create Appendix D** (setup details) in `13_appendix.adoc`

## Expected Output Files

1. **10.adoc** - Complete chapter (~1000-1500 lines)
2. **book.cu/8_distributed/tensor_parallel/8gpu_tensor_parallel.cu** - Working code
3. **book.cu/8_distributed/tensor_parallel/16gpu_tensor_parallel.cu** - Working code
4. **book.cu/8_distributed/pipeline/pipeline.cu** - Already exists, may need refinement
5. **book.cu/8_distributed/README.md** - How to build and run all examples
6. **Diagram instructions** - Embedded in 10.adoc as comments

## Success Criteria

- Chapter compiles with `./compile.py` - zero warnings
- All code examples reference actual files that compile and run
- Progressive learning: 8-GPU first, then 16-GPU
- Clear connection between hardware (NVLink/IB) and software (NCCL/MPI)
- Benchmark results match real-world expectations
- Reader can follow the progression: topology → single-GPU → multi-GPU → multi-node

## Performance Expectations (from DUAL_NODE_COMPLETE_SETUP.md)

- Single H100: ~770K GFLOPS (FP16 GEMM)
- 8 GPUs (single node): ~4.4M GFLOPS, 99% efficiency
- 16 GPUs (two nodes): ~8.9M GFLOPS, 99.8% efficiency
- Pipeline naive: 1.1x speedup on 8 GPUs (27% efficiency)
- Pipeline async: 7.8x speedup on 8 GPUs (98% efficiency)

Use these numbers in the chapter for concrete examples.

---

## Final Note

This chapter is about teaching readers how to scale beyond one GPU. Focus on:
1. **Hardware reality** - physical wires matter
2. **Progressive complexity** - single GPU → 8 GPU → 16 GPU
3. **Working examples** - all code must run and match benchmark results
4. **Practical insights** - when to use each parallelism strategy

You're writing for someone who just hit an OOM error and needs to understand their options. Make it practical, concrete, and runnable.

Good luck! 🚀

