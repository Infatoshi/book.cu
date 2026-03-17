# ONE-SHOT PROMPT: Generate Complete Chapter 10

Copy-paste this entire prompt to an LLM (Claude, GPT-4, etc.) to generate the complete Chapter 10 in one go.

---

## Your Task

Generate the complete **Chapter 10: Distributed Computing** for the book "CUDA for Deep Learning" starting from line 50 of `10.adoc`. The chapter must be production-ready, compile with zero warnings via `./compile.py`, and follow all AsciiDoc formatting rules.

## Context Files (Read These First)

### Chapter Structure & Rules
- **CHAPTER_10_SYSTEM_PROMPT.md** - Complete formatting rules, structure, and requirements
- **speculating/10_roadmap_dist.md** - Detailed chapter outline and section plan
- **docs/llms.md** - AsciiDoc formatting rules (callouts, captions, line length, etc.)

### Working Code Examples
- **book.cu/8_distributed/README.md** - Overview of all distributed examples
- **book.cu/8_distributed/tensor_parallel/** - 8-GPU and 16-GPU tensor parallelism
  - `8gpu_single_node.cu` - Single-node implementation (4096³ GEMM)
  - `16gpu_multi_node.cu` - Multi-node implementation (2048³ GEMM)
  - `Makefile` and `run_all.sh` - Build and execution scripts
- **book.cu/8_distributed/pipeline_parallel/** - Pipeline parallelism with CUDA streams
  - `pipeline.cu` - Complete implementation (naive vs optimized)
  - `README.md` - Usage guide and performance results (~4x speedup)
  - `ARCHITECTURE.md` - Deep dive into stream-based concurrency

### Reference Material
- **DUAL_NODE_COMPLETE_SETUP.md** - Proven 16 H100 setup from real cluster run
- **10.adoc (lines 1-50)** - Existing chapter intro (hardware topology, NVLink vs PCIe)

## Chapter Outline

Continue `10.adoc` from line 50 with these sections:

### Section 1: Understanding NCCL (~100 lines)
- What is NCCL and why it matters
- Collective operations: AllReduce, Broadcast, Gather
- Topology-aware communication (NVLink > PCIe)
- Brief mention of `nccl-tests` for benchmarking
- **No C++ code yet** - just terminal commands and concepts

### Section 2: Tensor Parallelism - 8 GPUs (~200 lines)
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

### Section 3: Pipeline Parallelism - 8 GPUs (~250 lines)
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

### Section 4: Multi-Node Introduction (~80 lines)
- When 8 GPUs isn't enough
- New bottleneck: Inter-node communication
- Hardware: InfiniBand (200 Gb/s = 25 GB/s) vs NVLink (600 GB/s)
- Software: MPI for process management
- Brief setup overview (details in Appendix D)

### Section 5: Tensor Parallelism - 16 GPUs (~150 lines)
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

### Section 6: Understanding Scaling (~100 lines)
- **Strong scaling:** Fixed problem size + more GPUs
- **Weak scaling:** Scale problem size with GPU count
- When to use each parallelism strategy:
  - Data parallel: Large batch training
  - Tensor parallel: Wide models (big hidden dims)
  - Pipeline parallel: Deep models (many layers)
  - Hybrid: Combine all (Megatron-LM, TensorRT-LLM)

### Section 7: Summary (~50 lines)
Bullet list covering:
- Hardware hierarchy: NVLink > IB > Ethernet
- NCCL for collectives, MPI for multi-node
- Streams + Events = Concurrency
- Real systems combine all approaches

## Critical Formatting Rules

### 1. Code Annotations (MANDATORY)
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

### 2. Code Content
- ❌ NO `#include` statements
- ❌ NO `import` statements
- ✅ Only core logic with annotation markers
- ✅ Max 76 chars per line (55 if annotations)
- ✅ Wrap long lines for readability

### 3. Listings Must Reference Files
After each code block:
```asciidoc
The complete implementation is available at:
`book.cu/8_distributed/tensor_parallel/8gpu_single_node.cu`
```

### 4. Figures (Create Placeholders)
```asciidoc
// Figure 10.X: Title
.Caption describing what diagram shows
image::diagram-name.png[]
```

Add detailed Excalidraw instructions in comments.

### 5. Sequential Numbering
- Figures: 10.1, 10.2, 10.3... (NO GAPS)
- Listings: 10.1, 10.2, 10.3... (NO GAPS)
- Track with comments: `// Figure 10.X: Description`

### 6. Headings
```asciidoc
== Main Section
=== Subsection
==== Deeper Level
```
- ❌ NO numbers in headings
- ✅ At least 1 paragraph between any two headings

### 7. Lists
```asciidoc
* First item

* Second item

* Third item
```
- ✅ Blank line between EVERY item
- ✅ Only use `*` (never `-`)

### 8. Language Style
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

## Diagram Placeholders Needed

Create placeholders with Excalidraw instructions for:
1. **nccl-operations.png** - Visual of AllReduce, Broadcast, Gather
2. **tensor-parallel-8gpu.png** - Matrix split across 8 GPUs
3. **pipeline-gantt-naive.png** - Timeline showing idle GPUs
4. **pipeline-gantt-optimized.png** - Staircase pattern (all GPUs active)
5. **multi-node-arch.png** - 2 nodes with IB connection
6. **strong-scaling.png** - Speedup vs GPU count chart

## Key Numbers to Include

### Tensor Parallelism
- 8 GPUs: ~6-7M GFLOPS, ~100% efficiency, NVLink bandwidth
- 16 GPUs: ~8.9M GFLOPS, 99.8% efficiency, IB + NVLink

### Pipeline Parallelism
- Naive: 167k samples/s, 1.1x speedup, 27% efficiency
- Optimized: 698k samples/s, 4.17x speedup, 104% efficiency

### Hardware
- NVLink: 600 GB/s (intra-node)
- InfiniBand HDR: 200 Gb/s = 25 GB/s (inter-node)
- PCIe Gen4: 32 GB/s (fallback)

## Output Requirements

1. **Start at line 50** - Don't repeat the intro (lines 1-50)
2. **~1000-1200 lines** - Complete chapter content
3. **Zero warnings** - Must compile with `./compile.py`
4. **All callouts in code** - Markers `<1>`, `<2>` on actual lines
5. **Sequential numbering** - No gaps in figures/listings
6. **File references** - Point to actual code in `book.cu/8_distributed/`

## Success Criteria

- [ ] Compiles with zero warnings
- [ ] All code blocks have captions and callouts
- [ ] Figures numbered sequentially
- [ ] No inline comments in code (only annotations)
- [ ] Line length under 76 chars
- [ ] Benchmark results integrated from README files
- [ ] Follows structure from CHAPTER_10_SYSTEM_PROMPT.md

## Begin Generation

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

