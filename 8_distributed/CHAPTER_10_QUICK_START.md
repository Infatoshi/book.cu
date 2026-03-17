# Chapter 10: Quick Start Guide

## TL;DR - Complete Workflow

### For Cluster Execution (Cost-Optimized)
1. **Prepare locally** (~15 min): All scripts in `book.cu/8_distributed/` ✅ DONE
2. **Start 16 H100s** (2 nodes × 8 GPUs)
3. **Run `CLUSTER_EXECUTION_ROADMAP.md`** (~30 min on cluster)
4. **Retrieve results** (~5 min)
5. **Shutdown cluster** (IMMEDIATELY)
6. **Generate chapter** (locally, post-cluster)

### For Chapter Writing (Post-Cluster)
1. **Feed `CHAPTER_10_SYSTEM_PROMPT.md` to LLM** with:
   - `DUAL_NODE_COMPLETE_SETUP.md` (your successful setup)
   - `book.cu/8_distributed/tensor_parallel/results_*.txt` (benchmark results)
   - `speculating/10_roadmap_dist.md` (structure guide)
2. **LLM generates complete `10.adoc`**
3. **Run `./compile.py`** and fix warnings
4. **Done!**

---

## What You Have Now

### Documentation
- ✅ **CHAPTER_10_SYSTEM_PROMPT.md** - Complete system prompt for LLM to generate 10.adoc
- ✅ **CLUSTER_EXECUTION_ROADMAP.md** - Step-by-step cluster execution plan (<45 min)
- ✅ **DUAL_NODE_COMPLETE_SETUP.md** - Your successful 16 H100 setup from previous run
- ✅ **book.cu/8_distributed/README.md** - Code documentation and usage guide

### Code (Ready to Deploy)
- ✅ **setup_scripts/** - Automated node setup scripts
- ✅ **tensor_parallel/** - 8-GPU and 16-GPU examples with Makefile
- ✅ **local_sync.sh** - Push code to cluster
- ✅ **local_retrieve.sh** - Pull results from cluster

### What's Missing (You'll Generate)
- ⏳ **10.adoc** (lines 50+) - Complete chapter content (LLM generates this)
- ⏳ **Excalidraw diagrams** - Follow instructions embedded in 10.adoc
- ⏳ **pipeline/pipeline.cu** - Pipeline parallelism code (adapt from existing or write fresh)

---

## Cluster Execution: 5-Step Process

### Step 1: Update IPs in Sync Scripts (2 min)
```bash
cd /Users/elliotarledge/cuda/cuda-book/book.cu/8_distributed

# Get Node 0 IP after starting cluster
# Edit local_sync.sh and local_retrieve.sh:
sed -i '' 's/<FILL_IN>/ACTUAL_NODE0_IP/g' local_sync.sh
sed -i '' 's/<FILL_IN>/ACTUAL_NODE0_IP/g' local_retrieve.sh
```

### Step 2: Sync to Cluster (1 min)
```bash
./local_sync.sh
```

### Step 3: Execute on Cluster (30 min)
**SSH to Node 0:**
```bash
cd ~/distributed/setup_scripts
./node_setup.sh  # Install CUDA, MPI, verify GPUs

# Configure multi-node (follow prompts)
./node0_only.sh  # SSH keys, hostfile

# Run benchmarks
cd ~/distributed/tensor_parallel
./run_all.sh  # Runs 8-GPU and 16-GPU tests
```

**SSH to Node 1 (parallel terminal):**
```bash
cd ~/distributed/setup_scripts
./node_setup.sh  # Install CUDA, MPI, verify GPUs
# Add Node 0's SSH key to authorized_keys (instructions from node0_only.sh)
```

### Step 4: Retrieve Results (2 min)
**From local machine:**
```bash
cd /Users/elliotarledge/cuda/cuda-book/book.cu/8_distributed
./local_retrieve.sh
```

### Step 5: Shutdown Cluster (1 min)
```bash
# Via cloud provider console or:
aws ec2 stop-instances --instance-ids i-xxxxx i-yyyyy
```

**Total cluster time: ~35 minutes (cost: $30-60 depending on provider)**

---

## Chapter Generation: 3-Step Process

### Step 1: Prepare Context for LLM
Open your LLM (Cursor, Claude, etc.) and provide:

**System Prompt:**
```
Use CHAPTER_10_SYSTEM_PROMPT.md as your system prompt.
```

**Context Files:**
1. `DUAL_NODE_COMPLETE_SETUP.md` (proven setup guide)
2. `book.cu/8_distributed/tensor_parallel/results_8gpu.txt` (benchmark results)
3. `book.cu/8_distributed/tensor_parallel/results_16gpu.txt` (benchmark results)
4. `speculating/10_roadmap_dist.md` (chapter structure)
5. `10.adoc` (current chapter, lines 1-50)

**Prompt:**
```
Generate the complete Chapter 10 (10.adoc) from line 50 onwards, following the structure in CHAPTER_10_SYSTEM_PROMPT.md. Include:

1. Section 1: NCCL Introduction
2. Section 2: 8-GPU Tensor Parallelism (use results_8gpu.txt)
3. Section 3: 8-GPU Pipeline Parallelism
4. Section 4: Multi-Node Introduction
5. Section 5: 16-GPU Tensor Parallelism (use results_16gpu.txt)
6. Section 6: Scaling Analysis
7. Section 7: Summary

Use Jupyter-style code cells with AsciiDoc callouts (<1>, <2>). Include diagram placeholders with Excalidraw instructions. Follow all formatting rules from the system prompt.
```

### Step 2: Validate and Compile
```bash
cd /Users/elliotarledge/cuda/cuda-book
./compile.py

# Fix any AsciiDoc warnings
# Common issues:
# - Callout markers not in code
# - Missing captions on listings
# - Line length >76 chars
```

### Step 3: Create Diagrams (Optional, Can Be Later)
```bash
# Open Excalidraw
# Follow instructions embedded in 10.adoc comments
# Export as PNG to assets/10/

# Diagrams needed:
# - nccl-operations.png (AllReduce, Broadcast, Gather)
# - tensor-parallel-8gpu.png (Matrix split across 8 GPUs)
# - pipeline-gantt-naive.png (Idle GPUs)
# - pipeline-gantt-optimized.png (Staircase pattern)
# - multi-node-architecture.png (2 nodes, IB connection)
# - strong-scaling.png (Speedup chart)
# - weak-scaling.png (Efficiency chart)
```

---

## Key Files Reference

### Documentation
| File | Purpose | Status |
|------|---------|--------|
| `CHAPTER_10_SYSTEM_PROMPT.md` | Complete instructions for LLM | ✅ Ready |
| `CLUSTER_EXECUTION_ROADMAP.md` | Step-by-step cluster execution | ✅ Ready |
| `DUAL_NODE_COMPLETE_SETUP.md` | Proven setup from previous run | ✅ Ready |
| `book.cu/8_distributed/README.md` | Code documentation | ✅ Ready |

### Code
| File | Purpose | Status |
|------|---------|--------|
| `setup_scripts/node_setup.sh` | Node initialization (both nodes) | ✅ Ready |
| `setup_scripts/node0_only.sh` | SSH keys, hostfile (Node 0) | ✅ Ready |
| `tensor_parallel/8gpu_single_node.cu` | 8-GPU tensor parallel | ✅ Ready |
| `tensor_parallel/16gpu_multi_node.cu` | 16-GPU tensor parallel | ✅ Ready |
| `tensor_parallel/Makefile` | Build system | ✅ Ready |
| `tensor_parallel/run_all.sh` | Automated benchmark runner | ✅ Ready |
| `pipeline/pipeline.cu` | Pipeline parallelism | ⏳ Adapt/create |

### Chapter Content
| File | Purpose | Status |
|------|---------|--------|
| `10.adoc` (lines 1-50) | Intro + hardware topology | ✅ Done |
| `10.adoc` (lines 50+) | Main content | ⏳ LLM generates |
| `assets/10/*.png` | Diagrams | ⏳ Create later |

---

## What Makes This Different from Previous Attempts

### Previous Issues (from your notes)
- ❌ Setup scripts broke mid-execution
- ❌ Had to debug on expensive cluster
- ❌ No clear roadmap → wasted time

### This Approach
- ✅ **All scripts prepared and validated locally first**
- ✅ **Clear separation: cluster (benchmarks) vs local (chapter writing)**
- ✅ **Precise time estimates for each phase**
- ✅ **Automated execution scripts → minimal manual steps**
- ✅ **Complete system prompt for LLM → one-shot chapter generation**
- ✅ **Based on DUAL_NODE_COMPLETE_SETUP.md → proven to work**

---

## Expected Timeline

### Cluster Phase (On Cloud)
| Task | Time | Cost |
|------|------|------|
| Start cluster + get IPs | 5 min | $8 |
| Sync code to cluster | 1 min | $2 |
| Node setup (both nodes) | 10 min | $16 |
| Run benchmarks | 10 min | $16 |
| Retrieve results | 2 min | $3 |
| **Buffer for issues** | 10 min | $16 |
| **TOTAL** | **38 min** | **~$61** |

*Assumes $98/hour for 2× p5.48xlarge (AWS) = $1.63/min for both nodes*

### Local Phase (Free)
| Task | Time |
|------|------|
| Prepare LLM context | 5 min |
| LLM generates 10.adoc | 10 min |
| Review and fix warnings | 20 min |
| Create diagrams | 60 min (optional) |
| **TOTAL** | **35-95 min** |

---

## Risk Mitigation

### If Cluster Fails
- **Checkpoint:** Results saved to `~/distributed/tensor_parallel/results_*.txt`
- **Retrieve partial results:** `./local_retrieve.sh` works even if benchmark crashes
- **Restart:** Scripts are idempotent (can run multiple times safely)

### If Chapter Generation Fails
- **Fallback:** Use DUAL_NODE_COMPLETE_SETUP.md as prose template
- **Iterate:** System prompt is comprehensive; LLM can regenerate sections
- **Manual:** Worst case, write chapter manually using system prompt as guide

### If Diagrams Take Too Long
- **Defer:** Placeholders in 10.adoc allow compilation without diagrams
- **Commission:** Can hire designer later using embedded Excalidraw instructions
- **Screenshot:** Use `nvidia-smi topo -m` output as interim figure

---

## Success Criteria

### Cluster Phase
- [ ] `results_8gpu.txt` shows ~6-7M GFLOPS total (efficiency >95%)
- [ ] `results_16gpu.txt` shows ~8-9M GFLOPS total (efficiency >95%)
- [ ] Total cluster time <60 minutes
- [ ] All results transferred to local machine
- [ ] Cluster shut down (verify billing stopped)

### Chapter Phase
- [ ] `10.adoc` compiles with `./compile.py` (zero warnings)
- [ ] All code listings have captions and callouts
- [ ] Figures numbered sequentially (10.1, 10.2, ...)
- [ ] Benchmark results match cluster output
- [ ] Chapter follows structure from CHAPTER_10_SYSTEM_PROMPT.md

---

## Next Steps

1. **Review these files:**
   - [ ] Read `CHAPTER_10_SYSTEM_PROMPT.md` (understand LLM instructions)
   - [ ] Read `CLUSTER_EXECUTION_ROADMAP.md` (memorize cluster workflow)
   - [ ] Skim `book.cu/8_distributed/README.md` (understand code structure)

2. **Prepare for cluster:**
   - [ ] Choose cloud provider (AWS, Lambda, Vast.ai)
   - [ ] Provision 2 nodes × 8 H100 GPUs each
   - [ ] Get Node 0 and Node 1 IPs
   - [ ] Update `local_sync.sh` and `local_retrieve.sh` with Node 0 IP

3. **Execute cluster phase:**
   - [ ] Follow `CLUSTER_EXECUTION_ROADMAP.md` step-by-step
   - [ ] Use `tmux` on both nodes (survive SSH drops)
   - [ ] Save all terminal output (use `tee`)

4. **Generate chapter:**
   - [ ] Feed `CHAPTER_10_SYSTEM_PROMPT.md` to LLM
   - [ ] Provide context files (DUAL_NODE_COMPLETE_SETUP.md, results_*.txt)
   - [ ] Review generated `10.adoc`
   - [ ] Run `./compile.py` and fix warnings

5. **Finalize:**
   - [ ] Create diagrams (or defer to later)
   - [ ] Commit to git
   - [ ] Celebrate! 🎉

---

## Questions?

If anything is unclear:
1. Check the relevant file:
   - Setup questions → `CLUSTER_EXECUTION_ROADMAP.md`
   - Chapter structure → `CHAPTER_10_SYSTEM_PROMPT.md`
   - Code usage → `book.cu/8_distributed/README.md`
2. Refer to proven guide: `DUAL_NODE_COMPLETE_SETUP.md`
3. Test locally first with smaller examples (1-2 GPUs on local machine)

**Good luck with your cluster run!** 🚀

---

## Appendix: One-Command Cluster Execution

If everything is set up correctly, you can run everything from Node 0 in one go:

```bash
# On Node 0 (after SSH keys and hostfile configured)
cd ~/distributed
(cd setup_scripts && ./node_setup.sh) && \
(cd tensor_parallel && ./run_all.sh) && \
echo "DONE! Retrieve results now."
```

**On local machine:**
```bash
# Retrieve results
cd /Users/elliotarledge/cuda/cuda-book/book.cu/8_distributed
./local_retrieve.sh

# Shutdown cluster (AWS example)
aws ec2 stop-instances --instance-ids i-xxxxx i-yyyyy
```

**Total hands-on time: ~5 minutes** (rest is automated)

