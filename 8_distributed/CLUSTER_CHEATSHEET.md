# Chapter 10 Cluster Execution - One-Page Cheatsheet

## Pre-Cluster (Local Machine)
```bash
cd /Users/elliotarledge/cuda/cuda-book/book.cu/8_distributed

# Edit these files with Node 0 IP:
vim local_sync.sh        # Replace <FILL_IN> with Node 0 IP
vim local_retrieve.sh    # Replace <FILL_IN> with Node 0 IP

# Sync code to cluster
./local_sync.sh
```

## On Cluster (Node 0)
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

## On Cluster (Node 1, parallel terminal)
```bash
ssh ubuntu@<node1-ip>

cd ~/distributed/setup_scripts
./node_setup.sh

# When node0_only.sh prompts, add Node 0's SSH key:
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo "ssh-rsa AAAA..." >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

## Post-Cluster (Local Machine)
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

## Chapter Generation (Local Machine)
```bash
# Use LLM with:
# - System prompt: CHAPTER_10_SYSTEM_PROMPT.md
# - Context: DUAL_NODE_COMPLETE_SETUP.md, results_*.txt, 10_roadmap_dist.md

# After LLM generates 10.adoc:
./compile.py          # Fix any warnings
git add 10.adoc
git commit -m "Chapter 10: Complete"
```

## Troubleshooting Quick Reference

| Problem | Solution |
|---------|----------|
| SSH fails | `ssh ubuntu@<node1-ip> "echo test"` |
| MPI can't find hosts | Check `~/distributed/hosts` (no tabs!) |
| CUDA OOM | Reduce M/N/K in .cu files (4096→2048) |
| mpi.h not found | `sudo apt install openmpi-bin libopenmpi-dev` |

## Expected Results

**8-GPU:** ~6-7M GFLOPS total, ~100% efficiency  
**16-GPU:** ~8-9M GFLOPS total, ~99% efficiency

## Time Budget

- Setup: 15 min
- Benchmarks: 10 min  
- Buffer: 15 min
- **Total: 40 min (~$20-60 depending on provider)**

## Key Files

- **CLUSTER_EXECUTION_ROADMAP.md** - Detailed step-by-step guide
- **CHAPTER_10_SYSTEM_PROMPT.md** - Complete LLM instructions
- **CHAPTER_10_QUICK_START.md** - 5-step workflow summary
- **DISTRIBUTED_CHAPTER_STATUS.md** - Full status and checklist

