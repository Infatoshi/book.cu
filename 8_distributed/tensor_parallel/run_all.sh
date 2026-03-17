#!/bin/bash
# This script is a benchmark runner for tensor parallelism examples. It compiles
# the CUDA code and then executes the benchmarks for both single-node (8 GPUs)
# and multi-node (16 GPUs) configurations.

# `set -e` ensures that the script will exit immediately if any command fails.
set -e

echo "========================================"
echo " Tensor Parallel Benchmark Suite"
echo "========================================"
echo ""

# Check for the existence of the MPI hostfile.
# The hostfile is required for multi-node execution and is created by the
# `node0_only.sh` setup script. If it's not found, the multi-node test
# will be skipped.
HOSTFILE="../hosts"
if [ ! -f "$HOSTFILE" ]; then
    echo "Warning: Hostfile not found at $HOSTFILE"
    echo "Will only run single-node (8 GPU) test."
    echo "To run multi-node test, create hostfile first:"
    echo "  cd ../setup_scripts && ./node0_only.sh"
    echo ""
    MULTI_NODE=false
else
    MULTI_NODE=true
fi

# Compile the CUDA source files.
# `make clean` removes old binaries and object files.
# `make` builds the executables for the benchmarks.
echo "=== Building ==="
make clean
make
echo ""

# --- Single-Node Benchmark ---
echo "========================================"
echo " Test 1: 8 GPUs (Single Node)"
echo "========================================"
echo ""
echo "Running: mpirun -np 8 --mca btl tcp,self ./8gpu_single_node"
echo ""

# Execute the single-node benchmark for 8 GPUs.
# - `mpirun -np 8`: Launch 8 processes.
# - `--mca btl tcp,self`: Use the TCP and self BTL components for inter-process
#   communication. This is standard for single-node runs.
# - `./8gpu_single_node`: The executable to run.
# The output is piped to `tee`, which writes it to both the console and the
# specified results file.
mpirun -np 8 --mca btl tcp,self ./8gpu_single_node | tee results_8gpu.txt

echo ""
echo "✓ Results saved to: results_8gpu.txt"
echo ""

# --- Multi-Node Benchmark ---
if [ "$MULTI_NODE" = true ]; then
    echo "========================================"
    echo " Test 2: 16 GPUs (Two Nodes)"
    echo "========================================"
    echo ""
    echo "Running: mpirun -np 16 --hostfile $HOSTFILE --mca btl tcp,self ./16gpu_multi_node"
    echo ""
    
    # Execute the multi-node benchmark for 16 GPUs across two nodes.
    # - `mpirun -np 16`: Launch 16 total processes.
    # - `--hostfile $HOSTFILE`: Use the provided hostfile to distribute the
    #   processes across the specified nodes.
    # - `timeout 60`: The command is wrapped in a 60-second timeout. This is a
    #   workaround for a potential issue where `MPI_Finalize` can hang, causing
    #   the script to stall indefinitely.
    timeout 60 mpirun -np 16 --hostfile $HOSTFILE --mca btl tcp,self ./16gpu_multi_node | tee results_16gpu.txt || {
        EXIT_CODE=$?
        # Check if the command failed due to the timeout (exit code 124).
        if [ $EXIT_CODE -eq 124 ]; then
            echo ""
            echo "Warning: Benchmark timed out after 60s (MPI_Finalize may hang)"
            echo "This is a known issue with some MPI configurations."
            echo "Results should still be valid (check results_16gpu.txt)"
        else
            # If it failed for another reason, report the error and exit.
            echo ""
            echo "Error: Multi-node test failed with exit code $EXIT_CODE"
            exit $EXIT_CODE
        fi
    }
    
    echo ""
    echo "✓ Results saved to: results_16gpu.txt"
    echo ""
else
    # If the hostfile was not found, this block is executed, and the
    # multi-node test is skipped. Instructions are provided for the user.
    echo "========================================"
    echo " Test 2: Skipped (No Hostfile)"
    echo "========================================"
    echo ""
    echo "To run multi-node test:"
    echo "  1. cd ../setup_scripts"
    echo "  2. ./node0_only.sh"
    echo "  3. Return here and run: ./run_all.sh"
    echo ""
fi

echo "========================================"
echo " All Tests Complete"
echo "========================================"
echo ""
echo "Results:"
if [ -f "results_8gpu.txt" ]; then
    echo "  ✓ 8-GPU:  results_8gpu.txt"
fi
if [ -f "results_16gpu.txt" ]; then
    echo "  ✓ 16-GPU: results_16gpu.txt"
fi
echo ""

