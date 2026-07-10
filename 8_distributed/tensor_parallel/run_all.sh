#!/bin/bash
# Benchmark runner for the tensor parallelism example. Builds the binary and
# runs the single-node benchmark on all local GPUs. If an MPI hostfile exists
# at ../hosts, it also runs the multi-node configuration.
set -e

echo "========================================"
echo " Tensor Parallel Benchmark"
echo "========================================"
echo ""

echo "=== Building ==="
make clean
make
echo ""

NGPU=$(nvidia-smi --list-gpus | wc -l)
echo "========================================"
echo " Test 1: ${NGPU} GPUs (single node)"
echo "========================================"
mpirun -np "$NGPU" ./tensor_parallel | tee results_single_node.txt
echo "Results saved to: results_single_node.txt"
echo ""

HOSTFILE="../hosts"
if [ -f "$HOSTFILE" ]; then
    NRANKS=$(awk -F'slots=' '{s+=$2} END {print s}' "$HOSTFILE")
    echo "========================================"
    echo " Test 2: ${NRANKS} GPUs (multi node, hostfile ${HOSTFILE})"
    echo "========================================"
    mpirun -np "$NRANKS" --hostfile "$HOSTFILE" ./tensor_parallel | tee results_multi_node.txt
    echo "Results saved to: results_multi_node.txt"
else
    echo "No hostfile at ${HOSTFILE}; skipping multi-node run."
fi
