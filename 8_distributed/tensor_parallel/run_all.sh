set -e

echo "========================================"
echo " Tensor Parallel Benchmark Suite"
echo "========================================"
echo ""

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

echo "=== Building ==="
make clean
make
echo ""

echo "========================================"
echo " Test 1: 8 GPUs (Single Node)"
echo "========================================"
echo ""
echo "Running: mpirun -np 8 --mca btl tcp,self ./8gpu_single_node"
echo ""

mpirun -np 8 --mca btl tcp,self ./8gpu_single_node | tee results_8gpu.txt

echo ""
echo "✓ Results saved to: results_8gpu.txt"
echo ""

if [ "$MULTI_NODE" = true ]; then
    echo "========================================"
    echo " Test 2: 16 GPUs (Two Nodes)"
    echo "========================================"
    echo ""
    echo "Running: mpirun -np 16 --hostfile $HOSTFILE --mca btl tcp,self ./16gpu_multi_node"
    echo ""
    
    
    timeout 60 mpirun -np 16 --hostfile $HOSTFILE --mca btl tcp,self ./16gpu_multi_node | tee results_16gpu.txt || {
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 124 ]; then
            echo ""
            echo "Warning: Benchmark timed out after 60s (MPI_Finalize may hang)"
            echo "This is a known issue with some MPI configurations."
            echo "Results should still be valid (check results_16gpu.txt)"
        else
            echo ""
            echo "Error: Multi-node test failed with exit code $EXIT_CODE"
            exit $EXIT_CODE
        fi
    }
    
    echo ""
    echo "✓ Results saved to: results_16gpu.txt"
    echo ""
else
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

