
NODE0_IP="<FILL_IN>"
NODE0_USER="ubuntu"

if [ "$NODE0_IP" = "<FILL_IN>" ]; then
    echo "Error: Please edit this script and replace <FILL_IN> with your Node 0 IP"
    echo ""
    echo "Example:"
    echo '  NODE0_IP="172.16.0.55"'
    exit 1
fi

echo "========================================"
echo " Retrieving Results from Cluster"
echo "========================================"
echo "Source: ${NODE0_USER}@${NODE0_IP}:~/distributed/"
echo ""

rsync -avz --progress \
    ${NODE0_USER}@${NODE0_IP}:~/distributed/tensor_parallel/results_*.txt \
    ./tensor_parallel/ || {
    echo "Warning: Failed to retrieve some files (they may not exist yet)"
}

echo ""

if [ -f "tensor_parallel/results_8gpu.txt" ]; then
    echo "✓ Retrieved: tensor_parallel/results_8gpu.txt"
    echo ""
    echo "--- 8-GPU Results Preview ---"
    tail -n 10 tensor_parallel/results_8gpu.txt
    echo ""
else
    echo "✗ Not found: tensor_parallel/results_8gpu.txt"
fi

if [ -f "tensor_parallel/results_16gpu.txt" ]; then
    echo "✓ Retrieved: tensor_parallel/results_16gpu.txt"
    echo ""
    echo "--- 16-GPU Results Preview ---"
    tail -n 10 tensor_parallel/results_16gpu.txt
    echo ""
else
    echo "✗ Not found: tensor_parallel/results_16gpu.txt"
fi

echo "========================================"
echo " Retrieval Complete"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Review results: cat tensor_parallel/results_*.txt"
echo "  2. Shutdown cluster to stop billing"
echo "  3. Generate chapter: Use CHAPTER_10_SYSTEM_PROMPT.md with LLM"
echo "  4. Commit results: git add tensor_parallel/results_*.txt && git commit"
echo ""

