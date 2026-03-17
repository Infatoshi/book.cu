#!/bin/bash
# This script is designed to retrieve the results of distributed training jobs
# from a remote cluster, specifically from Node 0. It uses rsync to copy
# the result files back to the local machine for analysis.

# Configuration for the remote cluster's Node 0.
# TODO: These variables must be configured by the user.
NODE0_IP="<FILL_IN>"
NODE0_USER="ubuntu"

# Check if the user has updated the placeholder IP address.
# The script will exit if the default value is still present, preventing
# connection errors.
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

# Use rsync to securely copy the result files from Node 0.
# rsync is efficient, transferring only the differences between files.
# The ` --progress` flag shows transfer progress.
# The script attempts to copy all files matching `results_*.txt` from the
# `~/distributed/tensor_parallel/` directory on the remote node.
rsync -avz --progress \
    ${NODE0_USER}@${NODE0_IP}:~/distributed/tensor_parallel/results_*.txt \
    ./tensor_parallel/ || {
    # If rsync fails, it might be because the result files do not exist yet.
    # A warning is printed, but the script continues execution.
    echo "Warning: Failed to retrieve some files (they may not exist yet)"
}

echo ""

# After attempting to retrieve the files, check for their local existence
# and provide a status update to the user.

# Check for the 8-GPU results file and show a preview if it exists.
if [ -f "tensor_parallel/results_8gpu.txt" ]; then
    echo "✓ Retrieved: tensor_parallel/results_8gpu.txt"
    echo ""
    echo "--- 8-GPU Results Preview ---"
    tail -n 10 tensor_parallel/results_8gpu.txt
    echo ""
else
    echo "✗ Not found: tensor_parallel/results_8gpu.txt"
fi

# Check for the 16-GPU results file and show a preview if it exists.
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
# Provide the user with guidance on the next steps after retrieving the results.
echo "Next steps:"
echo "  1. Review results: cat tensor_parallel/results_*.txt"
echo "  2. Shutdown cluster to stop billing"
echo "  3. Generate chapter: Use CHAPTER_10_SYSTEM_PROMPT.md with LLM"
echo "  4. Commit results: git add tensor_parallel/results_*.txt && git commit"
echo ""

