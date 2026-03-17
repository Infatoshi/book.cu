#!/bin/bash
# This script synchronizes the local distributed training code repository with
# Node 0 of a remote cluster. It is the first step in setting up and running
# a distributed training job. It uses rsync to efficiently transfer the necessary
# files.

# Configuration for the remote cluster's Node 0.
# TODO: These variables must be configured by the user.
NODE0_IP="<FILL_IN>"
NODE0_USER="ubuntu"

# Ensures that the user has configured the IP address for Node 0.
# The script will exit with an error if the placeholder is still present.
if [ "$NODE0_IP" = "<FILL_IN>" ]; then
    echo "Error: Please edit this script and replace <FILL_IN> with your Node 0 IP"
    echo ""
    echo "Example:"
    echo '  NODE0_IP="172.16.0.55"'
    exit 1
fi

echo "========================================"
echo " Syncing Code to Cluster"
echo "========================================"
echo "Target: ${NODE0_USER}@${NODE0_IP}:~/distributed/"
echo ""

# Create the `distributed` directory on the remote Node 0 if it doesn't already exist.
# This is where the code will be synced.
ssh ${NODE0_USER}@${NODE0_IP} "mkdir -p ~/distributed"

# Use rsync to copy the local directory to the remote `~/distributed/` directory.
# rsync is used for its efficiency, as it only transfers changed files.
#
# --exclude patterns are used to avoid transferring unnecessary files:
#   - `results_*.txt`: Result files from previous runs are not needed on the cluster.
#   - `*.o`: Object files, which are intermediate compilation artifacts.
#   - `8gpu_single_node`, `16gpu_multi_node`: Compiled binaries. These will be rebuilt on the cluster.
#   - `hosts`: This file is generated on the cluster itself.
rsync -avz --progress \
    --exclude 'results_*.txt' \
    --exclude '*.o' \
    --exclude '8gpu_single_node' \
    --exclude '16gpu_multi_node' \
    --exclude 'hosts' \
    ./ \
    ${NODE0_USER}@${NODE0_IP}:~/distributed/

echo ""
echo "✓ Sync complete"
echo ""
# Provides instructions for the user on what to do after the code is synced.
# This typically involves SSH-ing into the cluster and running setup/execution scripts.
echo "Next steps:"
echo "  1. SSH to Node 0: ssh ${NODE0_USER}@${NODE0_IP}"
echo "  2. cd ~/distributed/setup_scripts"
echo "  3. ./node_setup.sh"
echo "  4. ./node0_only.sh (if multi-node)"
echo "  5. cd ~/distributed/tensor_parallel"
echo "  6. ./run_all.sh"
echo ""

