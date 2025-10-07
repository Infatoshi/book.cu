
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
echo " Syncing Code to Cluster"
echo "========================================"
echo "Target: ${NODE0_USER}@${NODE0_IP}:~/distributed/"
echo ""

ssh ${NODE0_USER}@${NODE0_IP} "mkdir -p ~/distributed"

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
echo "Next steps:"
echo "  1. SSH to Node 0: ssh ${NODE0_USER}@${NODE0_IP}"
echo "  2. cd ~/distributed/setup_scripts"
echo "  3. ./node_setup.sh"
echo "  4. ./node0_only.sh (if multi-node)"
echo "  5. cd ~/distributed/tensor_parallel"
echo "  6. ./run_all.sh"
echo ""

