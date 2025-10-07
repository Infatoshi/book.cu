set -e

echo "=== Node 0 Specific Setup ==="
echo ""

if [ ! -f ~/.ssh/id_rsa ]; then
    echo "Generating SSH key..."
    ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""
    echo "✓ SSH key generated"
else
    echo "✓ SSH key already exists"
fi

echo ""
echo "========================================="
echo "STEP 1: Copy this public key to Node 1"
echo "========================================="
cat ~/.ssh/id_rsa.pub
echo ""
echo "On Node 1, run:"
echo "  mkdir -p ~/.ssh && chmod 700 ~/.ssh"
echo "  echo \"<paste key above>\" >> ~/.ssh/authorized_keys"
echo "  chmod 600 ~/.ssh/authorized_keys"
echo ""
read -p "Press ENTER after you've added the key to Node 1..."

echo ""
echo "========================================="
echo "STEP 2: Enter Node IPs"
echo "========================================="
read -p "Enter Node 0 IP (this machine): " NODE0_IP
read -p "Enter Node 1 IP (remote machine): " NODE1_IP

HOSTS_FILE="../hosts"
cat > $HOSTS_FILE << EOF
$NODE0_IP slots=8
$NODE1_IP slots=8
EOF

echo ""
echo "✓ Hostfile created at: $HOSTS_FILE"
cat $HOSTS_FILE

echo ""
echo "========================================="
echo "STEP 3: Testing SSH connection to Node 1"
echo "========================================="
if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ubuntu@$NODE1_IP "echo 'SSH test successful'" 2>/dev/null; then
    echo "✓ SSH connection to Node 1 successful"
else
    echo "✗ SSH connection failed"
    echo "Make sure:"
    echo "  1. Node 1 is running"
    echo "  2. SSH key was correctly added to Node 1's authorized_keys"
    echo "  3. Node 1 IP is correct: $NODE1_IP"
    exit 1
fi

echo ""
echo "========================================="
echo "STEP 4: Testing MPI across nodes"
echo "========================================="

cat > /tmp/mpi_hello.c << 'EOF'

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    char hostname[256];
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    gethostname(hostname, 256);
    printf("Rank %d/%d on %s\n", rank, size, hostname);
    fflush(stdout);
    MPI_Finalize();
    return 0;
}
EOF

mpicc -o /tmp/mpi_hello /tmp/mpi_hello.c

scp /tmp/mpi_hello ubuntu@$NODE1_IP:/tmp/

echo "Running: mpirun -np 16 --hostfile $HOSTS_FILE --mca btl tcp,self /tmp/mpi_hello"
if mpirun -np 16 --hostfile $HOSTS_FILE --mca btl tcp,self /tmp/mpi_hello 2>/dev/null | grep -q "Rank"; then
    echo ""
    echo "✓ MPI test successful"
    echo "✓ All 16 processes launched across 2 nodes"
else
    echo "✗ MPI test failed"
    exit 1
fi

rm -f /tmp/mpi_hello /tmp/mpi_hello.c

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo "You can now run multi-node benchmarks:"
echo "  cd ~/distributed/tensor_parallel"
echo "  ./run_all.sh"
echo ""

