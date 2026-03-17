#!/bin/bash
# This script performs setup tasks that are only required on Node 0 of a
# multi-node cluster. It is responsible for setting up passwordless SSH access
# to other nodes and creating the hostfile required by MPI for launching
# distributed jobs.

# `set -e` ensures that the script will exit immediately if any command fails.
set -e

echo "=== Node 0 Specific Setup ==="
echo ""

# Step 1: Set up passwordless SSH for Node 0 to communicate with other nodes.
# This is crucial for MPI to be able to launch processes on all nodes without
# requiring a password for each one.
if [ ! -f ~/.ssh/id_rsa ]; then
    echo "Generating SSH key..."
    # Generate a new RSA SSH key pair without a passphrase.
    ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""
    echo "✓ SSH key generated"
else
    echo "✓ SSH key already exists"
fi

echo ""
echo "========================================="
echo "STEP 1: Copy this public key to Node 1"
echo "========================================="
# Display the public key and provide instructions for the user to manually
# add it to the `authorized_keys` file on Node 1. This grants Node 0
# the ability to SSH into Node 1 without a password.
cat ~/.ssh/id_rsa.pub
echo ""
echo "On Node 1, run:"
echo "  mkdir -p ~/.ssh && chmod 700 ~/.ssh"
echo "  echo \"<paste key above>\" >> ~/.ssh/authorized_keys"
echo "  chmod 600 ~/.ssh/authorized_keys"
echo ""
# The script pauses here, waiting for the user to complete the manual step.
read -p "Press ENTER after you've added the key to Node 1..."

echo ""
echo "========================================="
echo "STEP 2: Enter Node IPs"
echo "========================================="
# Prompt the user for the IP addresses of both nodes in the cluster.
read -p "Enter Node 0 IP (this machine): " NODE0_IP
read -p "Enter Node 1 IP (remote machine): " NODE1_IP

# Define the path for the MPI hostfile.
HOSTS_FILE="../hosts"
# Create the hostfile. This file tells MPI which nodes are available in the
# cluster and how many process slots (typically corresponding to the number
# of GPUs) are available on each node.
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
# Verify that the passwordless SSH connection to Node 1 is working correctly.
# -o StrictHostKeyChecking=no bypasses the prompt for adding the host to known_hosts.
# -o ConnectTimeout=5 sets a timeout to prevent long hangs.
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

# This step verifies that MPI can successfully launch processes across both nodes.
# A simple "hello world" MPI program is created, compiled, and run.

# Create a temporary C source file for the MPI test.
cat > /tmp/mpi_hello.c << 'EOF'
#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

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

# Compile the MPI program using the mpicc wrapper.
mpicc -o /tmp/mpi_hello /tmp/mpi_hello.c

# Copy the compiled executable to the same location on Node 1.
scp /tmp/mpi_hello ubuntu@$NODE1_IP:/tmp/

# Run the MPI program using mpirun.
# -np 16: Launch 16 total processes.
# --hostfile: Use the specified hostfile to determine where to launch processes.
# --mca btl tcp,self: Use the TCP and self BTL components for communication.
echo "Running: mpirun -np 16 --hostfile $HOSTS_FILE --mca btl tcp,self /tmp/mpi_hello"
if mpirun -np 16 --hostfile $HOSTS_FILE --mca btl tcp,self /tmp/mpi_hello 2>/dev/null | grep -q "Rank"; then
    echo ""
    echo "✓ MPI test successful"
    echo "✓ All 16 processes launched across 2 nodes"
else
    echo "✗ MPI test failed"
    exit 1
fi

# Clean up the temporary source and binary files.
rm -f /tmp/mpi_hello /tmp/mpi_hello.c

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo "You can now run multi-node benchmarks:"
echo "  cd ~/distributed/tensor_parallel"
echo "  ./run_all.sh"
echo ""

