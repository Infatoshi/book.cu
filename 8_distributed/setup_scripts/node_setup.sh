#!/bin/bash
# This script is intended to be run on every node in the distributed cluster.
# It sets up the necessary environment for running distributed CUDA applications,
# including configuring environment variables for the CUDA Toolkit and installing
# OpenMPI if it is not already present. It also verifies that the required
# software (NVIDIA drivers, CUDA, MPI) is correctly installed.

# `set -e` ensures that the script will exit immediately if any command fails.
set -e

echo "=== Node Setup Starting on $(hostname) ==="

# Set up environment variables for the CUDA Toolkit.
# These are necessary for the compiler (nvcc) and runtime to find the CUDA
# libraries and executables.
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Persist the CUDA environment variables by adding them to the user's .bashrc file.
# This ensures that they are set automatically in future shell sessions.
if ! grep -q "CUDA_HOME" ~/.bashrc; then
    cat >> ~/.bashrc << 'EOF'

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF
    echo "CUDA environment added to ~/.bashrc"
fi

# Check for and install OpenMPI if it's not found.
# OpenMPI is used to launch and manage the processes in a distributed training job.
# `mpirun` is the command used to start MPI applications.
if ! command -v mpirun &> /dev/null; then
    echo "Installing OpenMPI..."
    # Update package lists and install OpenMPI packages.
    sudo apt update
    sudo apt install -y openmpi-bin libopenmpi-dev build-essential
else
    echo "OpenMPI already installed"
fi

echo ""
echo "=== Verification ==="

# Verify that the NVIDIA drivers are installed and GPUs are visible.
# `nvidia-smi` is the NVIDIA System Management Interface.
if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi found"
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "✓ GPU count: $GPU_COUNT"
else
    echo "✗ nvidia-smi not found (install NVIDIA drivers)"
    exit 1
fi

# Verify that the CUDA Toolkit is installed.
# `nvcc` is the NVIDIA CUDA Compiler.
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
    echo "✓ CUDA version: $CUDA_VERSION"
else
    echo "✗ nvcc not found (install CUDA toolkit)"
    exit 1
fi

# Verify that OpenMPI is installed correctly.
if command -v mpirun &> /dev/null; then
    MPI_VERSION=$(mpirun --version | head -1)
    echo "✓ MPI: $MPI_VERSION"
else
    echo "✗ mpirun not found"
    exit 1
fi

echo ""
echo "=== Node Setup Complete on $(hostname) ==="
echo "Hostname: $(hostname)"
echo "IP: $(hostname -I | awk '{print $1}')"
echo ""
echo "Ready for multi-GPU or multi-node execution."

