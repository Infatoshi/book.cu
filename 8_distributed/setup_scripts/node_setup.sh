set -e

echo "=== Node Setup Starting on $(hostname) ==="

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

if ! grep -q "CUDA_HOME" ~/.bashrc; then
    cat >> ~/.bashrc << 'EOF'

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF
    echo "CUDA environment added to ~/.bashrc"
fi

if ! command -v mpirun &> /dev/null; then
    echo "Installing OpenMPI..."
    sudo apt update
    sudo apt install -y openmpi-bin libopenmpi-dev build-essential
else
    echo "OpenMPI already installed"
fi

echo ""
echo "=== Verification ==="

if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi found"
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "✓ GPU count: $GPU_COUNT"
else
    echo "✗ nvidia-smi not found (install NVIDIA drivers)"
    exit 1
fi

if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
    echo "✓ CUDA version: $CUDA_VERSION"
else
    echo "✗ nvcc not found (install CUDA toolkit)"
    exit 1
fi

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

