# CUDA Naive Neural Network Operations

This repository contains **naive but correct** CUDA implementations of the fundamental operations in deep learning, extracted from **Chapter 03: Building the Core Operations from Scratch** of the CUDA Programming Book.

## Overview

These implementations prioritize **correctness over performance**. Each GPU kernel is paired with a CPU reference implementation for verification. This "CPU-first" approach ensures you understand the algorithms before optimizing them.

The operations implemented here form the computational core of modern neural networks:

- **Element-wise operations**: Vector/matrix addition (foundation of residual connections)
- **Matrix operations**: Transpose, GEMM (core of linear layers)
- **Neural network primitives**: Softmax, 1D/2D convolution, max pooling

## Repository Structure

```
naive/
├── common.h                 # Shared utilities (timing, CUDA helpers, verification)
├── Makefile                 # Build system for all examples
├── README.md               # This file
├── elementwise/            # Basic element-wise operations
│   ├── vector_add.cu       # 1D vector addition
│   └── matrix_add.cu       # 2D matrix addition
├── transpose/              # Matrix transposition
│   └── transpose.cu        # Out-of-place matrix transpose
├── gemm/                   # General Matrix Multiplication
│   └── gemm.cu            # C = A * B (naive implementation)
├── softmax/                # Softmax activation
│   └── softmax.cu         # Naive softmax (inefficient but correct)
├── conv1d/                 # 1D convolution
│   └── conv1d.cu          # 1D signal processing convolution
├── conv2d/                 # 2D convolution
│   └── conv2d.cu          # 2D image convolution (CNN core)
└── maxpool2d/              # 2D max pooling
    └── maxpool2d.cu       # Downsampling operation
```

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- GNU Make

### Build All Examples

```bash
cd naive
make all
```

### Run All Examples

```bash
make run
```

### Build and Run Individual Examples

```bash
# Build specific operation
make vector_add
./vector_add

# Or use run targets
make run_gemm
```

## Understanding the "Naive" Approach

### CPU-First Development

Every GPU kernel follows this pattern:

1. **Implement CPU version**: Clear, sequential code for verification
2. **Implement GPU kernel**: Parallelize the CPU logic
3. **Verify correctness**: Compare GPU output with CPU reference
4. **Measure performance**: Understand baseline performance

### Thread Indexing Patterns

All kernels use consistent indexing:

```cpp
// 1D indexing (vectors, 1D conv)
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D indexing (matrices, 2D operations)
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;

// Boundary checks
if (row < height && col < width) {
    // Compute...
}
```

## Operation Details

### Element-wise Operations

#### Vector Addition
- **Purpose**: `c[i] = a[i] + b[i]`
- **Parallelization**: 1 thread per element
- **Use case**: Bias addition, residual connections

#### Matrix Addition
- **Purpose**: `C[i,j] = A[i,j] + B[i,j]`
- **Parallelization**: 2D grid matching matrix dimensions
- **Use case**: Batch normalization, layer aggregation

### Matrix Operations

#### Transpose
- **Purpose**: `B[j,i] = A[i,j]` (swap rows/columns)
- **Parallelization**: 2D grid, each thread moves one element
- **Use case**: Preparing matrices for attention mechanisms

#### GEMM (General Matrix Multiplication)
- **Purpose**: `C = A * B` where `C[m,n] = sum(A[m,k] * B[k,n])`
- **Parallelization**: 2D grid for output matrix, dot product per thread
- **Use case**: Linear layers, attention mechanisms

### Neural Network Primitives

#### Softmax
- **Purpose**: Convert logits to probability distribution
- **Algorithm**: `exp(x - max) / sum(exp(x - max))`
- **Parallelization**: 1 thread per output element (naive implementation)
- **Note**: Highly inefficient - each thread recomputes max/sum

#### 1D Convolution
- **Purpose**: Sliding window dot products on 1D signals
- **Algorithm**: `out[i] = sum(signal[i+j] * kernel[j])`
- **Parallelization**: 1 thread per output element
- **Use case**: Time series analysis, text processing

#### 2D Convolution
- **Purpose**: 2D sliding window operations on images
- **Algorithm**: Nested loops over kernel dimensions
- **Parallelization**: 2D grid matching output feature map
- **Use case**: Convolutional Neural Networks (CNNs)

#### 2D Max Pooling
- **Purpose**: Downsampling with maximum value preservation
- **Algorithm**: Find max in each `pool_size x pool_size` window
- **Parallelization**: 2D grid matching downsampled output
- **Use case**: Spatial reduction in CNNs

## Performance Characteristics

### Why "Naive" = Correct but Slow

These implementations are intentionally simple:

- **No shared memory**: Each thread loads from global memory
- **No tiling/blocking**: Simple memory access patterns
- **No vectorization**: One thread, one output element
- **Redundant computation**: Softmax recomputes max/sum per thread

### Expected Performance

| Operation | Naive GFLOPS | Optimized GFLOPS | Speedup |
|-----------|---------------|------------------|---------|
| GEMM      | 10-50         | 1000-5000       | 100-200x |
| Conv2D    | 5-20          | 1000-10000      | 200-500x |
| Softmax   | 1-5           | 100-500         | 100-200x |

*Performance numbers are approximate and GPU-dependent*

## Learning Path

### Start Here: Element-wise Operations
1. `vector_add.cu` - Understand thread indexing
2. `matrix_add.cu` - Learn 2D grids and blocks

### Matrix Operations
3. `transpose.cu` - Data movement patterns
4. `gemm.cu` - Reduction operations

### Neural Network Building Blocks
5. `softmax.cu` - Row-wise reductions
6. `conv1d.cu` - Sliding window patterns
7. `conv2d.cu` - 2D sliding windows
8. `maxpool2d.cu` - Spatial reduction

## Verification and Debugging

### Built-in Verification

Every example includes:
- **CPU reference implementation**
- **Element-wise result comparison**
- **Tolerance-based floating point checks**
- **Sample output printing**

### Common Issues

#### Compilation Errors
```bash
# Check CUDA installation
nvcc --version

# Verify GPU presence
nvidia-smi
```

#### Runtime Errors
- **CUDA_ERROR_NO_DEVICE**: No CUDA-capable GPU
- **CUDA_ERROR_OUT_OF_MEMORY**: Allocate smaller matrices
- **Incorrect results**: Check boundary conditions and indexing

#### Debugging Tips
1. Start with small matrices (4x4, 8x8)
2. Print intermediate values
3. Verify CPU implementation first
4. Check thread/block dimensions

## Next Steps

### From Naive to Optimized

These implementations are your foundation. Chapter optimizations include:

- **Shared Memory**: Reduce global memory accesses
- **Tiling**: Process data in blocks
- **Vectorization**: Multiple elements per thread
- **Algorithmic improvements**: Reduce redundant computation

### Building Neural Networks

With these primitives, you can construct:
- **MLP**: Linear layers + activations
- **CNN**: Conv2D + MaxPool + Linear
- **Transformers**: GEMM + Softmax + Transpose

### Advanced Topics
- Memory coalescing
- Warp-level primitives
- Multi-GPU scaling
- cuBLAS/cuDNN integration

## License

These examples are part of the CUDA Programming educational materials. Use for learning and experimentation.
