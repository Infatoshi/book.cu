# CUDA Vector Addition Examples

This repository contains CUDA vector addition examples from **Chapter 02: GPU Memory Management and Kernel Launch** of the CUDA Programming Book.

## Overview

These examples demonstrate the fundamental concepts of CUDA programming:
- Host (CPU) and Device (GPU) memory management
- Kernel launches with thread hierarchies
- 1D, 2D, and 3D indexing patterns
- Boundary checks for scalable kernels

## Examples

### 1. Basic Vector Addition (`vecadd.cu`)

The simplest CUDA program: adds two 8-element vectors using a single thread block with 8 threads.

**Key Concepts:**
- `__global__` kernel functions
- `threadIdx.x` for thread identification
- Basic memory allocation (`cudaMalloc`, `cudaMemcpy`)
- Kernel launch syntax: `kernel<<<blocks, threads>>>(args)`

```bash
nvcc vecadd.cu -o vecadd
./vecadd
```

### 2. Scalable Vector Addition (`vecadd_scalable.cu`)

Handles large arrays (1 million elements) using multiple thread blocks with boundary checks.

**Key Concepts:**
- Global thread indexing: `blockIdx.x * blockDim.x + threadIdx.x`
- Boundary checks to prevent out-of-bounds access
- Calculating optimal grid dimensions: `ceil(n/threadsPerBlock)`

```bash
nvcc vecadd_scalable.cu -o vecadd_scalable
./vecadd_scalable
```

### 3. 3D Tensor Addition (`tensor_add_3d.cu`)

Demonstrates 3D indexing for tensor operations, common in deep learning applications.

**Key Concepts:**
- 3D thread blocks and grids using `dim3`
- Multi-dimensional indexing: `(blockIdx.x,y,z)` and `(threadIdx.x,y,z)`
- 3D-to-1D memory flattening: `d * (height * width) + h * width + w`

```bash
nvcc tensor_add_3d.cu -o tensor_add_3d
./tensor_add_3d
```

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- GNU Make (optional, for using Makefile)

### Build All Examples

```bash
# Using Makefile (recommended)
make all

# Or build individually
nvcc vecadd.cu -o vecadd
nvcc vecadd_scalable.cu -o vecadd_scalable
nvcc tensor_add_3d.cu -o tensor_add_3d
```

### Run All Examples

```bash
# Using Makefile
make run

# Or run individually
./vecadd
./vecadd_scalable
./tensor_add_3d
```

## Understanding the CUDA Programming Model

### Thread Hierarchy
- **Threads**: Individual workers executing kernel code
- **Blocks**: Groups of threads that can cooperate and share memory
- **Grid**: Collection of all blocks for a kernel launch

### Memory Model
- **Host Memory**: CPU RAM (allocated with `malloc`)
- **Device Memory**: GPU VRAM (allocated with `cudaMalloc`)
- Data transfer via `cudaMemcpy`

### Kernel Launch Syntax
```cpp
kernelName<<<blocksPerGrid, threadsPerBlock>>>(args...);
```

Where:
- `blocksPerGrid`: Number of blocks in the grid (can be 1D, 2D, or 3D)
- `threadsPerBlock`: Number of threads per block (can be 1D, 2D, or 3D)

## Learning Path

1. **Start with `vecadd.cu`**: Understand the basic CUDA program structure
2. **Move to `vecadd_scalable.cu`**: Learn how to scale to large datasets
3. **Explore `tensor_add_3d.cu`**: Master multi-dimensional indexing

## Common Issues

### Compilation Errors
- Ensure `nvcc` is in your PATH
- Check that CUDA Toolkit is properly installed

### Runtime Errors
- Verify you have a CUDA-capable GPU
- Check GPU memory availability for large arrays
- Ensure proper CUDA driver installation

### Performance Considerations
- Thread block sizes should typically be multiples of 32 (warp size)
- Avoid launching more threads than data elements
- Consider memory coalescing for optimal performance

## Next Steps

After mastering these examples, you're ready to explore:
- Shared memory and synchronization
- More complex kernel patterns
- Memory optimization techniques
- Multi-GPU programming

## License

These examples are part of the CUDA Programming educational materials.
