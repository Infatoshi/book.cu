# book.cu

CUDA code examples for the Manning book *CUDA for LLMs* by Elliot Arledge. Each directory maps to one chapter (`0_vecadd` is chapter 2, `9_cutlass` is chapter 11); appendix B of the book has the build and run commands for every chapter. Progress from basic vector operations to production-grade optimizations.

## Structure

```
0_vecadd/         Vector addition - CUDA fundamentals
1_naive/          Naive neural network operations
2_mnist/          MNIST training examples
3_transformer/    Transformer implementation  
4_optim/          Optimized kernels (GEMM, softmax, layernorm)
5_tensor_cores/   Tensor core programming (WMMA, WGMMA)
6_flash/          Flash attention implementations
7_quant/          Quantization techniques
8_distributed/    Single-node multi-GPU: tensor and pipeline parallelism
9_cutlass/        CUTLASS library examples
```

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit (appendix A of the book installs CUDA 13; the Hopper and Blackwell examples need a toolkit that supports those GPUs)
- Python 3 and [uv](https://docs.astral.sh/uv/) for the Python-driven benchmarks

## Quick Start

Each directory contains its own README with specific instructions. Start with:

```bash
cd 0_vecadd
make run
```

## Learning Path

1. **Start**: `0_vecadd` - Basic CUDA programming model
2. **Build**: `1_naive` → `2_mnist` - Implement neural network primitives
3. **Optimize**: `4_optim` → `5_tensor_cores` - Performance optimization techniques
4. **Scale**: `6_flash` → `7_quant` → `8_distributed` → `9_cutlass` - Fusion, quantization, multi-GPU, and CUTLASS

## License

Educational materials for CUDA programming.

