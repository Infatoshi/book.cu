# book.cu

CUDA code examples accompanying the book. Progress from basic vector operations to production-grade optimizations.

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
8_distributed/    Multi-GPU and distributed training
9_cutlass/        CUTLASS library examples
```

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit (11.0+)
- Python 3.8+ (for Python bindings)

## Quick Start

Each directory contains its own README with specific instructions. Start with:

```bash
cd 0_vecadd
make
./vecadd
```

## Learning Path

1. **Start**: `0_vecadd` - Basic CUDA programming model
2. **Build**: `1_naive` → `2_mnist` - Implement neural network primitives
3. **Optimize**: `4_optim` → `5_tensor_cores` - Performance optimization techniques
4. **Scale**: `8_distributed` → `9_cutlass` - Production deployment patterns

## License

Educational materials for CUDA programming.

