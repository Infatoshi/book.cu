# Flash Attention Implementations

Progressive implementations of Flash Attention from naive baselines to WMMA tensor core optimizations.

## Quick Start

```bash
# Compare all implementations
python main.py

# Test specific kernels
python main.py --kernels fa

# Different configurations
python main.py --seq-len 1024 --batch-size 8
```

## Performance Results (H100, B=16, H=8, N=512, D=64)

```
Implementation          Time (ms)    Speedup    Architecture
─────────────────────────────────────────────────────────────
PyTorch Flash (bf16)        0.087      1.00x    Optimized (reference)
PyTorch Naive (f32)         0.465      0.19x    3 matmuls
fa (fp16 + WMMA)            5.314      0.02x    Tensor cores
naive (fp32)               68.832      0.00x    3 separate kernels
```

**Key Achievement**: FA achieves **13x speedup** over naive using WMMA tensor cores!

## Implementations

### 1. Naive (`kernels/naive.cu`)
Simple baseline with 3 separate kernel launches:
- Compute S = Q @ K^T / √d
- Apply softmax(S)  
- Compute O = S @ V

**Characteristics**: Materializes full N×N matrix, no optimization

### 2. Flash Attention (`kernels/fa.cu`)
Fused kernel with WMMA tensor cores:
- Single kernel launch
- Tiling (Br=16, Bc=16)
- Online softmax (no N×N materialization)
- FP16 inputs with FP32 accumulation
- WMMA 16×16×16 tiles for Q@K^T and S@V
- Works on Volta, Turing, Ampere, Ada, Hopper (SM_70+)

**Speedup**: 13x faster than naive

**Key WMMA code**:
```cuda
// Q @ K^T using tensor cores
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

for (int k = 0; k < 64; k += 16) {
    wmma::load_matrix_sync(a_frag, Q + k, 64);
    wmma::load_matrix_sync(b_frag, K + k, 64);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);  // Tensor cores!
}
```

## Why Still 62x Slower Than PyTorch?

PyTorch's Flash Attention uses highly optimized CUTLASS/CuTe with:
- BFloat16 (2x memory bandwidth on H100)
- Larger tiles (128×128 vs our 16×16)
- Async memory pipeline (`cp.async`)
- Swizzled shared memory layouts
- Warp specialization
- 1024 threads/block (vs our 256)

See future CUTLASS chapter for these advanced optimizations!

## File Structure

```
6_flash/
├── main.py                           # Benchmark script
├── README.md                         # This file
├── kernels/
│   ├── naive.cu                     # Baseline implementation
│   ├── fa.cu                        # Flash Attention with WMMA tensor cores
│   ├── build_naive.cpp              # PyTorch bindings for naive
│   └── build_fa.cpp                 # PyTorch bindings for FA
└── build/                           # Compiled kernels (auto-generated)
```

## Adding Your Own Kernel

1. **Create kernel**:
   ```bash
   cp kernels/naive.cu kernels/my_kernel.cu
   cp kernels/build_naive.cpp kernels/build_my.cpp
   # Edit my_kernel.cu with your implementation
   ```

2. **Register in main.py**:
   ```python
   KERNELS = {
       'naive': KernelConfig(...),
       'fa': KernelConfig(...),
       'my_kernel': KernelConfig(
           name='my_kernel_attn',
           sources=['kernels/build_my.cpp', 'kernels/my_kernel.cu'],
           build_dir='./build/my_kernel'
       ),
   }
   ```

3. **Run**:
   ```bash
   mkdir -p build/my_kernel
   python main.py --kernels my_kernel
   ```

## Requirements

```bash
# GPU with tensor cores (Volta or newer)
nvidia-smi

# CUDA toolkit
nvcc --version  # Need CUDA 11.0+

# Python packages
pip install torch numpy
```

## Key Concepts Demonstrated

- **Memory Hierarchy**: HBM vs SRAM vs Registers
- **Tiling**: Breaking computation into blocks to fit in fast memory
- **Online Algorithms**: Computing softmax incrementally without materialization
- **Tensor Cores**: Using WMMA for 10-20x speedup on matrix operations
- **Mixed Precision**: FP16 compute with FP32 accumulation
- **Numerical Stability**: Max subtraction in softmax

## References

- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Flash Attention 2 Paper](https://arxiv.org/abs/2307.08691)
- [NVIDIA WMMA Docs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
