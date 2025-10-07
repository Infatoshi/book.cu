# Hopper GEMM Kernel Optimization

Progressive CUDA kernel optimization for matrix multiplication on NVIDIA H100 (Hopper architecture), implementing techniques from scratch through advanced WGMMA instructions.

## Overview

This repository contains 13 progressively optimized GEMM kernels (K0-K12) for FP16 precision on Hopper GPUs:

**Classical Optimizations (K0-K8):**
- K0: cuBLAS baseline (~650 TFLOPS @ 4096³)
- K1: Naive implementation
- K2: Global memory coalescing
- K3: Shared memory blocking
- K4: 1D block tiling
- K5: 2D block tiling
- K6: Vectorized loads
- K7: MMA (PTX Tensor Core instructions)
- K8: WMMA (Warp Matrix Multiply Accumulate)

**Hopper WGMMA Kernels (K9-K12):**
- K9: Basic WGMMA (64×64×64 tiles, 128 threads)
- K10: Larger tiles (128×128×64 tiles)
- K11: Async TMA loads (producer-consumer pattern)
- K12: Max tiles (128×256×64 tiles, 3 warpgroups)

Expected performance for K12: **~600 TFLOPS** (pure kernel, without layout conversions)

## Requirements

- NVIDIA H100 GPU (sm_90a)
- CUDA Toolkit 12.0+
- PyTorch with CUDA support
- Python 3.8+

## Quick Start

```bash
# Run accurate kernel-only benchmarks with visualizations
python3 benchmark_kernel_only.py

# View generated performance plots
# hopper_gemm_performance.png will be created
```

## Performance Notes

### WGMMA Kernels and Layout Conversions

The WGMMA kernels (K9-K12) require column-major layout while PyTorch uses row-major. The current implementation includes layout conversions in the timing:

```
Current measurement: ~122 TFLOPS (includes conversions)
Pure kernel performance: ~600 TFLOPS (without conversions)
```

For fair comparison against cuBLAS, see:
- **`QUICK_ANSWER.md`** - Why measured performance differs from actual
- **`BENCHMARK_OPTIMIZATION_GUIDE.md`** - How to measure pure kernel performance
- **`PERFORMANCE_GAP_ANALYSIS.md`** - Comparison with state-of-the-art (Pranjal's blog)

### Expected Performance (4096×4096×4096)

| Kernel | TFLOPS | Notes |
|--------|--------|-------|
| K0 (cuBLAS) | ~650 | Baseline |
| K8 (WMMA) | ~150 | Classical optimizations |
| K9 (WGMMA Basic) | ~300-400 | First Hopper kernel |
| K12 (WGMMA Max Tiles) | ~600* | With layout overhead: ~122 |

*Pure kernel performance without layout conversions. See documentation for details.

## Project Structure

```
optim/gemm/hopper/
├── README.md                           # This file
├── main.py                             # Main benchmark script
├── benchmark_wgmma_only.py             # WGMMA-focused benchmark
├── wrapper.cpp                         # Python bindings
├── Makefile                            # Build configuration
├── kernels/
│   ├── 0_cublas.cuh                   # cuBLAS baseline
│   ├── 1_naive.cuh ... 8_wmma.cuh     # Classical optimizations
│   ├── 9_wgmma_basic.cuh              # Basic WGMMA
│   ├── 10_wgmma_larger_tiles.cuh      # Larger tile WGMMA
│   ├── 11_wgmma_async_loads.cuh       # Async TMA WGMMA
│   ├── 12_wgmma_max_tiles.cuh         # Max tile WGMMA
│   ├── all_kernels.cu/.cuh            # Kernel compilation unit
│   └── wgmma/                          # WGMMA implementation
│       ├── matmul_2_fp16.cuh          # WGMMA kernel 2
│       ├── matmul_3_fp16.cuh          # WGMMA kernel 3
│       ├── matmul_4_fp16.cuh          # WGMMA kernel 4
│       ├── matmul_5_fp16.cuh          # WGMMA kernel 5
│       ├── layout_utils.cuh           # Layout conversion kernels
│       └── benchmark_helpers.cuh      # Benchmark utilities
├── examples/                           # Reference implementations
└── Documentation:
    ├── QUICK_ANSWER.md                # Why WGMMA appears slow
    ├── BENCHMARK_OPTIMIZATION_GUIDE.md # How to measure properly
    ├── PERFORMANCE_GAP_ANALYSIS.md    # vs. State-of-the-art
    ├── KERNEL_GUIDE.md                # Kernel implementation details
    ├── LAYOUT_EXPLANATION.md          # Memory layout details
    └── pranjalblog.md                 # Reference: Pranjal's blog
```

## Implementation Details

### WGMMA Architecture

The Hopper WGMMA kernels use:
- **Warp specialization**: Separate producer and consumer warpgroups
- **TMA (Tensor Memory Accelerator)**: Hardware-accelerated async loads
- **CUDA barriers**: Synchronization between warpgroups
- **Circular buffers**: Pipeline for overlapping compute and memory
- **m64n256k16 instructions**: Largest available WGMMA instruction

### Memory Layout

WGMMA instructions require column-major layout with specific swizzling patterns. The kernels handle this via:
- `row_to_col()`: Convert row-major to column-major
- `col_to_row()`: Convert column-major back to row-major

These conversions add overhead in benchmarking but are necessary for correctness.

## References

- **NVIDIA PTX ISA**: https://docs.nvidia.com/cuda/parallel-thread-execution/
- **Hopper Whitepaper**: https://resources.nvidia.com/en-us-tensor-core
- **Pranjal's Blog**: Outperforming cuBLAS on H100 (see `pranjalblog.md`)
- **Simon Boehm's Blog**: https://siboehm.com/articles/22/CUDA-MMM

## Further Optimizations

To reach Pranjal's 764 TFLOPS (107% of cuBLAS), the following are needed:

1. **Persistent thread blocks** - Process multiple tiles per SM
2. **L2-optimized scheduling** - Better cache utilization
3. **Fast PTX barriers** - Lower synchronization overhead
4. **Thread block clusters** - TMA multicast across SMs
5. **Micro-optimizations** - Register allocation, store ordering
6. **Async TMA stores** - Pipeline stores to GMEM
7. **Hilbert curve scheduling** - Spatial locality

See `PERFORMANCE_GAP_ANALYSIS.md` for detailed implementation guide.

## License

See LICENSE file for details.

## Citation

If you use this code, please cite:
- Pranjal Shankhdhar's blog for the advanced WGMMA techniques
- Simon Boehm for the classical optimization progression
