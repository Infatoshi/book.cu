# Chapter 5: Tensor Core Programming

This chapter covers tensor core programming, moving beyond manual CUDA core optimizations to hardware-accelerated matrix operations.

## Overview

After optimizing GEMM with CUDA cores in Chapter 4 (kernels 0-7), this chapter introduces **tensor cores** - specialized hardware units designed specifically for matrix multiply-accumulate operations.

## Structure

```
5_tensor_cores/
└── gemm/
    ├── kernels/
    │   ├── 7_cublas_tc.cuh                 # cuBLAS with Tensor Cores enabled
    │   ├── 8_wmma.cuh                      # WMMA (Warp Matrix Multiply Accumulate)
    │   ├── 9_wgmma.cuh                     # WGMMA (Warp Group MMA)
    │   ├── 10_wgmma_larger_tiles.cuh       # WGMMA with larger tile sizes
    │   ├── 11_wgmma_async_loads.cuh        # WGMMA with async memory operations
    │   ├── 12_wgmma_max_tiles.cuh          # WGMMA maximizing tile sizes
    │   ├── all_kernels.cu                  # Kernel implementations
    │   ├── all_kernels.cuh                 # Kernel headers
    │   └── wgmma/                          # Helper utilities
    │       ├── benchmark_helpers.cuh
    │       ├── layout_utils.cuh
    │       ├── wgmma_basic_fp16.cuh
    │       ├── wgmma_larger_tiles_fp16.cuh
    │       ├── wgmma_async_loads_fp16.cuh
    │       └── wgmma_max_tiles_fp16.cuh
    ├── wrapper.cpp                         # PyTorch C++ extension
    ├── main.py                             # Benchmark script
    └── Makefile                            # Build configuration
```

## Kernel Progression

### 7. cuBLAS with Tensor Cores
- **Architecture:** Volta+ (sm_70+)
- **Precision:** FP16 input → FP32 accumulate
- **Level:** Library call (zero-effort tensor cores)
- **API:** cuBLAS with `CUBLAS_TENSOR_OP_MATH` mode
- **Why:** Shows the baseline performance with tensor cores enabled

### 8. WMMA (Warp Matrix Multiply Accumulate)
- **Architecture:** Ampere+ (sm_80+)
- **Precision:** FP16 input → FP32 accumulate
- **Level:** Warp-level operations (32 threads)
- **API:** High-level C++ intrinsics
- **Why:** Easiest entry point to tensor core programming

### 9. WGMMA (Warp Group MMA)
- **Architecture:** Hopper (sm_90)
- **Precision:** BF16 input → BF16 accumulate
- **Level:** Warp group operations (128 threads / 4 warps)
- **API:** PTX-level programming
- **Why:** Shows modern asynchronous tensor core programming

### 10-12. WGMMA Optimizations
- **10:** Larger tile sizes for better occupancy
- **11:** Asynchronous memory loads with TMA (Tensor Memory Accelerator)
- **12:** Maximum tile sizes pushing hardware limits

## Key Concepts Covered

1. **From CUDA Cores to Tensor Cores**
   - What gets automated: memory coalescing, register tiling, shared memory staging
   - What remains manual: kernel launch configuration, overall tiling strategy

2. **WMMA Programming Model**
   - Fragment loading and storing
   - Warp-level synchronization
   - Fixed 16×16×16 tile operations

3. **WGMMA Programming Model**
   - Warp group cooperation (4 warps = 128 threads)
   - Asynchronous operations and barriers
   - TMA integration for efficient memory movement
   - Dynamic tile size configuration

4. **Performance Characteristics**
   - When tensor cores win over CUDA cores
   - Break-even analysis for matrix sizes
   - Theoretical vs. achieved TFLOPS

## Comparison: CUDA Cores vs. Tensor Cores

| Aspect | CUDA Cores (Ch 4) | Tensor Cores (Ch 5) |
|--------|------------------|---------------------|
| **Optimization** | Manual tiling, vectorization | Hardware-accelerated MMA |
| **Code Complexity** | High (100+ LOC kernels) | Medium (intrinsics/PTX) |
| **Performance** | Good | Excellent (5-10x faster) |
| **Hardware** | All GPUs | Volta+ (V100, A100, H100) |
| **Use Case** | General compute | Matrix multiply dominant |

## Building and Running

The tensor core kernels follow the same structure as the CUDA core kernels in Chapter 4:

```bash
cd 5_tensor_cores/gemm
make
python main.py
```

## Hardware Requirements

- **Kernel 7 (cuBLAS TC):** Requires Volta or later (V100, A100, H100)
- **Kernel 8 (WMMA):** Requires Ampere or later (A100, RTX 30/40 series)
- **Kernels 9-12 (WGMMA):** Requires Hopper (H100)

If you don't have the required hardware:
- Cloud options: Lambda Labs, CoreWeave, vast.ai
- Understand conceptually - the code teaches the programming model
- Chapter 9 (CUTLASS) will show library-based approach for newer hardware

## Learning Path

1. Start with kernel 7 (cuBLAS TC) - see the baseline performance
2. Progress to kernel 8 (WMMA) - learn manual tensor core programming
3. Advance to kernel 9 (WGMMA) - modern asynchronous programming
4. Study kernels 10-12 to understand optimization progression
5. Compare with Chapter 4's manual CUDA core optimizations
6. Prepare for Chapter 6 (Flash Attention) - fused tensor core operations
7. Look ahead to Chapter 9 (CUTLASS) - production-ready tensor core programming

## Next Steps

- **Chapter 6 (Flash Attention):** Shows how to fuse multiple tensor core operations
- **Chapter 9 (CUTLASS):** Professional library for tensor core programming on Hopper (FP8) and Blackwell (FP4)

## Notes

- MMA (PTX-level, Volta) is covered conceptually but not implemented (complexity vs. WMMA)
- TCGen05 (Blackwell FP4) is covered in Chapter 9 with CUTLASS
- Focus here is on understanding the programming model with minimal abstraction

