# Official CUTLASS Distributed GEMM

This directory contains the **official CUTLASS distributed GEMM implementation** adapted from CUTLASS Example 65 for educational and performance comparison purposes.

## Overview

- **Based on**: CUTLASS Example 65 (distributed GEMM with Hopper SM90)
- **GPU Configuration**: Currently fixed at 2 GPUs (TP=2)
- **Schedule**: `AllGather1D_TilingCD_RotatingA` - optimized distributed schedule
- **Features**:
  - Hopper's advanced distributed optimizations (GDC, TMA, WGMMA)
  - Peer-to-peer GPU communication over NVLink
  - Column-major layouts for B and C matrices

## Files

- `gemm_multi_sm90_official.cu` - Official distributed GEMM kernel (PyTorch-wrapped)
- `benchmark.py` - Full benchmark with numerical verification vs cuBLAS
- **`perf_only.py`** - **Performance-only script (recommended for raw TFLOPS)**

## Quick Start

### Performance-Only Benchmark (Recommended)

```bash
# Default: 8192x8192x8192, 100 iterations
python perf_only.py

# Custom problem size
python perf_only.py --m=4096 --n=4096 --k=4096 --iterations=50

# Quick test
python perf_only.py --m=2048 --n=2048 --k=2048 --iterations=10 --warmup=3
```

### Full Benchmark (with Verification)

```bash
python benchmark.py
```

## Performance Results (2x H100 80GB)

| Problem Size | Kernel Time | TFLOPS | Per-GPU TFLOPS |
|--------------|-------------|--------|----------------|
| 4096³        | 3.81 ms     | 36.09  | 18.04          |
| 8192³        | 5.89 ms     | 186.70 | 93.35          |

**Note**: These are kernel-only timings that exclude data distribution and gathering overheads.

## Comparison with From-Scratch Multi-GPU

The official CUTLASS implementation achieves **significantly higher performance** than the from-scratch version due to:

1. **Advanced Distributed Schedule**: `AllGather1D_TilingCD_RotatingA` with optimized tensor slicing
2. **Warp Specialization**: Different warps handle TMA loads vs compute
3. **Pipeline Depth**: Overlaps communication and computation
4. **Heuristic Tuning**: NVIDIA-tuned tile sizes and cluster shapes
5. **GDC/PDL**: Grid Dependency Control for reduced launch overhead
6. **Optimized Epilogue**: TMA-accelerated write-back

## Limitations

### Current Implementation
- **Fixed at 2 GPUs**: The `TP_` template parameter is hard-coded to 2
- **Hopper Only**: Requires SM90a architecture (H100)
- **CUDA 12.6+**: Requires CUDA Toolkit 12.6 or later
- **Layout Constraints**: A must be row-major, B/C must be column-major

### Why No 4 or 8 GPU Support?

The distributed GEMM API uses compile-time template parameters for the number of GPUs (`TP`). To support multiple GPU counts, we would need to:

1. Create separate template instantiations for each TP value
2. Recompile the entire kernel for each configuration
3. Handle the exponential growth in compilation time and binary size

For educational purposes and comparison, the 2-GPU configuration is sufficient to demonstrate the performance characteristics.

## Understanding the Code

### Key Components

```cpp
// 1. Distributed schedule
using DistSchedule = cutlass::distributed::schedules::AllGather1D_TilingCD_RotatingA<TP>;

// 2. Collective builders for mainloop and epilogue
using CollectiveEpilogue = /* CollectiveBuilder with TmaWarpSpecialized */
using CollectiveMainloop = /* CollectiveBuilder with TmaGmmaWarpSpecialized */

// 3. Distributed GEMM kernel wrapper
using DistGemmKernel = cutlass::distributed::kernel::DistributedGemmKernelWrapper<
  GemmKernel, DistSchedule>;

// 4. Device adapter
using DistGemm = cutlass::distributed::device::DistributedGemmUniversalAdapter<
  DistGemmKernel>;
```

### Execution Flow

1. **Enable P2P**: Set up peer-to-peer access between GPUs
2. **Compute Local Shapes**: Each GPU gets a slice of the full problem
3. **Distribute Data**: Use `cutlass::device_copy` to scatter data to local tensors
4. **Initialize Kernels**: Call `dist_gemm.initialize()` with workspace arrays
5. **Run GEMM**: Call `dist_gemm.run()` on each GPU
6. **Gather Results**: Collect local outputs back to global tensor

## Next Steps

To extend this for 4 or 8 GPUs:

1. Modify the template instantiation in `gemm_multi_sm90_official.cu`
2. Add conditional compilation for different TP values
3. Update the Python wrapper to accept `num_gpus` parameter
4. Create a dispatcher that selects the correct template instance

However, for most educational and benchmarking purposes, the 2-GPU version is sufficient to understand the distributed GEMM mechanics and performance characteristics.

## References

- [CUTLASS Example 65: Distributed GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/65_distributed_gemm)
- [CUTLASS Distributed Schedules](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/experimental/distributed/schedules)
- [Hopper Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core)
