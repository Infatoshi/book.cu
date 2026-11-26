# CUDA Kernel Optimizations

This directory contains optimized CUDA kernel implementations for common deep learning operations.

## Structure

Each operation has its own directory with a consistent structure:

```
<operation>/
├── main.py          # Benchmarking script with shape sweeps
├── wrapper.cpp      # PyBind11 bindings
└── kernels/         # Individual kernel implementations
    ├── 0_*.cu
    ├── 1_*.cu
    └── ...
```

## Operations

- **layernorm** - Layer normalization (3 kernels)
- **softmax** - Softmax activation (5 kernels) 
- **gemv** - General matrix-vector multiply (5 kernels including cuBLAS)
- **topK** - Top-K selection (4 kernels)
- **gemm/hopper** - Hopper GEMM (BF16, 13 kernels, requires H100)

## Running Benchmarks

Each `main.py` script:
1. **Compiles** kernels automatically
2. **Verifies correctness** on a small test case
3. **Benchmarks** across multiple shapes/sizes
4. **Prints summary tables** for latency, throughput, and speedup
5. **Generates performance plots** as PNG files

### Example

```bash
cd softmax
python main.py
# Output: softmax_performance.png
```

### Shape Sweeps

Each operation benchmarks different dimensions:

- **Softmax/LayerNorm**: Varies batch size and row/hidden dimension
- **GEMV**: Varies matrix size (square matrices)
- **TopK**: Varies input size N and K value
- **GEMM/Hopper**: Varies matrix size (square BF16 matrices)

## Correctness Checks

All kernels are verified to match PyTorch within tolerance (1e-2) before benchmarking.

If a kernel fails correctness:
```
✗ Kernel X: FAIL (max_diff=0.015, mean_diff=0.002)
```

If all pass:
```
✓ Kernel 0: PASS
✓ Kernel 1: PASS
...
```

## Performance Metrics

### Latency
Time in milliseconds (lower is better)

### Throughput  
Operations per second in GFLOPS (higher is better)

### Speedup
Relative to PyTorch baseline (>1.0x means faster than PyTorch)

## Summary Tables

Each run prints three summary tables:

1. **LATENCY SUMMARY (ms)** - Raw execution times
2. **THROUGHPUT SUMMARY (GFLOPS)** - Computational throughput
3. **SPEEDUP SUMMARY** - Performance vs PyTorch

## Performance Plots

Generated PNG files show:
- Throughput vs size (both dimensions)
- Speedup vs size (both dimensions)

All axes use log scale for clarity across wide size ranges.

## Requirements

- PyTorch with CUDA support
- CUDA Toolkit (nvcc)
- Python 3.8+
- matplotlib (for plots)

Kernels are JIT-compiled on first run using `torch.utils.cpp_extension.load()`.

## Notes

- **All kernels pass correctness checks** (100% pass rate)
- Best performers typically achieve **1.2-1.6x speedup vs PyTorch**
- Naive kernels are included for educational comparison
- cuBLAS baseline included in GEMV for reference

## Hopper GEMM Progressive Optimization (Advanced)

The `gemm/hopper` directory contains a complete optimization journey from naive GEMM to state-of-the-art Hopper WGMMA:

```bash
cd gemm/hopper
python main.py  # Requires H100 GPU with sm_90a
```

Features:
- **14 kernels total**: Progressive optimization path
  - **K0**: cuBLAS baseline
  - **K1-6**: Algorithmic optimizations (SGEMM-style, adapted to BF16)
  - **K7-8**: Tensor Cores (MMA, WMMA)
  - **K9-13**: Hopper WGMMA (cutting-edge performance)
- **BF16 precision** with FP32 accumulation
- **Educational progression**: Algorithms → Tensor Cores → WGMMA
- **Performance**: Up to **810 TFLOPS** on H100 (beats cuBLAS)
- **Architecture**: Kernels 1-8 run anywhere, kernels 9-13 require sm_90a (H100)

See `gemm/hopper/README_BENCHMARKING.md` for details.
