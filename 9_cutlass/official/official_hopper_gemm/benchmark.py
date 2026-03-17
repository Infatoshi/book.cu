"""
Official CUTLASS Hopper (SM90) FP16 GEMM Benchmark.

This script provides a comprehensive benchmark for a single-GPU GEMM kernel built
with the modern CUTLASS 3.x API and optimized for the NVIDIA Hopper architecture (SM90).

The benchmark performs two main functions:
1.  **Numerical Verification**: Compares the output of the custom CUTLASS kernel
    against PyTorch's `torch.mm` (cuBLAS) to ensure correctness.
2.  **Performance Benchmarking**: Measures the GFLOPS of the CUTLASS kernel and
    compares it to the performance of cuBLAS.

The script uses `torch.utils.cpp_extension.load` to just-in-time (JIT) compile
the `gemm_sm90_official.cu` file, which contains the C++/CUDA code for the
CUTLASS kernel. This kernel leverages key Hopper architectural features,
including:
-   **TMA (Tensor Memory Accelerator)**: For asynchronous data movement between
    global and shared memory.
-   **WGMMA (Warp Group MMA)**: For high-performance matrix multiply-accumulate
    operations at the warp-group level.
-   **Cluster Tiling**: Uses `ClusterShape` to organize thread blocks for better
    data locality and performance on large GEMMs.
"""
import os
import torch
import numpy as np
from torch.utils.cpp_extension import load

# Cache for the compiled CUDA module to avoid recompilation.
_CUTLASS_GEMM_MODULE = None

def load_cutlass_gemm():
    """
    Loads and JIT-compiles the CUTLASS Hopper GEMM CUDA kernel.

    This function configures the PyTorch C++ extension builder to compile the
    `gemm_sm90_official.cu` file. It sets the target architecture to SM90a (Hopper)
    and provides the necessary include paths to the CUTLASS library headers.

    Returns:
        The loaded Python module containing the compiled `gemm` function.
    """
    global _CUTLASS_GEMM_MODULE
    if _CUTLASS_GEMM_MODULE is not None:
        return _CUTLASS_GEMM_MODULE

    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set the target CUDA architecture for JIT compilation to Hopper.
    os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a'
    
    # Get the absolute path to the cutlass directory.
    cutlass_path = os.path.abspath(os.path.join(current_dir, '../../../cutlass'))

    # Use PyTorch's JIT compiler to load the CUDA source file.
    _CUTLASS_GEMM_MODULE = load(
        name='cutlass_gemm_official_sm90',
        sources=[os.path.join(current_dir, 'gemm_sm90_official.cu')],
        extra_cuda_cflags=[
            '-O3',
            '--use_fast_math',
            '-std=c++17',
            # Compile for SM90a architecture.
            '-gencode=arch=compute_90a,code=sm_90a',
            # Add include paths for the CUTLASS header files.
            '-I' + os.path.join(cutlass_path, 'include'),
            '-I' + os.path.join(cutlass_path, 'tools/util/include'),
        ],
        verbose=False,
    )
    return _CUTLASS_GEMM_MODULE

def benchmark_kernel(func, *args, warmup=5, iters=20):
    """
    Benchmarks a given function using CUDA events for accurate timing.

    Args:
        func: The function to benchmark (e.g., a CUDA kernel).
        *args: The arguments to pass to the function.
        warmup (int): The number of warmup iterations.
        iters (int): The number of timed iterations.

    Returns:
        A tuple of (mean execution time in ms, standard deviation in ms).
    """
    # Warmup iterations
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()
    
    # Timed iterations
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        func(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    return np.mean(times), np.std(times)

def verify_numerical(M, N, K, atol=0.5):
    """
    Verifies the numerical correctness of the CUTLASS kernel against torch.mm.

    Args:
        M, N, K (int): The dimensions of the matrices.
        atol (float): The absolute tolerance for the comparison.
    """
    print(f"\n{'='*60}")
    print(f"Verifying {M}x{N}x{K}")
    print(f"{'='*60}")
    
    cutlass_module = load_cutlass_gemm()
    
    # Run three passes with different random data.
    for pass_num in range(3):
        A = torch.randn(M, K, dtype=torch.float16, device='cuda')
        B = torch.randn(K, N, dtype=torch.float16, device='cuda')
        
        # Reference computation with cuBLAS.
        C_ref = torch.mm(A, B)
        
        # CUTLASS computation.
        C_cutlass = torch.empty(M, N, dtype=torch.float16, device='cuda')
        cutlass_module.gemm(A, B, C_cutlass)
        torch.cuda.synchronize()
        
        # Check if the results are close.
        max_diff = torch.max(torch.abs(C_cutlass - C_ref.half())).item()
        passed = max_diff < atol
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  Pass {pass_num + 1}/3: {status} (max_diff={max_diff:.6f})")
        
        if not passed:
            raise RuntimeError(f"Numerical verification failed on pass {pass_num + 1}")
    
    print("✅ All verification passes succeeded\n")

def benchmark_performance(M, N, K):
    """
    Benchmarks the performance of the CUTLASS kernel against torch.mm (cuBLAS).

    Args:
        M, N, K (int): The dimensions of the matrices.
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking {M}x{N}x{K}")
    print(f"{'='*60}\n")
    
    cutlass_module = load_cutlass_gemm()
    
    # Create input tensors.
    A = torch.randn(M, K, dtype=torch.float16, device='cuda')
    B = torch.randn(K, N, dtype=torch.float16, device='cuda')
    C = torch.empty(M, N, dtype=torch.float16, device='cuda')
    
    # Benchmark the custom CUTLASS kernel.
    cutlass_time, cutlass_std = benchmark_kernel(cutlass_module.gemm, A, B, C, iters=20)
    
    # Benchmark the default PyTorch kernel (cuBLAS).
    pytorch_time, pytorch_std = benchmark_kernel(torch.mm, A, B, iters=20)
    
    # Calculate GFLOPS (Giga-Floating-point Operations Per Second).
    gflops = (2.0 * M * N * K) / 1e9
    cutlass_gflops = gflops / (cutlass_time / 1000)
    pytorch_gflops = gflops / (pytorch_time / 1000)
    speedup = pytorch_time / cutlass_time
    
    # Print formatted results.
    print(f"{'Method':<20} {'Time (ms)':<20} {'GFLOPS':<15} {'Speedup':<10}")
    print(f"{'-'*65}")
    print(f"{'CUTLASS Official':<20} {cutlass_time:>6.3f} ± {cutlass_std:>5.3f}     {cutlass_gflops:>12.1f}    {speedup:>6.2f}x")
    print(f"{'PyTorch (cuBLAS)':<20} {pytorch_time:>6.3f} ± {pytorch_std:>5.3f}     {pytorch_gflops:>12.1f}    {'1.00x':>6}")

def main():
    """
    Main entry point for the benchmark script.
    """
    if not torch.cuda.is_available():
        print("CUDA not available. This benchmark requires a CUDA-enabled GPU.")
        return
    
    try:
        print("="*60)
        print("CUTLASS Official Hopper (SM90) FP16 GEMM Benchmark")
        print("="*60)
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Data type: FP16 (input/output), FP32 (accumulator)")
        print(f"CUTLASS Optimizations: TMA + WGMMA + Cluster Tiling")
        
        # Define the matrix sizes for testing.
        test_sizes = [
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (8192, 8192, 8192),
        ]
        
        # Run numerical verification first.
        print("\n" + "="*60)
        print("NUMERICAL VERIFICATION")
        print("="*60)
        
        for M, N, K in test_sizes:
            verify_numerical(M, N, K)
        
        # Run performance benchmarks.
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARKING")
        print("="*60)
        
        for M, N, K in test_sizes:
            benchmark_performance(M, N, K)
        
        print("\n" + "="*60)
        print("✅ All benchmarks completed successfully.")
        print("="*60)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main()

