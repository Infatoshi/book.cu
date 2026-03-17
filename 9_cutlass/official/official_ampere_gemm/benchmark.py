"""
Official CUTLASS Ampere (SM80) FP16 GEMM Benchmark.

This script provides a comprehensive benchmark for a GEMM (General Matrix-Matrix
Multiplication) kernel built with CUTLASS 2.x and optimized for the NVIDIA
Ampere architecture (SM80). It serves two main purposes:
1.  **Numerical Verification**: It checks the correctness of the CUTLASS GEMM
    implementation by comparing its output against PyTorch's native `torch.mm`,
    which typically uses the highly optimized cuBLAS library.
2.  **Performance Benchmarking**: It measures the execution time of the CUTLASS
    kernel and compares its performance (in GFLOPS) against `torch.mm`.

The script uses `torch.utils.cpp_extension.load` to just-in-time (JIT) compile
the `gemm_sm80_official.cu` file, which contains the C++/CUDA code for the
CUTLASS kernel.
"""
import os
import torch
import numpy as np
from torch.utils.cpp_extension import load

# A cache to store the loaded CUDA module, so we don't recompile it unnecessarily.
_CUBLAS_GEMM_MODULE = None

def load_cutlass_gemm():
    """
    Loads the CUTLASS GEMM CUDA kernel using PyTorch's C++ extension loader.

    This function compiles `gemm_sm80_official.cu` into a Python module on the fly.
    It sets the target architecture to SM80 (NVIDIA Ampere) and includes the
    necessary paths to the CUTLASS library headers.

    Returns:
        A loaded Python module containing the `gemm` function.
    """
    global _CUBLAS_GEMM_MODULE
    if _CUBLAS_GEMM_MODULE is not None:
        return _CUBLAS_GEMM_MODULE

    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set the target CUDA architecture for the JIT compilation.
    # This ensures the kernel is compiled for NVIDIA Ampere GPUs.
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'
    
    # Use PyTorch's JIT compiler to load the CUDA source file.
    _CUBLAS_GEMM_MODULE = load(
        name='cutlass_gemm_official_sm80',
        sources=[os.path.join(current_dir, 'gemm_sm80_official.cu')],
        extra_cuda_cflags=[
            '-O3',
            '--use_fast_math',
            '-std=c++17',
            # Ensure we are compiling for the correct architecture.
            '-gencode=arch=compute_80,code=sm_80',
            # Include paths for the CUTLASS header files.
            '-I' + os.path.join(current_dir, '../../../cutlass/include'),
            '-I' + os.path.join(current_dir, '../../../cutlass/tools/util/include'),
        ],
        verbose=False,
    )
    return _CUBLAS_GEMM_MODULE

def benchmark_kernel(func, *args, warmup=5, iters=20):
    """
    Benchmarks a given function using CUDA events for accurate timing.

    Args:
        func: The function to benchmark (e.g., a CUDA kernel).
        *args: The arguments to pass to the function.
        warmup (int): The number of warmup iterations to run before timing.
        iters (int): The number of timed iterations.

    Returns:
        A tuple containing the mean and standard deviation of the execution times in milliseconds.
    """
    # Warmup iterations to stabilize GPU clocks and cache state.
    for _ in range(warmup):
        func(*args)
    
    torch.cuda.synchronize()
    
    # Create CUDA events for timing.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(iters):
        start.record()
        func(*args)
        end.record()
        # Synchronize the CPU and GPU to ensure the kernel is finished.
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    return np.mean(times), np.std(times)

def verify_numerical(M: int, N: int, K: int, rtol: float = 1e-2, atol: float = 0.5):
    """
    Verifies the numerical correctness of the CUTLASS kernel against torch.mm.

    Args:
        M, N, K (int): The dimensions of the matrices.
        rtol (float): The relative tolerance for the comparison.
        atol (float): The absolute tolerance for the comparison.
    """
    print(f"\n{'='*60}")
    print(f"Verifying {M}x{N}x{K}")
    print(f"{'='*60}")
    
    cutlass_module = load_cutlass_gemm()
    
    # Run three passes with different random data to ensure robustness.
    for pass_num in range(3):
        A = torch.randn(M, K, dtype=torch.float16, device='cuda')
        B = torch.randn(K, N, dtype=torch.float16, device='cuda')
        
        # Reference computation using PyTorch's `torch.mm`.
        C_ref = torch.mm(A, B)
        
        # Computation using the custom CUTLASS kernel.
        C_cutlass = torch.empty(M, N, dtype=torch.float16, device='cuda')
        cutlass_module.gemm(A, B, C_cutlass)
        torch.cuda.synchronize()
        
        # Compare the results.
        # Due to the use of --use_fast_math and potential differences in operation order,
        # small discrepancies are expected. We check if the maximum difference is within
        # an acceptable tolerance.
        max_diff = torch.max(torch.abs(C_cutlass - C_ref.half())).item()
        passed = max_diff < atol
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  Pass {pass_num + 1}/3: {status} (max_diff={max_diff:.6f})")
        
        if not passed:
            raise RuntimeError(f"Numerical verification failed on pass {pass_num + 1}")
    
    print("✅ All verification passes succeeded\n")

def benchmark_performance(M: int, N: int, K: int):
    """
    Benchmarks the performance of the CUTLASS kernel against torch.mm.

    Args:
        M, N, K (int): The dimensions of the matrices for the benchmark.
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking {M}x{N}x{K}")
    print(f"{'='*60}\n")
    
    cutlass_module = load_cutlass_gemm()
    
    # Create input tensors.
    A = torch.randn(M, K, dtype=torch.float16, device='cuda')
    B = torch.randn(K, N, dtype=torch.float16, device='cuda')
    C = torch.empty(M, N, dtype=torch.float16, device='cuda')
    
    # Benchmark the CUTLASS kernel.
    cutlass_time, cutlass_std = benchmark_kernel(cutlass_module.gemm, A, B, C)
    
    # Benchmark the PyTorch `torch.mm` kernel (cuBLAS).
    pytorch_time, pytorch_std = benchmark_kernel(torch.mm, A, B)
    
    # Calculate performance in GFLOPS.
    # The number of floating-point operations in a GEMM is 2 * M * N * K.
    gflops = (2.0 * M * N * K) / 1e9
    cutlass_gflops = gflops / (cutlass_time / 1000)
    pytorch_gflops = gflops / (pytorch_time / 1000)
    speedup = pytorch_time / cutlass_time
    
    # Print a formatted results table.
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
        print("CUTLASS Official Ampere (SM80) FP16 GEMM Benchmark")
        print("="*60)
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Data type: FP16 (input/output), FP32 (accumulator)")
        print(f"CUTLASS Optimizations: WMMA + Large tiles + 3-stage pipeline")
        
        # Define the matrix sizes to test.
        test_sizes = [
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (8192, 8192, 8192),
        ]
        
        # First, run numerical verification on all sizes.
        print("\n" + "="*60)
        print("NUMERICAL VERIFICATION")
        print("="*60)
        
        for M, N, K in test_sizes:
            verify_numerical(M, N, K)
        
        # Then, run performance benchmarks.
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

