"""
Official Multi-GPU Distributed GEMM Benchmark (Hopper SM90, 2 GPUs).

This script provides a comprehensive benchmark for the 2-GPU distributed GEMM
kernel (`gemm_multi_sm90_official.cu`). It is designed to be run on a system
with at least two NVIDIA Hopper-class GPUs.

The script performs two key functions:
1.  **Numerical Verification**: It ensures that the output of the 2-GPU CUTLASS
    kernel is numerically consistent with the output of a standard single-GPU
    GEMM performed by `torch.matmul` (which uses cuBLAS).
2.  **Performance Benchmarking**: It compares the performance of the 2-GPU
    distributed kernel against a single-GPU cuBLAS kernel to evaluate the
    speedup and scaling efficiency of the distributed implementation.

The script uses PyTorch's C++ extension mechanism to just-in-time (JIT) compile
the CUDA source file containing the distributed CUTLASS kernel.
"""

import torch
import os
import sys
from torch.utils.cpp_extension import load

def check_gpu_count():
    """
    Verifies that at least two GPUs are available for the benchmark.
    Exits if the requirement is not met.
    """
    if torch.cuda.device_count() < 2:
        print(f"ERROR: This benchmark requires at least 2 GPUs, but found {torch.cuda.device_count()}.")
        sys.exit(1)
    print(f"Found {torch.cuda.device_count()} GPUs. Using GPU 0 and 1 for the benchmark.")
    for i in range(2):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

def load_cutlass_kernel():
    """
    Loads and JIT-compiles the distributed CUTLASS GEMM CUDA kernel.

    This function configures the PyTorch C++ extension builder to compile the
    `gemm_multi_sm90_official.cu` file. It sets the target architecture to SM90a
    (Hopper) and provides the necessary include paths and compiler flags
    for the CUTLASS distributed GEMM example.

    Returns:
        The loaded Python module containing the compiled `gemm` function.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set the target CUDA architecture for JIT compilation to Hopper.
    os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a'
    
    print("Compiling official CUTLASS multi-GPU kernel (this may take a few minutes)...")
    
    # Get the absolute path to the cutlass directory, assuming a fixed relative path.
    cutlass_path = os.path.abspath(os.path.join(current_dir, '../../../cutlass'))

    return load(
        name='cutlass_gemm_multi_sm90_official',
        sources=[os.path.join(current_dir, 'gemm_multi_sm90_official.cu')],
        extra_cuda_cflags=[
            '-O3',
            '--use_fast_math',
            '-std=c++17',
            '-gencode=arch=compute_90a,code=sm_90a',
            # Add include paths for CUTLASS headers
            '-I' + os.path.join(cutlass_path, 'include'),
            '-I' + os.path.join(cutlass_path, 'tools/util/include'),
            '-I' + os.path.join(cutlass_path, 'examples'),
            # This flag is required for SM90 distributed GEMM examples.
            '-DCUTLASS_ENABLE_GDC_FOR_SM90=1',
        ],
        verbose=False,
    )

def verify_numerical(cutlass_module, M, N, K, num_passes=3):
    """
    Verifies the numerical correctness of the distributed kernel against `torch.matmul`.

    Args:
        cutlass_module: The loaded CUDA extension module.
        M, N, K (int): The dimensions of the matrices.
        num_passes (int): The number of verification passes to run with random data.
    """
    print(f"\nVerifying numerical correctness for size {M}x{N}x{K} ({num_passes} passes)...")
    
    for pass_num in range(1, num_passes + 1):
        # All input tensors are created on the primary device (`cuda:0`).
        # The distributed kernel is responsible for distributing the data.
        A = torch.randn(M, K, dtype=torch.float16, device='cuda:0')
        B = torch.randn(K, N, dtype=torch.float16, device='cuda:0')
        
        # 1. Reference computation (single-GPU cuBLAS)
        C_ref = torch.matmul(A, B)
        
        # 2. Distributed CUTLASS computation
        C_cutlass = torch.zeros(M, N, dtype=torch.float16, device='cuda:0')
        cutlass_module.gemm(A, B, C_cutlass)
        
        # 3. Compare results
        # A higher tolerance (`atol`) may be needed for large matrices due to
        # floating-point accumulation differences between algorithms.
        atol = 0.25 if max(M, N, K) <= 2048 else 2.0
        if torch.allclose(C_cutlass, C_ref, atol=atol, rtol=1e-2):
            print(f"  Pass {pass_num}: ✅ Numerical verification passed (max diff: {torch.max(torch.abs(C_cutlass - C_ref)).item():.3f})")
        else:
            max_diff = torch.max(torch.abs(C_cutlass - C_ref)).item()
            print(f"  Pass {pass_num}: ❌ Numerical verification FAILED (max diff: {max_diff})")
            raise RuntimeError(f"Numerical verification failed on pass {pass_num}")

def benchmark_gemm(cutlass_module, M, N, K, num_iterations=10, num_warmup=3):
    """
    Benchmarks the performance of the 2-GPU distributed kernel vs. a single-GPU cuBLAS kernel.

    Args:
        cutlass_module: The loaded CUDA extension module.
        M, N, K (int): The dimensions of the matrices.
        num_iterations (int): The number of timed iterations.
        num_warmup (int): The number of warmup iterations.

    Returns:
        A dictionary containing the performance results.
    """
    print(f"\nBenchmarking size {M}x{N}x{K}...")
    
    # Input tensors are placed on the primary device.
    A = torch.randn(M, K, dtype=torch.float16, device='cuda:0')
    B = torch.randn(K, N, dtype=torch.float16, device='cuda:0')
    C_torch = torch.zeros(M, N, dtype=torch.float16, device='cuda:0')
    C_cutlass = torch.zeros(M, N, dtype=torch.float16, device='cuda:0')
    
    # Warmup runs
    for _ in range(num_warmup):
        torch.matmul(A, B, out=C_torch)
        cutlass_module.gemm(A, B, C_cutlass)
    torch.cuda.synchronize()
    
    # --- Benchmark single-GPU PyTorch (cuBLAS) ---
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iterations):
        torch.matmul(A, B, out=C_torch)
    end_event.record()
    torch.cuda.synchronize()
    
    torch_time = start_event.elapsed_time(end_event) / num_iterations
    
    # --- Benchmark 2-GPU CUTLASS ---
    start_event.record()
    for _ in range(num_iterations):
        cutlass_module.gemm(A, B, C_cutlass)
    end_event.record()
    torch.cuda.synchronize()
    
    cutlass_time = start_event.elapsed_time(end_event) / num_iterations
    
    # --- Calculate and report performance ---
    flops = 2 * M * N * K
    torch_tflops = (flops / torch_time) / 1e9
    cutlass_tflops = (flops / cutlass_time) / 1e9
    speedup = torch_time / cutlass_time
    
    print(f"  PyTorch (cuBLAS, 1 GPU):   {torch_time:.3f} ms  ({torch_tflops:.2f} TFLOPS)")
    print(f"  CUTLASS (Distributed, 2 GPUs): {cutlass_time:.3f} ms  ({cutlass_tflops:.2f} TFLOPS)")
    print(f"  Speedup vs. 1 GPU cuBLAS:    {speedup:.2f}x")
    
    return {
        'M': M, 'N': N, 'K': K,
        'torch_time': torch_time,
        'cutlass_time': cutlass_time,
        'torch_tflops': torch_tflops,
        'cutlass_tflops': cutlass_tflops,
        'speedup': speedup
    }

def main():
    """
    Main function to orchestrate the benchmark.
    """
    print("=" * 80)
    print("Official Multi-GPU Distributed GEMM Benchmark (Hopper SM90, 2 GPUs)")
    print("=" * 80)
    
    try:
        check_gpu_count()
        
        cutlass_module = load_cutlass_kernel()
        print("✅ Kernel compiled successfully\n")
        
        # Define problem sizes for benchmarking.
        sizes = [
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (8192, 8192, 8192),
        ]
        
        results = []
        
        for M, N, K in sizes:
            # First, verify numerical correctness.
            verify_numerical(cutlass_module, M, N, K, num_passes=3)
            
            # Then, measure performance.
            result = benchmark_gemm(cutlass_module, M, N, K, num_iterations=10, num_warmup=3)
            results.append(result)
        
        # Print a summary table of the results.
        print("\n" + "=" * 80)
        print("📊 SUMMARY")
        print("=" * 80)
        print(f"{'Size (MxNxK)':<20} {'PyTorch Time (ms)':<20} {'CUTLASS Time (ms)':<20} {'Speedup':<10}")
        print("-" * 80)
        for r in results:
            size_str = f"{r['M']}x{r['N']}x{r['K']}"
            print(f"{size_str:<20} {r['torch_time']:<20.3f} {r['cutlass_time']:<20.3f} {r['speedup']:<10.2f}x")
        print("=" * 80)
    except Exception as e:
        print(f"\n❌ An error occurred during the benchmark: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

