"""
Official Distributed GEMM - Performance-Only Benchmark.

This script provides a streamlined performance benchmark for the 2-GPU distributed
GEMM kernel (`gemm_multi_sm90_official.cu`). Unlike the more comprehensive
`benchmark.py`, this script focuses solely on measuring raw kernel performance in
TFLOPS, without performing numerical verification or comparison against other
implementations like cuBLAS.

It is designed for quick performance checks and profiling.

Key Features:
- **Performance Focus**: Measures only the execution time of the distributed
  GEMM kernel.
- **JIT Compilation**: Uses `torch.utils.cpp_extension.load` to compile the
  CUDA source file on the fly.
- **Command-Line Interface**: Allows specifying problem dimensions (M, N, K) and
  the number of iterations via command-line arguments.
- **Hardcoded for 2 GPUs**: The underlying C++ implementation is specifically
  written and hardcoded for a 2-GPU setup.

Usage:
    python perf_only.py --m=8192 --n=8192 --k=8192 --iterations=100
"""

import torch
from torch.utils.cpp_extension import load
import os
import argparse
import time

def load_cutlass_kernel():
    """
    Loads the distributed CUTLASS GEMM CUDA kernel using PyTorch's JIT compiler.

    This function compiles `gemm_multi_sm90_official.cu`, setting the target
    architecture to SM90 (NVIDIA Hopper) and including the necessary CUTLASS
    header paths and compilation flags for distributed GEMM.

    Returns:
        A loaded Python module containing the `gemm` function.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set the target CUDA architecture for JIT compilation to Hopper.
    os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a'
    
    # Get the absolute path to the cutlass directory.
    # Assumes this script is in a subdirectory of `9_cutlass/official`.
    cutlass_path = os.path.abspath(os.path.join(current_dir, '../../../cutlass'))

    return load(
        name='cutlass_gemm_multi_sm90_official',
        sources=[os.path.join(current_dir, 'gemm_multi_sm90_official.cu')],
        extra_cuda_cflags=[
            '-O3',
            '--use_fast_math',
            '-std=c++17',
            # Compile for SM90a (Hopper with CUDA 12+ GDS support)
            '-gencode=arch=compute_90a,code=sm_90a',
            # Add include paths for CUTLASS headers.
            '-I' + os.path.join(cutlass_path, 'include'),
            '-I' + os.path.join(cutlass_path, 'tools/util/include'),
            '-I' + os.path.join(cutlass_path, 'examples'),
            # This flag is required for the distributed GEMM example.
            '-DCUTLASS_ENABLE_GDC_FOR_SM90=1',
        ],
        verbose=False,
    )

def benchmark_gemm(cutlass_module, M, N, K, num_iterations=100, num_warmup=10):
    """
    Runs a pure performance benchmark for the loaded CUTLASS GEMM kernel.

    Args:
        cutlass_module: The loaded CUDA extension module.
        M, N, K (int): The dimensions of the GEMM problem.
        num_iterations (int): The number of timed benchmark iterations.
        num_warmup (int): The number of warmup iterations.

    Returns:
        A tuple containing the average execution time (ms) and performance (TFLOPS).
    """
    # Note: For this distributed kernel, all tensors must be on the primary device (`cuda:0`)
    # The kernel itself handles the distribution of data to other GPUs.
    A = torch.randn(M, K, dtype=torch.float16, device='cuda:0')
    B = torch.randn(K, N, dtype=torch.float16, device='cuda:0')
    C_cutlass = torch.zeros(M, N, dtype=torch.float16, device='cuda:0')
    
    # Warmup iterations
    for _ in range(num_warmup):
        cutlass_module.gemm(A, B, C_cutlass)
    torch.cuda.synchronize()
    
    # Timing using CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iterations):
        cutlass_module.gemm(A, B, C_cutlass)
    end_event.record()
    torch.cuda.synchronize()
    
    # Calculate performance metrics
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_ms / num_iterations
    flops = 2 * M * N * K
    tflops = (flops / (avg_time_ms / 1000.0)) / 1e12
    
    return avg_time_ms, tflops

def main():
    """
    Main function to parse arguments, run the benchmark, and print results.
    """
    parser = argparse.ArgumentParser(description='Official Distributed GEMM Performance-Only Benchmark')
    parser.add_argument('--gpus', type=int, default=2, help='Number of GPUs (currently fixed at 2 for this kernel)')
    parser.add_argument('--m', type=int, default=8192, help='M dimension of the GEMM')
    parser.add_argument('--n', type=int, default=8192, help='N dimension of the GEMM')
    parser.add_argument('--k', type=int, default=8192, help='K dimension of the GEMM')
    parser.add_argument('--iterations', type=int, default=100, help='Number of benchmark iterations')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup iterations')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Official Distributed GEMM - Performance-Only Benchmark")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a CUDA-enabled GPU.")
        return
    
    num_gpus_available = torch.cuda.device_count()
    print("\nGPU Configuration:")
    print(f"  Available GPUs: {num_gpus_available}")
    for i in range(num_gpus_available):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    if num_gpus_available < args.gpus:
        print(f"\nERROR: Requested {args.gpus} GPUs but only {num_gpus_available} are available.")
        return
    if args.gpus != 2:
        print(f"\nWARNING: This specific kernel is hard-coded for 2 GPUs, but {args.gpus} were requested. Exiting.")
        return

    print(f"\nCompiling CUTLASS kernel (this may take a few minutes)...")
    start_time = time.time()
    try:
        cutlass_module = load_cutlass_kernel()
        compile_time = time.time() - start_time
        print(f"✅ Compilation successful ({compile_time:.1f}s)\n")
    except Exception as e:
        print(f"❌ Compilation failed. Please check CUDA/compiler paths and CUTLASS headers.")
        print(f"   Error: {e}")
        return
    
    print(f"Problem size: {args.m} x {args.n} x {args.k}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Benchmark iterations: {args.iterations}")
    print("\nRunning benchmark...")
    
    avg_time_ms, tflops = benchmark_gemm(
        cutlass_module, args.m, args.n, args.k,
        num_iterations=args.iterations,
        num_warmup=args.warmup
    )
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"GPUs:              {args.gpus}")
    print(f"Problem size:      {args.m} x {args.n} x {args.k}")
    print(f"Avg kernel time:   {avg_time_ms:.3f} ms")
    print(f"Total Performance: {tflops:.2f} TFLOPS")
    print(f"Per-GPU Performance: {tflops / args.gpus:.2f} TFLOPS")
    print("=" * 80)
    
if __name__ == '__main__':
    main()

