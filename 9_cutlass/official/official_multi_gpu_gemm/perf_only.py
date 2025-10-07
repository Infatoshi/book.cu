"""
Official Distributed GEMM - Performance-Only Benchmark
=======================================================

Simple performance measurement script that focuses purely on TFLOPS.
No verification, no PyTorch comparison - just raw kernel performance.

Usage:
    python perf_only.py --gpus=2 --m=8192 --n=8192 --k=8192 --iterations=100
"""

import torch
from torch.utils.cpp_extension import load
import os
import argparse
import time

def load_cutlass_kernel():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'
    return load(
        name='cutlass_gemm_multi_sm90_official',
        sources=[os.path.join(current_dir, 'gemm_multi_sm90_official.cu')],
        extra_cuda_cflags=[
            '-O3', '--use_fast_math', '-std=c++17',
            '-gencode=arch=compute_90a,code=sm_90a',
            '-I/mnt/storage/cuda-book/cutlass/include',
            '-I/mnt/storage/cuda-book/cutlass/tools/util/include',
            '-I/mnt/storage/cuda-book/cutlass/examples',
            '-DCUTLASS_ENABLE_GDC_FOR_SM90=1',
        ],
        verbose=False,
    )

def benchmark_gemm(cutlass, M, N, K, num_iterations=100, num_warmup=10):
    """Pure performance benchmark - no verification"""
    
    A = torch.randn(M, K, dtype=torch.float16, device='cuda:0')
    B = torch.randn(K, N, dtype=torch.float16, device='cuda:0')
    C_cutlass = torch.zeros(M, N, dtype=torch.float16, device='cuda:0')
    
    
    for _ in range(num_warmup):
        cutlass.gemm(A, B, C_cutlass)
    torch.cuda.synchronize()
    
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iterations):
        cutlass.gemm(A, B, C_cutlass)
    end_event.record()
    torch.cuda.synchronize()
    
    
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_ms / num_iterations
    flops = 2 * M * N * K
    tflops = (flops / (avg_time_ms / 1000.0)) / 1e12
    
    return avg_time_ms, tflops

def main():
    parser = argparse.ArgumentParser(description='Official Distributed GEMM Performance Benchmark')
    parser.add_argument('--gpus', type=int, default=2, help='Number of GPUs (currently fixed at 2)')
    parser.add_argument('--m', type=int, default=8192, help='M dimension')
    parser.add_argument('--n', type=int, default=8192, help='N dimension')
    parser.add_argument('--k', type=int, default=8192, help='K dimension')
    parser.add_argument('--iterations', type=int, default=100, help='Number of benchmark iterations')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup iterations')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Official Distributed GEMM - Performance Benchmark")
    print("=" * 80)
    
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"\nGPU Configuration:")
    print(f"  Available GPUs: {num_gpus}")
    for i in range(min(num_gpus, args.gpus)):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    if num_gpus < args.gpus:
        print(f"\nWARNING: Requested {args.gpus} GPUs but only {num_gpus} available")
        print("Note: Current implementation is hard-coded for 2 GPUs")
        return
    
    
    print(f"\nCompiling CUTLASS kernel (may take a few minutes)...")
    start_time = time.time()
    try:
        cutlass = load_cutlass_kernel()
        compile_time = time.time() - start_time
        print(f"✓ Compilation successful ({compile_time:.1f}s)\n")
    except RuntimeError as e:
        print(f"✗ Compilation failed: {e}")
        return
    
    
    print(f"Problem size: {args.m} x {args.n} x {args.k}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Benchmark iterations: {args.iterations}")
    print("\nRunning benchmark...")
    
    avg_time_ms, tflops = benchmark_gemm(
        cutlass, args.m, args.n, args.k,
        num_iterations=args.iterations,
        num_warmup=args.warmup
    )
    
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"GPUs:              {args.gpus}")
    print(f"Problem size:      {args.m} x {args.n} x {args.k}")
    print(f"Avg kernel time:   {avg_time_ms:.3f} ms")
    print(f"Performance:       {tflops:.2f} TFLOPS")
    print(f"Per-GPU:           {tflops / args.gpus:.2f} TFLOPS")
    print("=" * 80)
    
if __name__ == '__main__':
    main()

