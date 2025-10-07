"""
Official CUTLASS Ampere (SM80) FP16 GEMM Benchmark

Uses CUTLASS's optimized Ampere path with:
- WMMA (Warp Matrix Multiply Accumulate) tensor core instructions
- Large tile sizes (128x256x32) for better occupancy
- 3-stage software pipeline to hide memory latency
- Optimized threadblock swizzling
"""
import os
import torch
import numpy as np
from torch.utils.cpp_extension import load

def load_cutlass_gemm():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'
    
    return load(
        name='cutlass_gemm_official_sm80',
        sources=[os.path.join(current_dir, 'gemm_sm80_official.cu')],
        extra_cuda_cflags=[
            '-O3',
            '--use_fast_math',
            '-std=c++17',
            '-arch=sm_80',
            '-I../cutlass/include',
            '-I../cutlass/tools/util/include',
        ],
        verbose=False,
    )

def benchmark_kernel(func, *args, warmup=3, iters=10):
    """Benchmark a kernel with CUDA events"""
    for _ in range(warmup):
        func(*args)
    
    torch.cuda.synchronize()
    
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

def verify_numerical(M, N, K, rtol=1e-2, atol=0.5):
    """Run numerical verification with 3 passes"""
    print(f"\n{'='*60}")
    print(f"Verifying {M}x{N}x{K}")
    print(f"{'='*60}")
    
    cutlass = load_cutlass_gemm()
    
    for pass_num in range(3):
        A = torch.randn(M, K, dtype=torch.float16, device='cuda')
        B = torch.randn(K, N, dtype=torch.float16, device='cuda')
        C_ref = torch.mm(A, B)
        C_cutlass = torch.empty(M, N, dtype=torch.float16, device='cuda')
        
        cutlass.gemm(A, B, C_cutlass)
        torch.cuda.synchronize()
        
        max_diff = torch.max(torch.abs(C_cutlass - C_ref.half())).item()
        passed = max_diff < atol
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Pass {pass_num + 1}/3: {status} (max_diff={max_diff:.6f})")
        
        if not passed:
            raise RuntimeError(f"Numerical verification failed on pass {pass_num + 1}")
    
    print("✓ All verification passes succeeded\n")

def benchmark_performance(M, N, K):
    """Benchmark CUTLASS vs PyTorch for a given problem size"""
    print(f"\n{'='*60}")
    print(f"Benchmarking {M}x{N}x{K}")
    print(f"{'='*60}\n")
    
    cutlass = load_cutlass_gemm()
    
    A = torch.randn(M, K, dtype=torch.float16, device='cuda')
    B = torch.randn(K, N, dtype=torch.float16, device='cuda')
    C = torch.empty(M, N, dtype=torch.float16, device='cuda')
    
    
    cutlass_time, cutlass_std = benchmark_kernel(cutlass.gemm, A, B, C)
    
    
    pytorch_time, pytorch_std = benchmark_kernel(torch.mm, A, B)
    
    
    gflops = (2.0 * M * N * K) / 1e9
    cutlass_gflops = gflops / (cutlass_time / 1000)
    pytorch_gflops = gflops / (pytorch_time / 1000)
    speedup = pytorch_time / cutlass_time
    
    print(f"{'Method':<20} {'Time (ms)':<20} {'GFLOPS':<15} {'Speedup':<10}")
    print(f"{'-'*65}")
    print(f"{'CUTLASS Official':<20} {cutlass_time:>6.3f} ± {cutlass_std:>5.3f}     {cutlass_gflops:>12.1f}    {speedup:>6.2f}x")
    print(f"{'PyTorch cuBLAS':<20} {pytorch_time:>6.3f} ± {pytorch_std:>5.3f}     {pytorch_gflops:>12.1f}    {'1.00x':>6}")

def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    print("="*60)
    print("CUTLASS Official Ampere (SM80) FP16 GEMM Benchmark")
    print("="*60)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Data type: FP16 (input/output), FP32 (accumulator)")
    print(f"Optimizations: WMMA + Large tiles + 3-stage pipeline")
    print(f"Warmup iterations: 3")
    print(f"Benchmark iterations: 10")
    
    
    test_sizes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]
    
    print("\n" + "="*60)
    print("NUMERICAL VERIFICATION")
    print("="*60)
    
    for M, N, K in test_sizes:
        verify_numerical(M, N, K)
    
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARKING")
    print("="*60)
    
    for M, N, K in test_sizes:
        benchmark_performance(M, N, K)
    
    print("\n" + "="*60)
    print("✓ All benchmarks completed")
    print("="*60)

if __name__ == "__main__":
    main()

