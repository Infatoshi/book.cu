"""
Official Multi-GPU GEMM Benchmark (Hopper SM90 - 2 GPUs)
Uses CUTLASS Example 65 distributed GEMM approach with AllGather schedule
"""

import torch
import os
import sys
from torch.utils.cpp_extension import load

def check_gpu_count():
    """Verify we have at least 2 GPUs"""
    if torch.cuda.device_count() < 2:
        print(f"Error: This benchmark requires at least 2 GPUs, but found {torch.cuda.device_count()}")
        sys.exit(1)
    print(f"Found {torch.cuda.device_count()} GPUs")
    for i in range(min(2, torch.cuda.device_count())):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

def load_cutlass_kernel():
    """Load the CUTLASS multi-GPU kernel"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    
    os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'
    
    print("Compiling official CUTLASS multi-GPU kernel (this may take a few minutes)...")
    
    return load(
        name='cutlass_gemm_multi_sm90_official',
        sources=[os.path.join(current_dir, 'gemm_multi_sm90_official.cu')],
        extra_cuda_cflags=[
            '-O3',
            '--use_fast_math',
            '-std=c++17',
            '-gencode=arch=compute_90a,code=sm_90a',  
            '-I/mnt/storage/cuda-book/cutlass/include',
            '-I/mnt/storage/cuda-book/cutlass/tools/util/include',
            '-I/mnt/storage/cuda-book/cutlass/examples',
            '-DCUTLASS_ENABLE_GDC_FOR_SM90=1',  
        ],
        verbose=False,
    )

def verify_numerical(cutlass, M, N, K, num_passes=3):
    """Verify numerical correctness against PyTorch cuBLAS"""
    print(f"\nVerifying numerical correctness for size {M}x{N}x{K} ({num_passes} passes)...")
    
    for pass_num in range(1, num_passes + 1):
        
        A = torch.randn(M, K, dtype=torch.float16, device='cuda:0')
        B = torch.randn(K, N, dtype=torch.float16, device='cuda:0')
        
        
        C_ref = torch.matmul(A, B)
        
        
        C_cutlass = torch.zeros(M, N, dtype=torch.float16, device='cuda:0')
        cutlass.gemm(A, B, C_cutlass)
        
        
        atol = 0.25 if max(M, N, K) <= 2048 else 2.0
        if torch.allclose(C_cutlass, C_ref, atol=atol, rtol=1e-2):
            print(f"  Pass {pass_num}: ✓ Numerical verification passed (max diff: {torch.max(torch.abs(C_cutlass - C_ref)).item():.3f})")
        else:
            max_diff = torch.max(torch.abs(C_cutlass - C_ref)).item()
            print(f"  Pass {pass_num}: ✗ Numerical verification FAILED (max diff: {max_diff})")
            raise RuntimeError(f"Numerical verification failed on pass {pass_num}")

def benchmark_gemm(cutlass, M, N, K, num_iterations=10, num_warmup=3):
    """Benchmark GEMM performance"""
    print(f"\nBenchmarking size {M}x{N}x{K}...")
    
    
    A = torch.randn(M, K, dtype=torch.float16, device='cuda:0')
    B = torch.randn(K, N, dtype=torch.float16, device='cuda:0')
    C_torch = torch.zeros(M, N, dtype=torch.float16, device='cuda:0')
    
    C_cutlass = torch.zeros(M, N, dtype=torch.float16, device='cuda:0')
    
    
    for _ in range(num_warmup):
        torch.matmul(A, B, out=C_torch)
        cutlass.gemm(A, B, C_cutlass)
    torch.cuda.synchronize()
    
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iterations):
        torch.matmul(A, B, out=C_torch)
    end_event.record()
    torch.cuda.synchronize()
    
    torch_time = start_event.elapsed_time(end_event) / num_iterations
    
    
    start_event.record()
    for _ in range(num_iterations):
        cutlass.gemm(A, B, C_cutlass)
    end_event.record()
    torch.cuda.synchronize()
    
    cutlass_time = start_event.elapsed_time(end_event) / num_iterations
    
    
    flops = 2 * M * N * K
    torch_tflops = (flops / torch_time) / 1e9
    cutlass_tflops = (flops / cutlass_time) / 1e9
    speedup = torch_time / cutlass_time
    
    print(f"  PyTorch (cuBLAS, 1xSM90):    {torch_time:.3f} ms  ({torch_tflops:.2f} TFLOPS)")
    print(f"  CUTLASS Official 2xSM90:     {cutlass_time:.3f} ms  ({cutlass_tflops:.2f} TFLOPS)")
    print(f"  Speedup: {speedup:.2f}x")
    
    return {
        'M': M, 'N': N, 'K': K,
        'torch_time': torch_time,
        'cutlass_time': cutlass_time,
        'torch_tflops': torch_tflops,
        'cutlass_tflops': cutlass_tflops,
        'speedup': speedup
    }

def main():
    print("=" * 80)
    print("Official Multi-GPU GEMM Benchmark (Hopper SM90 - 2 GPUs)")
    print("=" * 80)
    
    
    check_gpu_count()
    
    
    cutlass = load_cutlass_kernel()
    print("✓ Kernel compiled successfully\n")
    
    
    sizes = [
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]
    
    results = []
    
    for M, N, K in sizes:
        
        verify_numerical(cutlass, M, N, K, num_passes=3)
        
        
        result = benchmark_gemm(cutlass, M, N, K, num_iterations=10, num_warmup=3)
        results.append(result)
    
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Size':<20} {'PyTorch (ms)':<15} {'CUTLASS (ms)':<15} {'Speedup':<10}")
    print("-" * 80)
    for r in results:
        size_str = f"{r['M']}x{r['N']}x{r['K']}"
        print(f"{size_str:<20} {r['torch_time']:<15.3f} {r['cutlass_time']:<15.3f} {r['speedup']:<10.2f}x")
    print("=" * 80)

if __name__ == '__main__':
    main()

