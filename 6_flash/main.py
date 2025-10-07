"""
Modular Attention Kernel Comparison Framework

This script compares different attention implementations:
- PyTorch Naive: Manual attention (Q@K^T, softmax, S@V) - materializes full N×N matrix
- PyTorch Flash: PyTorch's built-in Flash Attention using SDPA
- Naive CUDA: Basic CUDA kernel with separate QK^T, softmax, SV operations
- FA2 CUDA: Custom Flash Attention 2 kernel with fused operations and tiling

To add a new kernel:
1. Create kernels/your_kernel.cu and kernels/build_your_kernel.cpp
2. Add entry to KERNELS dict below
3. Run this script!
"""

import os
import math
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from dataclasses import dataclass
from typing import Callable, Dict, List
import time

os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"

@dataclass
class KernelConfig:
    """Configuration for a kernel implementation"""
    name: str
    sources: List[str]
    build_dir: str
    extra_cflags: List[str] = None
    
    def __post_init__(self):
        if self.extra_cflags is None:
            self.extra_cflags = ['-O3']

@dataclass
class BenchmarkResult:
    """Results from benchmarking a kernel"""
    kernel_name: str
    avg_time_ms: float
    throughput_tflops: float
    correctness: bool
    max_diff: float


KERNELS = {
    'naive': KernelConfig(
        name='naive_attn',
        sources=['kernels/build_naive.cpp', 'kernels/naive.cu'],
        build_dir='./build/naive'
    ),
    'fa': KernelConfig(
        name='fa_attn',
        sources=['kernels/build_fa.cpp', 'kernels/fa.cu'],
        build_dir='./build/fa'
    ),
}


def load_kernel(config: KernelConfig):
    """Load a CUDA kernel using JIT compilation"""
    print(f"Loading {config.name}...")
    return load(
        name=config.name,
        sources=config.sources,
        build_directory=config.build_dir,
        extra_cuda_cflags=config.extra_cflags,
        verbose=False
    )

def pytorch_naive_attention(q, k, v):
    """Naive PyTorch attention implementation (materializes full attention matrix)"""
    scale = 1.0 / math.sqrt(q.size(-1))
    att = (q @ k.transpose(-2, -1)) * scale
    att = F.softmax(att, dim=-1)
    return att @ v

def pytorch_flash_attention(q, k, v):
    """PyTorch Flash Attention implementation using SDPA"""
    
    q_bf16 = q.to(torch.bfloat16)
    k_bf16 = k.to(torch.bfloat16)
    v_bf16 = v.to(torch.bfloat16)

    
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        result = F.scaled_dot_product_attention(q_bf16, k_bf16, v_bf16)

    
    return result.to(q.dtype)

def compute_flops(batch_size, n_heads, seq_len, head_dim):
    """
    Compute FLOPs for attention:
    - Q @ K^T: 2 * B * H * N * N * D
    - Softmax: ~5 * B * H * N * N (approximation)
    - S @ V: 2 * B * H * N * N * D
    Total ≈ 4 * B * H * N^2 * D
    """
    return 4 * batch_size * n_heads * (seq_len ** 2) * head_dim

def benchmark_kernel(
    func: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_warmup: int = 5,
    n_iter: int = 20
) -> tuple:
    """Benchmark a kernel and return (output, avg_time_ms)"""
    
    
    for _ in range(n_warmup):
        _ = func(q, k, v)
    torch.cuda.synchronize()
    
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(n_iter):
        output = func(q, k, v)
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_ms / n_iter
    
    return output, avg_time_ms

def check_correctness(output: torch.Tensor, reference: torch.Tensor, atol: float = 1e-2) -> tuple:
    """Check if output matches reference within tolerance"""
    diff = torch.abs(output - reference)
    max_diff = diff.max().item()
    is_correct = torch.allclose(output, reference, rtol=0, atol=atol)
    return is_correct, max_diff


def compare_kernels(
    batch_size: int = 16,
    n_heads: int = 8,
    seq_len: int = 512,
    head_dim: int = 64,
    dtype: torch.dtype = torch.float32,
    kernels_to_test: List[str] = None
):
    """
    Compare multiple attention kernel implementations
    
    Args:
        batch_size: Batch size
        n_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Head dimension
        dtype: Data type (only float32 supported for now)
        kernels_to_test: List of kernel names to test (None = test all)
    """
    
    print("=" * 80)
    print("ATTENTION KERNEL COMPARISON")
    print("=" * 80)
    print(f"Config: B={batch_size}, H={n_heads}, N={seq_len}, D={head_dim}")
    print(f"Total parameters: {batch_size * n_heads * seq_len * head_dim:,}")
    print("=" * 80)
    
    
    q = torch.randn((batch_size, n_heads, seq_len, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((batch_size, n_heads, seq_len, head_dim), device="cuda", dtype=dtype)
    v = torch.randn((batch_size, n_heads, seq_len, head_dim), device="cuda", dtype=dtype)
    
    
    pytorch_baselines = [
        ("PyTorch Naive", pytorch_naive_attention),
        ("PyTorch Flash", pytorch_flash_attention),
    ]

    
    print("\n[1/3] Benchmarking PyTorch baselines...")
    pytorch_results = []
    reference_output = None
    for name, func in pytorch_baselines:
        output, avg_time = benchmark_kernel(func, q, k, v)
        if reference_output is None:
            reference_output = output  
        pytorch_results.append((name, avg_time))
        print(f"    ✓ {name}: {avg_time:.3f} ms")
    
    
    if kernels_to_test is None:
        kernels_to_test = list(KERNELS.keys())
    
    
    results: List[BenchmarkResult] = []
    total_flops = compute_flops(batch_size, n_heads, seq_len, head_dim)
    
    print(f"\n[2/3] Loading and benchmarking {len(kernels_to_test)} CUDA kernels...")
    for kernel_name in kernels_to_test:
        if kernel_name not in KERNELS:
            print(f"    ⚠ Warning: Unknown kernel '{kernel_name}', skipping")
            continue
        
        config = KERNELS[kernel_name]
        try:
            
            kernel = load_kernel(config)
            
            
            print(f"    Benchmarking {kernel_name}...")
            output, avg_time = benchmark_kernel(kernel.forward, q, k, v)
            
            
            is_correct, max_diff = check_correctness(output, reference_output)
            
            
            throughput_tflops = (total_flops / (avg_time * 1e-3)) / 1e12
            
            results.append(BenchmarkResult(
                kernel_name=kernel_name,
                avg_time_ms=avg_time,
                throughput_tflops=throughput_tflops,
                correctness=is_correct,
                max_diff=max_diff
            ))
            
        except Exception as e:
            print(f"    ✗ Error loading/running {kernel_name}: {e}")
    
    
    print("\n[3/3] Results Summary")
    print("=" * 80)
    print(f"{'Implementation':<18} {'Time (ms)':<12} {'Throughput':<15} {'Speedup':<10} {'Correct':<10} {'Max Diff':<10}")
    print("-" * 80)

    
    fastest_pytorch_time = min(time for _, time in pytorch_results)
    for name, pytorch_time in pytorch_results:
        pytorch_throughput = (total_flops / (pytorch_time * 1e-3)) / 1e12
        speedup = fastest_pytorch_time / pytorch_time
        print(f"{name:<18} {pytorch_time:>10.3f}  {pytorch_throughput:>11.2f} TF/s  {speedup:>6.2f}x    {'✓':<10} {'-':<10}")

    
    
    results.sort(key=lambda x: x.avg_time_ms)

    for result in results:
        speedup = fastest_pytorch_time / result.avg_time_ms
        correct_mark = '✓' if result.correctness else '✗'
        print(f"{result.kernel_name:<18} {result.avg_time_ms:>10.3f}  "
              f"{result.throughput_tflops:>11.2f} TF/s  "
              f"{speedup:>6.2f}x    "
              f"{correct_mark:<10} "
              f"{result.max_diff:<10.6f}")

    print("=" * 80)
    
    
    if results:
        best = results[0]
        best_speedup = fastest_pytorch_time / best.avg_time_ms
        print(f"\n🏆 Best CUDA kernel: {best.kernel_name} ({best_speedup:.2f}x faster than fastest PyTorch)")

        if len(results) > 1:
            second_best = results[1]
            improvement = second_best.avg_time_ms / best.avg_time_ms
            print(f"   ({improvement:.2f}x faster than {second_best.kernel_name})")
    
    print()
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare attention kernel implementations")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--kernels", type=str, nargs="+", default=None,
                        help="Kernels to test (default: all)")
    
    args = parser.parse_args()
    
    compare_kernels(
        batch_size=args.batch_size,
        n_heads=args.n_heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        kernels_to_test=args.kernels
    )
