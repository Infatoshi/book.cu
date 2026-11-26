"""
This script serves as a modular framework for comparing the performance and
correctness of various attention mechanism implementations. It is designed to
benchmark custom CUDA kernels against PyTorch's native implementations.

The script compares four main types of attention:
- PyTorch Naive: A manual implementation in PyTorch that follows the standard
  mathematical definition: `softmax((Q @ K.T) / sqrt(d)) @ V`. This version
  is simple but memory-inefficient as it materializes the full N x N attention
  score matrix.
- PyTorch Flash: PyTorch's optimized, built-in Flash Attention, accessed via
  `torch.nn.functional.scaled_dot_product_attention` with the Flash Attention
  backend enabled. This is the high-performance industry standard.
- Naive CUDA Kernel: A custom CUDA implementation that mirrors the naive PyTorch
  approach but with separate kernel launches for each step (QK^T, softmax, SV).
  It also materializes the full attention matrix and serves as a baseline for
  custom kernel development.
- FlashAttention-2.5 CUDA Kernel: A custom CUDA implementation of a FlashAttention-like
  algorithm that uses tiling and an online softmax to avoid materializing the
  full attention matrix. It is highly optimized for performance and memory
  efficiency, using WMMA tensor cores for matrix multiplications.

To add a new custom kernel to the comparison:
1.  Create the CUDA source file (e.g., `kernels/my_kernel.cu`) and a
    corresponding C++ binding file (e.g., `kernels/build_my_kernel.cpp`).
2.  Add a new `KernelConfig` entry to the `KERNELS` dictionary in this script.
    The configuration should specify the module name, source files, and build
    directory.
3.  Run the script from the command line, optionally specifying the kernel name
    to test (e.g., `python main.py --kernels my_kernel`).
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
    """
    A data class to hold the configuration for a single CUDA kernel.
    This includes metadata needed for Just-In-Time (JIT) compilation by PyTorch.

    Attributes:
        name (str): The name of the compiled module.
        sources (List[str]): A list of paths to the source files (.cu, .cpp) to be compiled.
        build_dir (str): The directory where build artifacts will be stored.
        extra_cflags (List[str]): Optional list of extra compiler flags (e.g., '-O3').
    """
    name: str
    sources: List[str]
    build_dir: str
    extra_cflags: List[str] = None
    
    def __post_init__(self):
        if self.extra_cflags is None:
            self.extra_cflags = ['-O3']

@dataclass
class BenchmarkResult:
    """
    A data class to store the results of a single benchmark run for a kernel.

    Attributes:
        kernel_name (str): The name of the kernel that was benchmarked.
        avg_time_ms (float): The average execution time in milliseconds.
        throughput_tflops (float): The calculated throughput in TFLOP/s.
        correctness (bool): True if the output passed the correctness check.
        max_diff (float): The maximum absolute difference compared to the reference output.
    """
    kernel_name: str
    avg_time_ms: float
    throughput_tflops: float
    correctness: bool
    max_diff: float


# Dictionary mapping kernel names to their build configurations.
# This is the central registry for all custom kernels to be tested.
KERNELS = {
    # Configuration for the naive, non-optimized CUDA attention kernel.
    'naive': KernelConfig(
        name='naive_attn',
        sources=['kernels/build_naive.cpp', 'kernels/naive.cu'],
        build_dir='./build/naive'
    ),
    # Configuration for the custom FlashAttention-2.5 WMMA kernel.
    'fa': KernelConfig(
        name='fa_attn',
        sources=['kernels/build_fa.cpp', 'kernels/fa.cu'],
        build_dir='./build/fa'
    ),
}


def load_kernel(config: KernelConfig):
    """
    Loads and JIT-compiles a CUDA kernel using `torch.utils.cpp_extension.load`.

    Args:
        config (KernelConfig): The configuration object for the kernel.

    Returns:
        A loaded Python module containing the bound CUDA function.
    """
    print(f"Loading {config.name}...")
    return load(
        name=config.name,
        sources=config.sources,
        build_directory=config.build_dir,
        extra_cuda_cflags=config.extra_cflags,
        verbose=False
    )

def pytorch_naive_attention(q, k, v):
    """
    A naive PyTorch implementation of the standard attention mechanism.

    This function explicitly materializes the N x N attention matrix, making it
    a good baseline for demonstrating the memory and performance costs that
    FlashAttention aims to solve.

    Args:
        q (torch.Tensor): The query tensor.
        k (torch.Tensor): The key tensor.
        v (torch.Tensor): The value tensor.

    Returns:
        torch.Tensor: The output of the attention mechanism.
    """
    scale = 1.0 / math.sqrt(q.size(-1))
    att = (q @ k.transpose(-2, -1)) * scale
    att = F.softmax(att, dim=-1)
    return att @ v

def pytorch_flash_attention(q, k, v):
    """
    PyTorch's built-in Flash Attention implementation using SDPA.

    This function uses `torch.nn.functional.scaled_dot_product_attention`, which
    under the hood dispatches to a highly optimized FlashAttention implementation
    if the hardware and input conditions are suitable. It serves as the primary
    high-performance baseline for comparison.

    Args:
        q (torch.Tensor): The query tensor.
        k (torch.Tensor): The key tensor.
        v (torch.Tensor): The value tensor.

    Returns:
        torch.Tensor: The output from the scaled dot-product attention.
    """
    # Note: As of PyTorch 2.0, SDPA automatically handles Flash Attention.
    # The context manager is used here to explicitly request the Flash Attention backend.
    # Inputs are converted to bfloat16 as it's often faster on modern GPUs.
    q_bf16 = q.to(torch.bfloat16)
    k_bf16 = k.to(torch.bfloat16)
    v_bf16 = v.to(torch.bfloat16)

    
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        result = F.scaled_dot_product_attention(q_bf16, k_bf16, v_bf16)

    
    return result.to(q.dtype)

def compute_flops(batch_size, n_heads, seq_len, head_dim):
    """
    Computes the approximate total floating-point operations for an attention layer.

    The calculation is based on the two main matrix multiplications:
    1.  Q @ K^T: `2 * B * H * N * N * D` (2 for mul/add)
    2.  Softmax @ V: `2 * B * H * N * N * D`

    The FLOPs for the softmax operation itself are minor in comparison and are
    approximated. The total is roughly `4 * B * H * N^2 * D`.

    Args:
        batch_size (int): Batch size.
        n_heads (int): Number of attention heads.
        seq_len (int): Sequence length.
        head_dim (int): Dimension of each attention head.

    Returns:
        int: The total approximate FLOPs.
    """
    # FLOPs for Q @ K^T: B * H * N * D * N * 2
    # FLOPs for S @ V:   B * H * N * N * D * 2
    # Total is approx. 4 * B * H * N^2 * D
    return 4 * batch_size * n_heads * (seq_len ** 2) * head_dim

def benchmark_kernel(
    func: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_warmup: int = 5,
    n_iter: int = 20
) -> tuple:
    """
    Benchmarks a given attention function using CUDA events for precise timing.

    Args:
        func (Callable): The attention function to benchmark.
        q, k, v (torch.Tensor): Input tensors.
        n_warmup (int): Number of warmup iterations to run before timing.
        n_iter (int): Number of timed iterations.

    Returns:
        A tuple containing:
        - The output tensor from the last function call.
        - The average execution time in milliseconds.
    """
    
    # Warmup iterations to stabilize GPU clocks and cache state
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
    """
    Compares the output of a kernel against a reference tensor to check for correctness.

    Args:
        output (torch.Tensor): The output tensor from the kernel being tested.
        reference (torch.Tensor): The ground truth tensor.
        atol (float): The absolute tolerance for the comparison.

    Returns:
        A tuple containing:
        - A boolean indicating if the check passed.
        - The maximum absolute difference found between the two tensors.
    """
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
    Main function to drive the comparison of multiple attention implementations.

    It generates random input data, benchmarks PyTorch's native implementations to
    establish a baseline, then JIT-compiles, benchmarks, and verifies the
    correctness of each custom CUDA kernel specified in the `KERNELS` dictionary.
    Finally, it prints a summary table of the results.

    Args:
        batch_size (int): Batch size for the input tensors.
        n_heads (int): Number of attention heads.
        seq_len (int): Sequence length of the input.
        head_dim (int): Dimension of each attention head.
        dtype (torch.dtype): The data type for the tensors.
        kernels_to_test (List[str]): A list of specific kernel names from the
            `KERNELS` dictionary to test. If None, all kernels are tested.
    """
    
    print("=" * 80)
    print("ATTENTION KERNEL COMPARISON")
    print("=" * 80)
    print(f"Config: B={batch_size}, H={n_heads}, N={seq_len}, D={head_dim}")
    print(f"Total parameters: {batch_size * n_heads * seq_len * head_dim:,}")
    print("=" * 80)
    
    # Generate random input tensors on the GPU
    q = torch.randn((batch_size, n_heads, seq_len, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((batch_size, n_heads, seq_len, head_dim), device="cuda", dtype=dtype)
    v = torch.randn((batch_size, n_heads, seq_len, head_dim), device="cuda", dtype=dtype)
    
    # Define the PyTorch implementations to be used as baselines.
    pytorch_baselines = [
        ("PyTorch Naive", pytorch_naive_attention),
        ("PyTorch Flash", pytorch_flash_attention),
    ]

    # --- Benchmarking PyTorch Baselines ---
    print("\n[1/3] Benchmarking PyTorch baselines...")
    pytorch_results = []
    reference_output = None
    for name, func in pytorch_baselines:
        output, avg_time = benchmark_kernel(func, q, k, v)
        if reference_output is None:
            # Use the output of the first baseline (PyTorch Naive) as the ground truth.
            reference_output = output  
        pytorch_results.append((name, avg_time))
        print(f"    ✓ {name}: {avg_time:.3f} ms")
    
    # --- Benchmarking Custom CUDA Kernels ---
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
            # JIT-compile the kernel
            kernel = load_kernel(config)
            
            # Benchmark the forward pass of the custom kernel
            print(f"    Benchmarking {kernel_name}...")
            output, avg_time = benchmark_kernel(kernel.forward, q, k, v)
            
            # Check correctness against the reference output
            is_correct, max_diff = check_correctness(output, reference_output)
            
            # Calculate throughput in TFLOP/s
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
    
    # --- Print Results Summary ---
    print("\n[3/3] Results Summary")
    print("=" * 80)
    print(f"{'Implementation':<18} {'Time (ms)':<12} {'Throughput':<15} {'Speedup':<10} {'Correct':<10} {'Max Diff':<10}")
    print("-" * 80)

    # Print baseline results
    fastest_pytorch_time = min(time for _, time in pytorch_results)
    for name, pytorch_time in pytorch_results:
        pytorch_throughput = (total_flops / (pytorch_time * 1e-3)) / 1e12
        speedup = fastest_pytorch_time / pytorch_time
        print(f"{name:<18} {pytorch_time:>10.3f}  {pytorch_throughput:>11.2f} TF/s  {speedup:>6.2f}x    {'✓':<10} {'-':<10}")

    
    # Sort and print custom kernel results
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
    
    
    # --- Print Final Conclusion ---
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
    
    # --- Command-line argument parsing ---
    parser = argparse.ArgumentParser(description="Compare attention kernel implementations")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--kernels", type=str, nargs="+", default=None,
                        help="Space-separated list of kernels to test (e.g., 'naive fa'). Default is all.")
    
    args = parser.parse_args()
    
    compare_kernels(
        batch_size=args.batch_size,
        n_heads=args.n_heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        kernels_to_test=args.kernels
    )
