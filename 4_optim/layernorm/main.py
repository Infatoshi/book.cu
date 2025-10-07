"""
LayerNorm kernel benchmarking across different shapes
"""
import torch
from torch.utils.cpp_extension import load
import os
import sys
import matplotlib.pyplot as plt

def load_kernels():
    """Load and compile CUDA kernels"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    layernorm_cuda = load(
        name='layernorm_cuda',
        sources=[
            os.path.join(current_dir, 'wrapper.cpp'),
            os.path.join(current_dir, 'kernels/0_naive.cu'),
            os.path.join(current_dir, 'kernels/1_parallel.cu'),
            os.path.join(current_dir, 'kernels/2_warp.cu'),
        ],
        extra_cuda_cflags=['-O3', '--use_fast_math', '-std=c++17'],
        extra_cflags=['-O3', '-std=c++17'],
        verbose=False
    )
    return layernorm_cuda

def benchmark_kernel(func, *args, warmup=2, iters=10):
    """Benchmark a CUDA kernel"""
    for _ in range(warmup):
        result = func(*args)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(iters):
        start.record()
        result = func(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    return result, sum(times) / len(times)

def compute_throughput(N, C, ms):
    """Compute throughput in GFLOPS"""
    ops_per_row = 7 * C  
    total_ops = N * ops_per_row
    gflops = (total_ops / (ms * 1e-3)) / 1e9
    return gflops

def verify_correctness(cuda_out, torch_out, tol=1e-2):
    """Verify numerical correctness"""
    max_diff = (cuda_out - torch_out).abs().max().item()
    mean_diff = (cuda_out - torch_out).abs().mean().item()
    return max_diff <= tol, max_diff, mean_diff

def main():
    print("="*80)
    print("LayerNorm Kernel Benchmarking - Shape Sweep")
    print("="*80)
    
    
    shapes = [
        (8, 256),
        (16, 512),
        (32, 768),
        (64, 1024),
        (128, 2048),
        (256, 4096),
    ]
    
    print("\nCompiling kernels...")
    try:
        kernels = load_kernels()
    except Exception as e:
        print(f"Error loading kernels: {e}")
        sys.exit(1)
    print("✓ Kernels compiled\n")
    
    kernel_configs = [
        ("Kernel 0", "Naive", kernels.kernel_0),
        ("Kernel 1", "Parallel", kernels.kernel_1),
        ("Kernel 2", "Warp", kernels.kernel_2),
    ]
    
    
    results = {name: {'shapes': [], 'latency': [], 'throughput': [], 'speedup': []} 
               for name, _, _ in kernel_configs}
    results['PyTorch'] = {'shapes': [], 'latency': [], 'throughput': [], 'speedup': []}
    
    
    print("Correctness Check (N=8, C=256):")
    N, C = 8, 256
    inp = torch.randn(N, C, device='cuda', dtype=torch.float32)
    weight = torch.ones(C, device='cuda', dtype=torch.float32)
    bias = torch.zeros(C, device='cuda', dtype=torch.float32)
    torch_ln = torch.nn.LayerNorm(C, eps=1e-5).cuda()
    torch_ln.weight.data = weight
    torch_ln.bias.data = bias
    torch_out = torch_ln(inp)
    
    all_pass = True
    for name, desc, kernel_func in kernel_configs:
        cuda_out = kernel_func(inp, weight, bias)
        passed, max_diff, mean_diff = verify_correctness(cuda_out, torch_out)
        if passed:
            print(f"  ✓ {name}: PASS")
        else:
            print(f"  ✗ {name}: FAIL (max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})")
            all_pass = False
    
    if not all_pass:
        print("\n❌ Some kernels failed correctness checks. Exiting.")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("Performance Benchmarking")
    print("="*80)
    
    for N, C in shapes:
        print(f"\nShape: N={N}, C={C} (batch_size={N}, hidden_dim={C})")
        
        inp = torch.randn(N, C, device='cuda', dtype=torch.float32)
        weight = torch.ones(C, device='cuda', dtype=torch.float32)
        bias = torch.zeros(C, device='cuda', dtype=torch.float32)
        torch_ln = torch.nn.LayerNorm(C, eps=1e-5).cuda()
        torch_ln.weight.data = weight
        torch_ln.bias.data = bias
        
        
        torch_out, torch_time = benchmark_kernel(lambda: torch_ln(inp))
        torch_gflops = compute_throughput(N, C, torch_time)
        
        results['PyTorch']['shapes'].append(f"{N}x{C}")
        results['PyTorch']['latency'].append(torch_time)
        results['PyTorch']['throughput'].append(torch_gflops)
        results['PyTorch']['speedup'].append(1.0)
        
        print(f"  PyTorch: {torch_time:.3f} ms, {torch_gflops:.1f} GFLOPS")
        
        
        for name, desc, kernel_func in kernel_configs:
            _, cuda_time = benchmark_kernel(kernel_func, inp, weight, bias)
            cuda_gflops = compute_throughput(N, C, cuda_time)
            speedup = torch_time / cuda_time
            
            results[name]['shapes'].append(f"{N}x{C}")
            results[name]['latency'].append(cuda_time)
            results[name]['throughput'].append(cuda_gflops)
            results[name]['speedup'].append(speedup)
            
            print(f"  {name}: {cuda_time:.3f} ms, {cuda_gflops:.1f} GFLOPS, {speedup:.2f}x")
    
    
    print("\n" + "="*80)
    print("LATENCY SUMMARY (ms)")
    print("="*80)
    print(f"{'Shape':<12} {'PyTorch':<10} " + " ".join([f"{name:<10}" for name, _, _ in kernel_configs]))
    print("-"*80)
    for i, (N, C) in enumerate(shapes):
        row = f"{N}x{C:<9} {results['PyTorch']['latency'][i]:<10.3f} "
        row += " ".join([f"{results[name]['latency'][i]:<10.3f}" for name, _, _ in kernel_configs])
        print(row)
    
    print("\n" + "="*80)
    print("THROUGHPUT SUMMARY (GFLOPS)")
    print("="*80)
    print(f"{'Shape':<12} {'PyTorch':<10} " + " ".join([f"{name:<10}" for name, _, _ in kernel_configs]))
    print("-"*80)
    for i, (N, C) in enumerate(shapes):
        row = f"{N}x{C:<9} {results['PyTorch']['throughput'][i]:<10.1f} "
        row += " ".join([f"{results[name]['throughput'][i]:<10.1f}" for name, _, _ in kernel_configs])
        print(row)
    
    print("\n" + "="*80)
    print("SPEEDUP SUMMARY (vs PyTorch)")
    print("="*80)
    print(f"{'Shape':<12} " + " ".join([f"{name:<10}" for name, _, _ in kernel_configs]))
    print("-"*80)
    for i, (N, C) in enumerate(shapes):
        row = f"{N}x{C:<9} "
        row += " ".join([f"{results[name]['speedup'][i]:<10.2f}" for name, _, _ in kernel_configs])
        print(row)
    
    
    plot_results(results, shapes, kernel_configs)

def plot_results(results, shapes, kernel_configs):
    """Generate performance plots"""
    batch_sizes = [s[0] for s in shapes]
    hidden_dims = [s[1] for s in shapes]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    
    ax = axes[0, 0]
    for name, _, _ in kernel_configs:
        ax.plot(batch_sizes, results[name]['throughput'], marker='o', label=name)
    ax.plot(batch_sizes, results['PyTorch']['throughput'], marker='s', label='PyTorch', linestyle='--')
    ax.set_xlabel('Batch Size (N)')
    ax.set_ylabel('Throughput (GFLOPS)')
    ax.set_title('Throughput vs Batch Size')
    ax.set_xscale('log', base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    
    ax = axes[0, 1]
    for name, _, _ in kernel_configs:
        ax.plot(hidden_dims, results[name]['throughput'], marker='o', label=name)
    ax.plot(hidden_dims, results['PyTorch']['throughput'], marker='s', label='PyTorch', linestyle='--')
    ax.set_xlabel('Hidden Dimension (C)')
    ax.set_ylabel('Throughput (GFLOPS)')
    ax.set_title('Throughput vs Hidden Dimension')
    ax.set_xscale('log', base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    
    ax = axes[1, 0]
    for name, _, _ in kernel_configs:
        ax.plot(batch_sizes, results[name]['speedup'], marker='o', label=name)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='PyTorch baseline')
    ax.set_xlabel('Batch Size (N)')
    ax.set_ylabel('Speedup vs PyTorch')
    ax.set_title('Speedup vs Batch Size')
    ax.set_xscale('log', base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    
    ax = axes[1, 1]
    for name, _, _ in kernel_configs:
        ax.plot(hidden_dims, results[name]['speedup'], marker='o', label=name)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='PyTorch baseline')
    ax.set_xlabel('Hidden Dimension (C)')
    ax.set_ylabel('Speedup vs PyTorch')
    ax.set_title('Speedup vs Hidden Dimension')
    ax.set_xscale('log', base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('layernorm_performance.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Performance plots saved to layernorm_performance.png")

if __name__ == "__main__":
    main()
