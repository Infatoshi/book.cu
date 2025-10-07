"""
TopK kernel benchmarking across different sizes
"""
import torch
from torch.utils.cpp_extension import load
import os
import sys
import matplotlib.pyplot as plt

def load_kernels():
    """Load and compile CUDA kernels"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    topk_cuda = load(
        name='topk_cuda',
        sources=[
            os.path.join(current_dir, 'wrapper.cpp'),
            os.path.join(current_dir, 'kernels/0_naive.cu'),
            os.path.join(current_dir, 'kernels/1_heap.cu'),
            os.path.join(current_dir, 'kernels/2_warp.cu'),
        ],
        extra_cuda_cflags=['-O3', '--use_fast_math', '-std=c++17'],
        extra_cflags=['-O3', '-std=c++17'],
        verbose=False
    )
    return topk_cuda

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

def verify_correctness(cuda_vals, cuda_inds, torch_vals, torch_inds):
    """Verify numerical correctness"""
    cuda_set = set(zip(cuda_vals.cpu().numpy(), cuda_inds.cpu().numpy()))
    torch_set = set(zip(torch_vals.cpu().numpy(), torch_inds.cpu().numpy()))
    
    if cuda_set == torch_set:
        return True, "Exact match"
    else:
        diff = len(torch_set.symmetric_difference(cuda_set))
        return False, f"{diff} mismatches"

def main():
    print("="*80)
    print("TopK Kernel Benchmarking - Size Sweep")
    print("="*80)
    
    
    
    configs = [
        (256, 8),
        (512, 16),
        (1024, 32),
        (2048, 64),
        (4096, 128),
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
        ("Kernel 1", "Heap", kernels.kernel_1),
        ("Kernel 2", "Warp", kernels.kernel_2),
    ]
    
    
    results = {name: {'configs': [], 'latency': [], 'speedup': []} 
               for name, _, _ in kernel_configs}
    results['PyTorch'] = {'configs': [], 'latency': [], 'speedup': []}
    
    
    print("Correctness Check (N=128, K=8):")
    N, K = 128, 8
    torch.manual_seed(42)
    inp = torch.randn(N, device='cuda', dtype=torch.float32)
    torch_vals, torch_inds = torch.topk(inp, K, largest=True)
    
    all_pass = True
    for name, desc, kernel_func in kernel_configs:
        cuda_vals, cuda_inds = kernel_func(inp, K)
        passed, msg = verify_correctness(cuda_vals, cuda_inds, torch_vals, torch_inds)
        if passed:
            print(f"  ✓ {name}: PASS ({msg})")
        else:
            print(f"  ✗ {name}: FAIL ({msg})")
            all_pass = False
    
    if not all_pass:
        print("\n❌ Some kernels failed correctness checks. Exiting.")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("Performance Benchmarking")
    print("="*80)
    
    for N, K in configs:
        print(f"\nConfig: N={N}, K={K}")
        
        torch.manual_seed(42)
        inp = torch.randn(N, device='cuda', dtype=torch.float32)
        
        
        (torch_vals, torch_inds), torch_time = benchmark_kernel(lambda: torch.topk(inp, K, largest=True))
        
        results['PyTorch']['configs'].append(f"N={N},K={K}")
        results['PyTorch']['latency'].append(torch_time)
        results['PyTorch']['speedup'].append(1.0)
        
        print(f"  PyTorch: {torch_time:.3f} ms")
        
        
        for name, desc, kernel_func in kernel_configs:
            _, cuda_time = benchmark_kernel(kernel_func, inp, K)
            speedup = torch_time / cuda_time
            
            results[name]['configs'].append(f"N={N},K={K}")
            results[name]['latency'].append(cuda_time)
            results[name]['speedup'].append(speedup)
            
            print(f"  {name}: {cuda_time:.3f} ms, {speedup:.2f}x")
    
    
    print("\n" + "="*80)
    print("LATENCY SUMMARY (ms)")
    print("="*80)
    print(f"{'Config':<15} {'PyTorch':<10} " + " ".join([f"{name:<10}" for name, _, _ in kernel_configs]))
    print("-"*80)
    for i, (N, K) in enumerate(configs):
        row = f"N={N},K={K:<8} {results['PyTorch']['latency'][i]:<10.3f} "
        row += " ".join([f"{results[name]['latency'][i]:<10.3f}" for name, _, _ in kernel_configs])
        print(row)
    
    print("\n" + "="*80)
    print("SPEEDUP SUMMARY (vs PyTorch)")
    print("="*80)
    print(f"{'Config':<15} " + " ".join([f"{name:<10}" for name, _, _ in kernel_configs]))
    print("-"*80)
    for i, (N, K) in enumerate(configs):
        row = f"N={N},K={K:<8} "
        row += " ".join([f"{results[name]['speedup'][i]:<10.2f}" for name, _, _ in kernel_configs])
        print(row)
    
    
    plot_results(results, configs, kernel_configs)

def plot_results(results, configs, kernel_configs):
    """Generate performance plots"""
    input_sizes = [c[0] for c in configs]
    k_values = [c[1] for c in configs]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    
    ax = axes[0, 0]
    for name, _, _ in kernel_configs:
        ax.plot(input_sizes, results[name]['latency'], marker='o', label=name)
    ax.plot(input_sizes, results['PyTorch']['latency'], marker='s', label='PyTorch', linestyle='--')
    ax.set_xlabel('Input Size (N)')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Latency vs Input Size')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    
    ax = axes[0, 1]
    for name, _, _ in kernel_configs:
        ax.plot(k_values, results[name]['latency'], marker='o', label=name)
    ax.plot(k_values, results['PyTorch']['latency'], marker='s', label='PyTorch', linestyle='--')
    ax.set_xlabel('K Value')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Latency vs K Value')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    
    ax = axes[1, 0]
    for name, _, _ in kernel_configs:
        ax.plot(input_sizes, results[name]['speedup'], marker='o', label=name)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='PyTorch baseline')
    ax.set_xlabel('Input Size (N)')
    ax.set_ylabel('Speedup vs PyTorch')
    ax.set_title('Speedup vs Input Size')
    ax.set_xscale('log', base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    
    ax = axes[1, 1]
    for name, _, _ in kernel_configs:
        ax.plot(k_values, results[name]['speedup'], marker='o', label=name)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='PyTorch baseline')
    ax.set_xlabel('K Value')
    ax.set_ylabel('Speedup vs PyTorch')
    ax.set_title('Speedup vs K Value')
    ax.set_xscale('log', base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('topk_performance.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Performance plots saved to topk_performance.png")

if __name__ == "__main__":
    main()
