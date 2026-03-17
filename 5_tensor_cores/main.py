"""
Tensor Core GEMM Benchmarks
Chapter 5: Tensor Core Programming

Kernels:
- K7:  cuBLAS with Tensor Cores (baseline)
- K8:  WMMA (Warp Matrix Multiply Accumulate)
- K9:  WGMMA Basic (Warp Group MMA)
- K10: WGMMA Larger Tiles
- K11: WGMMA Async Loads
- K12: WGMMA Max Tiles
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np

os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a'

from torch.utils.cpp_extension import load

print("=" * 80)
print("Chapter 5: Tensor Core GEMM Benchmarks")
print("=" * 80)

print("\n[1/3] Compiling tensor core kernels...")
kernels = load(
    name='tensor_core_gemm',
    sources=['wrapper.cpp', 'kernels/all_kernels.cu'],
    extra_cuda_cflags=['-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '-O3'],
    extra_ldflags=['-lcublas', '-lcuda'],
    verbose=False,
)
print("✓ Compilation complete")

M, N, K = 4096, 4096, 4096
NUM_WARMUP = 5
NUM_ITERS = 20
TorchDevice = torch::device('cuda')

print(f"\n[2/3] Preparing inputs (M=N=K={M})...")
A = torch.randn(M, K, device=TorchDevice, dtype=torch.float16)
B = torch.randn(K, N, device=TorchDevice, dtype=torch.float16)

C = torch.zeros(M, N, device=TorchDevice, dtype=torch.float16)
rowmajor_ptrs = (A.data_ptr(), B.data_ptr(), C.data_ptr(), 0)

B_col = B.transpose(0, 1).contiguous()
C_col = torch.zeros(N, M, device=TorchDevice, dtype=torch.float16)
colmajor_ptrs = (A.data_ptr(), B_col.data_ptr(), C_col.data_ptr(), 0)

print("✓ Inputs ready")

KERNELS = [
    ("kernel_7_raw", kernels.kernel_7_raw, rowmajor_ptrs, False, "cuBLAS + Tensor Cores"),
    ("kernel_8_raw", kernels.kernel_8_raw, rowmajor_ptrs, False, "WMMA"),
    ("kernel_9_raw", kernels.kernel_9_raw, colmajor_ptrs, True, "WGMMA Basic"),
    ("kernel_10_raw", kernels.kernel_10_raw, colmajor_ptrs, True, "WGMMA Larger Tiles"),
    ("kernel_11_raw", kernels.kernel_11_raw, colmajor_ptrs, True, "WGMMA Async Loads"),
    ("kernel_12_raw", kernels.kernel_12_raw, colmajor_ptrs, True, "WGMMA Max Tiles"),
]

def benchmark_kernel(func, args, uses_col_layout=False):
    if uses_col_layout:
        C_col.zero_()
    else:
        C.zero_()

    for _ in range(NUM_WARMUP):
        func(M, N, K, *args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    if uses_col_layout:
        C_col.zero_()
    else:
        C.zero_()

    start.record()
    for _ in range(NUM_ITERS):
        func(M, N, K, *args)
    end.record()
    torch.cuda.synchronize()

    elapsed = start.elapsed_time(end) / NUM_ITERS
    tflops = (2.0 * M * N * K) / (elapsed * 1e9)
    return elapsed, tflops

print("\n[3/3] Benchmarking tensor core kernels...")
results = []
for name, func, ptrs, is_col_layout, desc in KERNELS:
    layout = 'col-major' if is_col_layout else 'row-major'
    print(f"  {name} ({desc}) [{layout}] ...", end='', flush=True)
    try:
        elapsed_ms, tflops = benchmark_kernel(func, ptrs, is_col_layout)
        results.append((name, desc, elapsed_ms, tflops))
        print(f" {elapsed_ms:.3f} ms | {tflops:.1f} TFLOPS")
    except Exception as err:
        results.append((name, desc, float('nan'), 0.0))
        print(f" ERROR ({err})")

print("\n" + "=" * 80)
print(f"Tensor Core Performance (M=N=K={M})")
print("=" * 80)
print(f"{'Kernel':<25}{'Description':<30}{'Time (ms)':>12}{'TFLOPS':>12}")
print("-" * 80)
for name, desc, elapsed_ms, tflops in results:
    if torch.isnan(torch.tensor(elapsed_ms)):
        print(f"{name:<25}{desc:<30}{'ERROR':>12}{'ERROR':>12}")
    else:
        print(f"{name:<25}{desc:<30}{elapsed_ms:>12.3f}{tflops:>12.1f}")

print("-" * 80)

print("\n" + "=" * 80)
print("Generating performance visualization...")

fig, ax = plt.subplots(1, 1, figsize=(14, 8))
fig.suptitle(f'Tensor Core GEMM Performance (M=N=K={M})', fontsize=20, fontweight='bold')

colors = {
    'kernel_7_raw': 'black',      
    'kernel_8_raw': '
    'kernel_9_raw': '
    'kernel_10_raw': '
    'kernel_11_raw': '
    'kernel_12_raw': '
}

kernel_names = [r[1] for r in results]  
kernel_tflops = [r[3] for r in results]
kernel_ids = [r[0] for r in results]

ax.set_title('Tensor Core Throughput', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Kernel', fontsize=14)
ax.set_ylabel('TFLOPS (Higher is Better)', fontsize=14)
ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.8)

bars = ax.bar(range(len(kernel_names)), kernel_tflops,
               color=[colors.get(k, 'gray') for k in kernel_ids],
               edgecolor='black', linewidth=1.2)
ax.set_xticks(range(len(kernel_names)))
ax.set_xticklabels([f"K{7+i}" for i in range(len(kernel_names))], rotation=0, fontsize=12)

for i, (bar, val) in enumerate(zip(bars, kernel_tflops)):
    if val > 0 and not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width()/2, val + 15, f'{val:.0f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()

output_file = os.path.join(os.path.dirname(__file__), 'tensor_core_performance.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"✓ Performance plot saved to: {output_file}")

print("\n" + "=" * 80)
print("Analysis Complete!")
if len(kernel_tflops) >= 6 and kernel_tflops[5] > 0:
    print(f"Best Kernel: K12 (WGMMA Max Tiles) = {kernel_tflops[5]:.1f} TFLOPS")
    print(f"cuBLAS Baseline: K7 = {kernel_tflops[0]:.1f} TFLOPS")
    print(f"WMMA Performance: K8 = {kernel_tflops[1]:.1f} TFLOPS")
    print(f"Efficiency vs cuBLAS: {(kernel_tflops[5]/kernel_tflops[0]*100):.1f}%")
print("=" * 80)

