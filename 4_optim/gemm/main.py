"""
Clean benchmark: Only time the kernel execution, no allocations
Pre-allocate all tensors before timing loop
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np

os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a'

from torch.utils.cpp_extension import load

print("=" * 80)
print("Progressive Hopper kernels – kernel-only timing (raw entry points)")
print("=" * 80)

print("\n[1/3] Compiling extension...")
kernels = load(
    name='gemm_kernels_raw_bench',
    sources=['wrapper.cpp', 'kernels/all_kernels.cu'],
    extra_cuda_cflags=['-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '-O3'],
    extra_ldflags=['-lcublas', '-lcuda'],
    verbose=False,
)
print("✓ Compilation done")

M, N, K = 4096, 4096, 4096
NUM_WARMUP = 5
NUM_ITERS = 20
TorchDevice = torch.device('cuda')

print(f"\n[2/3] Preparing inputs (M=N=K={M})...")
A = torch.randn(M, K, device=TorchDevice, dtype=torch.float16)
B = torch.randn(K, N, device=TorchDevice, dtype=torch.float16)

C = torch.zeros(M, N, device=TorchDevice, dtype=torch.float16)
rowmajor_ptrs = (A.data_ptr(), B.data_ptr(), C.data_ptr())

B_col = B.transpose(0, 1).contiguous()
C_col = torch.zeros(N, M, device=TorchDevice, dtype=torch.float16)
colmajor_ptrs = (A.data_ptr(), B_col.data_ptr(), C_col.data_ptr(), 0)

print("✓ Inputs ready")

KERNELS = [
    ("kernel_0_raw", kernels.kernel_0_raw, rowmajor_ptrs, False),
    ("kernel_1_raw", kernels.kernel_1_raw, rowmajor_ptrs, False),
    ("kernel_2_raw", kernels.kernel_2_raw, rowmajor_ptrs, False),
    ("kernel_3_raw", kernels.kernel_3_raw, rowmajor_ptrs, False),
    ("kernel_4_raw", kernels.kernel_4_raw, rowmajor_ptrs, False),
    ("kernel_5_raw", kernels.kernel_5_raw, rowmajor_ptrs, False),
    ("kernel_6_raw", kernels.kernel_6_raw, rowmajor_ptrs, False),
    ("kernel_7_raw", kernels.kernel_7_raw, rowmajor_ptrs, False),
    ("kernel_8_raw", kernels.kernel_8_raw, colmajor_ptrs, True),
    ("kernel_9_raw", kernels.kernel_9_raw, colmajor_ptrs, True),
    ("kernel_10_raw", kernels.kernel_10_raw, colmajor_ptrs, True),
    ("kernel_11_raw", kernels.kernel_11_raw, colmajor_ptrs, True),
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

print("\n[3/3] Benchmarking kernels...")
results = []
for name, func, ptrs, is_col_layout in KERNELS:
    layout = 'col-major' if is_col_layout else 'row-major'
    print(f"  {name} [{layout}] ...", end='', flush=True)
    try:
        elapsed_ms, tflops = benchmark_kernel(func, ptrs, is_col_layout)
        results.append((name, elapsed_ms, tflops))
        print(f" {elapsed_ms:.3f} ms | {tflops:.1f} TFLOPS")
    except Exception as err:
        results.append((name, float('nan'), 0.0))
        print(f" ERROR ({err})")

print("\n" + "=" * 80)
print(f"Kernel-only TFLOPS (M=N=K={M})")
print("=" * 80)
print(f"{'Kernel':<20}{'Time (ms)':>12}{'TFLOPS':>12}")
print("-" * 80)
for name, elapsed_ms, tflops in results:
    if torch.isnan(torch.tensor(elapsed_ms)):
        print(f"{name:<20}{'ERROR':>12}{'ERROR':>12}")
    else:
        print(f"{name:<20}{elapsed_ms:>12.3f}{tflops:>12.1f}")

print("-" * 80)
print("Benchmark complete. All measurements exclude layout conversions and allocations.")

print("\n" + "=" * 80)
print("Generating performance visualization...")

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
fig.suptitle(f'Hopper GEMM Kernels Throughput (M=N=K={M})', fontsize=20, fontweight='bold')

colors = {
    'kernel_0_raw': 'black',
    'kernel_1_raw': '
    'kernel_2_raw': '
    'kernel_3_raw': '
    'kernel_4_raw': '
    'kernel_5_raw': '
    'kernel_6_raw': '
    'kernel_7_raw': '
    'kernel_8_raw': '
    'kernel_9_raw': '
    'kernel_10_raw': '
    'kernel_11_raw': '
}

kernel_names = [r[0] for r in results]
kernel_times = [r[1] for r in results]
kernel_tflops = [r[2] for r in results]

ax.set_title('Kernel Throughput', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Kernel', fontsize=14)
ax.set_ylabel('TFLOPS (Higher is Better)', fontsize=14)
ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.8)

bars = ax.bar(range(len(kernel_names)), kernel_tflops,
               color=[colors.get(k, 'gray') for k in kernel_names],
               edgecolor='black', linewidth=1.2)
ax.set_xticks(range(len(kernel_names)))
ax.set_xticklabels([f"K{i}" for i in range(len(kernel_names))], rotation=0, fontsize=12)

for i, (bar, val) in enumerate(zip(bars, kernel_tflops)):
    if val > 0 and not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width()/2, val + 15, f'{val:.0f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()

output_file = os.path.join(os.path.dirname(__file__), 'hopper_gemm_performance.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"✓ Performance plots saved to: {output_file}")

print("\n" + "=" * 80)
print("Analysis Complete!")
print(f"Best Custom Kernel: K11 (WGMMA Max Tiles) = {kernel_tflops[11]:.1f} TFLOPS")
print(f"cuBLAS Baseline: K0 = {kernel_tflops[0]:.1f} TFLOPS")
print(f"Efficiency: {(kernel_tflops[11]/kernel_tflops[0]*100):.1f}% of cuBLAS")
print(f"Speedup vs Naive: {(kernel_times[1]/kernel_times[11]):.0f}x")
print("=" * 80)

