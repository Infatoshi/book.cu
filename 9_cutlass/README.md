# CUTLASS Examples

CUTLASS-based GEMM examples from **Chapter 11** of the book. These build on top
of NVIDIA's [CUTLASS](https://github.com/NVIDIA/cutlass) library, which the
provided build scripts clone automatically.

## Subdirectories

| Directory | Description | GPU required |
|-----------|-------------|--------------|
| `unofficial/` | Hand-written single- and multi-GPU GEMM using the CUTLASS 3.x API (`single_gpu_gemm.cu`, `multi_gpu_gemm.cu`) | Hopper (`sm_90a`, e.g. H100) |
| `official/` | NVIDIA's official CUTLASS example kernels for reference | Hopper (`sm_90a`) |
| `fp4/` | Blackwell nvFP4 GEMM, isolating CUTLASS example 72a | Blackwell (`sm_100a`, e.g. B200) |

## Hardware note

These examples target specific architectures and will not run on older or
mismatched GPUs. The `unofficial/` and `official/` examples require Hopper
(`sm_90a`); the `fp4/` example requires Blackwell (`sm_100a`). No single GPU runs
all three, so treat the chapter as a hardware-specific tour rather than a set you
run end to end on one machine.

## Building

Each subdirectory contains its own scripts (`build.sh`, `single_gpu.sh`,
`multi_gpu.sh`) or `Makefile`. Start there, for example:

```bash
cd fp4
./build.sh        # clones CUTLASS and builds example 72a for Blackwell
./verify.sh       # correctness check
./benchmark.sh    # performance benchmark
```
