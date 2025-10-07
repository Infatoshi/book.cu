# nvFP4 GEMM - Educational Example

This project demonstrates NVIDIA's FP4 precision GEMM operations using CUTLASS on the Blackwell architecture (SM100/B200 GPU). It isolates and benchmarks the kernel 72a from CUTLASS examples.

## Quick Start

### Three Simple Steps

#### Step 1: Build
```bash
./build.sh
```
This clones CUTLASS and builds the kernel 72a example for SM100 architecture.

#### Step 2: Verify (1024x1024 matrices)
```bash
./verify.sh
```
This runs a correctness check comparing GPU results against CPU reference on 1024x1024 matrices.

#### Step 3: Benchmark (8192x8192, batch=8)
```bash
./benchmark.sh
```
This runs the performance benchmark and reports PFLOPS/TFLOPS.

## Overview

- **Precision**: FP4 inputs (A, B matrices) with BF16 outputs (C, D matrices)
- **Architecture**: SM100 (Blackwell - B200 GPU)
- **Kernel**: CUTLASS example 72a with block-scaled tensor operations
- **Purpose**: Educational demonstration of low-precision matrix multiplication

## Project Structure

```
fp4/
├── build.sh           # Clones CUTLASS and builds kernel 72a
├── verify.sh          # Verifies correctness on 1024x1024 matrices
├── benchmark.cu       # Isolated kernel implementation for benchmarking
├── benchmark.sh       # Builds and runs benchmark on 8192x8192 matrices
├── README.md          # This comprehensive documentation
└── cutlass/          # CUTLASS library (created by build.sh)
```

## Detailed Workflow

### Step 1: BUILD
**Command:** `./build.sh`

**Actions:**
- Clone CUTLASS repository (if not present)
- Configure CMake with `-DCUTLASS_NVCC_ARCHS=100a`
- Build example 72a: `72a_blackwell_nvfp4_bf16_gemm`

**Output:**
```
cutlass/build/examples/72_blackwell_narrow_precision_gemm/
                         72a_blackwell_nvfp4_bf16_gemm (binary)
```

### Step 2: VERIFY
**Command:** `./verify.sh`

**Configuration:**
- Matrix Size: 1024 × 1024 × 1024
- Batch: 1

**Process:**
1. Initialize random FP4 matrices A, B
2. Initialize random BF16 matrix C
3. Generate scale factors (block-scaled)
4. Run GPU kernel: D = A × B + C
5. Run CPU reference computation
6. Compare results with tolerance

**Success Criteria:**
- ✓ GPU and CPU outputs match within FP4 tolerance
- ✓ Output tensors have non-zero norm

**Expected Output:**
```
Disposition: Passed ✓
```

### Step 3: BENCHMARK
**Command:** `./benchmark.sh`

**Build Phase:**
- Compile benchmark.cu with nvcc
- Target architecture: `-arch=sm_100a`
- Include CUTLASS headers
- Enable SM100 support: `-DCUTLASS_ARCH_MMA_SM100_SUPPORTED`

**Benchmark Configuration:**
- Matrix Size: 8192 × 8192 × 8192
- Batch: 8 (8x scaling)
- Warmup: 5 iterations
- Timing: 20 iterations

**Benchmark Phases:**
1. Allocate and initialize tensors on HOST
2. Transfer data to GPU (H2D) ⚠ **NOT TIMED**
3. Setup CUTLASS kernel arguments
4. Run warmup iterations ⚠ **NOT TIMED**
5. ⏱️ **START TIMING** ⏱️
   - Record CUDA event (start)
   - Execute kernel 20 times
   - Record CUDA event (stop)
   - Synchronize and compute elapsed time
6. ⏱️ **STOP TIMING** ⏱️
7. Calculate performance metrics

**Performance Calculation:**
```
FLOPs = 2 × M × N × K × Batch
     = 2 × 8192 × 8192 × 8192 × 8
     = ~8.8 PetaFLOPs per run

PFLOPS = FLOPs / (time_ms / 1000.0) / 1e15
```

**Expected Output:**
```
=== nvFP4 GEMM Benchmark ===
Problem size: 8192 x 8192 x 8192 (batch=8)
Architecture: SM100 (Blackwell)
Precision: FP4 (A, B) x BF16 (C, D)

=== Results ===
Average kernel time: X.XX ms
Performance: X.XX PFLOPS
Performance: XXX.XX TFLOPS

✓ Benchmark completed successfully
```

## Requirements

- **CUDA**: 12.8 or newer
- **GPU**: SM100, SM101, or SM103 (Blackwell architecture - B200 GPU)
- **Compiler**: nvcc with C++17 support
- **CMake**: 3.18 or newer
- **System**: Linux (tested on Ubuntu)

## Key Features

### Accurate Timing Methodology

The benchmark measures **only kernel execution time**, not data movement:

**What IS timed:**
- ✓ Kernel execution only
- ✓ Pure computational performance
- ✓ Time from kernel launch to completion

**What is NOT timed:**
- ✗ Host-to-Device (H2D) data transfers
- ✗ Device-to-Host (D2H) data transfers
- ✗ Memory allocation
- ✗ Kernel initialization
- ✗ Warmup iterations

This ensures the benchmark measures raw computational throughput, not system overhead or data movement bottlenecks.

### FP4 Precision (E2M1 Format)

FP4 uses the E2M1 format:
- 1 sign bit
- 2 exponent bits
- 1 mantissa bit

This extremely low precision format:
- Reduces memory bandwidth requirements
- Increases computational throughput (4x vs FP8, 8x vs FP16)
- Requires careful scaling to maintain accuracy (block-scaled operations)

### SM100 Architecture Features

The Blackwell SM100 architecture introduces:
- **Block-Scaled Tensor Cores**: Native support for FP4 with per-block scaling
- **Tensor Memory (TMEM)**: Per-SM memory for improved data locality
- **2x throughput** compared to FP8 operations
- **4x throughput** compared to FP8 on Hopper

**SM100 Features Used:**
- `tcgen05.mma.blockscaled` instructions (native FP4 support)
- Tensor Memory (TMEM) for efficient data staging
- Warp-specialized execution pattern
- Dynamic cluster scheduling

### Kernel Configuration (from 72a)

- **MMA Tile**: 256 × 256 × 256
- **Cluster**: 2 × 4 × 1 thread blocks
- **Input A**: FP4 (row major) + scale factors
- **Input B**: FP4 (column major) + scale factors
- **Output C/D**: BF16 (row major)
- **Accumulator**: FP32
- **Operation**: D = alpha * (A @ B) + beta * C

## Performance Expectations

On a B200 GPU, you should expect:
- **Several PFLOPS** for large matrix operations
- Performance scales with matrix size and batch size
- Block-scaled FP4 provides 2x speedup over FP8 operations

**Recent Results:**
- **Average kernel time**: 1.93 ms
- **Performance**: 4.55 PFLOPS (4553 TFLOPS)
- **Problem size**: 8192 × 8192 × 8192 (batch=8)

## Customization

You can modify the benchmark parameters by editing `benchmark.sh`:

```bash
M=8192        # Number of rows in A
N=8192        # Number of columns in B
K=8192        # Shared dimension
BATCH=8       # Batch size
WARMUP=5      # Warmup iterations
ITERS=20      # Timing iterations
```

Or pass them directly to the compiled benchmark:

```bash
./benchmark --m=4096 --n=4096 --k=4096 --batch=16 --warmup=10 --iters=50
```


## Educational Notes

### Why FP4?

Modern AI workloads, especially inference and some training tasks, can tolerate lower precision:
- **Memory-bound workloads** benefit from reduced data movement
- **Compute-bound workloads** benefit from increased throughput
- **Quantization** techniques maintain accuracy despite low precision

### Block Scaling

The kernel uses block-scaled operations where:
- Matrices are divided into blocks
- Each block has a scaling factor (higher precision)
- Operations are performed in low precision with scaling applied
- This maintains numerical stability

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "CUDA 12.8 required" | Update CUDA Toolkit |
| "SM100 required" | Need B200 GPU |
| Build fails | Check CMake version (≥3.18) |
| Verify fails | Expected with FP4 - check tolerance |
| Compilation errors | Ensure CUTLASS is properly built: `rm -rf cutlass && ./build.sh` |
| Performance lower than expected | Check GPU is in performance mode, no other processes using GPU |

### Common Issues

**"This example requires CUDA 12.8 or newer"**
Update your CUDA Toolkit to version 12.8 or later.

**"This benchmark requires SM100"**
This code is specifically for Blackwell GPUs (B200). It will not run on older architectures.

**Compilation errors**
Ensure CUTLASS is properly built:
```bash
rm -rf cutlass
./build.sh
```

**Performance lower than expected**
- Check GPU is in performance mode (not throttled)
- Ensure no other processes are using the GPU
- Verify you're using the correct architecture flag (`-arch=sm_100a`)

## References

- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
- [CUDA 12.8 Documentation](https://docs.nvidia.com/cuda/)
- [PTX ISA: Block-Scaled Operations](https://docs.nvidia.com/cuda/parallel-thread-execution)
- [Blackwell Architecture Whitepaper](https://www.nvidia.com/en-us/data-center/blackwell-architecture/)

## License

This example follows the CUTLASS BSD-3-Clause license. See the CUTLASS repository for details.

