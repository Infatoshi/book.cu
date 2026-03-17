# Chapter 8: Quantization - Individual CUDA Examples

This chapter demonstrates **9 fundamental quantization concepts** through standalone CUDA programs. Each `.cu` file is a complete, self-contained example showing one quantization technique with accuracy validation.

## Structure

```
08/
├── quant/                      # Individual quantization examples
│   ├── fp32_int8.cu            # Basic FP32 -> INT8 quantization
│   ├── fp32_int4.cu            # FP32 -> INT4 with packing
│   ├── dynamic_static.cu       # Runtime vs pre-computed scales
│   ├── int8_uint8.cu           # Symmetric vs asymmetric quantization
│   ├── calibrator.cu           # Min-max vs percentile calibration
│   ├── tensorwise.cu           # Single scale for entire tensor
│   ├── groupwise.cu            # Different scales for groups
│   ├── blockwise.cu            # Different scales for blocks (spatial)
│   └── channelwise.cu          # Different scales per channel/feature
├── assets/                     # Diagrams and figures
├── Makefile                    # Build automation
└── README.md                   # This file
```

**Ultra-minimal:** Each example is a single `.cu` file with inline kernels using C primitives - no C++ standard library bloat, maximum simplicity.

## Minimal NVCC Commands

Each example is a single `.cu` file. Compile and run with these minimal commands:

```bash
cd /mnt/storage/cuda-book/08

nvcc -O3 -o fp32_int8 quant/fp32_int8.cu && ./fp32_int8
nvcc -O3 -o fp32_int4 quant/fp32_int4.cu && ./fp32_int4
nvcc -O3 -o dynamic_static quant/dynamic_static.cu && ./dynamic_static
nvcc -O3 -o int8_uint8 quant/int8_uint8.cu && ./int8_uint8
nvcc -O3 -o calibrator quant/calibrator.cu && ./calibrator
nvcc -O3 -o tensorwise quant/tensorwise.cu && ./tensorwise
nvcc -O3 -o groupwise quant/groupwise.cu && ./groupwise
nvcc -O3 -o blockwise quant/blockwise.cu && ./blockwise
nvcc -O3 -o channelwise quant/channelwise.cu && ./channelwise
```

Each example generates test data, quantizes/dequantizes, and reports accuracy metrics.

## The 9 Fundamental Quantization Concepts

### Data Type Conversions
- **fp32_int8.cu**: Basic FP32→INT8 symmetric quantization
- **fp32_int4.cu**: FP32→INT4 with 2:1 packing for 8x compression

### Quantization Strategies
- **dynamic_static.cu**: Dynamic (per-batch) vs static (pre-computed) scales
- **int8_uint8.cu**: INT8 symmetric vs UINT8 asymmetric quantization

### Scale Computation
- **calibrator.cu**: Min-max vs percentile-based calibration

### Granularity Schemes
- **tensorwise.cu**: Single scale for entire tensor (simplest)
- **groupwise.cu**: Different scales for groups of elements
- **blockwise.cu**: Different scales for spatial blocks (images/features)
- **channelwise.cu**: Different scales per channel/feature dimension

## Key Insights Each Example Demonstrates

- **Accuracy vs compression trade-offs**
- **Memory savings (4x INT8, 8x INT4)**
- **When different schemes work best**
- **Calibration sensitivity**
- **Per-dimension quantization strategies**

Perfect for understanding quantization fundamentals before applying to complex architectures!

## Key Concepts Demonstrated

### 1. Basic Quant/Dequant (FP32 ↔ INT8/INT4)
- Symmetric quantization with scale computation
- Memory compression ratios (4x for INT8, 8x for INT4)
- Quantization error analysis

### 2. Quantization Schemes
- **Tensor-wise**: Simplest, single scale for entire tensor
- **Group-wise**: Multiple scales, balances accuracy vs overhead
- **Per-channel**: Different scale per feature/channel dimension
- **Linear Projection**: Quantizing matrix multiplication operations
- **Weight Matrix**: Different strategies for quantizing weight matrices

### 3. Calibration
- Computing quantization parameters from representative data
- Symmetric vs asymmetric calibration approaches
- Impact of calibration quality on final accuracy

### 4. Dynamic vs Static Quantization
- **Static**: Pre-computed scales (faster inference)
- **Dynamic**: Runtime scale computation (more accurate)

### 5. Symmetric vs Asymmetric Quantization
- **Symmetric**: Zero-point = 0, simpler but less efficient
- **Asymmetric**: Zero-point ≠ 0, uses full range but more complex

### 6. Signed vs Unsigned Quantization
- **Signed (INT8)**: Better for weights and zero-mean data
- **Unsigned (UINT8)**: Better for activations and asymmetric data

## The Quantization Mindset

This chapter shows that quantization is about:
- **Memory efficiency**: Reducing precision to fit more in GPU memory
- **Accuracy trade-offs**: Understanding when quantization noise matters
- **Adaptation**: Choosing the right scheme for your data distribution
- **Calibration**: Getting the scales right is half the battle

Each example is designed to be minimal and educational - showing exactly what happens at the kernel level when you quantize data. The matplotlib visualizations help build intuition about quantization effects.

## Performance Insights

- **INT8**: ~4x memory reduction with minimal accuracy loss
- **INT4**: **8x memory reduction** - this creates the "free lunch" feeling
- **Per-channel**: Critical for feature dimensions with different ranges
- **Linear projections**: Shows how to quantize the core ML operation (matmul)
- **Weight matrices**: Per-output quantization often best for learned parameters
- **Calibration**: Poor calibration can hurt accuracy more than the quantization itself
- **Dynamic**: More accurate but slower - use for critical paths only

## Fundamental Tensor Operations

### 3D Tensors (B × T × C)
The canonical tensor shape for demonstrating quantization concepts:
- **B**: Batch dimension (different examples/sequences)
- **T**: Sequence/time dimension (steps in a sequence)
- **C**: Channel/feature dimension (different learned features)

### Linear Projections (B × T × C) @ (C × H)
The fundamental ML operation that gets quantized:
- Input activations: `[B, T, C]` (3D tensor)
- Weight matrix: `[C, H]` (2D projection)
- Output: `[B, T, H]` (transformed features)

### Weight Matrix Granularities
- **Full matrix**: Simple, one scale for entire weight matrix
- **Per-output**: Different scale per output feature (recommended)
- **Per-input**: Different scale per input feature (alternative)

## Reader Takeaways

By the end of this chapter, readers should understand:
1. How to implement basic quant/dequant operations
2. When to choose different quantization schemes
3. The importance of proper calibration
4. Trade-offs between accuracy, speed, and memory usage
5. How to adapt quantization to different model architectures

The examples encourage experimentation: "Here are the building blocks - now go build your own quantization schemes!"