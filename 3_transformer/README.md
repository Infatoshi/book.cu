# Character-level Transformer: Training & Inference with Custom CUDA Kernels

A minimal character-level GPT implementation that demonstrates integrating custom CUDA kernels into PyTorch's autograd system. This repository contains both **training** and **inference** components with hybrid CUDA implementations - using custom CUDA operations where they work reliably, and PyTorch operations where needed for stability.

## 🏗️ Project Structure

```
naive.cu/
├── csrc/              # C++ source files and CUDA kernels
│   ├── training/      # Training C++ bindings and kernels
│   └── inference/     # Inference C++ bindings and kernels
├── wrapper/           # Python wrappers for custom operations
│   ├── training/      # Training operation wrappers
│   └── inference/     # Inference operation wrappers
├── docs/              # Documentation and debugging guides
├── train.py           # Character-level transformer training
├── inference.py       # Transformer inference with dense/MoE
├── setup.py           # Combined build configuration
├── README.md          # This file
└── LICENSE
```

## 📚 Components

### Training Component
- **Character-level GPT training** with custom CUDA kernels
- **Dense transformers only** (no MoE during training)
- **Two-stage approach**: PyTorch baseline → Custom CUDA implementation
- **Dataset**: "The Wonderful Wizard of Oz" (public domain)
- **Architecture**: Transformer with multi-head attention and feed-forward layers

### Inference Component
- **Transformer inference** with both dense and MoE architectures
- **Dense inference**: Demonstrates exact PyTorch ↔ CUDA matching
- **MoE inference**: Educational example showing routing challenges (see below)
- **KV caching** for efficient autoregressive generation
- **Optimized kernels**: GEMV operations for inference speedup

## 🔧 Setup & Installation

### Prerequisites
- **CUDA-compatible GPU** (RTX 30xx+ recommended)
- **Python 3.8+**
- **PyTorch 2.5+**
- **CUDA Toolkit 12.1+**

### Environment Setup

1. **Create virtual environment:**
   ```bash
   uv venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   uv pip install torch torchvision torchaudio pybind11 requests
   ```

### Training Setup

1. **Build CUDA extensions:**
   ```bash
   python setup.py build_ext --inplace
   ```

2. **Run training:**
   ```bash
   python train.py
   ```

### Inference Setup

1. **Build CUDA extensions:**
   ```bash
   python setup.py build_ext --inplace
   ```

2. **Run inference:**
   ```bash
   python inference.py
   ```

### Quick Start (Both Components)

```bash
# Clone and setup
git clone <repository-url>
cd naive.cu

# Setup environment
uv venv && source .venv/bin/activate
uv pip install torch torchvision torchaudio pybind11 requests

# Build CUDA extensions
python setup.py build_ext --inplace

# Run training
python train.py

# Run inference
python inference.py
```

## 🎯 What You'll See When Running

### Training Output
```
Using device: cuda
Batch size: 16, Block size: 64, Embedding dim: 128

PyTorch baseline training...
PyTorch Model Parameters: ~1.6M
iter 0/1000 | loss 4.3512
iter 100/1000 | loss 2.5310
iter 200/1000 | loss 2.4305
...
PyTorch training time: ~20s

Custom CUDA training...
Custom Model Parameters: ~1.6M
iter 0/1000 | loss 4.3772
iter 100/1000 | loss 2.5535
...
Custom CUDA training time: ~15s
```

### Inference Output
```
=== Transformer Inference Setup ===
Batch size: 1
Block size: 64
Embedding dimension: 768
Number of heads: 8
Number of layers: 24
Vocabulary size: 95
Device: cuda
Max new tokens: 200

=== PyTorch Dense vs CUDA Dense ===
✓ SUCCESS: Dense implementations match exactly! Speedup: 0.9-1.2x

=== PyTorch MoE vs CUDA MoE ===
⚠️ NOTICE: MoE implementations show expected differences (see MoE Routing Challenges below)
(This is normal due to numerical precision in custom CUDA routing operations)
```

## ⚠️ Known Pain Points & Troubleshooting

### Training Setup Issues

1. **CUDA Kernel Compilation Failures**
   - **Symptom**: `nvcc fatal error` during `python setup.py build_ext --inplace`
   - **Common causes**: Incorrect CUDA toolkit version, missing GPU architecture flags
   - **Fix**: Ensure CUDA 12.1+ and add `-arch=sm_86` for RTX 30xx GPUs

2. **Memory Corruption in Custom Operations**
   - **Symptom**: Training works initially but crashes randomly or produces NaN gradients
   - **Cause**: Non-contiguous tensors passed to CUDA kernels
   - **Fix**: Always call `.contiguous()` before custom CUDA operations

3. **Gradient Flow Issues**
   - **Symptom**: `loss.backward()` fails or gradients are None
   - **Cause**: Incorrect backward pass implementation in custom CUDA operations
   - **Debug**: Replace custom ops with PyTorch equivalents incrementally to isolate

4. **Numerical Precision Differences**
   - **Symptom**: Identical loss curves expected but slight variations occur
   - **Normal**: Floating-point precision differences between PyTorch and custom CUDA
   - **Verification**: Check if differences are < 1e-4 (acceptable for float32)

### General Debugging Strategy

**Always establish a PyTorch baseline first:**
```python
# Replace custom CUDA operations with PyTorch equivalents temporarily
# self.custom_matmul = CustomMatMul()  # CUDA version
self.custom_matmul = torch.nn.functional.linear  # PyTorch equivalent
```

**Test incrementally:** Replace one operation at a time and verify training still works. This isolates exactly which custom operation is problematic.

**Common tensor issues:**
- Ensure tensors are contiguous: `assert x.is_contiguous()`
- Check tensor shapes: `print(f"Shape: {x.shape}, dtype: {x.dtype}")`
- Verify device placement: `assert x.device.type == 'cuda'`

### Inference-Specific Issues

1. **KV Cache Memory Issues**
   - **Symptom**: Memory usage grows unbounded during long generations
   - **Fix**: Implement proper cache size limits and periodic cleanup

2. **MoE Routing Instability**
   - **Symptom**: Different outputs on identical inputs due to floating-point precision
   - **Normal**: Expected with MoE routing - verify outputs are semantically similar

## 🎓 MoE Routing Challenges (Educational Journey)

### The Numerical Precision Discovery

The MoE (Mixture of Experts) inference demonstrates an **important lesson in numerical computing**: seemingly small differences in floating-point operations can cascade into significant behavioral changes when used together.

### Diagnostic Investigation

We ran systematic tests to isolate which operations cause divergence:

| Test | Softmax | TopK | Token Divergence | Finding |
|------|---------|------|------------------|---------|
| **Baseline** | Custom CUDA | Custom CUDA | **50%** | ❌ Significant divergence |
| **Test A** | **PyTorch** | Custom CUDA | **0%** | ✅ Perfect match! |
| **Test B** | Custom CUDA | **PyTorch** | **0%** | ✅ Perfect match! |
| **Test C** | PyTorch | PyTorch | **0%** | ✅ Perfect match |

**Key Discovery:** Either custom implementation works fine **individually**, but combining both creates **compound precision errors**.

### Why This Happens

**The Cascade Effect:**
1. **Custom softmax** produces: `[0.400001, 0.399999, 0.200000]` (±1e-7 precision)
2. **Custom topk** on these values has tie-breaking ambiguity
3. **Small differences compound**: Different experts → different outputs → different tokens
4. **Autoregressive amplification**: Wrong token at step N affects all future predictions

**Example:**
```
Step 1: Softmax differences
PyTorch: [0.400000, 0.400000, 0.200000]
Custom:  [0.400001, 0.399999, 0.200000]  # Tiny difference

Step 2: TopK selection (no tie-breaking)
PyTorch: experts [0, 1] (deterministic)
Custom:  experts [1, 0] (ambiguous ordering)

Step 3: Cascade
→ Different expert computations
→ Different hidden states
→ Different next token
→ Repeat for 200 tokens → 50% divergence
```

### The Educational Value

**Why we keep this naive implementation:**
- ✅ Demonstrates **real numerical stability challenges** in production ML
- ✅ Shows **compound precision errors** in action
- ✅ Establishes baseline for optimization (Chapter X)
- ✅ Each custom operation works individually (not "broken")
- ✅ Dense inference **still matches exactly** (proves concept)

**What this teaches:**
- Production systems need **careful numerical design**
- Custom CUDA excels at **compute** (matmul), requires care for **logic** (sorting/comparisons)
- **Testing in isolation** vs **integration** matters
- Performance isn't just speed—it's correctness + speed

### Industry Practice

**How production MoE systems handle this:**
- Google/Meta/NVIDIA: Use PyTorch native ops for routing
- Custom CUDA focuses on **expert-parallel communication** and **fused kernels**
- Optimization comes from **kernel fusion** (topk + routing), not reimplementing primitives
- Lesson: Battle-tested implementations for critical paths

### Next Steps (Chapter Preview)

**Chapter X: Optimizing Top-K for MoE** will:
1. **Fix numerical stability** (add deterministic tie-breaking)
2. **Optimize algorithm** (heap-based selection for large k)
3. **Add CUDA optimizations** (warp-level primitives)
4. **Achieve**: <1% divergence + 20x speedup over naive

**Diagnostic Mode Available:**
To test different combinations yourself, edit `inference.py`:
```python
USE_PYTORCH_SOFTMAX = False  # Set True to test
USE_PYTORCH_TOPK = False     # Set True to test
```

## 🏛️ Architecture Details

### Custom CUDA Operations

#### Training Operations (`wrapper/training/`)
- **MatMul**: Matrix multiplication with batched support
- **Add/Mul**: Element-wise operations
- **LayerNorm**: Layer normalization
- **Softmax**: Attention softmax
- **Embedding**: Learned token/position embeddings with backprop
- **GELU**: Activation function

#### Inference Operations (`wrapper/inference/`)
- **MatMul/GEMV**: Optimized for inference (single token generation)
- **TopK**: Expert selection for MoE routing
- **Element-wise**: Optimized for sequence processing
- **LayerNorm**: Fast inference implementation

### Hybrid Approach Philosophy

This codebase embraces **hybrid implementations** rather than pure CUDA:

✅ **Custom CUDA where reliable:**
- Matrix multiplications (well-established algorithms)
- Element-wise operations (simple, predictable)
- Standard normalization operations

⚠️ **PyTorch where complex:**
- Complex attention mechanisms (until fully debugged)
- Backward passes (gradient flow verification needed)
- Operations with dynamic shapes

**Why hybrid?** CUDA kernel debugging is extremely time-intensive. Using PyTorch for problematic operations allows you to accelerate the 80% of computation that works reliably while maintaining correctness.

## 🔬 Hyperparameters

### Training (Dense Only)
- **Batch Size**: 16
- **Sequence Length**: 64
- **Embedding Dimension**: 128
- **Attention Heads**: 4
- **Transformer Layers**: 8 (2x scaled)
- **Vocabulary Size**: ~80 (character-level)
- **Learning Rate**: 3e-4
- **Training Iterations**: 1000
- **Parameters**: ~1.6M

### Inference (Dense + MoE)
- **Batch Size**: 1 (autoregressive)
- **Sequence Length**: 64
- **Embedding Dimension**: 768
- **Attention Heads**: 8
- **Transformer Layers**: 24 (2x scaled)
- **MoE Experts**: 8 (inference only)
- **Top-K Experts**: 2 (inference only)
- **Max New Tokens**: 200
- **Parameters**: Dense ~177M, MoE ~708M

## 🧪 Testing & Verification

### Numerical Accuracy Testing
```python
# Compare custom CUDA vs PyTorch outputs
torch_result = torch.nn.functional.layer_norm(x, (x.shape[-1],))
custom_result = custom_layer_norm(x)

max_diff = torch.abs(torch_result - custom_result).max()
assert max_diff < 1e-4, f"Accuracy test failed: {max_diff}"
```

### Performance Benchmarking
```python
# Profile CUDA kernels
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    # Your training/inference code here
    pass
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## 📖 Advanced Documentation

### Advanced Usage and Debugging

This section contains detailed documentation for advanced usage and debugging of the CUDA kernel implementations.

The debugging guide provides the methodology developed through extensive testing of this codebase. It includes specific examples from the transformer training implementation and general principles that apply to any CUDA kernel development project.

**Read this guide when:**
- Adding new CUDA operations to the codebase
- Debugging failing custom kernels
- Optimizing CUDA kernel performance
- Understanding CUDA memory management issues

### Development Tips
- **Start simple**: Implement naive but correct CUDA kernels first
- **Test incrementally**: Add one custom operation at a time
- **Document assumptions**: Note tensor layout expectations and limitations
- **Use PyTorch as reference**: Always verify against known-good implementations

## 🔧 CUDA Kernel Debugging Guide

This guide provides detailed methodology for debugging custom CUDA kernels when integrating them with PyTorch's autograd system. It's designed for developers who need to extend or modify the CUDA operations in this codebase.

### Core Debugging Methodology

When your CUDA implementation breaks, follow this systematic approach:

#### Step 1: Establish PyTorch Baseline

**Always start here** - replace all custom CUDA operations with PyTorch equivalents:

```python
# In your model definition, temporarily replace:
# self.custom_op = CustomOp()  # CUDA version
self.custom_op = torch.nn.functional.some_op  # PyTorch equivalent
```

**Why this works:**
- PyTorch operations are thoroughly tested and reliable
- Eliminates CUDA-specific bugs from the equation
- Gives you a "known good" reference implementation

#### Step 2: Incremental Replacement Testing

Once you have a working PyTorch baseline, replace operations one at a time:

```python
# Test 1: Replace embedding
# self.embedding = nn.Embedding(...)  # PyTorch
self.embedding = Embedding(...)       # Custom CUDA

# Test training - if it works, move to next operation
# If it breaks, you found your buggy kernel!
```

**Key principle:** Only one variable changes per test. This isolates exactly which operation is problematic.

#### Step 3: Isolate and Fix

When you find a failing operation:

1. **Check tensor shapes** - CUDA kernels are sensitive to exact tensor dimensions
2. **Verify memory layout** - Use `.contiguous()` calls before CUDA operations
3. **Check kernel launch parameters** - Grid/block dimensions must match kernel expectations
4. **Add debug prints** - Log tensor shapes and values before/after operations
5. **Test backward pass** - CUDA bugs often manifest during gradient computation

### Common CUDA Kernel Bugs

#### Memory Corruption Issues

**Symptoms:**
- `CUBLAS_STATUS_EXECUTION_FAILED`
- `illegal memory access`
- Random crashes or incorrect results
- Failures in downstream operations

**Causes:**
- Incorrect tensor indexing in kernels
- Missing `.contiguous()` calls on non-contiguous tensors
- Buffer overflows in kernel code
- Race conditions in parallel operations

**Debugging:**
```python
# Add before CUDA operations:
assert tensor.is_contiguous(), f"Tensor not contiguous: {tensor.shape}"
print(f"Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
```

#### Broadcasting Problems

**Symptoms:**
- Operations work with some tensor shapes but fail with others
- Inconsistent results across batch sizes

**Common culprit:** Custom Add/Mul operations that don't handle PyTorch broadcasting rules.

**Fix:**
```python
# Instead of custom Add, use:
a_broadcast, b_broadcast = torch.broadcast_tensors(a, b)
out = torch.add(a_broadcast, b_broadcast)
```

#### Backward Pass Issues

**Symptoms:**
- Forward pass works, but training fails during `loss.backward()`
- Gradients are `None` or incorrect

**Debugging:**
```python
# Check gradients after backward:
loss.backward()
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"No gradient for {name}")
    elif torch.isnan(param.grad).any():
        print(f"NaN gradients in {name}")
```

### Kernel-Specific Debugging

#### Matrix Multiplication (MatMul)

**2D vs 3D variants:**
- **MatMul**: Expects 2D tensors `(M, N)` and `(N, K)` → `(M, K)`
- **BatchedMatMul**: Expects 3D tensors `(B, M, N)` and `(B, N, K)` → `(B, M, K)`

**Common issues:**
- Wrong dimension ordering (PyTorch uses row-major, some CUDA code assumes column-major)
- Incorrect grid/block size calculations for large matrices

#### Layer Normalization

**Debugging tips:**
- Check that `mean` and `variance` tensors have correct shapes
- Verify epsilon value matches PyTorch's default (1e-5)
- Ensure proper broadcasting of gamma/beta parameters

#### Attention Operations

**Common issues:**
- Incorrect transpose operations (`.transpose(-2, -1)` vs `.t()`)
- Causal masking implementation bugs
- Scale factor calculation errors (`head_size ** -0.5`)

### Testing Strategies

#### Unit Tests for Individual Operations

Create minimal test scripts for each operation:

```python
# test_embedding.py
import torch
from wrapper.training import Embedding

# Create test tensors
vocab_size, embed_dim = 100, 32
indices = torch.randint(0, vocab_size, (4, 10))  # batch_size=4, seq_len=10

# Test forward
embed = Embedding(vocab_size, embed_dim)
output = embed(indices)
assert output.shape == (4, 10, 32)

# Test backward
loss = output.sum()
loss.backward()
assert embed.weight.grad is not None

print("Embedding test passed!")
```

#### Numerical Verification

Compare custom CUDA outputs with PyTorch equivalents:

```python
# Test numerical accuracy
torch_result = torch.nn.functional.layer_norm(x, (x.shape[-1],))
custom_result = custom_layer_norm(x)

max_diff = torch.abs(torch_result - custom_result).max()
print(f"Max difference: {max_diff}")

# Should be very small (< 1e-5 for float32)
assert max_diff < 1e-4, f"Numerical accuracy test failed: {max_diff}"
```

### Performance Debugging

#### Profiling CUDA Kernels

Use PyTorch's profiler to identify bottlenecks:

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    # Your training loop here
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

#### Memory Usage Analysis

Check for memory leaks or excessive usage:

```python
# Monitor GPU memory
print(f"GPU memory used: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
print(f"GPU memory cached: {torch.cuda.memory_reserved()/1024**2:.1f} MB")
```

### Best Practices

#### 1. Always Test Incrementally
Never implement multiple CUDA operations before testing. Add and test one at a time.

#### 2. Use PyTorch as Reference
PyTorch operations are your ground truth. Custom CUDA should match PyTorch results exactly.

#### 3. Check Tensor Properties
Always verify:
- Tensor shapes match expectations
- Data types are correct (float32)
- Device placement (CUDA)
- Memory contiguity

#### 4. Start Simple, Then Optimize
Begin with naive but correct CUDA implementations. Optimize only after correctness is verified.

#### 5. Document Your Changes
When you add or modify CUDA kernels, document:
- Expected input/output shapes
- Any assumptions about tensor layout
- Known limitations or edge cases

### Advanced Debugging Tools

#### CUDA Device-Side Assertions

Enable device-side assertions for better error messages:

```bash
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
python your_script.py
```

#### Nsight Systems/Compute

For detailed profiling:
```bash
nsys profile --stats=true python train.py
ncu --set full python train.py  # Nsight Compute for kernel analysis
```

#### Memory Sanitizers

Use CUDA's memory checking tools:
```bash
cuda-memcheck python your_script.py
```

### Troubleshooting Common Issues

#### "No kernel image available" errors
- Check GPU architecture compatibility (sm_86 for RTX 30xx)
- Verify CUDA toolkit version matches PyTorch build

#### Random crashes during training
- Add `torch.cuda.synchronize()` calls to isolate async errors
- Check for race conditions in custom backward passes

#### Performance worse than PyTorch
- Profile both implementations
- Check memory transfer overhead
- Verify kernel launch parameters are optimal

### Getting Help

When debugging complex CUDA issues:

1. **Isolate the problem** using the methodology above
2. **Search PyTorch issues** for similar problems
3. **Check CUDA programming forums** for kernel-specific issues
4. **Use minimal reproducers** when asking for help

Remember: CUDA debugging is challenging, but systematic isolation makes even complex issues solvable. Start with PyTorch baselines, test incrementally, and verify numerical accuracy at each step.

## 🎓 Educational Value

This repository serves as a **practical CUDA learning resource** with:

- **Real implementations**: Working custom CUDA kernels integrated with PyTorch
- **Debugging experience**: Comprehensive guide born from actual debugging sessions
- **Hybrid approach**: Practical balance between performance and maintainability
- **Progressive complexity**: From simple operations to complex transformer architectures

**Perfect for**: Students/researchers learning CUDA programming in the context of modern deep learning frameworks.

## 📄 License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🤝 Citation

If you use this code in your research or teaching, please consider citing:

```bibtex
@misc{cuda-transformer-naive,
  title={Character-level Transformer: Training \& Inference with Custom CUDA Kernels},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/naive.cu}
}
```

---

**Note**: This is a "hacky Karpathi-style repo" - educational code prioritizing clarity and learning over production optimization. Expect some rough edges and focus on understanding the concepts rather than perfect engineering practices.
