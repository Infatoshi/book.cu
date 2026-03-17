# CUTLASS-Learn V2 Status

## ✅ What's Working

### Repository Structure
- **7 files total**: Drastically simplified from 34+ files
- **3 examples**: Ampere, Hopper, Multi-GPU
- **Consistent interface**: All use same Python API (tensor-based)
- **Single README**: All documentation in one place

### Code Quality
- Clean separation: 1 `.cu` file + 1 `.py` file per example
- Proper tensor interface: Takes `torch.Tensor` directly
- Correct CUTLASS paths: `/mnt/storage/cuda-book/cutlass`
- Consistent benchmarking: 3 warmup + 10 iterations

## ⚠️ Current Issue

### Hopper (SM90) Compilation Problem

**Symptom**: "Arch conditional MMA instruction used without targeting appropriate compute capability"

**Root Cause**: PyTorch's JIT compiler is not using `sm_90a` (with async suffix)

**What We've Tried**:
1. ✅ Set `TORCH_CUDA_ARCH_LIST='9.0'`
2. ✅ Added `-arch=sm_90` to CUDA flags
3. ✅ Correct CUTLASS include paths
4. ✅ Proper tensor-based interface
5. ❌ **Missing**: Need `sm_90a` not just `sm_90`

**The Problem**:
- Hopper WGMMA/TMA requires `sm_90a` architecture flag
- PyTorch's `load()` doesn't support the 'a' suffix in `TORCH_CUDA_ARCH_LIST`
- Need to either:
  - Use `90a` in gencode flags directly
  - OR simplify to Ampere-only examples

## 📊 What We Have

### Ampere (`ampere_gemm/`)
- ✅ **Should work** (SM80, standard WMMA)
- Uses old-style CUTLASS API
- Compatible with A100, RTX 3090

### Hopper (`hopper_gemm/`)  
- ❌ **Blocked** by SM90a compilation issue
- Uses new CollectiveBuilder API
- Requires H100 with special flags

### Multi-GPU (`multi_gpu_gemm/`)
- ❌ **Blocked** by same SM90a issue  
- 2-GPU data-parallel GEMM
- Uses NCCL for communication

## 🔧 Recommended Fix

### Option 1: Fix SM90a Compilation (Complex)
Manually construct the gencode flags to use `compute_90a,code=sm_90a`:
```python
extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a', ...]
```

### Option 2: Ampere-Only (Simple)
Keep only `ampere_gemm/` working example:
- Simpler, guaranteed to work
- Still demonstrates CUTLASS concepts
- Works on more hardware (A100, RTX 3090)

### Option 3: Use Old cutlass-learn
The previous `cutlass-learn/` with CMake builds DOES work for SM90:
```bash
cd /mnt/storage/cuda-book/cutlass-learn/basic_gemm/build_sm90
./basic_gemm_sm90  # This works!
```

## 📝 Next Steps

**Immediate**: Test if Ampere example works:
```bash
cd /mnt/storage/cuda-book/cutlass-learn-v2/ampere_gemm
python benchmark.py
```

**If Ampere works**: Document as educational Ampere-only repo

**If need Hopper**: Either fix SM90a flags OR use CMake-based approach from old repo

