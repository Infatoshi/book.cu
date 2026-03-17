/**
 * @file all_kernels.cuh
 * @brief Header file declaring all GEMM kernel functions
 * 
 * This header provides declarations for all kernel implementations:
 * - Kernel 7: cuBLAS with Tensor Cores
 * - Kernel 8: WMMA (Volta/Turing Tensor Cores)
 * - Kernel 9: WGMMA Basic (Hopper)
 * - Kernel 10: WGMMA Larger Tiles
 * - Kernel 11: WGMMA Async Loads
 * - Kernel 12: WGMMA Max Tiles
 * 
 * Also includes validation utilities and PyTorch integration functions.
 */

typedef __half fp16;

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(1);
  }
}


