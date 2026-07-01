#include <torch/extension.h>
#include "all_kernels.cuh"
#include "0_cublas.cuh"
#include "1_naive.cuh"
#include "2_gmem_coalesce.cuh"
#include "3_smem_blocking.cuh"
#include "4_1d_blocktiling.cuh"
#include "5_2d_blocktiling.cuh"
#include "6_vectorize.cuh"


using at_fp16 = c10::Half;

void runCublasGemmFP16(int M, int N, int K, at_fp16 *A, at_fp16 *B, at_fp16 *C) {
  static cublasHandle_t cublas_handle;
  static bool cublas_initialized = false;
  
  if (!cublas_initialized) {
    cublasCreate(&cublas_handle);
    cublas_initialized = true;
  }
  
  float alpha = 1.0f, beta = 0.0f;
  
  
  
  cublasStatus_t status = cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, 
    reinterpret_cast<fp16*>(B), CUDA_R_16F, N,  
    reinterpret_cast<fp16*>(A), CUDA_R_16F, K,  
    &beta, reinterpret_cast<fp16*>(C), CUDA_R_16F, N,  
    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("CUBLAS error: %d\n", status);
  }
}

void runKernel1(int M, int N, int K, at_fp16 *A, at_fp16 *B, at_fp16 *C) {
  dim3 blockDim(32, 32);
  dim3 gridDim((M + 31) / 32, (N + 31) / 32);
  gemm_naive<<<gridDim, blockDim>>>(M, N, K, 
    reinterpret_cast<fp16*>(A), reinterpret_cast<fp16*>(B), reinterpret_cast<fp16*>(C));
}

void runKernel2(int M, int N, int K, at_fp16 *A, at_fp16 *B, at_fp16 *C) {
  const uint BLOCKSIZE = 32;
  dim3 blockDim(BLOCKSIZE * BLOCKSIZE);
  dim3 gridDim((M + BLOCKSIZE - 1) / BLOCKSIZE, (N + BLOCKSIZE - 1) / BLOCKSIZE);
  gemm_gmem_coalesce<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, 
    reinterpret_cast<fp16*>(A), reinterpret_cast<fp16*>(B), reinterpret_cast<fp16*>(C));
}

void runKernel3(int M, int N, int K, at_fp16 *A, at_fp16 *B, at_fp16 *C, int *DB) {
  const uint BLOCKSIZE = 32;
  dim3 blockDim(BLOCKSIZE * BLOCKSIZE);
  dim3 gridDim((M + BLOCKSIZE - 1) / BLOCKSIZE, (N + BLOCKSIZE - 1) / BLOCKSIZE);
  gemm_smem_blocking<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, 
    reinterpret_cast<fp16*>(A), reinterpret_cast<fp16*>(B), reinterpret_cast<fp16*>(C));
}

void runKernel4(int M, int N, int K, at_fp16 *A, at_fp16 *B, at_fp16 *C, int *DB) {
  const uint BK = 8;
  const uint TM = 8;
  const uint BM = 64;
  const uint BN = 64;
  dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 blockDim((BM * BN) / TM);
  gemm_1d_blocktiling<BM, BN, BK, TM><<<gridDim, blockDim>>>(M, N, K, 
    reinterpret_cast<fp16*>(A), reinterpret_cast<fp16*>(B), reinterpret_cast<fp16*>(C));
}

void runKernel5(int M, int N, int K, at_fp16 *A, at_fp16 *B, at_fp16 *C, int *DB) {
  const uint BM = 64;
  const uint BN = 64;
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 blockDim((BM * BN) / (TM * TN));
  gemm_2d_blocktiling<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, 
    reinterpret_cast<fp16*>(A), reinterpret_cast<fp16*>(B), reinterpret_cast<fp16*>(C));
}

void runKernel6(int M, int N, int K, at_fp16 *A, at_fp16 *B, at_fp16 *C, int *DB) {
  const uint BM = 128;
  const uint BN = 128;
  const uint BK = 16;
  const uint TM = 8;
  const uint TN = 8;
  dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 blockDim((BM * BN) / (TM * TN));
  gemm_vectorize<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, 
    reinterpret_cast<fp16*>(A), reinterpret_cast<fp16*>(B), reinterpret_cast<fp16*>(C));
}
