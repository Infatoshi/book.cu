
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

void runKernel7(int M, int N, int K, at_fp16 *A, at_fp16 *B, at_fp16 *C, int *DB) {
  const int WMMA_M = 16;
  const int WMMA_N = 16;
  const int WMMA_K = 16;
  const int BM = WMMA_M * 4 * 2;
  const int BN = WMMA_N * 2 * 4;
  dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 blockDim(256);
  gemm_wmma_tiled<WMMA_M, WMMA_N, WMMA_K><<<gridDim, blockDim>>>(M, N, K,
    reinterpret_cast<fp16*>(A), reinterpret_cast<fp16*>(B), reinterpret_cast<fp16*>(C));
}

void runKernel8(int M, int N, int K, at_fp16 *A, at_fp16 *B, at_fp16 *C, int *DB) {
  wgmma_k9::run(M, N, K,
                reinterpret_cast<fp16*>(A),
                reinterpret_cast<fp16*>(B),
                reinterpret_cast<fp16*>(C));
}

void runKernel9(int M, int N, int K, at_fp16 *A, at_fp16 *B, at_fp16 *C, int *DB) {
  wgmma_k10::run(M, N, K,
                 reinterpret_cast<fp16*>(A),
                 reinterpret_cast<fp16*>(B),
                 reinterpret_cast<fp16*>(C),
                 DB);
}

void runKernel10(int M, int N, int K, at_fp16 *A, at_fp16 *B, at_fp16 *C, int *DB) {
  wgmma_k11::run(M, N, K,
                 reinterpret_cast<fp16*>(A),
                 reinterpret_cast<fp16*>(B),
                 reinterpret_cast<fp16*>(C),
                 DB);
}

void runKernel11(int M, int N, int K, at_fp16 *A, at_fp16 *B, at_fp16 *C, int *DB) {
  wgmma_k12::run(M, N, K,
                 reinterpret_cast<fp16*>(A),
                 reinterpret_cast<fp16*>(B),
                 reinterpret_cast<fp16*>(C),
                 DB);
}

void kernel_8_raw(int M, int N, int K, uint64_t A_ptr, uint64_t B_ptr, uint64_t C_ptr, uint64_t /*DB_ptr*/) {
  wgmma_k9::run_preconverted_ptrs(M, N, K, A_ptr, B_ptr, C_ptr);
}

void kernel_9_raw(int M, int N, int K, uint64_t A_ptr, uint64_t B_ptr, uint64_t C_ptr, uint64_t DB_ptr) {
  wgmma_k10::run_preconverted_ptrs(M, N, K, A_ptr, B_ptr, C_ptr, DB_ptr);
}

void kernel_10_raw(int M, int N, int K, uint64_t A_ptr, uint64_t B_ptr, uint64_t C_ptr, uint64_t DB_ptr) {
  wgmma_k11::run_preconverted_ptrs(M, N, K, A_ptr, B_ptr, C_ptr, DB_ptr);
}

void kernel_11_raw(int M, int N, int K, uint64_t A_ptr, uint64_t B_ptr, uint64_t C_ptr, uint64_t DB_ptr) {
  wgmma_k12::run_preconverted_ptrs(M, N, K, A_ptr, B_ptr, C_ptr, DB_ptr);
}
