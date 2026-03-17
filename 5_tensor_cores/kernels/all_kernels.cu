/**
 * @file all_kernels.cu
 * @brief Implementation file for kernel launcher functions
 * 
 * Provides wrapper functions that convert between different tensor types
 * (PyTorch Half vs CUDA __half) and launch the appropriate kernels.
 */

using at_fp16 = c10::Half;

void runKernel7(int M, int N, int K, at_fp16 *A, at_fp16 *B, at_fp16 *C, int *DB) {
  runCublasGemmFP16_TensorCores(M, N, K,
    reinterpret_cast<fp16*>(A),
    reinterpret_cast<fp16*>(B),
    reinterpret_cast<fp16*>(C));
}

void runKernel8(int M, int N, int K, at_fp16 *A, at_fp16 *B, at_fp16 *C, int *DB) {
  const int WMMA_M = 16;
  const int WMMA_N = 16;
  const int WMMA_K = 16;
  const int BM = WMMA_M * 4 * 2;  
  const int BN = WMMA_N * 2 * 4;  
  dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 blockDim(256);  
  gemm_wmma_tiled<WMMA_M, WMMA_N, WMMA_K><<<gridDim, blockDim>>>(M, N, K,
    reinterpret_cast<fp16*>(A),
    reinterpret_cast<fp16*>(B),
    reinterpret_cast<fp16*>(C));
}

void runKernel9(int M, int N, int K, at_fp16 *A, at_fp16 *B, at_fp16 *C, int *DB) {
  wgmma_k9::run(M, N, K,
                reinterpret_cast<fp16*>(A),
                reinterpret_cast<fp16*>(B),
                reinterpret_cast<fp16*>(C));
}

void runKernel10(int M, int N, int K, at_fp16 *A, at_fp16 *B, at_fp16 *C, int *DB) {
  wgmma_k10::run(M, N, K,
                 reinterpret_cast<fp16*>(A),
                 reinterpret_cast<fp16*>(B),
                 reinterpret_cast<fp16*>(C),
                 DB);
}

void runKernel11(int M, int N, int K, at_fp16 *A, at_fp16 *B, at_fp16 *C, int *DB) {
  wgmma_k11::run(M, N, K,
                 reinterpret_cast<fp16*>(A),
                 reinterpret_cast<fp16*>(B),
                 reinterpret_cast<fp16*>(C),
                 DB);
}

void runKernel12(int M, int N, int K, at_fp16 *A, at_fp16 *B, at_fp16 *C, int *DB) {
  wgmma_k12::run(M, N, K,
                 reinterpret_cast<fp16*>(A),
                 reinterpret_cast<fp16*>(B),
                 reinterpret_cast<fp16*>(C),
                 DB);
}

void kernel_7_raw(int M, int N, int K, uint64_t A_ptr, uint64_t B_ptr, uint64_t C_ptr, uint64_t /*DB_ptr*/) {
  runCublasGemmFP16_TensorCores(M, N, K,
    reinterpret_cast<fp16*>(A_ptr),
    reinterpret_cast<fp16*>(B_ptr),
    reinterpret_cast<fp16*>(C_ptr));
}

void kernel_8_raw(int M, int N, int K, uint64_t A_ptr, uint64_t B_ptr, uint64_t C_ptr, uint64_t /*DB_ptr*/) {
  const int WMMA_M = 16;
  const int WMMA_N = 16;
  const int WMMA_K = 16;
  const int BM = WMMA_M * 4 * 2;
  const int BN = WMMA_N * 2 * 4;
  dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 blockDim(256);
  gemm_wmma_tiled<WMMA_M, WMMA_N, WMMA_K><<<gridDim, blockDim>>>(M, N, K,
    reinterpret_cast<fp16*>(A_ptr),
    reinterpret_cast<fp16*>(B_ptr),
    reinterpret_cast<fp16*>(C_ptr));
}

void kernel_9_raw(int M, int N, int K, uint64_t A_ptr, uint64_t B_ptr, uint64_t C_ptr, uint64_t /*DB_ptr*/) {
  wgmma_k9::run_preconverted_ptrs(M, N, K, A_ptr, B_ptr, C_ptr);
}

void kernel_10_raw(int M, int N, int K, uint64_t A_ptr, uint64_t B_ptr, uint64_t C_ptr, uint64_t DB_ptr) {
  wgmma_k10::run_preconverted_ptrs(M, N, K, A_ptr, B_ptr, C_ptr, DB_ptr);
}

void kernel_11_raw(int M, int N, int K, uint64_t A_ptr, uint64_t B_ptr, uint64_t C_ptr, uint64_t DB_ptr) {
  wgmma_k11::run_preconverted_ptrs(M, N, K, A_ptr, B_ptr, C_ptr, DB_ptr);
}

void kernel_12_raw(int M, int N, int K, uint64_t A_ptr, uint64_t B_ptr, uint64_t C_ptr, uint64_t DB_ptr) {
  wgmma_k12::run_preconverted_ptrs(M, N, K, A_ptr, B_ptr, C_ptr, DB_ptr);
}

