
/**
 * @file 12_wgmma_max_tiles.cuh
 * @brief Wrapper interface for WGMMA GEMM with maximum tile sizes
 * 
 * Provides interface to the maximum tiles implementation (128×256×64 blocks).
 * Uses the largest supported WGMMA tile sizes for maximum throughput.
 */

namespace wgmma_k12 {

inline void run(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C, int *DB = nullptr) {
  constexpr int BM = 128;
  constexpr int BN = 256;
  constexpr int BK = 64;

  if ((M % BM) != 0 || (N % BN) != 0 || (K % BK) != 0) {
    wgmma_k9::run(M, N, K, A, B, C);
    return;
  }

  fp16 *A_col = nullptr;
  fp16 *B_col = nullptr;
  fp16 *C_col = nullptr;

  size_t sizeA = static_cast<size_t>(M) * K * sizeof(fp16);
  size_t sizeB = static_cast<size_t>(K) * N * sizeof(fp16);
  size_t sizeC = static_cast<size_t>(M) * N * sizeof(fp16);

  if (cudaMalloc(&A_col, sizeA) != cudaSuccess) goto cleanup;
  if (cudaMalloc(&B_col, sizeB) != cudaSuccess) goto cleanup;
  if (cudaMalloc(&C_col, sizeC) != cudaSuccess) goto cleanup;

  
  cudaMemcpy(A_col, A, sizeA, cudaMemcpyDeviceToDevice);
  wgmma_layout::row_to_col(B, B_col, K, N);

  WGMMA_MaxTiles_fp16::runKernel_fp16(M, N, K, A_col, B_col, C_col, DB);

  wgmma_layout::col_to_row(C_col, C, M, N);

  cudaDeviceSynchronize();

cleanup:
  if (A_col) cudaFree(A_col);
  if (B_col) cudaFree(B_col);
  if (C_col) cudaFree(C_col);
}

inline void run_preconverted(int M, int N, int K, fp16 *A_col, fp16 *B_col, fp16 *C_col, int *DB = nullptr) {
  constexpr int BM = 128;
  constexpr int BN = 256;
  constexpr int BK = 64;

  if ((M % BM) != 0 || (N % BN) != 0 || (K % BK) != 0) {
    wgmma_k9::run_preconverted(M, N, K, A_col, B_col, C_col);
    return;
  }

  WGMMA_MaxTiles_fp16::runKernel_fp16(M, N, K, A_col, B_col, C_col, DB);
}

inline void run_preconverted_ptrs(int M, int N, int K,
                                  uint64_t A_ptr, uint64_t B_ptr, uint64_t C_ptr,
                                  uint64_t DB_ptr = 0) {
  WGMMA_MaxTiles_fp16::runKernel_fp16(
      M, N, K,
      reinterpret_cast<fp16 *>(A_ptr),
      reinterpret_cast<fp16 *>(B_ptr),
      reinterpret_cast<fp16 *>(C_ptr),
      DB_ptr ? reinterpret_cast<int *>(DB_ptr) : nullptr);
}

}  
