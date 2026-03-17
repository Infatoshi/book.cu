
/**
 * @file 9_wgmma.cuh
 * @brief Wrapper interface for basic WGMMA GEMM kernel
 * 
 * This file provides a high-level interface to the basic WGMMA implementation.
 * It handles matrix layout conversion (row-major to column-major) required by WGMMA
 * and provides multiple entry points for different use cases.
 * 
 * Matrix Layout Requirements:
 * - WGMMA requires column-major layout for optimal performance
 * - Matrix A: Already in column-major format
 * - Matrix B: Converted from row-major to column-major
 * - Matrix C: Output in column-major, converted back to row-major
 * 
 * Entry Points:
 * - run(): Handles layout conversion automatically
 * - run_preconverted(): Assumes matrices are already in correct layout
 * - run_preconverted_ptrs(): Same as above but with raw pointers
 */

namespace wgmma_k9 {

/**
 * @brief Runs WGMMA GEMM with automatic layout conversion
 * @param M Number of rows in matrices A and C
 * @param N Number of columns in matrices B and C
 * @param K Number of columns in A and rows in B
 * @param A Input matrix A (M×K, FP16, device memory, row-major)
 * @param B Input matrix B (K×N, FP16, device memory, row-major)
 * @param C Output matrix C (M×N, FP16, device memory, row-major)
 * 
 * This function:
 * 1. Allocates temporary buffers for column-major layout
 * 2. Converts A and B to column-major format
 * 3. Calls the WGMMA kernel
 * 4. Converts output C back to row-major format
 * 5. Cleans up temporary buffers
 */
inline void run(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C) {
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

  WGMMA_Basic_fp16::runKernel_fp16(M, N, K, A_col, B_col, C_col);

  
  wgmma_layout::col_to_row(C_col, C, M, N);

  cudaDeviceSynchronize();

cleanup:
  if (A_col) cudaFree(A_col);
  if (B_col) cudaFree(B_col);
  if (C_col) cudaFree(C_col);
}

/**
 * @brief Runs WGMMA GEMM assuming matrices are already in column-major layout
 * @param M Number of rows in matrices A and C
 * @param N Number of columns in matrices B and C
 * @param K Number of columns in A and rows in B
 * @param A_col Input matrix A (M×K, FP16, device memory, column-major)
 * @param B_col Input matrix B (K×N, FP16, device memory, column-major)
 * @param C_col Output matrix C (M×N, FP16, device memory, column-major)
 * 
 * Skips layout conversion overhead when matrices are already in correct format.
 * Use this when you can maintain column-major layout throughout your pipeline.
 */
inline void run_preconverted(int M, int N, int K, fp16 *A_col, fp16 *B_col, fp16 *C_col) {
  WGMMA_Basic_fp16::runKernel_fp16(M, N, K, A_col, B_col, C_col);
}

/**
 * @brief Runs WGMMA GEMM with raw pointer arguments
 * @param M Number of rows in matrices A and C
 * @param N Number of columns in matrices B and C
 * @param K Number of columns in A and rows in B
 * @param A_ptr Raw pointer to matrix A (FP16, column-major)
 * @param B_ptr Raw pointer to matrix B (FP16, column-major)
 * @param C_ptr Raw pointer to matrix C (FP16, column-major)
 * 
 * Low-level interface for integration with other systems that use raw pointers.
 * Assumes all matrices are already in column-major layout.
 */
inline void run_preconverted_ptrs(int M, int N, int K,
                                  uint64_t A_ptr, uint64_t B_ptr, uint64_t C_ptr) {
  WGMMA_Basic_fp16::runKernel_fp16(
      M, N, K,
      reinterpret_cast<fp16 *>(A_ptr),
      reinterpret_cast<fp16 *>(B_ptr),
      reinterpret_cast<fp16 *>(C_ptr));
}

}  
