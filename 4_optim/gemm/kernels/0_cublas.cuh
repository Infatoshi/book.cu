
/**
 * CUBLAS GEMM implementation using BF16 precision
 * Demonstrates high-performance matrix multiplication using cuBLAS library
 * Uses BF16 (Brain Floating Point) format for reduced memory bandwidth
 */

typedef __nv_bfloat16 bf16;  // NVIDIA BF16 type definition

// Global CUBLAS handle for reuse across function calls
static cublasHandle_t cublas_handle_global;
static bool cublas_initialized = false;

/**
 * High-performance GEMM using CUBLAS with BF16 precision
 * Computes C = A^T * B where A is K×M, B is K×N, C is M×N
 * Uses BF16 input matrices with FP32 accumulation for numerical stability
 * 
 * @param M Number of rows in A^T and C
 * @param N Number of columns in B and C  
 * @param K Number of columns in A^T and rows in B
 * @param A Input matrix A (K×M, BF16, device memory)
 * @param B Input matrix B (K×N, BF16, device memory)
 * @param C Output matrix C (M×N, BF16, device memory)
 */
void runCublasGemmBF16(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
  // Initialize CUBLAS handle if not already done
  if (!cublas_initialized) {
    cublasCreate(&cublas_handle_global);
    cublas_initialized = true;
  }
  
  // GEMM parameters: C = alpha * A^T * B + beta * C
  float alpha = 1.0f;  // Scaling factor for A^T * B
  float beta = 0.0f;   // Scaling factor for C (0 means overwrite)
  
  // Call CUBLAS GEMM with BF16 precision
  // CUBLAS_OP_T: A is transposed (A^T)
  // CUBLAS_OP_N: B is not transposed
  // CUDA_R_16BF: BF16 data type
  // CUBLAS_COMPUTE_32F: FP32 accumulation for numerical stability
  cublasStatus_t status = cublasGemmEx(cublas_handle_global, CUBLAS_OP_T, CUBLAS_OP_N, 
    M, N, K, &alpha, A, CUDA_R_16BF, N, B, CUDA_R_16BF, K, &beta, C, CUDA_R_16BF, N, 
    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

  // Check for errors
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("CUBLAS error: %d\n", status);
    exit(1);
  }
}
