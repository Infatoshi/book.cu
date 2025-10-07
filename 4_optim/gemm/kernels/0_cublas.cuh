
typedef __nv_bfloat16 bf16;

static cublasHandle_t cublas_handle_global;
static bool cublas_initialized = false;

void runCublasGemmBF16(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
  if (!cublas_initialized) {
    cublasCreate(&cublas_handle_global);
    cublas_initialized = true;
  }
  
  float alpha = 1, beta = 0;
  
  cublasStatus_t status = cublasGemmEx(cublas_handle_global, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, A, CUDA_R_16BF,
    N, B, CUDA_R_16BF, K, &beta, C, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("CUBLAS error: %d\n", status);
    exit(1);
  }
}
