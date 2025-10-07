
typedef __half fp16;

__global__ void gemm_naive(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < M && y < N) {
    fp16 tmp = __float2half(0.0f);
    for (int i = 0; i < K; ++i) {
      
      tmp = __hadd(tmp, __hmul(A[x * K + i], B[i * N + y]));
    }
    C[x * N + y] = tmp;
  }
}

void runKernel1(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C) {
  dim3 blockDim(32, 32);
  dim3 gridDim((M + 31) / 32, (N + 31) / 32);
  gemm_naive<<<gridDim, blockDim>>>(M, N, K, A, B, C);
}