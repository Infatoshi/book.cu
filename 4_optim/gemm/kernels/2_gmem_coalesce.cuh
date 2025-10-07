
typedef __half fp16;

template <const uint BLOCKSIZE>
__global__ void gemm_gmem_coalesce(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C) {
  const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (cRow < M && cCol < N) {
    fp16 tmp = __float2half(0.0f);
    for (int i = 0; i < K; ++i) {
      tmp = __hadd(tmp, __hmul(A[cRow * K + i], B[i * N + cCol]));
    }
    C[cRow * N + cCol] = tmp;
  }
}

void runKernel2(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C) {
  const uint BLOCKSIZE = 32;
  dim3 blockDim(BLOCKSIZE * BLOCKSIZE);
  dim3 gridDim((M + BLOCKSIZE - 1) / BLOCKSIZE, (N + BLOCKSIZE - 1) / BLOCKSIZE);
  gemm_gmem_coalesce<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, A, B, C);
}