
typedef __half fp16;

template <const int BLOCKSIZE>
__global__ void gemm_smem_blocking(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C) {
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  __shared__ fp16 As[BLOCKSIZE * BLOCKSIZE];
  __shared__ fp16 Bs[BLOCKSIZE * BLOCKSIZE];

  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  A += cRow * BLOCKSIZE * K;
  B += cCol * BLOCKSIZE;
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

  fp16 tmp = __float2half(0.0f);
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
    __syncthreads();
    
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp = __hadd(tmp, __hmul(As[threadRow * BLOCKSIZE + dotIdx],
                                Bs[dotIdx * BLOCKSIZE + threadCol]));
    }
    __syncthreads();
  }
  C[threadRow * N + threadCol] = tmp;
}

void runKernel3(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C, int *DB = nullptr) {
  const uint BLOCKSIZE = 32;
  dim3 blockDim(BLOCKSIZE * BLOCKSIZE);
  dim3 gridDim(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE));
  gemm_smem_blocking<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, A, B, C);
}