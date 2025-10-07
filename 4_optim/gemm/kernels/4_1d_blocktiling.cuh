
typedef __half fp16;

template <const int BM, const int BN, const int BK, const int TM>
__global__ void gemm_1d_blocktiling(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  __shared__ fp16 As[BM * BK];
  __shared__ fp16 Bs[BK * BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  const uint innerColA = threadIdx.x % BK;
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN;
  const uint innerRowB = threadIdx.x / BN;

  fp16 threadResults[TM];
  for (int i = 0; i < TM; i++) threadResults[i] = __float2half(0.0f);

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    A += BK;
    B += BK * N;

    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      fp16 tmpB = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] = __hadd(threadResults[resIdx],
                                        __hmul(As[(threadRow * TM + resIdx) * BK + dotIdx], tmpB));
      }
    }
    __syncthreads();
  }

  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] = threadResults[resIdx];
  }
}

void runKernel4(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C, int *DB = nullptr) {
  const uint BK = 8;
  const uint TM = 8;
  const uint BM = 64;
  const uint BN = 64;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / TM);
  gemm_1d_blocktiling<BM, BN, BK, TM><<<gridDim, blockDim>>>(M, N, K, A, B, C);
}