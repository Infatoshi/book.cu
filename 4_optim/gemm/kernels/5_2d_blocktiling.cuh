
typedef __half fp16;

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    gemm_2d_blocktiling(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  __shared__ fp16 As[BM * BK];
  __shared__ fp16 Bs[BK * BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  
  const uint numThreadsBlocktile = (BM * BN) / (TM * TN);
  const uint strideA = numThreadsBlocktile / BK;
  const uint strideB = numThreadsBlocktile / BN;

  fp16 threadResults[TM * TN];
  fp16 regM[TM];
  fp16 regN[TN];
  for (int i = 0; i < TM * TN; i++) threadResults[i] = __float2half(0.0f);

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    A += BK;
    B += BK * N;

    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] =
              __hadd(threadResults[resIdxM * TN + resIdxN],
                     __hmul(regM[resIdxM], regN[resIdxN]));
        }
      }
    }
    __syncthreads();
  }

  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
          threadResults[resIdxM * TN + resIdxN];
    }
  }
}

void runKernel5(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C, int *DB = nullptr) {
  const uint BM = 64;
  const uint BN = 64;
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / (TM * TN));
  gemm_2d_blocktiling<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, A, B, C);
}