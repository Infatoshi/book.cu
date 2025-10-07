
typedef __half fp16;

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    gemm_vectorize(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C) {
  constexpr int VecElems = 4;                
  using VecType = int2;                      

  const uint blockRow = blockIdx.y;
  const uint blockCol = blockIdx.x;

  const uint threadCol = threadIdx.x % (BN / TN);
  const uint threadRow = threadIdx.x / (BN / TN);
  const uint blockThreads = blockDim.x;

  __shared__ fp16 As[BM * BK];
  __shared__ fp16 Bs[BK * BN];

  fp16 threadResults[TM * TN];
  for (int i = 0; i < TM * TN; ++i) threadResults[i] = __float2half(0.0f);

  fp16 *C_block = C + blockRow * BM * N + blockCol * BN;
  const fp16 *A_block = A + blockRow * BM * K;
  const fp16 *B_block = B + blockCol * BN;

  const int vecsPerRowA = (BK + VecElems - 1) / VecElems;
  const int totalVecsA = BM * vecsPerRowA;
  const int vecsPerRowB = (BN + VecElems - 1) / VecElems;
  const int totalVecsB = BK * vecsPerRowB;
  const int loadsPerThreadA = CEIL_DIV(totalVecsA, (int)blockThreads);
  const int loadsPerThreadB = CEIL_DIV(totalVecsB, (int)blockThreads);

  const fp16 zero = __float2half(0.0f);

  fp16 regM[TM];
  fp16 regN[TN];

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    const fp16 *A_panel = A_block + bkIdx;
    const fp16 *B_panel = B_block + bkIdx * N;

    
    for (int iter = 0; iter < loadsPerThreadA; ++iter) {
      int vecIndex = iter * blockThreads + threadIdx.x;
      if (vecIndex >= totalVecsA) continue;

      int row = vecIndex / vecsPerRowA;
      int vecCol = vecIndex % vecsPerRowA;
      int kCol = vecCol * VecElems;

      fp16 *smemDst = &As[row * BK + kCol];
      int globalRow = blockRow * BM + row;
      int globalK = bkIdx + kCol;
      const fp16 *gmemSrc = A_panel + row * K + kCol;
      bool canVectorize = (globalRow < M) && (globalK + VecElems) <= K &&
                          (kCol + VecElems) <= BK &&
                          (((reinterpret_cast<uintptr_t>(gmemSrc)) &
                            (sizeof(VecType) - 1)) == 0) &&
                          (((reinterpret_cast<uintptr_t>(smemDst)) &
                            (sizeof(VecType) - 1)) == 0);

      if (canVectorize) {
        *reinterpret_cast<VecType *>(smemDst) =
            *reinterpret_cast<const VecType *>(gmemSrc);
      } else {
        for (int v = 0; v < VecElems && (kCol + v) < BK; ++v) {
          int kIdx = globalK + v;
          smemDst[v] = (globalRow < M && kIdx < K)
                           ? A_panel[row * K + kCol + v]
                           : zero;
        }
      }
    }

    
    for (int iter = 0; iter < loadsPerThreadB; ++iter) {
      int vecIndex = iter * blockThreads + threadIdx.x;
      if (vecIndex >= totalVecsB) continue;

      int row = vecIndex / vecsPerRowB;
      int vecCol = vecIndex % vecsPerRowB;
      int nCol = vecCol * VecElems;

      fp16 *smemDst = &Bs[row * BN + nCol];
      int globalK = bkIdx + row;
      int globalN = blockCol * BN + nCol;
      const fp16 *gmemSrc = B_panel + row * N + nCol;
      bool canVectorize = (globalK < K) && (globalN + VecElems) <= N &&
                          (nCol + VecElems) <= BN &&
                          (((reinterpret_cast<uintptr_t>(gmemSrc)) &
                            (sizeof(VecType) - 1)) == 0) &&
                          (((reinterpret_cast<uintptr_t>(smemDst)) &
                            (sizeof(VecType) - 1)) == 0);

      if (canVectorize) {
        *reinterpret_cast<VecType *>(smemDst) =
            *reinterpret_cast<const VecType *>(gmemSrc);
      } else {
        for (int v = 0; v < VecElems && (nCol + v) < BN; ++v) {
          int nIdx = globalN + v;
          smemDst[v] = (globalK < K && nIdx < N)
                           ? B_panel[row * N + nCol + v]
                           : zero;
        }
      }
    }

    __syncthreads();

    
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      for (uint i = 0; i < TM; ++i) {
        int localRow = threadRow * TM + i;
        regM[i] = As[localRow * BK + dotIdx];
      }
      for (uint j = 0; j < TN; ++j) {
        int localCol = threadCol * TN + j;
        regN[j] = Bs[dotIdx * BN + localCol];
      }
      for (uint i = 0; i < TM; ++i) {
        for (uint j = 0; j < TN; ++j) {
          threadResults[i * TN + j] =
              __hfma(regM[i], regN[j], threadResults[i * TN + j]);
        }
      }
    }

    __syncthreads();
  }

  
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    int globalRow = blockRow * BM + threadRow * TM + resIdxM;
    if (globalRow >= M) continue;
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      int globalCol = blockCol * BN + threadCol * TN + resIdxN;
      if (globalCol < N) {
        C_block[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
            threadResults[resIdxM * TN + resIdxN];
      }
    }
  }
}

void runKernel6(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C, int *DB = nullptr) {
  const uint BM = 128;
  const uint BN = 128;
  const uint BK = 16;
  const uint TM = 8;
  const uint TN = 8;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / (TM * TN));
  gemm_vectorize<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, A, B, C);
}
