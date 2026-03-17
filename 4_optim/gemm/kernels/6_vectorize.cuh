/**
 * @file 6_vectorize.cuh
 * @brief GEMM kernel with vectorized memory access optimization
 * 
 * This kernel extends the 2D block tiling approach by adding vectorized
 * memory loads/stores to maximize memory bandwidth utilization.
 * 
 * Optimization Journey:
 * Kernel 5: 2D block tiling - register blocking in both dimensions
 * Kernel 6: Vectorized memory access (this file) - vector loads for better bandwidth
 * 
 * Key Optimization:
 * - Uses vectorized memory loads (int2 = 4×fp16 = 8 bytes per load)
 * - Loads 4 elements at once from global memory to shared memory
 * - Reduces memory instruction count and improves bandwidth utilization
 * - Fallback to scalar loads when vectorization is not possible
 * - Uses FMA (fused multiply-add) instructions for better throughput
 * - Larger block tiles (BM=128, BN=128) for better memory reuse
 * 
 * Performance Impact:
 * - Reduces memory instruction overhead (4x fewer instructions)
 * - Better memory bandwidth utilization (up to 4x improvement)
 * - Higher instruction throughput through FMA operations
 * - Better cache utilization through larger tiles
 */

typedef __half fp16;

/**
 * @brief GEMM kernel with vectorized memory access
 * 
 * Computes C = A * B using:
 * - Shared memory blocking for data reuse
 * - 2D register blocking: each thread computes TM×TN output elements
 * - Vectorized memory loads: loads 4 fp16 elements at once (int2 = 8 bytes)
 * - FMA instructions for better floating-point throughput
 * 
 * Algorithm:
 * 1. Each thread block computes a BM×BN tile of output C
 * 2. Each thread computes TM×TN elements (stored in register array)
 * 3. Loop over K dimension in tiles of size BK:
 *    a. Load BM×BK tile of A into shared memory using vectorized loads
 *    b. Load BK×BN tile of B into shared memory using vectorized loads
 *    c. Compute partial dot products using shared memory
 *    d. Use FMA instructions for better throughput
 * 4. Write TM×TN results per thread to global memory with bounds checking
 * 
 * Vectorized loading:
 * - Uses int2 type to load 4 fp16 elements at once (8 bytes)
 * - Requires alignment checks: both source and destination must be aligned
 * - Falls back to scalar loads when vectorization is not possible
 * - Cooperative loading: multiple threads work together to load tiles
 * 
 * Memory access pattern:
 * - Global memory: Vectorized loads (4 elements per instruction)
 * - Shared memory: Regular access (already fast)
 * - Bounds checking for partial tiles at matrix boundaries
 * 
 * @tparam BM Block tile size in M dimension (rows, typically 128)
 * @tparam BN Block tile size in N dimension (columns, typically 128)
 * @tparam BK Block tile size in K dimension (reduction dimension, typically 16)
 * @tparam TM Number of output rows computed per thread
 * @tparam TN Number of output columns computed per thread
 * 
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param A Input matrix A (M×K, device memory, row-major)
 * @param B Input matrix B (K×N, device memory, row-major)
 * @param C Output matrix C (M×N, device memory, row-major)
 */
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    gemm_vectorize(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C) {
  // Vectorization parameters: load 4 fp16 elements at once
  constexpr int VecElems = 4;                // Number of elements per vector load
  using VecType = int2;                      // int2 = 8 bytes = 4×fp16

  // Block coordinates: which tile of output matrix this block computes
  const uint blockRow = blockIdx.y;
  const uint blockCol = blockIdx.x;

  // Thread position within block (2D layout)
  const uint threadCol = threadIdx.x % (BN / TN);
  const uint threadRow = threadIdx.x / (BN / TN);
  const uint blockThreads = blockDim.x;

  // Shared memory for caching tiles
  __shared__ fp16 As[BM * BK];  // Tile of A: BM×BK
  __shared__ fp16 Bs[BK * BN];  // Tile of B: BK×BN

  // Register array for storing intermediate results
  fp16 threadResults[TM * TN];
  for (int i = 0; i < TM * TN; ++i) threadResults[i] = __float2half(0.0f);

  // Set base pointers to the beginning of this block's tile
  fp16 *C_block = C + blockRow * BM * N + blockCol * BN;
  const fp16 *A_block = A + blockRow * BM * K;
  const fp16 *B_block = B + blockCol * BN;

  // Calculate vectorization parameters for cooperative loading
  // Each thread may need to load multiple vectors to cover the entire tile
  const int vecsPerRowA = (BK + VecElems - 1) / VecElems;  // Vectors per row of A tile
  const int totalVecsA = BM * vecsPerRowA;                 // Total vectors in A tile
  const int vecsPerRowB = (BN + VecElems - 1) / VecElems; // Vectors per row of B tile
  const int totalVecsB = BK * vecsPerRowB;                 // Total vectors in B tile
  const int loadsPerThreadA = CEIL_DIV(totalVecsA, (int)blockThreads);
  const int loadsPerThreadB = CEIL_DIV(totalVecsB, (int)blockThreads);

  const fp16 zero = __float2half(0.0f);

  // Register arrays for A and B elements
  fp16 regM[TM];
  fp16 regN[TN];

  // Loop over K dimension in tiles
  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    const fp16 *A_panel = A_block + bkIdx;
    const fp16 *B_panel = B_block + bkIdx * N;

    // Phase 1: Load A tile into shared memory using vectorized loads
    // Each thread loads multiple vectors to cover the entire tile
    for (int iter = 0; iter < loadsPerThreadA; ++iter) {
      int vecIndex = iter * blockThreads + threadIdx.x;
      if (vecIndex >= totalVecsA) continue;

      // Calculate which vector to load (row and column within tile)
      int row = vecIndex / vecsPerRowA;
      int vecCol = vecIndex % vecsPerRowA;
      int kCol = vecCol * VecElems;

      // Destination in shared memory and source in global memory
      fp16 *smemDst = &As[row * BK + kCol];
      int globalRow = blockRow * BM + row;
      int globalK = bkIdx + kCol;
      const fp16 *gmemSrc = A_panel + row * K + kCol;
      
      // Check if vectorized load is possible:
      // 1. Bounds check: ensure all elements are within matrix bounds
      // 2. Alignment check: both source and destination must be aligned to VecType size
      bool canVectorize = (globalRow < M) && (globalK + VecElems) <= K &&
                          (kCol + VecElems) <= BK &&
                          (((reinterpret_cast<uintptr_t>(gmemSrc)) &
                            (sizeof(VecType) - 1)) == 0) &&
                          (((reinterpret_cast<uintptr_t>(smemDst)) &
                            (sizeof(VecType) - 1)) == 0);

      if (canVectorize) {
        // Vectorized load: load 4 elements at once (8 bytes)
        *reinterpret_cast<VecType *>(smemDst) =
            *reinterpret_cast<const VecType *>(gmemSrc);
      } else {
        // Fallback to scalar loads when vectorization is not possible
        for (int v = 0; v < VecElems && (kCol + v) < BK; ++v) {
          int kIdx = globalK + v;
          smemDst[v] = (globalRow < M && kIdx < K)
                           ? A_panel[row * K + kCol + v]
                           : zero;
        }
      }
    }

    // Phase 2: Load B tile into shared memory using vectorized loads
    for (int iter = 0; iter < loadsPerThreadB; ++iter) {
      int vecIndex = iter * blockThreads + threadIdx.x;
      if (vecIndex >= totalVecsB) continue;

      // Calculate which vector to load (row and column within tile)
      int row = vecIndex / vecsPerRowB;
      int vecCol = vecIndex % vecsPerRowB;
      int nCol = vecCol * VecElems;

      // Destination in shared memory and source in global memory
      fp16 *smemDst = &Bs[row * BN + nCol];
      int globalK = bkIdx + row;
      int globalN = blockCol * BN + nCol;
      const fp16 *gmemSrc = B_panel + row * N + nCol;
      
      // Check if vectorized load is possible
      bool canVectorize = (globalK < K) && (globalN + VecElems) <= N &&
                          (nCol + VecElems) <= BN &&
                          (((reinterpret_cast<uintptr_t>(gmemSrc)) &
                            (sizeof(VecType) - 1)) == 0) &&
                          (((reinterpret_cast<uintptr_t>(smemDst)) &
                            (sizeof(VecType) - 1)) == 0);

      if (canVectorize) {
        // Vectorized load: load 4 elements at once (8 bytes)
        *reinterpret_cast<VecType *>(smemDst) =
            *reinterpret_cast<const VecType *>(gmemSrc);
      } else {
        // Fallback to scalar loads when vectorization is not possible
        for (int v = 0; v < VecElems && (nCol + v) < BN; ++v) {
          int nIdx = globalN + v;
          smemDst[v] = (globalK < K && nIdx < N)
                           ? B_panel[row * N + nCol + v]
                           : zero;
        }
      }
    }

    // Synchronize to ensure all tiles are loaded before computation
    __syncthreads();

    // Phase 3: Compute partial dot products using shared memory
    // Load TM elements of A and TN elements of B into registers
    // Then compute outer product using FMA instructions
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // Load TM elements of A into registers (reused for all TN columns)
      for (uint i = 0; i < TM; ++i) {
        int localRow = threadRow * TM + i;
        regM[i] = As[localRow * BK + dotIdx];
      }
      // Load TN elements of B into registers (reused for all TM rows)
      for (uint j = 0; j < TN; ++j) {
        int localCol = threadCol * TN + j;
        regN[j] = Bs[dotIdx * BN + localCol];
      }
      // Compute outer product: regM[TM] × regN[TN] -> threadResults[TM×TN]
      // Use FMA (fused multiply-add) for better throughput
      for (uint i = 0; i < TM; ++i) {
        for (uint j = 0; j < TN; ++j) {
          threadResults[i * TN + j] =
              __hfma(regM[i], regN[j], threadResults[i * TN + j]);
        }
      }
    }

    __syncthreads();
  }

  // Phase 4: Write results to global memory with bounds checking
  // Handle partial tiles at matrix boundaries
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    int globalRow = blockRow * BM + threadRow * TM + resIdxM;
    if (globalRow >= M) continue;  // Skip if out of bounds
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      int globalCol = blockCol * BN + threadCol * TN + resIdxN;
      if (globalCol < N) {
        C_block[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
            threadResults[resIdxM * TN + resIdxN];
      }
    }
  }
}

/**
 * @brief Launcher function for vectorized GEMM kernel
 * 
 * Configures kernel launch parameters for vectorized tiled matrix multiplication.
 * Uses larger tiles (BM=128, BN=128) to maximize memory reuse and vectorization benefits.
 * 
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param A Input matrix A (M×K, device memory)
 * @param B Input matrix B (K×N, device memory)
 * @param C Output matrix C (M×N, device memory)
 * @param DB Optional debug buffer (unused in this kernel)
 */
void runKernel6(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C, int *DB = nullptr) {
  const uint BM = 128;  // Larger M-tile for better memory reuse
  const uint BN = 128;  // Larger N-tile for better memory reuse
  const uint BK = 16;   // K-tile size
  const uint TM = 8;    // Register blocking: threads compute 8 rows
  const uint TN = 8;    // Register blocking: threads compute 8 columns
  
  // Grid dimensions: one block per output tile
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  // Block size: (BM * BN) / (TM * TN) threads per block
  dim3 blockDim((BM * BN) / (TM * TN));
  gemm_vectorize<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, A, B, C);
}
