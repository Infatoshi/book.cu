/**
 * @file 5_2d_blocktiling.cuh
 * @brief GEMM kernel with 2D block tiling and register blocking optimization
 * 
 * This kernel extends the 1D block tiling approach by computing a 2D tile
 * of output elements per thread (TMxTN), improving register utilization.
 * 
 * Optimization Journey:
 * Kernel 4: 1D block tiling - register blocking in one dimension
 * Kernel 5: 2D block tiling (this file) - register blocking in both dimensions
 * 
 * Key Optimization:
 * - Each thread computes a TMxTN tile of output elements (instead of TMx1)
 * - Uses register arrays for both A and B elements (regM[TM], regN[TN])
 * - Better register reuse: each element loaded once, used multiple times
 * - Strided loading pattern for better memory coalescing
 * - Uses __launch_bounds__ to hint compiler about optimal occupancy
 * 
 * Performance Impact:
 * - Higher arithmetic intensity: more FLOPs per memory access
 * - Better register pressure management
 * - Improved instruction-level parallelism
 * - Better memory bandwidth utilization through strided loads
 */

typedef __half fp16;

/**
 * @brief GEMM kernel with 2D block tiling and register blocking
 * 
 * Computes C = A @ B using:
 * - Shared memory blocking for data reuse
 * - 2D register blocking: each thread computes TMxTN output elements
 * - Strided loading pattern for better memory coalescing
 * - Register arrays for A and B elements to maximize reuse
 * 
 * Algorithm:
 * 1. Each thread block computes a BMxBN tile of output C
 * 2. Each thread computes TMxTN elements (stored in register array)
 * 3. Loop over K dimension in tiles of size BK:
 *    a. Load BMxBK tile of A into shared memory (strided for coalescing)
 *    b. Load BKxBN tile of B into shared memory (strided for coalescing)
 *    c. Load TM elements of A and TN elements of B into registers
 *    d. Compute outer product: regM[TM] @ regN[TN] -> threadResults[TMxTN]
 * 4. Write TMxTN results per thread to global memory
 * 
 * Thread-to-data mapping:
 * - Each thread is responsible for TMxTN output elements
 * - Threads arranged in 2D layout: (BN/TN) columns x (BM/TM) rows per block
 * - Total threads per block: (BM * BN) / (TM * TN)
 * 
 * Register blocking benefits:
 * - Each element of A loaded once, reused TN times (once per output column)
 * - Each element of B loaded once, reused TM times (once per output row)
 * - Maximizes register reuse and arithmetic intensity
 * 
 * Strided loading pattern:
 * - Multiple threads cooperate to load each tile
 * - Improves memory coalescing by ensuring consecutive threads access
 *   consecutive memory locations
 * 
 * @tparam BM Block tile size in M dimension (rows)
 * @tparam BN Block tile size in N dimension (columns)
 * @tparam BK Block tile size in K dimension (reduction dimension)
 * @tparam TM Number of output rows computed per thread
 * @tparam TN Number of output columns computed per thread
 * 
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param A Input matrix A (MxK, device memory, row-major)
 * @param B Input matrix B (KxN, device memory, row-major)
 * @param C Output matrix C (MxN, device memory, row-major)
 */
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    gemm_2d_blocktiling(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C) {
  // Block coordinates: which tile of output matrix this block computes
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // Thread position within block (2D layout)
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // Shared memory for caching tiles
  __shared__ fp16 As[BM * BK];  // Tile of A: BM×BK
  __shared__ fp16 Bs[BK * BN];  // Tile of B: BK×BN

  // Set base pointers to the beginning of this block's tile
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // Thread positions for loading tiles into shared memory
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  
  // Calculate stride for cooperative loading (multiple threads load same tile)
  const uint numThreadsBlocktile = (BM * BN) / (TM * TN);
  const uint strideA = numThreadsBlocktile / BK;  // Stride for loading A tile
  const uint strideB = numThreadsBlocktile / BN;  // Stride for loading B tile

  // Register arrays for storing intermediate results
  fp16 threadResults[TM * TN];  // Final results: TMxTN elements per thread
  fp16 regM[TM];                 // Register array for A elements
  fp16 regN[TN];                 // Register array for B elements
  for (int i = 0; i < TM * TN; i++) threadResults[i] = __float2half(0.0f);

  // Loop over K dimension in tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // Phase 1: Load tiles into shared memory using strided pattern
    // Multiple threads cooperate to load the tile for better coalescing
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    // Advance pointers to next K-tile
    A += BK;
    B += BK * N;

    // Phase 2: Compute partial dot products using shared memory
    // Load TM elements of A and TN elements of B into registers
    // Then compute outer product to update all TM*TN results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // Load TM values from A-tile into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      // Load TN values from B-tile into registers
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      
      // Compute Outer Product (TM x TN updates)
      // reuse each loaded regM and regN value multiple times
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          int flatIdx = resIdxM * TN + resIdxN;
          threadResults[flatIdx] =
              __hadd(threadResults[flatIdx],
                     __hmul(regM[resIdxM], regN[resIdxN]));
        }
      }
    }
    __syncthreads();
  }

  // Phase 3: Write results to global memory
  // Each thread writes TM×TN elements
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
          threadResults[resIdxM * TN + resIdxN];
    }
  }
}

/**
 * @brief Launcher function for 2D block tiling GEMM kernel
 * 
 * Configures kernel launch parameters for 2D register-blocked tiled matrix multiplication.
 * Uses __launch_bounds__ to optimize register usage and occupancy.
 * 
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param A Input matrix A (M×K, device memory)
 * @param B Input matrix B (K×N, device memory)
 * @param C Output matrix C (M×N, device memory)
 * @param DB Optional debug buffer (unused in this kernel)
 */
void runKernel5(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C, int *DB = nullptr) {
  const uint BM = 64;  // M-tile size
  const uint BN = 64;  // N-tile size
  const uint BK = 8;  // K-tile size
  const uint TM = 8;  // Register blocking: threads compute 8 rows
  const uint TN = 8;  // Register blocking: threads compute 8 columns
  
  // Grid dimensions: one block per output tile
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  // Block size: (BM * BN) / (TM * TN) threads per block
  dim3 blockDim((BM * BN) / (TM * TN));
  gemm_2d_blocktiling<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, A, B, C);
}