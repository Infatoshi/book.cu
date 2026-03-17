/**
 * @file 4_1d_blocktiling.cuh
 * @brief GEMM kernel with 1D block tiling and register blocking optimization
 * 
 * This kernel introduces thread-level register blocking to increase
 * arithmetic intensity and reduce memory traffic per thread.
 * 
 * Optimization Journey:
 * Kernel 3: Shared memory blocking - data reuse through shared memory
 * Kernel 4: 1D block tiling (this file) - register blocking for better compute/memory ratio
 * 
 * Key Optimization:
 * - Each thread computes multiple output elements (TM elements per thread)
 * - Uses register arrays to store intermediate results
 * - Increases arithmetic intensity: more FLOPs per memory access
 * - Better instruction-level parallelism through register reuse
 * - Larger block tiles (BM×BN) for better memory reuse
 * 
 * Performance Impact:
 * - Reduces memory traffic per floating-point operation
 * - Better GPU occupancy due to fewer threads per block
 * - Register blocking enables better instruction scheduling
 */

typedef __half fp16;

/**
 * @brief GEMM kernel with 1D block tiling and register blocking
 * 
 * Computes C = A * B using:
 * - Shared memory blocking for data reuse
 * - Register blocking: each thread computes TM output elements
 * - Larger block tiles (BM×BN) for better memory efficiency
 * 
 * Algorithm:
 * 1. Each thread block computes a BM×BN tile of output C
 * 2. Each thread computes TM elements (stored in registers)
 * 3. Loop over K dimension in tiles of size BK:
 *    a. Load BM×BK tile of A into shared memory
 *    b. Load BK×BN tile of B into shared memory
 *    c. Each thread computes TM partial results using shared memory
 * 4. Write TM results per thread to global memory
 * 
 * Thread-to-data mapping:
 * - Each thread is responsible for TM consecutive rows in the output tile
 * - Threads are arranged in a 1D layout: (BM * BN) / TM threads per block
 * - Within each thread's TM rows, it computes one column (threadCol)
 * 
 * Register blocking benefits:
 * - Each element of B loaded once, reused TM times (once per output row)
 * - Reduces memory operations per floating-point operation
 * - Better instruction-level parallelism through register reuse
 * 
 * @tparam BM Block tile size in M dimension (rows)
 * @tparam BN Block tile size in N dimension (columns)
 * @tparam BK Block tile size in K dimension (reduction dimension)
 * @tparam TM Number of output elements computed per thread (register blocking factor)
 * 
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param A Input matrix A (M×K, device memory, row-major)
 * @param B Input matrix B (K×N, device memory, row-major)
 * @param C Output matrix C (M×N, device memory, row-major)
 */
template <const int BM, const int BN, const int BK, const int TM>
__global__ void gemm_1d_blocktiling(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C) {
  // Block coordinates: which tile of output matrix this block computes
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // Thread position within block (1D layout)
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  // Shared memory for caching tiles
  __shared__ fp16 As[BM * BK];  // Tile of A: BM×BK
  __shared__ fp16 Bs[BK * BN];  // Tile of B: BK×BN

  // Set base pointers to the beginning of this block's tile
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // Thread positions for loading tiles into shared memory
  const uint innerColA = threadIdx.x % BK;
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN;
  const uint innerRowB = threadIdx.x / BN;

  // Register array: each thread stores TM partial results
  // This enables register blocking and better instruction scheduling
  fp16 threadResults[TM];
  for (int i = 0; i < TM; i++) threadResults[i] = __float2half(0.0f);

  // Loop over K dimension in tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // Phase 1: Load tiles into shared memory (coalesced access)
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    // Advance pointers to next K-tile
    A += BK;
    B += BK * N;

    // Phase 2: Compute partial dot products using shared memory
    // Each thread computes TM results using register blocking
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // Load B element once (reused for all TM rows)
      fp16 tmpB = Bs[dotIdx * BN + threadCol];
      
      // Update all TM results using the same B element
      // This is the key register blocking optimization
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] = __hadd(threadResults[resIdx],
                                        __hmul(As[(threadRow * TM + resIdx) * BK + dotIdx], tmpB));
      }
    }
    __syncthreads();
  }

  // Phase 3: Write results to global memory
  // Each thread writes TM elements
  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] = threadResults[resIdx];
  }
}

/**
 * @brief Launcher function for 1D block tiling GEMM kernel
 * 
 * Configures kernel launch parameters for register-blocked tiled matrix multiplication.
 * 
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param A Input matrix A (M×K, device memory)
 * @param B Input matrix B (K×N, device memory)
 * @param C Output matrix C (M×N, device memory)
 * @param DB Optional debug buffer (unused in this kernel)
 */
void runKernel4(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C, int *DB = nullptr) {
  const uint BK = 8;   // K-tile size
  const uint TM = 8;   // Register blocking: threads compute 8 output elements
  const uint BM = 64;  // M-tile size (larger than previous kernels)
  const uint BN = 64;  // N-tile size (larger than previous kernels)
  
  // Grid dimensions: one block per output tile
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  // Block size: (BM * BN) / TM threads per block
  dim3 blockDim((BM * BN) / TM);
  gemm_1d_blocktiling<BM, BN, BK, TM><<<gridDim, blockDim>>>(M, N, K, A, B, C);
}