/**
 * @file 3_smem_blocking.cuh
 * @brief GEMM kernel with shared memory blocking/tiling optimization
 * 
 * This kernel introduces shared memory to reduce global memory traffic
 * by reusing loaded data across multiple threads.
 * 
 * Optimization Journey:
 * Kernel 2: Memory coalescing - improved global memory access
 * Kernel 3: Shared memory blocking (this file) - data reuse through shared memory
 * 
 * Key Optimization:
 * - Uses shared memory to cache tiles of A and B matrices
 * - Each tile is loaded once and reused by all threads in the block
 * - Reduces global memory accesses from O(K) per thread to O(K/BLOCKSIZE)
 * - Implements the classic "tiled matrix multiplication" algorithm
 * 
 * Performance Impact:
 * - Dramatically reduces global memory bandwidth requirements
 * - Shared memory is ~100x faster than global memory
 * - Enables better instruction-level parallelism
 */

typedef __half fp16;

/**
 * @brief GEMM kernel with shared memory blocking/tiling
 * 
 * Computes C = A * B using tiled matrix multiplication with shared memory.
 * 
 * Algorithm:
 * 1. Each thread block computes a BLOCKSIZE×BLOCKSIZE tile of C
 * 2. The K dimension is divided into tiles of size BLOCKSIZE
 * 3. For each K-tile:
 *    a. Load tile of A into shared memory (coalesced)
 *    b. Load tile of B into shared memory (coalesced)
 *    c. Synchronize to ensure all data is loaded
 *    d. Compute partial dot product using shared memory data
 *    e. Synchronize before loading next tile
 * 4. Accumulate partial results across all K-tiles
 * 
 * Shared memory usage:
 * - As[BLOCKSIZE×BLOCKSIZE]: Cache for tile of A
 * - Bs[BLOCKSIZE×BLOCKSIZE]: Cache for tile of B
 * - Both tiles are reused by all threads in the block
 * 
 * Memory access pattern:
 * - Global memory: Load tiles coalesced (good bandwidth)
 * - Shared memory: All threads access cached tiles (high bandwidth)
 * - Each element of A and B loaded once per K-tile, used BLOCKSIZE times
 * 
 * @tparam BLOCKSIZE Size of the tile (typically 32, must be <= sqrt(shared memory))
 * 
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param A Input matrix A (M×K, device memory, row-major)
 * @param B Input matrix B (K×N, device memory, row-major)
 * @param C Output matrix C (M×N, device memory, row-major)
 */
template <const int BLOCKSIZE>
__global__ void gemm_smem_blocking(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C) {
  // Block coordinates: which tile of output matrix this block computes
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // Shared memory for caching tiles of A and B
  __shared__ fp16 As[BLOCKSIZE * BLOCKSIZE];
  __shared__ fp16 Bs[BLOCKSIZE * BLOCKSIZE];

  // Thread position within the block (for 2D thread layout)
  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  // Set base pointers to the beginning of this block's tile
  A += cRow * BLOCKSIZE * K;
  B += cCol * BLOCKSIZE;
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

  // Accumulator for dot product result
  fp16 tmp = __float2half(0.0f);
  
  // Loop over K dimension in tiles
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // Phase 1: Load tiles into shared memory (coalesced access)
    // Each thread loads one element of A and one element of B
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
    
    // Synchronize to ensure all threads have finished loading
    __syncthreads();
    
    // Advance pointers to next K-tile
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // Phase 2: Compute partial dot product using shared memory data
    // Each thread accumulates its contribution to the output element
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      // Multiply row of A tile by column of B tile
      tmp = __hadd(tmp, __hmul(As[threadRow * BLOCKSIZE + dotIdx],
                                Bs[dotIdx * BLOCKSIZE + threadCol]));
    }
    
    // Synchronize before loading next tile (ensures all threads finished computation)
    __syncthreads();
  }
  
  // Write final result to global memory
  C[threadRow * N + threadCol] = tmp;
}

/**
 * @brief Launcher function for shared memory blocking GEMM kernel
 * 
 * Configures kernel launch parameters for tiled matrix multiplication.
 * 
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param A Input matrix A (M×K, device memory)
 * @param B Input matrix B (K×N, device memory)
 * @param C Output matrix C (M×N, device memory)
 * @param DB Optional debug buffer (unused in this kernel)
 */
void runKernel3(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C, int *DB = nullptr) {
  const uint BLOCKSIZE = 32;
  // 1D thread block: BLOCKSIZE×BLOCKSIZE threads
  dim3 blockDim(BLOCKSIZE * BLOCKSIZE);
  // Grid dimensions: one block per output tile
  dim3 gridDim(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE));
  gemm_smem_blocking<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, A, B, C);
}