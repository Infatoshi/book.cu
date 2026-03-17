/**
 * @file 2_gmem_coalesce.cuh
 * @brief GEMM kernel with global memory coalescing optimization
 * 
 * This kernel improves upon the naive implementation by reorganizing
 * thread-to-data mapping to enable memory coalescing for better memory bandwidth.
 * 
 * Optimization Journey:
 * Kernel 1: Naive - basic implementation, poor memory access
 * Kernel 2: Memory coalescing (this file) - improved memory access pattern
 * 
 * Key Optimization:
 * - Reorganizes thread mapping so consecutive threads access consecutive
 *   memory locations, enabling memory coalescing
 * - Uses 1D thread blocks instead of 2D to better control memory access pattern
 * - Better cache utilization through improved memory access locality
 */

typedef __half fp16;

/**
 * @brief GEMM kernel with global memory coalescing optimization
 * 
 * Computes C = A * B with improved memory access pattern for coalescing.
 * 
 * Optimization technique:
 * - Maps threads to output elements in a way that enables coalesced memory reads
 * - Consecutive threads (threadIdx.x) access consecutive memory locations
 * - 1D thread block layout allows better control over memory access pattern
 * 
 * Memory access pattern:
 * - Threads in a warp access consecutive columns of B, enabling coalesced reads
 * - Still uses global memory (no shared memory yet)
 * - Better bandwidth utilization than naive kernel
 * 
 * Thread-to-output mapping:
 * - Thread with threadIdx.x maps to output element:
 *   row = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE)
 *   col = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE)
 * - This ensures threads 0,1,2... access consecutive columns within a block
 * 
 * @tparam BLOCKSIZE Size of the tile processed by each block (typically 32)
 * 
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param A Input matrix A (M×K, device memory, row-major)
 * @param B Input matrix B (K×N, device memory, row-major)
 * @param C Output matrix C (M×N, device memory, row-major)
 */
template <const uint BLOCKSIZE>
__global__ void gemm_gmem_coalesce(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C) {
  // Map thread index to output element position
  // Ensures consecutive threads access consecutive columns for coalescing
  const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  // Bounds check
  if (cRow < M && cCol < N) {
    // Initialize accumulator
    fp16 tmp = __float2half(0.0f);
    
    // Compute dot product - same as naive, but with better memory access pattern
    for (int i = 0; i < K; ++i) {
      // Accumulate: tmp += A[cRow][i] * B[i][cCol]
      tmp = __hadd(tmp, __hmul(A[cRow * K + i], B[i * N + cCol]));
    }
    // Write result
    C[cRow * N + cCol] = tmp;
  }
}

/**
 * @brief Launcher function for coalesced GEMM kernel
 * 
 * Configures kernel launch parameters for coalesced memory access.
 * Uses 1D thread blocks of size BLOCKSIZE×BLOCKSIZE threads.
 * 
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param A Input matrix A (M×K, device memory)
 * @param B Input matrix B (K×N, device memory)
 * @param C Output matrix C (M×N, device memory)
 */
void runKernel2(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C) {
  const uint BLOCKSIZE = 32;
  // 1D thread block: BLOCKSIZE×BLOCKSIZE threads
  dim3 blockDim(BLOCKSIZE * BLOCKSIZE);
  // Grid dimensions: ceil(M/BLOCKSIZE) × ceil(N/BLOCKSIZE) blocks
  dim3 gridDim((M + BLOCKSIZE - 1) / BLOCKSIZE, (N + BLOCKSIZE - 1) / BLOCKSIZE);
  gemm_gmem_coalesce<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, A, B, C);
}