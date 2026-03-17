/**
 * @file 1_naive.cuh
 * @brief Naive GEMM kernel implementation - baseline for optimization comparison
 * 
 * This kernel implements the most basic matrix multiplication C = A * B
 * without any optimizations. It serves as the baseline to demonstrate
 * the performance impact of subsequent optimizations.
 * 
 * Optimization Journey:
 * Kernel 0: cuBLAS (highly optimized library)
 * Kernel 1: Naive (this file) - baseline custom implementation
 * Kernel 2: Memory coalescing optimization
 * Kernel 3: Shared memory blocking
 * Kernel 4: 1D block tiling
 * Kernel 5: 2D block tiling
 * Kernel 6: Vectorized memory access
 */

typedef __half fp16;

/**
 * @brief Naive GEMM kernel - baseline implementation
 * 
 * Computes C = A * B where:
 * - A is M×K matrix
 * - B is K×N matrix  
 * - C is M×N matrix
 * 
 * Performance characteristics:
 * - Each thread computes one output element C[x][y]
 * - Direct global memory access for all reads
 * - No memory coalescing (poor cache utilization)
 * - Sequential reduction loop within each thread
 * - No reuse of loaded data (each element loaded once per thread)
 * 
 * Memory access pattern:
 * - Thread(x,y) reads A[x][:] (row-major, strided access)
 * - Thread(x,y) reads B[:][y] (column-major, poor coalescing)
 * - Each element of B is read M times by different threads
 * 
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param A Input matrix A (M×K, device memory, row-major)
 * @param B Input matrix B (K×N, device memory, row-major)
 * @param C Output matrix C (M×N, device memory, row-major)
 */
__global__ void gemm_naive(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C) {
  // Calculate output element position (x, y) for this thread
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // Bounds check to prevent out-of-bounds access
  if (x < M && y < N) {
    // Initialize accumulator for dot product
    fp16 tmp = __float2half(0.0f);
    
    // Compute dot product: sum over k of A[x][k] * B[k][y]
    for (int i = 0; i < K; ++i) {
      // Accumulate: tmp += A[x][i] * B[i][y]
      // Note: A accessed row-wise (coalesced), B accessed column-wise (not coalesced)
      tmp = __hadd(tmp, __hmul(A[x * K + i], B[i * N + y]));
    }
    // Write result to output matrix
    C[x * N + y] = tmp;
  }
}

/**
 * @brief Launcher function for naive GEMM kernel
 * 
 * Configures kernel launch parameters and invokes the naive kernel.
 * Uses a 2D thread block layout (32×32 = 1024 threads per block).
 * 
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param A Input matrix A (M×K, device memory)
 * @param B Input matrix B (K×N, device memory)
 * @param C Output matrix C (M×N, device memory)
 */
void runKernel1(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C) {
  // 2D thread block: 32×32 = 1024 threads per block
  dim3 blockDim(32, 32);
  // Grid dimensions: ceil(M/32) × ceil(N/32) blocks
  dim3 gridDim((M + 31) / 32, (N + 31) / 32);
  gemm_naive<<<gridDim, blockDim>>>(M, N, K, A, B, C);
}