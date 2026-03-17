// GEMV Coalesced Warp Implementation
// Based in part on Maharshi Pandya's CUDA optimization blog (Apache-2.0 license)
// https://github.com/Maharshi-Pandya/cuda-mode-resource-stream

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>

/**
 * CUDA error checking macro
 * Checks CUDA function calls for errors and exits on failure
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

namespace {
/**
 * Warp-level sum reduction using shuffle operations
 * Efficiently reduces values across all threads in a warp (32 threads)
 * Uses CUDA shuffle intrinsics for low-latency communication
 * 
 * @param val Input value to reduce
 * @return Sum of all values in the warp
 */
__device__ __forceinline__ float warpReduceSum(float val) {
    // Perform reduction in log2(32) = 5 steps
    // Each step halves the offset distance
    for (int offset = 16; offset > 0; offset /= 2) {
        // Shuffle down: get value from thread (current + offset)
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
} 

/**
 * Coalesced Warp SGEMM kernel for matrix-vector multiplication
 * Optimized version using warp-level parallelism and coalesced memory access
 * 
 * Key optimizations:
 * - Each block processes one row of the matrix
 * - Each block calculates one output element
 * - Columns are accessed in coalesced manner by threads (stride-pattern)
 * - Performs warp-level sum reduction only (no block-level reduction needed)
 * - Block size must equal warp size (32 threads)
 * 
 * Memory access pattern:
 * - Threads access consecutive columns, enabling coalesced memory reads
 * - Each thread processes multiple columns with stride = blockDim.x
 * 
 * @param matd Input matrix (M×N, device memory)
 * @param vecd Input vector (N, device memory)
 * @param resd Output vector (M, device memory)
 * @param M Number of rows in matrix
 * @param N Number of columns in matrix
 */
__global__ void coalesced_warp_sgmev_kernel(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N) {
    // Ensure block size equals warp size for optimal performance
    assert(blockDim.x == warpSize);

    // Get block index (one block per row)
    int bid = blockIdx.x;
    if (bid >= M) return;

    // Get thread index within block
    int tid = threadIdx.x;
    
    // Compute partial sum using coalesced memory access pattern
    // Each thread processes columns at stride = blockDim.x
    float partial_sum = 0.f;
    for (int col = tid; col < N; col += blockDim.x) {
        partial_sum += matd[bid * N + col] * vecd[col];
    }

    // Perform warp-level reduction to get final sum
    // Thread 0 will have the final result
    float sum = warpReduceSum(partial_sum);
    if (tid == 0) {
        resd[bid] = sum;
    }
}

/**
 * Launcher function for coalesced warp SGEMM kernel
 * Configures and launches the kernel with warp-sized blocks
 * 
 * @param matd Input matrix (M×N, device memory)
 * @param vecd Input vector (N, device memory)
 * @param resd Output vector (M, device memory)
 * @param M Number of rows in matrix
 * @param N Number of columns in matrix
 */
void run_kernel_2(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N) {
    // Use warp size (32 threads) for optimal performance
    int NUM_THREADS = 32;

    // Configure kernel launch parameters
    // One block per row, each block has warp size threads
    dim3 block_size(NUM_THREADS);
    dim3 grid_size(M);

    // Launch CUDA kernel
    coalesced_warp_sgmev_kernel<<<grid_size, block_size>>>(matd, vecd, resd, M, N);
}
