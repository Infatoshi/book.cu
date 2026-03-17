// GEMV Coalesced Warp+Block Implementation
// Based in part on Maharshi Pandya's CUDA optimization blog (Apache-2.0 license)
// https://github.com/Maharshi-Pandya/cuda-mode-resource-stream

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>

// Ceiling division macro
#ifndef CEIL_DIV
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#endif

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
 * @brief Warp-level sum reduction using shuffle operations
 * 
 * Efficiently reduces values across all threads in a warp (32 threads)
 * using CUDA shuffle intrinsics for low-latency communication.
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

/**
 * @brief Block-level sum reduction using shared memory
 * 
 * Reduces values across all threads in a block using a two-phase approach:
 * 1. Reduce within each warp using shuffle operations
 * 2. Store warp results in shared memory
 * 3. Reduce warp results using another warp-level reduction
 * 
 * This provides efficient reduction for blocks larger than a single warp.
 * 
 * @tparam T Type of value to reduce (float in this case)
 * @param val Input value to reduce
 * @param smem Shared memory array for intermediate results
 * @param tid Thread index within block
 * @param block_size Number of threads in block
 */
template<typename T>
__device__ __forceinline__ void blockReduceSum(T val, T* smem, int tid, int block_size) {
    int warp_size = 32;
    // Phase 1: Reduce within warp using shuffle operations
    val = warpReduceSum(val);
    
    // Phase 2: Store warp results in shared memory
    // Each warp writes its result to shared memory
    if (tid % warp_size == 0) smem[tid / warp_size] = val;
    __syncthreads();
    
    // Phase 3: Load warp results for final reduction
    if (tid < CEIL_DIV(block_size, warp_size)) {
        val = smem[tid];
    } else {
        val = 0.0f;
    }
    
    // Phase 4: Final reduction in first warp
    if (tid / warp_size == 0) {
        val = warpReduceSum(val);
    }
    
    // Phase 5: Store final result in shared memory
    if (tid == 0) smem[0] = val;
    __syncthreads();
}
} 

/**
 * @brief Coalesced warp+block GEMV kernel for matrix-vector multiplication
 * 
 * Optimized version using block-level parallelism with coalesced memory access.
 * 
 * Key optimizations:
 * - Each block processes one row of the matrix
 * - Each block calculates one output element
 * - Columns are accessed in coalesced manner by threads (stride-pattern)
 * - Performs two-level reduction: warp-level then block-level
 * - Uses shared memory for efficient block-level communication
 * 
 * Memory access pattern:
 * - Threads access consecutive columns, enabling coalesced memory reads
 * - Each thread processes multiple columns with stride = blockDim.x
 * - Vector elements accessed in coalesced pattern
 * 
 * Reduction strategy:
 * - Each thread computes partial sum over its assigned columns
 * - Warp-level reduction using shuffle operations (fast, no shared memory)
 * - Block-level reduction using shared memory (for multiple warps)
 * - Final result stored by thread 0
 * 
 * @param matd Input matrix (M×N, device memory)
 * @param vecd Input vector (N, device memory)
 * @param resd Output vector (M, device memory)
 * @param M Number of rows in matrix
 * @param N Number of columns in matrix
 */
__global__ void coalesced_warpblock_sgmev_kernel(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N) {
    // Shared memory for block-level reduction
    extern __shared__ float smem[];

    // Block index: which row this block processes
    int bid = blockIdx.x;
    if (bid >= M) return;

    // Thread index within block
    int tid = threadIdx.x;
    
    // Compute partial sum using coalesced memory access pattern
    // Each thread processes columns at stride = blockDim.x
    float partial_sum = 0.f;
    for (int col = tid; col < N; col += blockDim.x) {
        // Coalesced access: threads access consecutive memory locations
        partial_sum += matd[bid * N + col] * vecd[col];
    }

    // Perform block-level reduction (warp-level + block-level)
    // This reduces partial sums from all threads in the block
    blockReduceSum(partial_sum, smem, tid, blockDim.x);
    
    // Thread 0 writes the final result
    if (tid == 0) {
        float sum = smem[0];
        resd[bid] = sum;
    }
}

/**
 * @brief Launcher function for coalesced warp+block GEMV kernel
 * 
 * Configures and launches the kernel with block-level reduction.
 * Uses larger blocks (64 threads) for better GPU occupancy.
 * 
 * @param matd Input matrix (M×N, device memory)
 * @param vecd Input vector (N, device memory)
 * @param resd Output vector (M, device memory)
 * @param M Number of rows in matrix
 * @param N Number of columns in matrix
 */
void run_kernel_3(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N) {
    int NUM_THREADS = 64;  // Larger block for better occupancy
    int warp_size = 32;

    // Configure kernel launch parameters
    // One block per row, each block has NUM_THREADS threads
    dim3 block_size(NUM_THREADS);
    dim3 grid_size(M);
    
    // Allocate shared memory for block-level reduction
    // Need space for one float per warp
    size_t shared_mem_size = CEIL_DIV(block_size.x, warp_size) * sizeof(float);

    // Launch CUDA kernel with shared memory
    coalesced_warpblock_sgmev_kernel<<<grid_size, block_size, shared_mem_size>>>(matd, vecd, resd, M, N);
}
