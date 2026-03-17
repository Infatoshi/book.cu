// Softmax Shared Memory Implementation
// Based in part on Maharshi Pandya's CUDA optimization blog (Apache-2.0 license)
// https://github.com/Maharshi-Pandya/cuda-mode-resource-stream

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Ceiling division macro
#ifndef CEIL_DIV
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#endif

/**
 * Warp-level sum reduction using shuffle operations
 * Efficiently reduces values across all threads in a warp (32 threads)
 * 
 * @param val Input value to reduce
 * @return Sum of all values in the warp
 */
static __device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Warp-level max reduction using shuffle operations
 * Efficiently finds maximum value across all threads in a warp
 * 
 * @param val Input value to reduce
 * @return Maximum value in the warp
 */
static __device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

/**
 * Block-level sum reduction using shared memory
 * Reduces values across all threads in a block
 * 
 * @param val Input value to reduce
 * @param smem Shared memory array for intermediate results
 */
template<typename T>
static __device__ __forceinline__ void blockReduceSum(T val, T* smem) {
    int tid = threadIdx.x;
    int warp_size = 32;
    // First, reduce within warp
    val = warpReduceSum(val);
    // Store warp results in shared memory
    if (tid % warp_size == 0) smem[tid / warp_size] = val;
    __syncthreads();
    // Load warp results for final reduction
    if (tid < CEIL_DIV(blockDim.x, warp_size)) {
        val = smem[tid];
    } else {
        val = 0.0f;
    }
    // Final reduction in first warp
    if (tid / warp_size == 0) {
        val = warpReduceSum(val);
    }
    // Store final result in shared memory
    if (tid == 0) smem[0] = val;
    __syncthreads();
}

/**
 * Block-level max reduction using shared memory
 * Finds maximum value across all threads in a block
 * 
 * @param val Input value to reduce
 * @param smem Shared memory array for intermediate results
 * @param identity Identity value for max reduction (typically -INFINITY)
 */
template<typename T>
static __device__ __forceinline__ void blockReduceMax(T val, T* smem, T identity) {
    int tid = threadIdx.x;
    int warp_size = 32;
    // First, reduce within warp
    val = warpReduceMax(val);
    // Store warp results in shared memory
    if (tid % warp_size == 0) smem[tid / warp_size] = val;
    __syncthreads();
    // Load warp results for final reduction
    if (tid < CEIL_DIV(blockDim.x, warp_size)) {
        val = smem[tid];
    } else {
        val = identity;
    }
    // Final reduction in first warp
    if (tid / warp_size == 0) {
        val = warpReduceMax(val);
    }
    // Store final result in shared memory
    if (tid == 0) smem[0] = val;
    __syncthreads();
}

/**
 * Shared memory softmax kernel implementation
 * Optimized version using block-level parallelism and shared memory reductions
 * 
 * Key optimizations:
 * - Each block processes one row of the matrix
 * - Threads within block work together to process one row
 * - Uses shared memory for efficient block-level reductions
 * - Computes max and normalization factor in one pass
 * 
 * Algorithm:
 * 1. Each thread processes a subset of elements in the row
 * 2. Threads compute local max and local normalization factor
 * 3. Block-level reduction to compute final max and normalization factor
 * 4. Second pass: normalize each element using final values
 * 
 * Performance characteristics:
 * - Better GPU utilization than naive version
 * - Parallelizes both across rows and within rows
 * - Uses shared memory for efficient communication
 * 
 * @param xd Input matrix (M×N, device memory)
 * @param resd Output matrix (M×N, device memory)
 * @param M Number of rows (batch size)
 * @param N Number of columns (feature dimension)
 */
__global__ void softmax_kernel_2(float* __restrict__ xd, float* __restrict__ resd, int M, int N) {
    // Shared memory for block-level reductions
    __shared__ float smem[1024];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Bounds check
    if (row >= M) return;

    // Get pointers to current row
    float* input_row = xd + row * N;
    float* output_row = resd + row * N;
    float local_max = -INFINITY;
    float local_norm = 0.0f;

    // First pass: each thread processes subset of elements
    // Computes local max and local normalization factor
    for (int i = tid; i < N; i += blockDim.x) {
        float x = input_row[i];
        if (x > local_max) {
            // Scale existing sum when finding new max
            local_norm *= expf(local_max - x);
            local_max = x;
        }
        local_norm += expf(x - local_max);
    }
    __syncthreads();

    // Block-level max reduction
    // Store local maxes in shared memory
    smem[tid] = local_max;
    __syncthreads();

    // Perform reduction tree in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            smem[tid] = max(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }

    // Final max value for the row
    float row_max = smem[0];
    __syncthreads();

    // Block-level sum reduction for normalization factor
    // Scale local norms by exp(local_max - row_max) to account for final max
    smem[tid] = local_norm * expf(local_max - row_max);
    __syncthreads();

    // Perform reduction tree for sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    float row_norm = smem[0];
    __syncthreads();

    // Second pass: normalize each element using final max and norm
    for (int i = tid; i < N; i += blockDim.x) {
        output_row[i] = expf(input_row[i] - row_max) / row_norm;
    }
}

/**
 * Launcher function for shared memory softmax kernel
 * Configures and launches the block-parallel softmax implementation
 * 
 * @param matd Input matrix (M×N, device memory)
 * @param resd Output matrix (M×N, device memory)
 * @param M Number of rows (batch size)
 * @param N Number of columns (feature dimension)
 */
void run_kernel_2(float* __restrict__ matd, float* __restrict__ resd, int M, int N) {
    // Configure kernel launch parameters
    dim3 block_size(1024);  // Maximum threads per block
    dim3 grid_size(M);      // One block per row

    // Launch shared memory softmax kernel
    softmax_kernel_2<<<grid_size, block_size>>>(matd, resd, M, N);
}