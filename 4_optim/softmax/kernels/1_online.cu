// Softmax Online Implementation
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
 * Online softmax kernel implementation
 * Optimized version that reduces passes from 3 to 2 using exponential properties
 * 
 * Key optimization:
 * - Combines max finding and sum computation into a single pass
 * - Uses property: exp(x - m_new) = exp(x - m_old) * exp(m_old - m_new)
 * - When finding a new maximum, scales existing sum by exp(m_old - m_new)
 * 
 * Algorithm:
 * 1. Single pass: Compute max and sum of exponentials simultaneously
 *    - When encountering larger value, scale existing sum
 *    - Continuously update running sum
 * 2. Second pass: Normalize each element by the final sum
 * 
 * Performance characteristics:
 * - One thread processes one entire row
 * - Parallelizes over rows
 * - Reduces memory passes from 3 to 2
 * 
 * @param matd Input matrix (M×N, device memory)
 * @param resd Output matrix (M×N, device memory)
 * @param M Number of rows (batch size)
 * @param N Number of columns (feature dimension)
 */
__global__ void softmax_kernel_1(float* __restrict__ matd, float* __restrict__ resd, int M, int N) {
    // Calculate row index for this thread
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M) {
        float m = -1 * INFINITY;  // Current maximum value
        float L = 0.0f;           // Running sum of exponentials

        // Single pass: compute max and sum simultaneously
        // Uses exponential property to update sum when max changes
        for (int col = 0; col < N; col++) {
            int i = row * N + col;
            float curr = matd[i];
            if (curr > m) {
                // Scale existing sum when we find a new maximum
                // exp(x - m_new) = exp(x - m_old) * exp(m_old - m_new)
                L = L * expf(m - curr);
                m = curr;
            }
            // Add exponential of current value (shifted by current max)
            L += expf(curr - m);
        }
        
        // Second pass: normalize each element by final sum
        for (int col = 0; col < N; col++) {
            int i = row * N + col;
            resd[i] = expf(matd[i] - m) / L;
        }
    }
}

/**
 * Launcher function for online softmax kernel
 * Configures and launches the optimized 2-pass softmax implementation
 * 
 * @param matd Input matrix (M×N, device memory)
 * @param resd Output matrix (M×N, device memory)
 * @param M Number of rows (batch size)
 * @param N Number of columns (feature dimension)
 */
void run_kernel_1(float* __restrict__ matd, float* __restrict__ resd, int M, int N) {
    // Configure kernel launch parameters
    dim3 block_size(1024);  // Maximum threads per block
    dim3 grid_size(CEIL_DIV(M, block_size.x));  // One block per row

    // Launch online softmax kernel
    softmax_kernel_1<<<grid_size, block_size>>>(matd, resd, M, N);
}