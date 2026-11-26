// Softmax Naive Implementation
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
 * Naive softmax kernel implementation
 * This kernel implements a basic softmax operation on a matrix of size (M, N)
 * The softmax operation is performed on the last dimension (columns) of the matrix
 * 
 * Performance characteristics:
 * - One thread processes one entire row
 * - Only parallelizes over rows, not exploiting full GPU parallelism
 * - Sequential computation within each thread for max, sum, and normalization
 * - This is the baseline implementation for comparison with optimized versions
 * 
 * @param matd Input matrix (M×N, device memory)
 * @param resd Output matrix (M×N, device memory) 
 * @param M Number of rows (batch size)
 * @param N Number of columns (feature dimension)
 */
__global__ void softmax_kernel_0(float* __restrict__ matd, float* __restrict__ resd, int M, int N) {
    // Calculate row index for this thread
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    // Bounds check
    if (row < M) {
        // Step 1: Find maximum value in the row (for numerical stability)
        float m = -1 * INFINITY;
        
        // Step 2: Compute sum of exponentials (shifted by max for stability)
        float L = 0.0f;

        // First pass: find maximum
        for (int col = 0; col < N; col++) {
            int i = row * N + col;
            m = max(m, matd[i]);
        }
        
        // Second pass: compute sum of exponentials
        for (int col = 0; col < N; col++) {
            int i = row * N + col;
            L += expf(matd[i] - m);
        }
        
        // Third pass: compute softmax probabilities
        for (int col = 0; col < N; col++) {
            int i = row * N + col;
            resd[i] = expf(matd[i] - m) / L;
        }
    }
}

/**
 * Launcher function for naive softmax kernel
 * Configures and launches the baseline softmax implementation
 * 
 * @param matd Input matrix (M×N, device memory)
 * @param resd Output matrix (M×N, device memory)
 * @param M Number of rows (batch size)
 * @param N Number of columns (feature dimension)
 */
void run_kernel_0(float* __restrict__ matd, float* __restrict__ resd, int M, int N) {
    // Configure kernel launch parameters
    dim3 block_size(1024);  // Maximum threads per block
    dim3 grid_size(CEIL_DIV(M, block_size.x));  // One block per row

    // Launch naive softmax kernel
    softmax_kernel_0<<<grid_size, block_size>>>(matd, resd, M, N);
}