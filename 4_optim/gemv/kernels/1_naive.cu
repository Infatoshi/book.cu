// GEMV Naive Implementation
// Based in part on Maharshi Pandya's CUDA optimization blog (Apache-2.0 license)
// https://github.com/Maharshi-Pandya/cuda-mode-resource-stream

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

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

/**
 * Naive SGEMM kernel for matrix-vector multiplication
 * Computes: result = matrix * vector
 * 
 * Performance characteristics:
 * - Each thread calculates one element of the output vector
 * - The row index is calculated using block index and thread index
 * - Uses linearized indexing
 * - Memory accesses are not coalesced (poor performance)
 * 
 * @param matd Input matrix (M×N, device memory)
 * @param vecd Input vector (N, device memory)
 * @param resd Output vector (M, device memory)
 * @param M Number of rows in matrix and size of output vector
 * @param N Number of columns in matrix and size of input vector
 */
__global__ void naive_sgemv_kernel(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N) {
    // Calculate global thread index across all blocks
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    // Bounds check to ensure we don't access out-of-range elements
    if (row < M) {
        float sum = 0.0f;
        // Compute dot product of matrix row and input vector
        for (int col = 0; col < N; col++) {
            sum += matd[row * N + col] * vecd[col];
        }
        // Store result in output vector
        resd[row] = sum;
    }
}

/**
 * CUDA wrapper function for naive SGEMM kernel
 * Launches the kernel with appropriate grid and block dimensions
 * 
 * @param matd Input matrix (M×N, device memory)
 * @param vecd Input vector (N, device memory)
 * @param resd Output vector (M, device memory)
 * @param M Number of rows in matrix and size of output vector
 * @param N Number of columns in matrix and size of input vector
 */
void run_kernel_1(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N) {
    // Configure kernel launch parameters
    dim3 block_size(1024);  // Maximum threads per block
    dim3 grid_size(CEIL_DIV(M, block_size.x));  // Number of blocks needed

    // Launch CUDA kernel
    naive_sgemv_kernel<<<grid_size, block_size>>>(matd, vecd, resd, M, N);
}
