// GEMV cuBLAS Baseline Implementation

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

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
 * cuBLAS matrix-vector multiplication implementation
 * High-performance baseline using NVIDIA's optimized cuBLAS library
 * 
 * This function uses cuBLAS SGEMM (Single-precision General Matrix-Vector) operation
 * to compute: result = matrix^T * vector
 * 
 * Performance characteristics:
 * - Uses highly optimized cuBLAS library routines
 * - Static handle reuse to avoid overhead (~0.4ms overhead per create/destroy)
 * - Handle is created once on first use and reused for all subsequent calls
 * - Provides the best performance baseline for comparison
 * 
 * @param matd Input matrix (M×N, device memory, accessed as transposed)
 * @param vecd Input vector (N, device memory)
 * @param resd Output vector (M, device memory)
 * @param M Number of rows in matrix (after transpose)
 * @param N Number of columns in matrix
 */
void run_kernel_0(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N) {
    // Static handle for reuse across function calls
    // Avoids overhead of creating/destroying handle (~0.4ms per call)
    static cublasHandle_t handle = nullptr;
    
    // Create handle on first call only
    if (handle == nullptr) {
        cublasCreate(&handle);
    }

    // cuBLAS SGEMM parameters
    // CUBLAS_OP_T: Transpose matrix A (matd)
    // Alpha = 1.0, Beta = 0.0: result = 1.0 * A^T * x + 0.0 * result
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemv(handle, CUBLAS_OP_T, N, M, &alpha, matd, N, vecd, 1, &beta, resd, 1);
}
