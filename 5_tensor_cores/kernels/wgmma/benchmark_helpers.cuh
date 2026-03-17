
/**
 * @file benchmark_helpers.cuh
 * @brief Helper structures for benchmarking WGMMA kernels
 * 
 * Provides utilities for managing matrix memory allocation and layout conversion
 * during benchmarking. Encapsulates common operations needed for performance testing.
 */

namespace wgmma_benchmark {

using fp16 = __half;

/**
 * @brief Structure to manage matrix buffers for benchmarking
 * 
 * Maintains column-major buffers for A, B, and C matrices.
 * Handles allocation, layout conversion, and cleanup automatically.
 * Useful for repeated benchmark runs with same matrix dimensions.
 */
struct PreparedMatrices {
    fp16 *A_col = nullptr;
    fp16 *B_col = nullptr;
    fp16 *C_col = nullptr;
    int M = 0, N = 0, K = 0;
    
    ~PreparedMatrices() {
        cleanup();
    }
    
    void cleanup() {
        if (A_col) cudaFree(A_col);
        if (B_col) cudaFree(B_col);
        if (C_col) cudaFree(C_col);
        A_col = B_col = C_col = nullptr;
        M = N = K = 0;
    }
    
    /**
     * @brief Prepares matrices for WGMMA computation
     * @param m Number of rows in A and C
     * @param n Number of columns in B and C
     * @param k Number of columns in A and rows in B
     * @param A Input matrix A (row-major)
     * @param B Input matrix B (row-major)
     * @return true if preparation successful, false otherwise
     * 
     * Allocates buffers if needed and converts A, B to column-major format.
     * Reuses buffers if dimensions haven't changed.
     */
    bool prepare(int m, int n, int k, fp16* A, fp16* B) {
        
        if (M != m || N != n || K != k) {
            cleanup();
        }
        
        M = m; N = n; K = k;
        
        size_t sizeA = static_cast<size_t>(M) * K * sizeof(fp16);
        size_t sizeB = static_cast<size_t>(K) * N * sizeof(fp16);
        size_t sizeC = static_cast<size_t>(M) * N * sizeof(fp16);
        
        
        if (!A_col && cudaMalloc(&A_col, sizeA) != cudaSuccess) return false;
        if (!B_col && cudaMalloc(&B_col, sizeB) != cudaSuccess) return false;
        if (!C_col && cudaMalloc(&C_col, sizeC) != cudaSuccess) return false;
        
        
        cudaMemcpy(A_col, A, sizeA, cudaMemcpyDeviceToDevice);
        wgmma_layout::row_to_col(B, B_col, K, N);
        cudaDeviceSynchronize();
        
        return true;
    }
    
    /**
     * @brief Converts output matrix C back to row-major layout
     * @param C Output buffer (row-major) to write results to
     * 
     * Converts C_col (column-major) back to C (row-major) after computation.
     */
    void convert_output_back(fp16* C) {
        wgmma_layout::col_to_row(C_col, C, M, N);
        cudaDeviceSynchronize();
    }
};

} 

