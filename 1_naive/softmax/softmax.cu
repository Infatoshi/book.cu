#include "common.h"

/**
 * @file softmax.cu
 * @brief Naive CUDA implementation of softmax activation function
 * 
 * This file demonstrates row-wise reductions and numerical stability techniques.
 * Softmax is fundamental to many neural network operations including:
 * - Classification layers: converts logits to probability distributions
 * - Attention mechanisms: normalizes attention scores in transformers
 * - Multi-head attention: applies softmax to attention weights
 * 
 * Softmax Formula: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
 * 
 * Key Properties:
 * - Output sums to 1.0 for each row (probability distribution)
 * - Values are non-negative
 * - Numerical stability: subtracting max prevents overflow
 * 
 * CPU Implementation: Sequential three-pass algorithm per row
 * GPU Implementation: Each thread redundantly computes max and sum (highly inefficient)
 * 
 * Algorithm Steps:
 * 1. Find maximum value in row (for numerical stability)
 * 2. Compute sum of exp(x_i - max) for all elements in row
 * 3. Compute softmax(x_i) = exp(x_i - max) / sum for each element
 * 
 * Memory Access Pattern:
 * - CPU: Sequential row-major access (cache-friendly)
 * - GPU: Each thread reads entire row multiple times (extremely inefficient)
 * 
 * Performance Characteristics:
 * - CPU: O(M*N) time complexity, sequential processing
 * - GPU Naive: O(M*N*N) work per thread (extremely inefficient)
 * - Optimized GPU: Use parallel reductions for max and sum (O(M*N) work)
 */

/**
 * CPU implementation of softmax function
 * 
 * Applies softmax normalization to each row independently.
 * Uses numerically stable implementation by subtracting max before exponentiation.
 * 
 * Algorithm:
 * For each row:
 *   1. Find maximum value in row (max reduction)
 *   2. Compute sum of exp(x_i - max) for all elements (sum reduction)
 *   3. Compute softmax(x_i) = exp(x_i - max) / sum for each element
 * 
 * Numerical Stability:
 * - Subtracting max prevents exp() from overflowing with large inputs
 * - Formula: softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
 * - This is mathematically equivalent to exp(x_i) / sum(exp(x_j))
 * - But avoids numerical overflow issues
 * 
 * Time Complexity: O(M * N) where M = num_rows, N = num_cols
 * Space Complexity: O(1) excluding input/output matrices
 * 
 * @param in Input matrix (host memory, row-major order, size num_rows × num_cols)
 *           Contains logits (pre-softmax values)
 * @param out Output matrix (host memory, row-major order, size num_rows × num_cols)
 *            Stores softmax probabilities: out[i,j] = exp(in[i,j] - max) / sum(exp(in[i,:] - max))
 *            Each row sums to 1.0 (probability distribution)
 * @param num_rows Number of rows (batch size, must be >= 0)
 * @param num_cols Number of columns (feature dimension, must be >= 0)
 */
void softmax_cpu(const float* in, float* out, int num_rows, int num_cols) {
    // Process each row independently
    // Each row is normalized separately to form a probability distribution
    for (int row = 0; row < num_rows; ++row) {
        // Step 1: Find maximum value in the row (for numerical stability)
        // This is a reduction operation: max over all columns in this row
        // Initialize with first element, then compare with remaining elements
        float max_val = in[row * num_cols];
        for (int col = 1; col < num_cols; ++col) {
            if (in[row * num_cols + col] > max_val) {
                max_val = in[row * num_cols + col];
            }
        }

        // Step 2: Compute sum of exponentials (shifted by max for stability)
        // This is a reduction operation: sum over all columns in this row
        // Subtracting max prevents exp() from overflowing
        float sum_exp = 0.0f;
        for (int col = 0; col < num_cols; ++col) {
            sum_exp += expf(in[row * num_cols + col] - max_val);
        }

        // Step 3: Compute softmax probabilities for each element in row
        // Formula: softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
        // Result: Each row sums to 1.0 (probability distribution)
        for (int col = 0; col < num_cols; ++col) {
            out[row * num_cols + col] = expf(in[row * num_cols + col] - max_val) / sum_exp;
        }
    }
}

/**
 * Naive CUDA kernel for softmax function
 * 
 * This is a HIGHLY INEFFICIENT implementation that demonstrates the naive approach.
 * Each thread computes one output element but redundantly computes max and sum for the entire row.
 * 
 * Performance Issues:
 * - Redundant computation: All threads in same row compute same max and sum
 * - Work complexity: O(M*N*N) instead of O(M*N)
 * - Memory access: Each thread reads entire row multiple times
 * - No parallel reductions: Sequential loops within each thread
 * 
 * Algorithm (same as CPU but inefficient):
 * - Each thread computes one output element
 * - Thread at (row, column) redundantly computes:
 *   1. Max of entire row (same for all threads in row)
 *   2. Sum of exp(row - max) for entire row (same for all threads in row)
 *   3. Softmax value for one element
 * 
 * Thread Indexing (2D Grid):
 * - Grid dimensions: (blocksPerGrid.x, blocksPerGrid.y) covering output matrix
 * - Block dimensions: (blockDim.x, blockDim.y)
 * - Row index: row = blockIdx.y * blockDim.y + threadIdx.y
 * - Column index: column = blockIdx.x * blockDim.x + threadIdx.x
 * 
 * Optimization Opportunities:
 * - Parallel reduction for max: Use warp-level or block-level reductions
 * - Parallel reduction for sum: Use warp-level or block-level reductions
 * - Shared memory: Cache row data to reduce global memory accesses
 * - Two-pass kernel: First kernel computes max and sum, second computes softmax
 * 
 * @param in Input matrix (device memory, row-major order, size num_rows × num_cols)
 *           Contains logits (pre-softmax values)
 * @param out Output matrix (device memory, row-major order, size num_rows × num_cols)
 *            Stores softmax probabilities: out[i,j] = exp(in[i,j] - max) / sum(exp(in[i,:] - max))
 *            Each row sums to 1.0 (probability distribution)
 * @param num_rows Number of rows (batch size, must be >= 0)
 * @param num_cols Number of columns (feature dimension, must be >= 0)
 */
__global__ void softmax_naive_kernel(const float* in, float* out, int num_rows, int num_cols) {
    // Calculate 2D coordinates from thread indices
    // Y-dimension (rows): maps to blockIdx.y and threadIdx.y
    // Formula: row = block_index_y * threads_per_block_y + thread_index_y
    int row = blockIdx.y * blockDim.y + threadIdx.y;    // Row index
    
    // X-dimension (columns): maps to blockIdx.x and threadIdx.x
    // Formula: column = block_index_x * threads_per_block_x + thread_index_x
    int column = blockIdx.x * blockDim.x + threadIdx.x; // Column index

    // Bounds check: critical for correctness when dimensions are not divisible by block size
    // Checks both row and column bounds independently
    // Without this check, threads beyond matrix boundaries would access invalid memory
    if (row < num_rows && column < num_cols) {
        // Step 1: Find maximum value in the row (REDUNDANT computation per thread)
        // Problem: All threads in same row compute same max independently
        // Should use parallel reduction instead (warp-level or block-level)
        // Time complexity: O(N) per thread, O(M*N*N) total work
        float max_val = -1e20f;  // Initialize to very small value
        for (int col_idx = 0; col_idx < num_cols; ++col_idx) {
            if (in[row * num_cols + col_idx] > max_val) {
                max_val = in[row * num_cols + col_idx];
            }
        }
        
        // Step 2: Compute sum of exponentials (REDUNDANT computation per thread)
        // Problem: All threads in same row compute same sum independently
        // Should use parallel reduction instead (warp-level or block-level)
        // Time complexity: O(N) per thread, O(M*N*N) total work
        // Memory access: Each thread reads entire row from global memory
        float sum_exp = 0.0f;
        for (int col_idx = 0; col_idx < num_cols; ++col_idx) {
            sum_exp += expf(in[row * num_cols + col_idx] - max_val);
        }
        
        // Step 3: Compute softmax probability for this element
        // This is the only computation that is not redundant
        // Formula: softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
        out[row * num_cols + column] = expf(in[row * num_cols + column] - max_val) / sum_exp;
    }
}

int main() {
    const int num_rows = 128;  
    const int num_cols = 1000; 
    const int n = num_rows * num_cols;

    std::cout << "Softmax: " << num_rows << " rows x " << num_cols << " columns = "
              << n << " elements" << std::endl;

    float *h_in, *h_out_cpu, *h_out_gpu;
    allocate_host(&h_in, n);
    allocate_host(&h_out_cpu, n);
    allocate_host(&h_out_gpu, n);

    srand(42); 
    for (int i = 0; i < n; ++i) {
        h_in[i] = static_cast<float>(rand() % 20 - 10); 
    }

    {
        Timer cpu_timer("CPU Softmax");
        softmax_cpu(h_in, h_out_cpu, num_rows, num_cols);
    }

    float *d_in, *d_out;
    allocate_device(&d_in, n);
    allocate_device(&d_out, n);

    copy_to_device(d_in, h_in, n);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (num_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (num_rows + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    std::cout << "GPU: Launching " << blocksPerGrid.x << "x" << blocksPerGrid.y
              << " blocks with " << threadsPerBlock.x << "x" << threadsPerBlock.y
              << " threads per block" << std::endl;

    {
        Timer gpu_timer("GPU Softmax (naive)");
        softmax_naive_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, num_rows, num_cols);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    copy_to_host(h_out_gpu, d_out, n);

    if (verify_results(h_out_gpu, h_out_cpu, n, 1e-5f)) {
        std::cout << "✓ Softmax results match!" << std::endl;

        bool sum_check = true;
        for (int row = 0; row < num_rows && sum_check; ++row) {
            float row_sum = 0.0f;
            for (int col = 0; col < num_cols; ++col) {
                row_sum += h_out_gpu[row * num_cols + col];
            }
            if (std::abs(row_sum - 1.0f) > 1e-5f) {
                std::cout << "Row " << row << " sum: " << row_sum << " (expected ~1.0)" << std::endl;
                sum_check = false;
            }
        }

        if (sum_check) {
            std::cout << "✓ All rows sum to 1.0 (as expected for softmax)" << std::endl;
        }

        std::cout << "\nExample - First row input: ";
        for (int col = 0; col < std::min(5, num_cols); ++col) {
            std::cout << h_in[col] << " ";
        }
        if (num_cols > 5) std::cout << "...";
        std::cout << std::endl;

        std::cout << "Example - First row softmax output: ";
        for (int col = 0; col < std::min(5, num_cols); ++col) {
            std::cout << h_out_gpu[col] << " ";
        }
        if (num_cols > 5) std::cout << "...";
        std::cout << std::endl;
    }

    free_host(h_in);
    free_host(h_out_cpu);
    free_host(h_out_gpu);
    free_device(d_in);
    free_device(d_out);

    return 0;
}
