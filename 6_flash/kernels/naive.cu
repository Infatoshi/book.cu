/**
 * @file naive.cu
 * @brief This file contains a naive, non-optimized implementation of the
 *        standard attention mechanism.
 *
 * @details
 * This implementation serves as a baseline for understanding and comparison
 * against optimized versions like FlashAttention. It follows the standard
 * mathematical definition of scaled dot-product attention directly, without
 * memory-saving optimizations like tiling or online softmax.
 *
 * The key characteristic of this naive approach is that it "materializes" the
 * full N x N attention score matrix in global device memory (HBM). This is
* highly inefficient for long sequences (large N), as the memory required
 * (O(N^2)) quickly becomes a bottleneck, far exceeding the capacity of on-chip
 * SRAM. This leads to slow, memory-bound performance, which FlashAttention
 * is designed to solve.
 *
 * The process is broken into three separate kernel launches:
 * 1. `naive_qk_matmul_kernel`: Computes `S = (Q @ K^T) * scale`.
 * 2. `naive_softmax_kernel`: Applies softmax to the `S` matrix in-place.
 * 3. `naive_sv_matmul_kernel`: Computes `O = S @ V`.
 */
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

/**
 * CUDA error checking macro
 * Checks CUDA function calls and exits on failure with detailed error message
 * @param ans CUDA function call to check
 */
#define CUDA_CHECK(ans) {                                          \
        cudaAssert((ans), __FILE__, __LINE__); \
    }

/**
 * CUDA error assertion function
 * Prints detailed error information and exits on failure
 * @param code CUDA error code
 * @param file Source file name
 * @param line Line number
 */
inline void cudaAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s at %s: %d\n",
                cudaGetErrorName(code), cudaGetErrorString(code),
                file, line);
        exit(code);
    }
}

/**
 * @brief Naive kernel to compute the QK^T matrix multiplication.
 *
 * @tparam BLOCK_SIZE The size of the thread block in each dimension (e.g., 16).
 *
 * @param Q Pointer to the Query matrix (N x d) in global memory.
 * @param K Pointer to the Key matrix (N x d) in global memory.
 * @param S Pointer to the output attention score matrix (N x N) in global memory.
 * @param N The sequence length.
 * @param d The dimension of the attention heads.
 * @param scale The scaling factor (typically 1/sqrt(d)).
 *
 * @details
 * Each thread in the grid is responsible for computing a single element of the
 * output matrix `S`. It does this by computing the dot product of a row from `Q`
 * and a row from `K` (which is equivalent to a column from `K^T`). The result is
 * then scaled and written to global memory. This kernel materializes the full
 * `N x N` score matrix `S`.
 */
template <int BLOCK_SIZE>
__global__ void naive_qk_matmul_kernel(
    float* Q, float* K, float* S,
    int N, int d, float scale
) {
    // Calculate 2D coordinates from thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check
    if (row < N && col < N) {
        float sum = 0.0f;
        // Compute dot product: Q[row] @ K[col]^T
        for (int k = 0; k < d; k++) {
            sum += Q[row * d + k] * K[col * d + k];
        }
        // Apply scaling and store attention score
        S[row * N + col] = sum * scale;
    }
}

/**
 * @brief Naive kernel to apply softmax to the attention score matrix.
 *
 * @param S Pointer to the attention score matrix (N x N) to be normalized in-place.
 * @param N The dimension of the square matrix `S`.
 *
 * @details
 * This kernel applies the softmax function to each row of the input matrix `S`.
 * Each thread is responsible for processing one entire row.
 * The standard numerically stable softmax algorithm is used:
 * 1. Find the maximum value in the row.
 * 2. Subtract the max from each element in the row before exponentiating to
 *    prevent overflow.
 * 3. Compute the sum of the exponentiated values.
 * 4. Divide each exponentiated value by the sum to get the final probabilities.
 * The result is written back to the same memory location (in-place).
 */
__global__ void naive_softmax_kernel(float* S, int N) {
    // Calculate row index for this thread
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N) {
        float* row_ptr = S + row * N;
        
        // Step 1: Find maximum value in the row (for numerical stability)
        float max_val = -INFINITY;
        for (int i = 0; i < N; i++) {
            max_val = fmaxf(max_val, row_ptr[i]);
        }
        
        // Step 2: Compute exponentials and sum
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            row_ptr[i] = expf(row_ptr[i] - max_val);
            sum += row_ptr[i];
        }
        
        // Step 3: Normalize to get probabilities
        for (int i = 0; i < N; i++) {
            row_ptr[i] /= sum;
        }
    }
}

/**
 * @brief Naive kernel to compute the SV matrix multiplication.
 *
 * @tparam BLOCK_SIZE The size of the thread block in each dimension.
 *
 * @param S Pointer to the softmax probability matrix (N x N) in global memory.
 * @param V Pointer to the Value matrix (N x d) in global memory.
 * @param O Pointer to the final output matrix (N x d) in global memory.
 * @param N The sequence length.
 * @param d The dimension of the attention heads.
 *
 * @details
 * Each thread in the grid is responsible for computing one element of the final
 * output matrix `O`. It does this by computing the dot product of a row from the
 * softmax matrix `S` and a column from the `V` matrix. The result is the
 * weighted sum of value vectors, which is the definition of attention output.
 */
template <int BLOCK_SIZE>
__global__ void naive_sv_matmul_kernel(
    float* S, float* V, float* O,
    int N, int d
) {
    // Calculate 2D coordinates from thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check
    if (row < N && col < d) {
        float sum = 0.0f;
        // Compute dot product: S[row] @ V[:,col]
        for (int j = 0; j < N; j++) {
            sum += S[row * N + j] * V[j * d + col];
        }
        // Store result in output matrix
        O[row * d + col] = sum;
    }
}

/**
 * @brief PyTorch wrapper for the naive attention forward pass.
 *
 * @param Q Query tensor of shape [B, nh, N, d].
 * @param K Key tensor of shape [B, nh, N, d].
 * @param V Value tensor of shape [B, nh, N, d].
 * @return torch::Tensor The output tensor of shape [B, nh, N, d].
 *
 * @details
 * This function orchestrates the naive attention calculation for a batch of inputs.
 * It iterates through each batch item and each attention head, launching the
 * three separate kernels (`qk_matmul`, `softmax`, `sv_matmul`) for each one.
 *
 * A large temporary tensor `S` of size `N x N` is allocated on the GPU for each
 * head, which is the primary source of inefficiency in this method. This function
 * clearly demonstrates the memory bottleneck that FlashAttention avoids.
 */
torch::Tensor naive_attn_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // Extract tensor dimensions
    int B = Q.size(0);      // Batch size
    int nh = Q.size(1);      // Number of heads
    int N = Q.size(2);       // Sequence length
    int d = Q.size(3);       // Head dimension
    
    // Compute scaling factor: 1/sqrt(d)
    float scale = 1.0f / sqrtf((float)d);
    
    // Allocate output tensor
    auto O = torch::zeros_like(Q);

    // Pre-allocate scratch buffer for the N×N score matrix
    float* S;
    CUDA_CHECK(cudaMalloc(&S, N * N * sizeof(float)));

    const int BLOCK_SIZE = 16;
    dim3 block_qk(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_qk((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block_softmax(256);
    dim3 grid_softmax((N + 255) / 256);
    dim3 block_sv(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_sv((d + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Process each batch and head
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < nh; h++) {
            int offset = (b * nh + h) * N * d;

            float* Q_ptr = Q.data_ptr<float>() + offset;
            float* K_ptr = K.data_ptr<float>() + offset;
            float* V_ptr = V.data_ptr<float>() + offset;
            float* O_ptr = O.data_ptr<float>() + offset;

            naive_qk_matmul_kernel<BLOCK_SIZE><<<grid_qk, block_qk>>>(
                Q_ptr, K_ptr, S, N, d, scale);

            naive_softmax_kernel<<<grid_softmax, block_softmax>>>(S, N);

            naive_sv_matmul_kernel<BLOCK_SIZE><<<grid_sv, block_sv>>>(
                S, V_ptr, O_ptr, N, d);
        }
    }

    CUDA_CHECK(cudaFree(S));

    return O;
}
