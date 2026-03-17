/**
#include <c10/cuda/CUDAGuard.h>
 * CUDA kernels for embedding layer forward and backward passes
 * Implements lookup table for token embeddings
 */

/**
 * Forward pass kernel for embedding layer
 * Looks up embeddings for given token indices
 * 
 * @param weight Embedding weight matrix (vocab_size × n_embd, device memory)
 * @param indices Token indices (num_indices, device memory)
 * @param out Output embeddings (num_indices × n_embd, device memory)
 * @param num_indices Number of tokens to look up
 * @param n_embd Embedding dimension
 */
__global__ void embedding_fwd_kernel(const float* weight, const int* indices,
                                   float* out, int num_indices, int n_embd) {
    // Calculate global thread index across all elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_indices * n_embd;

    if (idx < total_elements) {
        // Extract token index and embedding dimension index
        int token_idx = idx / n_embd;
        int emb_idx = idx % n_embd;
        // Look up embedding value from weight matrix
        int weight_idx = indices[token_idx] * n_embd + emb_idx;
        out[idx] = weight[weight_idx];
    }
}

/**
 * Backward pass kernel for embedding layer
 * Accumulates gradients to embedding weights using atomic operations
 * 
 * @param grad_out Gradient from output (num_indices × n_embd, device memory)
 * @param indices Token indices (num_indices, device memory)
 * @param grad_weight Gradient for embedding weights (vocab_size × n_embd, device memory)
 * @param num_indices Number of tokens
 * @param n_embd Embedding dimension
 */
__global__ void embedding_bwd_kernel(const float* grad_out, const int* indices,
                                   float* grad_weight, int num_indices, int n_embd) {
    // Calculate global thread index across all elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_indices * n_embd;

    if (idx < total_elements) {
        // Extract token index and embedding dimension index
        int token_idx = idx / n_embd;
        int emb_idx = idx % n_embd;
        // Accumulate gradient to corresponding embedding weight
        // Use atomicAdd because multiple tokens may share the same embedding
        int weight_idx = indices[token_idx] * n_embd + emb_idx;
        atomicAdd(&grad_weight[weight_idx], grad_out[idx]);
    }
}

/**
 * CUDA wrapper for forward embedding pass
 * Launches the forward embedding kernel
 * 
 * @param weight Embedding weight matrix (vocab_size × n_embd, device memory)
 * @param indices Token indices (num_indices, device memory)
 * @param out Output embeddings (num_indices × n_embd, device memory)
 * @param num_indices Number of tokens to look up
 * @param n_embd Embedding dimension
 */
void embedding_fwd_cuda(const float* weight, const int* indices, float* out,
                       int num_indices, int n_embd) {
    int total_elements = num_indices * n_embd;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    embedding_fwd_kernel<<<blocks, threads>>>(weight, indices, out, num_indices, n_embd);
}

/**
 * CUDA wrapper for backward embedding pass
 * Launches the backward embedding kernel
 * 
 * @param grad_out Gradient from output (num_indices × n_embd, device memory)
 * @param indices Token indices (num_indices, device memory)
 * @param grad_weight Gradient for embedding weights (vocab_size × n_embd, device memory)
 * @param num_indices Number of tokens
 * @param n_embd Embedding dimension
 */
void embedding_bwd_cuda(const float* grad_out, const int* indices, float* grad_weight,
                       int num_indices, int n_embd) {
    int total_elements = num_indices * n_embd;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    embedding_bwd_kernel<<<blocks, threads>>>(grad_out, indices, grad_weight, num_indices, n_embd);
}
