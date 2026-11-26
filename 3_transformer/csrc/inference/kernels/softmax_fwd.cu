#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

/**
 * CUDA kernel for softmax activation
 * Applies softmax normalization to each position in the sequence independently
 * Formula: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
 * 
 * @param x Input logits (batch_size × seq_len × vocab_size, device memory)
 * @param y Output probabilities (batch_size × seq_len × vocab_size, device memory)
 * @param batch_size Number of samples in the batch
 * @param seq_len Length of the sequence
 * @param vocab_size Size of vocabulary (number of classes)
 */
__global__ void softmax_kernel(const float* x, float* y, int batch_size, int seq_len, int vocab_size) {
    // Calculate batch and sequence indices from block indices
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;

    // Bounds check
    if (batch_idx < batch_size && seq_idx < seq_len) {
        // Get pointer to the current row (one position in sequence)
        const float* x_row = x + batch_idx * seq_len * vocab_size + seq_idx * vocab_size;
        float* y_row = y + batch_idx * seq_len * vocab_size + seq_idx * vocab_size;

        // Step 1: Find maximum value for numerical stability
        float max_val = -INFINITY;
        for (int i = 0; i < vocab_size; i++) {
            max_val = fmaxf(max_val, x_row[i]);
        }

        // Step 2: Compute exponentials and sum
        float sum = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            float exp_val = expf(x_row[i] - max_val);
            y_row[i] = exp_val;
            sum += exp_val;
        }

        // Step 3: Normalize to get probabilities
        for (int i = 0; i < vocab_size; i++) {
            y_row[i] /= sum;
        }
    }
}

/**
 * PyTorch wrapper for softmax forward pass
 * Applies softmax normalization to 3D tensor (batch, sequence, vocabulary)
 * 
 * @param x Input tensor (batch_size × seq_len × vocab_size, CUDA tensor)
 * @return Output tensor with softmax probabilities (same shape as input)
 */
torch::Tensor softmax_forward(torch::Tensor x) {
    const c10::cuda::CUDAGuard device_guard(x.device());

    // Validate input tensor
    TORCH_CHECK(x.device().type() == torch::kCUDA, "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D tensor (batch, seq, vocab)");

    // Extract tensor dimensions
    int batch_size = x.size(0);
    int seq_len = x.size(1);
    int vocab_size = x.size(2);

    // Allocate output tensor
    auto y = torch::zeros_like(x);

    // Configure kernel launch parameters
    // Each block processes one position in the sequence
    dim3 threadsPerBlock(1, 1, 1);
    dim3 numBlocks(batch_size, seq_len, 1);

    // Launch CUDA kernel
    softmax_kernel<<<numBlocks, threadsPerBlock>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        batch_size, seq_len, vocab_size
    );

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return y;
}
