#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <math.h>

/**
 * CUDA kernel for Layer Normalization forward pass (inference)
 * Normalizes input across the hidden dimension for each (batch, sequence) position
 * 
 * Algorithm:
 * 1. Compute mean: mean = sum(x) / hidden_size
 * 2. Compute variance: var = sum((x - mean)^2) / hidden_size
 * 3. Normalize: normalized = (x - mean) / sqrt(var + eps)
 * 4. Scale and shift: y = normalized * weight + bias
 * 
 * This is a naive implementation where each thread processes one (batch, seq) position
 * sequentially over the hidden dimension. Used for inference where simplicity is prioritized.
 * 
 * @param x Input tensor (batch_size × seq_len × hidden_size, device memory)
 * @param weight Scale parameter (hidden_size, device memory)
 * @param bias Shift parameter (hidden_size, device memory)
 * @param y Output tensor (batch_size × seq_len × hidden_size, device memory)
 * @param batch_size Batch dimension
 * @param seq_len Sequence length dimension
 * @param hidden_size Hidden dimension (embedding size)
 */
__global__ void layernorm_kernel(const float* x, const float* weight, const float* bias,
                                float* y, int batch_size, int seq_len, int hidden_size) {
    // Each block processes one (batch, sequence) position
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;

    if (batch_idx < batch_size && seq_idx < seq_len) {
        // Get pointer to the row for this (batch, sequence) position
        const float* x_row = x + batch_idx * seq_len * hidden_size + seq_idx * hidden_size;
        float* y_row = y + batch_idx * seq_len * hidden_size + seq_idx * hidden_size;

        // Step 1: Compute mean across hidden dimension
        float mean = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            mean += x_row[i];
        }
        mean /= hidden_size;

        // Step 2: Compute variance across hidden dimension
        float var = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            float diff = x_row[i] - mean;
            var += diff * diff;
        }
        var /= hidden_size;

        // Step 3 & 4: Normalize and apply affine transformation
        float eps = 1e-5f;  // Small epsilon to prevent division by zero
        for (int i = 0; i < hidden_size; i++) {
            float normalized = (x_row[i] - mean) / sqrtf(var + eps);
            y_row[i] = normalized * weight[i] + bias[i];
        }
    }
}

/**
 * PyTorch wrapper for Layer Normalization forward pass (inference)
 * Applies layer normalization to input tensor across the last dimension
 * 
 * @param x Input tensor (batch_size × seq_len × hidden_size, CUDA tensor)
 * @param weight Scale parameter (hidden_size, CUDA tensor)
 * @param bias Shift parameter (hidden_size, CUDA tensor)
 * @return Output tensor with layer normalization applied (same shape as input)
 */
torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    const c10::cuda::CUDAGuard device_guard(x.device());

    // Validate input tensors
    TORCH_CHECK(x.device().type() == torch::kCUDA, "x must be a CUDA tensor");
    TORCH_CHECK(weight.device().type() == torch::kCUDA, "weight must be a CUDA tensor");
    TORCH_CHECK(bias.device().type() == torch::kCUDA, "bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D tensor (batch, seq, hidden)");
    TORCH_CHECK(weight.dim() == 1 && bias.dim() == 1, "weight and bias must be 1D");
    TORCH_CHECK(x.size(2) == weight.size(0) && x.size(2) == bias.size(0),
               "Hidden dimensions must match");

    // Extract tensor dimensions
    int batch_size = x.size(0);
    int seq_len = x.size(1);
    int hidden_size = x.size(2);

    // Allocate output tensor
    auto y = torch::zeros_like(x);

    // Configure kernel launch: one thread per (batch, sequence) position
    dim3 threadsPerBlock(1, 1, 1);
    dim3 numBlocks(batch_size, seq_len, 1);

    // Launch CUDA kernel
    layernorm_kernel<<<numBlocks, threadsPerBlock>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        y.data_ptr<float>(),
        batch_size, seq_len, hidden_size
    );

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return y;
}
