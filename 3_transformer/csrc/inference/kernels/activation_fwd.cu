#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <math.h>

/**
 * CUDA kernel for GELU (Gaussian Error Linear Unit) activation function
 * GELU is commonly used in transformer architectures (e.g., GPT, BERT)
 * 
 * GELU formula: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
 * 
 * This implementation uses the approximation:
 * - Computes x³ = x * x * x
 * - Applies tanh to transformed input
 * - Scales by 0.5 * x * (1 + tanh(...))
 * 
 * @param x Input tensor (device memory)
 * @param y Output tensor (device memory)
 * @param size Number of elements
 */
__global__ void gelu_kernel(const float* x, float* y, int size) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        // Precompute sqrt(2/π) constant
        float sqrt_2_pi = sqrtf(2.0f / 3.141592653589793f);
        // Compute x³ for the GELU approximation
        float x3 = val * val * val;
        // Compute inner term: sqrt(2/π) * (x + 0.044715 * x³)
        float inner = sqrt_2_pi * (val + 0.044715f * x3);
        // Apply tanh activation
        float tanh_inner = tanhf(inner);
        // Compute GELU: 0.5 * x * (1 + tanh(inner))
        y[idx] = 0.5f * val * (1.0f + tanh_inner);
    }
}

/**
 * PyTorch wrapper for GELU forward pass
 * Applies GELU activation element-wise to input tensor
 * 
 * @param x Input tensor (CUDA tensor, any shape)
 * @return Output tensor with GELU activation applied (same shape as input)
 */
torch::Tensor gelu_forward(torch::Tensor x) {
    const c10::cuda::CUDAGuard device_guard(x.device());

    // Validate input tensor
    TORCH_CHECK(x.device().type() == torch::kCUDA, "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    // Get total number of elements
    int size = x.numel();
    // Allocate output tensor
    auto y = torch::zeros_like(x);

    // Configure kernel launch parameters
    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Launch CUDA kernel
    gelu_kernel<<<numBlocks, threadsPerBlock>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        size
    );

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return y;
}
