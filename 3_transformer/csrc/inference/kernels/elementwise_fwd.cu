#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

/**
 * CUDA kernel for element-wise addition
 * Computes: c = a + b (element-wise)
 * 
 * @param a Input tensor a (device memory)
 * @param b Input tensor b (device memory)
 * @param c Output tensor c (device memory)
 * @param size Number of elements in each tensor
 */
__global__ void add_kernel(const float* a, const float* b, float* c, int size) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check
    if (idx < size) {
        // Perform element-wise addition
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * CUDA kernel for element-wise multiplication
 * Computes: c = a * b (element-wise)
 * 
 * @param a Input tensor a (device memory)
 * @param b Input tensor b (device memory)
 * @param c Output tensor c (device memory)
 * @param size Number of elements in each tensor
 */
__global__ void mul_kernel(const float* a, const float* b, float* c, int size) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check
    if (idx < size) {
        // Perform element-wise multiplication
        c[idx] = a[idx] * b[idx];
    }
}

/**
 * PyTorch wrapper for element-wise addition forward pass
 * 
 * @param a Input tensor a (CUDA tensor)
 * @param b Input tensor b (CUDA tensor, must match shape of a)
 * @return Output tensor c = a + b (same shape as inputs)
 */
torch::Tensor add_forward(torch::Tensor a, torch::Tensor b) {
    const c10::cuda::CUDAGuard device_guard(a.device());

    // Validate input tensors
    TORCH_CHECK(a.device().type() == torch::kCUDA, "a must be a CUDA tensor");
    TORCH_CHECK(b.device().type() == torch::kCUDA, "b must be a CUDA tensor");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "b must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensor shapes must match");

    // Get total number of elements
    int size = a.numel();
    // Allocate output tensor
    auto c = torch::zeros_like(a);

    // Configure kernel launch parameters
    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Launch CUDA kernel
    add_kernel<<<numBlocks, threadsPerBlock>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        size
    );

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return c;
}

/**
 * PyTorch wrapper for element-wise multiplication forward pass
 * 
 * @param a Input tensor a (CUDA tensor)
 * @param b Input tensor b (CUDA tensor, must match shape of a)
 * @return Output tensor c = a * b (same shape as inputs)
 */
torch::Tensor mul_forward(torch::Tensor a, torch::Tensor b) {
    const c10::cuda::CUDAGuard device_guard(a.device());

    // Validate input tensors
    TORCH_CHECK(a.device().type() == torch::kCUDA, "a must be a CUDA tensor");
    TORCH_CHECK(b.device().type() == torch::kCUDA, "b must be a CUDA tensor");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "b must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensor shapes must match");

    // Get total number of elements
    int size = a.numel();
    // Allocate output tensor
    auto c = torch::zeros_like(a);

    // Configure kernel launch parameters
    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Launch CUDA kernel
    mul_kernel<<<numBlocks, threadsPerBlock>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        size
    );

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return c;
}
