
__global__ void add_kernel(const float* a, const float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void mul_kernel(const float* a, const float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] * b[idx];
    }
}

torch::Tensor add_forward(torch::Tensor a, torch::Tensor b) {
    const at::cuda::CUDAGuard device_guard(a.device());

    TORCH_CHECK(a.device().type() == torch::kCUDA, "a must be a CUDA tensor");
    TORCH_CHECK(b.device().type() == torch::kCUDA, "b must be a CUDA tensor");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "b must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensor shapes must match");

    int size = a.numel();
    auto c = torch::zeros_like(a);

    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    add_kernel<<<numBlocks, threadsPerBlock>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        size
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return c;
}

torch::Tensor mul_forward(torch::Tensor a, torch::Tensor b) {
    const at::cuda::CUDAGuard device_guard(a.device());

    TORCH_CHECK(a.device().type() == torch::kCUDA, "a must be a CUDA tensor");
    TORCH_CHECK(b.device().type() == torch::kCUDA, "b must be a CUDA tensor");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "b must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensor shapes must match");

    int size = a.numel();
    auto c = torch::zeros_like(a);

    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    mul_kernel<<<numBlocks, threadsPerBlock>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        size
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return c;
}
