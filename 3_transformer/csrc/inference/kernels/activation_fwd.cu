
__global__ void gelu_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float sqrt_2_pi = sqrtf(2.0f / 3.141592653589793f);
        float x3 = val * val * val;
        float inner = sqrt_2_pi * (val + 0.044715f * x3);
        float tanh_inner = tanhf(inner);
        y[idx] = 0.5f * val * (1.0f + tanh_inner);
    }
}

torch::Tensor gelu_forward(torch::Tensor x) {
    const at::cuda::CUDAGuard device_guard(x.device());

    TORCH_CHECK(x.device().type() == torch::kCUDA, "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    int size = x.numel();
    auto y = torch::zeros_like(x);

    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    gelu_kernel<<<numBlocks, threadsPerBlock>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        size
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return y;
}
