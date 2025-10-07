
__global__ void layernorm_kernel(const float* x, const float* weight, const float* bias,
                                float* y, int batch_size, int seq_len, int hidden_size) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;

    if (batch_idx < batch_size && seq_idx < seq_len) {
        const float* x_row = x + batch_idx * seq_len * hidden_size + seq_idx * hidden_size;
        float* y_row = y + batch_idx * seq_len * hidden_size + seq_idx * hidden_size;

        float mean = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            mean += x_row[i];
        }
        mean /= hidden_size;

        float var = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            float diff = x_row[i] - mean;
            var += diff * diff;
        }
        var /= hidden_size;

        float eps = 1e-5f;
        for (int i = 0; i < hidden_size; i++) {
            float normalized = (x_row[i] - mean) / sqrtf(var + eps);
            y_row[i] = normalized * weight[i] + bias[i];
        }
    }
}

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    const at::cuda::CUDAGuard device_guard(x.device());

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

    int batch_size = x.size(0);
    int seq_len = x.size(1);
    int hidden_size = x.size(2);

    auto y = torch::zeros_like(x);

    dim3 threadsPerBlock(1, 1, 1);
    dim3 numBlocks(batch_size, seq_len, 1);

    layernorm_kernel<<<numBlocks, threadsPerBlock>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        y.data_ptr<float>(),
        batch_size, seq_len, hidden_size
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return y;
}
