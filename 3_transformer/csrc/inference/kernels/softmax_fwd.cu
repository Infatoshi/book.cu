
__global__ void softmax_kernel(const float* x, float* y, int batch_size, int seq_len, int vocab_size) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;

    if (batch_idx < batch_size && seq_idx < seq_len) {
        const float* x_row = x + batch_idx * seq_len * vocab_size + seq_idx * vocab_size;
        float* y_row = y + batch_idx * seq_len * vocab_size + seq_idx * vocab_size;

        float max_val = -INFINITY;
        for (int i = 0; i < vocab_size; i++) {
            max_val = fmaxf(max_val, x_row[i]);
        }

        float sum = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            float exp_val = expf(x_row[i] - max_val);
            y_row[i] = exp_val;
            sum += exp_val;
        }

        for (int i = 0; i < vocab_size; i++) {
            y_row[i] /= sum;
        }
    }
}

torch::Tensor softmax_forward(torch::Tensor x) {
    const at::cuda::CUDAGuard device_guard(x.device());

    TORCH_CHECK(x.device().type() == torch::kCUDA, "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D tensor (batch, seq, vocab)");

    int batch_size = x.size(0);
    int seq_len = x.size(1);
    int vocab_size = x.size(2);

    auto y = torch::zeros_like(x);

    dim3 threadsPerBlock(1, 1, 1);
    dim3 numBlocks(batch_size, seq_len, 1);

    softmax_kernel<<<numBlocks, threadsPerBlock>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        batch_size, seq_len, vocab_size
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return y;
}
