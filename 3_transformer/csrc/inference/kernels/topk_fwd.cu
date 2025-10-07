
__global__ void topk_kernel(const float* input, float* values, int* indices,
                           int batch_size, int n, int k) {
    int batch_idx = blockIdx.x;

    if (batch_idx < batch_size) {
        const float* input_row = input + batch_idx * n;
        float* values_row = values + batch_idx * k;
        int* indices_row = indices + batch_idx * k;

        for (int i = 0; i < k; i++) {
            values_row[i] = -INFINITY;
            indices_row[i] = -1;
        }

        for (int i = 0; i < n; i++) {
            float val = input_row[i];

            for (int j = 0; j < k; j++) {
                if (val > values_row[j]) {
                    for (int m = k - 1; m > j; m--) {
                        values_row[m] = values_row[m - 1];
                        indices_row[m] = indices_row[m - 1];
                    }
                    values_row[j] = val;
                    indices_row[j] = i;
                    break;
                }
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> topk_forward(torch::Tensor input, int k) {
    const at::cuda::CUDAGuard device_guard(input.device());

    TORCH_CHECK(input.device().type() == torch::kCUDA, "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 2, "input must be 2D tensor (batch, n)");
    TORCH_CHECK(k > 0 && k <= input.size(1), "k must be > 0 and <= input.size(1)");

    int batch_size = input.size(0);
    int n = input.size(1);

    auto values = torch::zeros({batch_size, k}, torch::dtype(torch::kFloat32).device(input.device()));
    auto indices = torch::zeros({batch_size, k}, torch::dtype(torch::kInt32).device(input.device()));

    int threads_per_block = 1;  
    int num_blocks = batch_size;

    topk_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        values.data_ptr<float>(),
        indices.data_ptr<int>(),
        batch_size, n, k
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return std::make_tuple(values, indices);
}
