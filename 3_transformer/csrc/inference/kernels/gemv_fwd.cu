
__global__ void gemv_kernel(const float* A, const float* x, float* y, int batch, int M, int N) {
    int batch_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx < batch && row < M) {
        float sum = 0.0f;
        const float* A_batch = A + batch_idx * M * N;
        const float* x_batch = x + batch_idx * N;

        for (int col = 0; col < N; col++) {
            sum += A_batch[row * N + col] * x_batch[col];
        }
        y[batch_idx * M + row] = sum;
    }
}

torch::Tensor gemv_forward(torch::Tensor A, torch::Tensor x) {
    const at::cuda::CUDAGuard device_guard(A.device());

    TORCH_CHECK(A.device().type() == torch::kCUDA, "A must be a CUDA tensor");
    TORCH_CHECK(x.device().type() == torch::kCUDA, "x must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    if (A.dim() == 2 && x.dim() == 1) {
        TORCH_CHECK(A.size(1) == x.size(0), "Matrix-vector dimensions don't match");

        int M = A.size(0);
        int N = A.size(1);

        auto y = torch::zeros({M}, torch::dtype(torch::kFloat32).device(A.device()));

        dim3 threadsPerBlock(1, 256, 1);
        dim3 numBlocks(1, (M + threadsPerBlock.y - 1) / threadsPerBlock.y, 1);

        gemv_kernel<<<numBlocks, threadsPerBlock>>>(
            A.data_ptr<float>(),
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            1, M, N
        );

        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

        return y;
    } else if (A.dim() == 3 && x.dim() == 2) {
        TORCH_CHECK(A.size(0) == x.size(0) && A.size(2) == x.size(1),
                   "Batch matrix-vector dimensions don't match");

        int batch = A.size(0);
        int M = A.size(1);
        int N = A.size(2);

        auto y = torch::zeros({batch, M}, torch::dtype(torch::kFloat32).device(A.device()));

        dim3 threadsPerBlock(1, 16, 16);
        dim3 numBlocks(1,
                      (M + threadsPerBlock.y - 1) / threadsPerBlock.y,
                      (batch + threadsPerBlock.z - 1) / threadsPerBlock.z);

        gemv_kernel<<<numBlocks, threadsPerBlock>>>(
            A.data_ptr<float>(),
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            batch, M, N
        );

        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

        return y;
    } else {
        TORCH_CHECK(false, "Unsupported tensor dimensions for GEMV");
    }
}
