
void matrix_add_cpu(const float* A, const float* B, float* C, int num_rows, int num_cols) {
    for (int row = 0; row < num_rows; ++row) {
        for (int col = 0; col < num_cols; ++col) {
            int index = row * num_cols + col;
            C[index] = A[index] + B[index];
        }
    }
}

__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int num_rows, int num_cols) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < num_rows && column < num_cols) {
        int index = row * num_cols + column;
        C[index] = A[index] + B[index];
    }
}

int main() {
    const int num_rows = 1024;
    const int num_cols = 1024;
    const int n = num_rows * num_cols;
    size_t size = n * sizeof(float);

    std::cout << "Matrix Addition: " << num_rows << "x" << num_cols << " = " << n << " elements" << std::endl;

    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    allocate_host(&h_A, n);
    allocate_host(&h_B, n);
    allocate_host(&h_C_cpu, n);
    allocate_host(&h_C_gpu, n);

    for (int i = 0; i < n; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    {
        Timer cpu_timer("CPU Matrix Addition");
        matrix_add_cpu(h_A, h_B, h_C_cpu, num_rows, num_cols);
    }

    float *d_A, *d_B, *d_C;
    allocate_device(&d_A, n);
    allocate_device(&d_B, n);
    allocate_device(&d_C, n);

    copy_to_device(d_A, h_A, n);
    copy_to_device(d_B, h_B, n);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (num_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (num_rows + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    std::cout << "GPU: Launching " << blocksPerGrid.x << "x" << blocksPerGrid.y
              << " blocks with " << threadsPerBlock.x << "x" << threadsPerBlock.y
              << " threads per block" << std::endl;

    {
        Timer gpu_timer("GPU Matrix Addition");
        matrix_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, num_rows, num_cols);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    copy_to_host(h_C_gpu, d_C, n);

    if (verify_results(h_C_gpu, h_C_cpu, n)) {
        std::cout << "✓ Matrix addition results match!" << std::endl;
    }

    free_host(h_A);
    free_host(h_B);
    free_host(h_C_cpu);
    free_host(h_C_gpu);
    free_device(d_A);
    free_device(d_B);
    free_device(d_C);

    return 0;
}
