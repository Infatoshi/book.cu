
void gemm_cpu(const float* A, const float* B, float* C, int M_rows, int N_cols, int K_shared_dim) {
    for (int row = 0; row < M_rows; ++row) {
        for (int column = 0; column < N_cols; ++column) {
            float sum = 0.0f;
            for (int k_idx = 0; k_idx < K_shared_dim; ++k_idx) {
                sum += A[row * K_shared_dim + k_idx] * B[k_idx * N_cols + column];
            }
            C[row * N_cols + column] = sum;
        }
    }
}

__global__ void gemm_kernel(const float* A, const float* B, float* C, int M_rows, int N_cols, int K_shared_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M_rows && column < N_cols) {
        float sum = 0.0f;
        for (int k_idx = 0; k_idx < K_shared_dim; ++k_idx) {
            sum += A[row * K_shared_dim + k_idx] * B[k_idx * N_cols + column];
        }
        C[row * N_cols + column] = sum;
    }
}

int main() {
    const int M_rows = 512, N_cols = 512, K_shared_dim = 256;
    const int size_A = M_rows * K_shared_dim;
    const int size_B = K_shared_dim * N_cols;
    const int size_C = M_rows * N_cols;

    std::cout << "GEMM: C[" << M_rows << "x" << N_cols << "] = A[" << M_rows << "x" << K_shared_dim
              << "] * B[" << K_shared_dim << "x" << N_cols << "]" << std::endl;
    std::cout << "Total operations: " << (long long)M_rows * N_cols * K_shared_dim * 2 << " FLOPs" << std::endl;

    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    allocate_host(&h_A, size_A);
    allocate_host(&h_B, size_B);
    allocate_host(&h_C_cpu, size_C);
    allocate_host(&h_C_gpu, size_C);

    for (int i = 0; i < size_A; ++i) {
        h_A[i] = static_cast<float>(i % 10);  
    }
    for (int i = 0; i < size_B; ++i) {
        h_B[i] = static_cast<float>((i * 2) % 10);  
    }

    {
        Timer cpu_timer("CPU GEMM");
        gemm_cpu(h_A, h_B, h_C_cpu, M_rows, N_cols, K_shared_dim);
    }

    float *d_A, *d_B, *d_C;
    allocate_device(&d_A, size_A);
    allocate_device(&d_B, size_B);
    allocate_device(&d_C, size_C);

    copy_to_device(d_A, h_A, size_A);
    copy_to_device(d_B, h_B, size_B);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (N_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M_rows + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    std::cout << "GPU: Launching " << blocksPerGrid.x << "x" << blocksPerGrid.y
              << " blocks with " << threadsPerBlock.x << "x" << threadsPerBlock.y
              << " threads per block" << std::endl;

    {
        Timer gpu_timer("GPU GEMM");
        gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M_rows, N_cols, K_shared_dim);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    copy_to_host(h_C_gpu, d_C, size_C);

    if (verify_results(h_C_gpu, h_C_cpu, size_C)) {
        std::cout << "✓ GEMM results match!" << std::endl;

        std::cout << "\nSample results (first 3x3 of output matrix):" << std::endl;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                std::cout << h_C_gpu[i * N_cols + j] << " ";
            }
            std::cout << std::endl;
        }
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
