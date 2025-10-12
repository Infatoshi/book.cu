
/**
 * CPU implementation of General Matrix Multiply (GEMM)
 * Computes C = A * B where A is M×K, B is K×N, C is M×N
 * 
 * @param A Centered matrix A (M×K, row-major order)
 * @param B Centered matrix B (K×N, row-major order)
 * @param C Output matrix C (M×N, row-major order)
 * @param M_rows Number of rows in A and C
 * @param N_cols Number of columns in B and C
 * @param K_shared_dim Number of columns in A and rows in B
 */
void gemm_cpu(const float* A, const float* B, float* C, int M_rows, int N_cols, int K_shared_dim) {
    // Iterate through each element of output matrix C
    for (int row = 0; row < M_rows; ++row) {
        for (int column = 0; column < N_cols; ++column) {
            float sum = 0.0f;
            // Compute dot product of row A[row,:] and column B[:,column]
            for (int k_idx = 0; k_idx < K_shared_dim; ++k_idx) {
                sum += A[row * K_shared_dim + k_idx] * B[k_idx * N_cols + column];
            }
            C[row * N_cols + column] = sum;
        }
    }
}

/**
 * CUDA kernel for General Matrix Multiply (GEMM)
 * Each thread computes one element of output matrix C
 * Uses 2D thread indexing to map threads to matrix elements
 * 
 * @param A Input matrix A (M×K, device memory)
 * @param B Input matrix B (K×N, device memory)
 * @param C Output matrix C (M×N, device memory)
 * @param M_rows Number of rows in A and C
 * @param N_cols Number of columns in B and C
 * @param K_shared_dim Number of columns in A and rows in B
 */
__global__ void gemm_kernel(const float* A, const float* B, float* C, int M_rows, int N_cols, int K_shared_dim) {
    // Calculate 2D coordinates from thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;    // Row index in output matrix
    int column = blockIdx.x * blockDim.x + threadIdx.x; // Column index in output matrix

    // Bounds check to ensure we don't access out-of-range elements
    if (row < M_rows && column < N_cols) {
        float sum = 0.0f;
        // Compute dot product of row A[row,:] and column B[:,column]
        for (int k_idx = 0; k_idx < K_shared_dim; ++k_idx) {
            sum += A[row * K_shared_dim + k_idx] * B[k_idx * N_cols + column];
        }
        // Store result in output matrix
        C[row * N_cols + column] = sum;
    }
}

int main() {
    // Matrix dimensions: C[M×N] = A[M×K] * B[K×N]
    const int M_rows = 512, N_cols = 512, K_shared_dim = 256;
    const int size_A = M_rows * K_shared_dim;
    const int size_B = K_shared_dim * N_cols;
    const int size_C = M_rows * N_cols;

    std::cout << "GEMM: C[" << M_rows << "x" << N_cols << "] = A[" << M_rows << "x" << K_shared_dim
              << "] * B[" << K_shared_dim << "x" << N_cols << "]" << std::endl;
    std::cout << "Total operations: " << (long long)M_rows * N_cols * K_shared_dim * 2 << " FLOPs" << std::endl;

    // Allocate host memory for matrices
    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    allocate_host(&h_A, size_A);
    allocate_host(&h_B, size_B);
    allocate_host(&h_C_cpu, size_C);
    allocate_host(&h_C_gpu, size_C);

    // Initialize input matrices with test data
    for (int i = 0; i < size_A; ++i) {
        h_A[i] = static_cast<float>(i % 10);  // Matrix A: values 0-9 repeating
    }
    for (int i = 0; i < size_B; ++i) {
        h_B[i] = static_cast<float>((i * 2) % 10);  // Matrix B: values 0,2,4,6,8 repeating
    }

    // Run CPU version with timing
    {
        Timer cpu_timer("CPU GEMM");
        gemm_cpu(h_A, h_B, h_C_cpu, M_rows, N_cols, K_shared_dim);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    allocate_device(&d_A, size_A);
    allocate_device(&d_B, size_B);
    allocate_device(&d_C, size_C);

    // Copy data from host to device
    copy_to_device(d_A, h_A, size_A);
    copy_to_device(d_B, h_B, size_B);

    // Configure 2D kernel launch parameters
    dim3 threadsPerBlock(16, 16);  // 16x16 = 256 threads per block
    dim3 blocksPerGrid(
        (N_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,  // Blocks in x-dimension
        (M_rows + threadsPerBlock.y - 1) / threadsPerBlock.y   // Blocks in y-dimension
    );

    std::cout << "GPU: Launching " << blocksPerGrid.x << "x" << blocksPerGrid.y
              << " blocks with " << threadsPerBlock.x << "x" << threadsPerBlock.y
              << " threads per block" << std::endl;

    // Run GPU version with timing
    {
        Timer gpu_timer("GPU GEMM");
        gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M_rows, N_cols, K_shared_dim);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Copy result back from device to host
    copy_to_host(h_C_gpu, d_C, size_C);

    // Verify GPU results against CPU results
    if (verify_results(h_C_gpu, h_C_cpu, size_C)) {
        std::cout << "✓ GEMM results match!" << std::endl;

        // Display sample results
        std::cout << "\nSample results (first 3x3 of output matrix):" << std::endl;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                std::cout << h_C_gpu[i * N_cols + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    // Clean up memory
    free_host(h_A);
    free_host(h_B);
    free_host(h_C_cpu);
    free_host(h_C_gpu);
    free_device(d_A);
    free_device(d_B);
    free_device(d_C);

    return 0;
}
