
void transpose_cpu(const float* in, float* out, int num_rows, int num_cols) {
    for (int row = 0; row < num_rows; ++row) {
        for (int column = 0; column < num_cols; ++column) {
            out[column * num_rows + row] = in[row * num_cols + column];
        }
    }
}

__global__ void transpose_kernel(const float* in, float* out, int num_rows, int num_cols) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < num_rows && column < num_cols) {
        int in_index = row * num_cols + column;
        int out_index = column * num_rows + row;
        out[out_index] = in[in_index];
    }
}

int main() {
    const int num_rows = 1024;
    const int num_cols = 1024;
    const int n = num_rows * num_cols;
    size_t size = n * sizeof(float);

    std::cout << "Matrix Transpose: " << num_rows << "x" << num_cols << " -> "
              << num_cols << "x" << num_rows << std::endl;

    float *h_in, *h_out_cpu, *h_out_gpu;
    allocate_host(&h_in, n);
    allocate_host(&h_out_cpu, n);
    allocate_host(&h_out_gpu, n);

    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            h_in[i * num_cols + j] = static_cast<float>(i * num_cols + j);
        }
    }

    {
        Timer cpu_timer("CPU Matrix Transpose");
        transpose_cpu(h_in, h_out_cpu, num_rows, num_cols);
    }

    float *d_in, *d_out;
    allocate_device(&d_in, n);
    allocate_device(&d_out, n);

    copy_to_device(d_in, h_in, n);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (num_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (num_rows + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    std::cout << "GPU: Launching " << blocksPerGrid.x << "x" << blocksPerGrid.y
              << " blocks with " << threadsPerBlock.x << "x" << threadsPerBlock.y
              << " threads per block" << std::endl;

    {
        Timer gpu_timer("GPU Matrix Transpose");
        transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, num_rows, num_cols);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    copy_to_host(h_out_gpu, d_out, n);

    if (verify_results(h_out_gpu, h_out_cpu, n)) {
        std::cout << "✓ Matrix transpose results match!" << std::endl;

        std::cout << "\nOriginal matrix (first 3x3):" << std::endl;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                std::cout << h_in[i * num_cols + j] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "\nTransposed matrix (first 3x3):" << std::endl;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                std::cout << h_out_gpu[i * num_rows + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    free_host(h_in);
    free_host(h_out_cpu);
    free_host(h_out_gpu);
    free_device(d_in);
    free_device(d_out);

    return 0;
}
