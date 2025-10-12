
/**
 * CPU implementation of softmax function
 * Applies softmax normalization to each row independently
 * Formula: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
 * 
 * @param in Input matrix (num_rows × num_cols, host memory)
 * @param out Output matrix (num_rows × num_cols, host memory)
 * @param num_rows Number of rows (batch size)
 * @param num_cols Number of columns (feature dimension)
 */
void softmax_cpu(const float* in, float* out, int num_rows, int num_cols) {
    // Process each row independently
    for (int row = 0; row < num_rows; ++row) {
        // Step 1: Find maximum value in the row (for numerical stability)
        float max_val = in[row * num_cols];
        for (int col = 1; col < num_cols; ++col) {
            if (in[row * num_cols + col] > max_val) {
                max_val = in[row * num_cols + col];
            }
        }

        // Step 2: Compute sum of exponentials (shifted by max for stability)
        float sum_exp = 0.0f;
        for (int col = 0; col < num_cols; ++col) {
            sum_exp += expf(in[row * num_cols + col] - max_val);
        }

        // Step 3: Compute softmax probabilities
        for (int col = 0; col < num_cols; ++col) {
            out[row * num_cols + col] = expf(in[row * num_cols + col] - max_val) / sum_exp;
        }
    }
}

/**
 * Naive CUDA kernel for softmax function
 * Each thread computes one element, but redundantly computes max and sum for the entire row
 * This is inefficient but demonstrates the basic concept
 * 
 * @param in Input matrix (num_rows × num_cols, device memory)
 * @param out Output matrix (num_rows × num_cols, device memory)
 * @param num_rows Number of rows (batch size)
 * @param num_cols Number of columns (feature dimension)
 */
__global__ void softmax_naive_kernel(const float* in, float* out, int num_rows, int num_cols) {
    // Calculate 2D coordinates from thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;    // Row index
    int column = blockIdx.x * blockDim.x + threadIdx.x; // Column index

    // Bounds check
    if (row < num_rows && column < num_cols) {
        // Step 1: Find maximum value in the row (redundant computation per thread)
        float max_val = -1e20f;
        for (int col_idx = 0; col_idx < num_cols; ++col_idx) {
            if (in[row * num_cols + col_idx] > max_val) max_val = in[row * num_cols + col_idx];
        }
        
        // Step 2: Compute sum of exponentials (redundant computation per thread)
        float sum_exp = 0.0f;
        for (int col_idx = 0; col_idx < num_cols; ++col_idx) {
            sum_exp += expf(in[row * num_cols + col_idx] - max_val);
        }
        
        // Step 3: Compute softmax probability for this element
        out[row * num_cols + column] = expf(in[row * num_cols + column] - max_val) / sum_exp;
    }
}

int main() {
    const int num_rows = 128;  
    const int num_cols = 1000; 
    const int n = num_rows * num_cols;
    size_t size = n * sizeof(float);

    std::cout << "Softmax: " << num_rows << " rows x " << num_cols << " columns = "
              << n << " elements" << std::endl;

    float *h_in, *h_out_cpu, *h_out_gpu;
    allocate_host(&h_in, n);
    allocate_host(&h_out_cpu, n);
    allocate_host(&h_out_gpu, n);

    srand(42); 
    for (int i = 0; i < n; ++i) {
        h_in[i] = static_cast<float>(rand() % 20 - 10); 
    }

    {
        Timer cpu_timer("CPU Softmax");
        softmax_cpu(h_in, h_out_cpu, num_rows, num_cols);
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
        Timer gpu_timer("GPU Softmax (naive)");
        softmax_naive_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, num_rows, num_cols);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    copy_to_host(h_out_gpu, d_out, n);

    if (verify_results(h_out_gpu, h_out_cpu, n, 1e-5f)) {
        std::cout << "✓ Softmax results match!" << std::endl;

        bool sum_check = true;
        for (int row = 0; row < num_rows && sum_check; ++row) {
            float row_sum = 0.0f;
            for (int col = 0; col < num_cols; ++col) {
                row_sum += h_out_gpu[row * num_cols + col];
            }
            if (std::abs(row_sum - 1.0f) > 1e-5f) {
                std::cout << "Row " << row << " sum: " << row_sum << " (expected ~1.0)" << std::endl;
                sum_check = false;
            }
        }

        if (sum_check) {
            std::cout << "✓ All rows sum to 1.0 (as expected for softmax)" << std::endl;
        }

        std::cout << "\nExample - First row input: ";
        for (int col = 0; col < std::min(5, num_cols); ++col) {
            std::cout << h_in[col] << " ";
        }
        if (num_cols > 5) std::cout << "...";
        std::cout << std::endl;

        std::cout << "Example - First row softmax output: ";
        for (int col = 0; col < std::min(5, num_cols); ++col) {
            std::cout << h_out_gpu[col] << " ";
        }
        if (num_cols > 5) std::cout << "...";
        std::cout << std::endl;
    }

    free_host(h_in);
    free_host(h_out_cpu);
    free_host(h_out_gpu);
    free_device(d_in);
    free_device(d_out);

    return 0;
}
