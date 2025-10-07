
void maxpool2d_cpu(const float* in, float* out, int height, int width, int pool_dim) {
    int output_h = height / pool_dim;
    int output_w = width / pool_dim;
    for (int r = 0; r < output_h; ++r) {
        for (int c = 0; c < output_w; ++c) {
            float max_val = -1e20f;
            for (int pr = 0; pr < pool_dim; ++pr) {
                for (int pc = 0; pc < pool_dim; ++pc) {
                    int input_row = r * pool_dim + pr;
                    int input_col = c * pool_dim + pc;
                    float val = in[input_row * width + input_col];
                    if (val > max_val) max_val = val;
                }
            }
            out[r * output_w + c] = max_val;
        }
    }
}

__global__ void maxpool2d_kernel(const float* in, float* out, int height, int width, int pool_dim) {
    int output_col = blockIdx.x * blockDim.x + threadIdx.x;
    int output_row = blockIdx.y * blockDim.y + threadIdx.y;

    int output_h = height / pool_dim;
    int output_w = width / pool_dim;

    if (output_row < output_h && output_col < output_w) {
        float max_val = -1e20f;
        for (int pool_row = 0; pool_row < pool_dim; ++pool_row) {
            for (int pool_col = 0; pool_col < pool_dim; ++pool_col) {
                int input_row = output_row * pool_dim + pool_row;
                int input_col = output_col * pool_dim + pool_col;
                float val = in[input_row * width + input_col];
                if (val > max_val) max_val = val;
            }
        }
        out[output_row * output_w + output_col] = max_val;
    }
}

int main() {
    const int height = 256, width = 256;  
    const int pool_dim = 2;               
    const int output_h = height / pool_dim;
    const int output_w = width / pool_dim;

    const int input_size = height * width;
    const int output_size = output_h * output_w;

    std::cout << "2D Max Pooling: Input[" << height << "x" << width << "] with "
              << pool_dim << "x" << pool_dim << " pooling -> Output[" << output_h << "x" << output_w << "]" << std::endl;
    std::cout << "Downsampling factor: " << pool_dim * pool_dim << "x" << std::endl;

    float *h_in, *h_out_cpu, *h_out_gpu;
    allocate_host(&h_in, input_size);
    allocate_host(&h_out_cpu, output_size);
    allocate_host(&h_out_gpu, output_size);

    srand(42); 
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float base_val = static_cast<float>((i / 16 + j / 16) % 10);
            float noise = static_cast<float>(rand() % 100) / 100.0f; 
            h_in[i * width + j] = base_val + noise;
        }
    }

    {
        Timer cpu_timer("CPU 2D Max Pooling");
        maxpool2d_cpu(h_in, h_out_cpu, height, width, pool_dim);
    }

    float *d_in, *d_out;
    allocate_device(&d_in, input_size);
    allocate_device(&d_out, output_size);

    copy_to_device(d_in, h_in, input_size);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (output_w + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (output_h + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    std::cout << "GPU: Launching " << blocksPerGrid.x << "x" << blocksPerGrid.y
              << " blocks with " << threadsPerBlock.x << "x" << threadsPerBlock.y
              << " threads per block" << std::endl;

    {
        Timer gpu_timer("GPU 2D Max Pooling");
        maxpool2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, height, width, pool_dim);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    copy_to_host(h_out_gpu, d_out, output_size);

    if (verify_results(h_out_gpu, h_out_cpu, output_size)) {
        std::cout << "✓ 2D max pooling results match!" << std::endl;

        std::cout << "\nInput feature map (top-left 8x8):" << std::endl;
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                std::cout << std::fixed << std::setprecision(1) << h_in[i * width + j] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "\nOutput after 2x2 max pooling (top-left 4x4):" << std::endl;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                std::cout << std::fixed << std::setprecision(1) << h_out_gpu[i * output_w + j] << " ";
            }
            std::cout << std::endl;
        }

        bool pooling_correct = true;
        for (int r = 0; r < output_h && pooling_correct; ++r) {
            for (int c = 0; c < output_w && pooling_correct; ++c) {
                float max_in_block = -1e20f;
                for (int pr = 0; pr < pool_dim; ++pr) {
                    for (int pc = 0; pc < pool_dim; ++pc) {
                        int input_row = r * pool_dim + pr;
                        int input_col = c * pool_dim + pc;
                        max_in_block = std::max(max_in_block, h_in[input_row * width + input_col]);
                    }
                }
                if (std::abs(h_out_gpu[r * output_w + c] - max_in_block) > 1e-6f) {
                    pooling_correct = false;
                }
            }
        }

        if (pooling_correct) {
            std::cout << "\n✓ Verified: Each output value is the maximum of its corresponding " << pool_dim << "x" << pool_dim << " input block" << std::endl;
        }
    }

    free_host(h_in);
    free_host(h_out_cpu);
    free_host(h_out_gpu);
    free_device(d_in);
    free_device(d_out);

    return 0;
}
