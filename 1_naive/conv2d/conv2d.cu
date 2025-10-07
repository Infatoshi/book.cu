
void conv2d_cpu(const float* in, float* out, const float* kernel, int height, int width, int kernel_dim) {
    int output_h = height - kernel_dim + 1;
    int output_w = width - kernel_dim + 1;
    for (int r = 0; r < output_h; ++r) {
        for (int c = 0; c < output_w; ++c) {
            float sum = 0.0f;
            for (int kr = 0; kr < kernel_dim; ++kr) {
                for (int kc = 0; kc < kernel_dim; ++kc) {
                    int input_row = r + kr;
                    int input_col = c + kc;
                    sum += in[input_row * width + input_col] * kernel[kr * kernel_dim + kc];
                }
            }
            out[r * output_w + c] = sum;
        }
    }
}

__global__ void conv2d_kernel(const float* in, float* out, const float* kernel, int height, int width, int kernel_dim) {
    int output_col = blockIdx.x * blockDim.x + threadIdx.x;
    int output_row = blockIdx.y * blockDim.y + threadIdx.y;

    int output_h = height - kernel_dim + 1;
    int output_w = width - kernel_dim + 1;

    if (output_row < output_h && output_col < output_w) {
        float sum = 0.0f;
        for (int kernel_row = 0; kernel_row < kernel_dim; ++kernel_row) {
            for (int kernel_col = 0; kernel_col < kernel_dim; ++kernel_col) {
                int input_row = output_row + kernel_row;
                int input_col = output_col + kernel_col;
                sum += in[input_row * width + input_col] * kernel[kernel_row * kernel_dim + kernel_col];
            }
        }
        out[output_row * output_w + output_col] = sum;
    }
}

int main() {
    const int height = 256, width = 256;  
    const int kernel_dim = 3;             
    const int output_h = height - kernel_dim + 1;
    const int output_w = width - kernel_dim + 1;

    const int input_size = height * width;
    const int kernel_size = kernel_dim * kernel_dim;
    const int output_size = output_h * output_w;

    std::cout << "2D Convolution: Input[" << height << "x" << width << "] * Kernel["
              << kernel_dim << "x" << kernel_dim << "] -> Output[" << output_h << "x" << output_w << "]" << std::endl;

    float *h_in, *h_kernel, *h_out_cpu, *h_out_gpu;
    allocate_host(&h_in, input_size);
    allocate_host(&h_kernel, kernel_size);
    allocate_host(&h_out_cpu, output_size);
    allocate_host(&h_out_gpu, output_size);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            h_in[i * width + j] = static_cast<float>((i + j) % 10);
        }
    }

    float kernel_data[9] = {
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    };
    for (int i = 0; i < kernel_size; ++i) {
        h_kernel[i] = kernel_data[i];
    }

    {
        Timer cpu_timer("CPU 2D Convolution");
        conv2d_cpu(h_in, h_out_cpu, h_kernel, height, width, kernel_dim);
    }

    float *d_in, *d_kernel, *d_out;
    allocate_device(&d_in, input_size);
    allocate_device(&d_kernel, kernel_size);
    allocate_device(&d_out, output_size);

    copy_to_device(d_in, h_in, input_size);
    copy_to_device(d_kernel, h_kernel, kernel_size);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (output_w + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (output_h + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    std::cout << "GPU: Launching " << blocksPerGrid.x << "x" << blocksPerGrid.y
              << " blocks with " << threadsPerBlock.x << "x" << threadsPerBlock.y
              << " threads per block" << std::endl;

    {
        Timer gpu_timer("GPU 2D Convolution");
        conv2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, d_kernel, height, width, kernel_dim);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    copy_to_host(h_out_gpu, d_out, output_size);

    if (verify_results(h_out_gpu, h_out_cpu, output_size)) {
        std::cout << "✓ 2D convolution results match!" << std::endl;

        std::cout << "\n3x3 Edge Detection Kernel:" << std::endl;
        for (int i = 0; i < kernel_dim; ++i) {
            for (int j = 0; j < kernel_dim; ++j) {
                std::cout << kernel_data[i * kernel_dim + j] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "\nInput image (top-left 5x5):" << std::endl;
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                std::cout << h_in[i * width + j] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "\nOutput feature map (top-left 5x5):" << std::endl;
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                std::cout << h_out_gpu[i * output_w + j] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "\nNote: Edge detection kernel highlights edges where values change." << std::endl;
        std::cout << "Center pixel (2,2) of output should be high due to edge detection." << std::endl;
    }

    free_host(h_in);
    free_host(h_kernel);
    free_host(h_out_cpu);
    free_host(h_out_gpu);
    free_device(d_in);
    free_device(d_kernel);
    free_device(d_out);

    return 0;
}
