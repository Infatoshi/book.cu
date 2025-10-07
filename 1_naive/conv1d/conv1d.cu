
void conv1d_cpu(const float* in, float* out, const float* kernel, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    for (int i = 0; i < output_size; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < kernel_size; ++j) {
            sum += in[i + j] * kernel[j];
        }
        out[i] = sum;
    }
}

__global__ void conv1d_kernel(const float* in, float* out, const float* kernel, int input_size, int kernel_size) {
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int output_size = input_size - kernel_size + 1;

    if (output_idx < output_size) {
        float sum = 0.0f;
        for (int k_idx = 0; k_idx < kernel_size; ++k_idx) {
            sum += in[output_idx + k_idx] * kernel[k_idx];
        }
        out[output_idx] = sum;
    }
}

int main() {
    const int input_size = 100000;  
    const int kernel_size = 32;     
    const int output_size = input_size - kernel_size + 1;

    std::cout << "1D Convolution: Input[" << input_size << "] * Kernel[" << kernel_size
              << "] -> Output[" << output_size << "]" << std::endl;

    float *h_in, *h_kernel, *h_out_cpu, *h_out_gpu;
    allocate_host(&h_in, input_size);
    allocate_host(&h_kernel, kernel_size);
    allocate_host(&h_out_cpu, output_size);
    allocate_host(&h_out_gpu, output_size);

    for (int i = 0; i < input_size; ++i) {
        h_in[i] = static_cast<float>(i % 10);  
    }

    for (int i = 0; i < kernel_size; ++i) {
        h_kernel[i] = 1.0f / kernel_size;  
    }

    {
        Timer cpu_timer("CPU 1D Convolution");
        conv1d_cpu(h_in, h_out_cpu, h_kernel, input_size, kernel_size);
    }

    float *d_in, *d_kernel, *d_out;
    allocate_device(&d_in, input_size);
    allocate_device(&d_kernel, kernel_size);
    allocate_device(&d_out, output_size);

    copy_to_device(d_in, h_in, input_size);
    copy_to_device(d_kernel, h_kernel, kernel_size);

    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "GPU: Launching " << blocksPerGrid << " blocks with "
              << threadsPerBlock << " threads per block" << std::endl;

    {
        Timer gpu_timer("GPU 1D Convolution");
        conv1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, d_kernel, input_size, kernel_size);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    copy_to_host(h_out_gpu, d_out, output_size);

    if (verify_results(h_out_gpu, h_out_cpu, output_size)) {
        std::cout << "✓ 1D convolution results match!" << std::endl;

        std::cout << "\nInput signal (first 10): ";
        for (int i = 0; i < 10; ++i) {
            std::cout << h_in[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "Kernel (first 10): ";
        for (int i = 0; i < std::min(10, kernel_size); ++i) {
            std::cout << h_kernel[i] << " ";
        }
        if (kernel_size > 10) std::cout << "...";
        std::cout << std::endl;

        std::cout << "Output (first 10): ";
        for (int i = 0; i < 10; ++i) {
            std::cout << h_out_gpu[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "\nVerification: For averaging kernel, output[" << kernel_size/2 << "] should be ~"
                  << (kernel_size * 4.5f) / kernel_size << " (average of 0-9 pattern)" << std::endl;
        std::cout << "Actual output[" << kernel_size/2 << "] = " << h_out_gpu[kernel_size/2] << std::endl;
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
