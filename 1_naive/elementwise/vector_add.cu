
void vector_add_cpu(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 1000000; 
    size_t size = n * sizeof(float);

    std::cout << "Vector Addition: " << n << " elements" << std::endl;

    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    allocate_host(&h_a, n);
    allocate_host(&h_b, n);
    allocate_host(&h_c_cpu, n);
    allocate_host(&h_c_gpu, n);

    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    {
        Timer cpu_timer("CPU Vector Addition");
        vector_add_cpu(h_a, h_b, h_c_cpu, n);
    }

    float *d_a, *d_b, *d_c;
    allocate_device(&d_a, n);
    allocate_device(&d_b, n);
    allocate_device(&d_c, n);

    copy_to_device(d_a, h_a, n);
    copy_to_device(d_b, h_b, n);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "GPU: Launching " << blocksPerGrid << " blocks with "
              << threadsPerBlock << " threads per block" << std::endl;

    {
        Timer gpu_timer("GPU Vector Addition");
        vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    copy_to_host(h_c_gpu, d_c, n);

    if (verify_results(h_c_gpu, h_c_cpu, n)) {
        std::cout << "✓ Vector addition results match!" << std::endl;
    }

    free_host(h_a);
    free_host(h_b);
    free_host(h_c_cpu);
    free_host(h_c_gpu);
    free_device(d_a);
    free_device(d_b);
    free_device(d_c);

    return 0;
}
