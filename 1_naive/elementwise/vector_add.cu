
/**
 * CPU implementation of vector addition
 * Sequential element-wise addition of two vectors
 * 
 * @param a Input vector A (host memory)
 * @param b Input vector B (host memory)
 * @param c Output vector C (host memory)
 * @param n Number of elements in vectors
 */
void vector_add_cpu(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

/**
 * CUDA kernel for vector addition
 * Each thread processes one element with global indexing and bounds checking
 * 
 * @param a Input vector A (device memory)
 * @param b Input vector B (device memory)
 * @param c Output vector C (device memory)
 * @param n Number of elements in vectors
 */
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    // Calculate global thread index across all blocks
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check to ensure we don't access out-of-range elements
    if (i < n) {
        // Perform element-wise addition: c[i] = a[i] + b[i]
        c[i] = a[i] + b[i];
    }
}

int main() {
    // Large vector size for performance demonstration
    const int n = 1000000; 
    size_t size = n * sizeof(float);

    std::cout << "Vector Addition: " << n << " elements" << std::endl;

    // Allocate host memory for input and output vectors
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    allocate_host(&h_a, n);
    allocate_host(&h_b, n);
    allocate_host(&h_c_cpu, n);
    allocate_host(&h_c_gpu, n);

    // Initialize input vectors with test data
    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);        // Vector A: [0, 1, 2, ..., 999999]
        h_b[i] = static_cast<float>(i * 2);    // Vector B: [0, 2, 4, ..., 1999998]
    }

    // Run CPU version with timing
    {
        Timer cpu_timer("CPU Vector Addition");
        vector_add_cpu(h_a, h_b, h_c_cpu, n);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    allocate_device(&d_a, n);
    allocate_device(&d_b, n);
    allocate_device(&d_c, n);

    // Copy data from host to device
    copy_to_device(d_a, h_a, n);
    copy_to_device(d_b, h_b, n);

    // Configure kernel launch parameters
    int threadsPerBlock = 256;  // Standard block size
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "GPU: Launching " << blocksPerGrid << " blocks with "
              << threadsPerBlock << " threads per block" << std::endl;

    // Run GPU version with timing
    {
        Timer gpu_timer("GPU Vector Addition");
        vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Copy result back from device to host
    copy_to_host(h_c_gpu, d_c, n);

    // Verify GPU results against CPU results
    if (verify_results(h_c_gpu, h_c_cpu, n)) {
        std::cout << "✓ Vector addition results match!" << std::endl;
    }

    // Clean up memory
    free_host(h_a);
    free_host(h_b);
    free_host(h_c_cpu);
    free_host(h_c_gpu);
    free_device(d_a);
    free_device(d_b);
    free_device(d_c);

    return 0;
}
