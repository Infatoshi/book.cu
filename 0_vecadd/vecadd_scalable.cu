
/**
 * Scalable CUDA kernel for vector addition
 * Uses global thread indexing to handle vectors of any size
 * Each thread processes one element, with bounds checking
 * 
 * @param a Input vector A (device memory)
 * @param b Input vector B (device memory)
 * @param c Output vector C (device memory)
 * @param n Size of the vectors
 */
__global__ void vectorAddScalable(float *a, float *b, float *c, int n) {
    // Calculate global thread index across all blocks
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check to ensure we don't access out-of-range elements
    if (i < n) {
        // Perform element-wise addition: c[i] = a[i] + b[i]
        c[i] = a[i] + b[i];
    }
}

int main() {
    // Large vector size to demonstrate scalability
    int n = 1000000;
    size_t size = n * sizeof(float);

    // Allocate host (CPU) memory for input and output vectors
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // Initialize input vectors with test data
    for (int i = 0; i < n; ++i) {
        h_a[i] = (float)i;        // Vector A: [0, 1, 2, ..., 999999]
        h_b[i] = (float)(i * 2);  // Vector B: [0, 2, 4, ..., 1999998]
    }

    // Allocate device (GPU) memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Configure kernel launch parameters
    int threadsPerBlock = 256;  // Standard block size
    // Calculate number of blocks needed to cover all elements
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Launching " << blocksPerGrid << " blocks with " << threadsPerBlock
              << " threads each (total: " << blocksPerGrid * threadsPerBlock << " threads)" << std::endl;

    // Launch scalable CUDA kernel
    vectorAddScalable<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Copy result back from device to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify results by checking first 10 and last 10 elements
    // (checking all 1M elements would be too slow)
    bool success = true;
    for (int i = 0; i < 10 && success; ++i) {
        if (h_c[i] != (h_a[i] + h_b[i])) {
            std::cout << "Error at index " << i << ": Got " << h_c[i] << ", expected " << (h_a[i] + h_b[i]) << std::endl;
            success = false;
        }
    }
    for (int i = n - 10; i < n && success; ++i) {
        if (h_c[i] != (h_a[i] + h_b[i])) {
            std::cout << "Error at index " << i << ": Got " << h_c[i] << ", expected " << (h_a[i] + h_b[i]) << std::endl;
            success = false;
        }
    }
    if (success) {
        std::cout << "Success! All elements are correct." << std::endl;
    }

    // Clean up memory
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
