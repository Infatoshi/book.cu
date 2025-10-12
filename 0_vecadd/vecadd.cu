
/**
 * CUDA kernel for vector addition
 * Each thread adds one element from arrays a and b, stores result in c
 * 
 * @param a Input vector A (device memory)
 * @param b Input vector B (device memory) 
 * @param c Output vector C (device memory)
 */
__global__ void vectorAdd(float *a, float *b, float *c) {
    // Get the thread index within the block
    int i = threadIdx.x;
    
    // Perform element-wise addition: c[i] = a[i] + b[i]
    c[i] = a[i] + b[i];
}

int main() {
    // Vector size and memory allocation size
    int n = 8;
    size_t size = n * sizeof(float);

    // Allocate host (CPU) memory for input and output vectors
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // Initialize input vectors with test data
    for (int i = 0; i < n; ++i) {
        h_a[i] = (float)i;        // Vector A: [0, 1, 2, 3, 4, 5, 6, 7]
        h_b[i] = (float)(i * 2);  // Vector B: [0, 2, 4, 6, 8, 10, 12, 14]
    }

    // Allocate device (GPU) memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch CUDA kernel: 1 block with 8 threads
    // Each thread processes one element
    vectorAdd<<<1, 8>>>(d_a, d_b, d_c);

    // Copy result back from device to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify the results
    bool success = true;
    for (int i = 0; i < n; ++i) {
        if (h_c[i] != (h_a[i] + h_b[i])) {
            std::cout << "Error at index " << i << ": Got " << h_c[i] << ", expected " << (h_a[i] + h_b[i]) << std::endl;
            success = false;
            break;
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
