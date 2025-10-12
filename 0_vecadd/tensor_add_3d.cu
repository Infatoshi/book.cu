
/**
 * CPU implementation of 3D tensor addition
 * Iterates through all elements in 3D tensor and performs element-wise addition
 * 
 * @param A Input tensor A (host memory)
 * @param B Input tensor B (host memory)
 * @param C Output tensor C (host memory)
 * @param depth Number of depth layers
 * @param height Number of height rows
 * @param width Number of width columns
 */
void tensorAdd3D_cpu(const float* A, const float* B, float* C, int depth, int height, int width) {
    // Iterate through all 3D coordinates
    for (int d = 0; d < depth; ++d) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                // Convert 3D coordinates to 1D index (row-major order)
                int index = d * (height * width) + h * width + w;
                // Perform element-wise addition: C[index] = A[index] + B[index]
                C[index] = A[index] + B[index];
            }
        }
    }
}

/**
 * CUDA kernel for 3D tensor addition
 * Uses 3D thread indexing to map threads to 3D tensor coordinates
 * Each thread processes one element with bounds checking
 * 
 * @param A Input tensor A (device memory)
 * @param B Input tensor B (device memory)
 * @param C Output tensor C (device memory)
 * @param depth Number of depth layers
 * @param height Number of height rows
 * @param width Number of width columns
 */
__global__ void tensorAdd3D_kernel(const float* A, const float* B, float* C, int depth, int height, int width) {
    // Calculate 3D coordinates from thread indices
    int w = blockIdx.x * blockDim.x + threadIdx.x;  // Width (x-dimension)
    int h = blockIdx.y * blockDim.y + threadIdx.y;   // Height (y-dimension)
    int d = blockIdx.z * blockDim.z + threadIdx.z;   // Depth (z-dimension)

    // Bounds check to ensure we don't access out-of-range elements
    if (d < depth && h < height && w < width) {
        // Convert 3D coordinates to 1D index (row-major order)
        int index = d * (height * width) + h * width + w;
        // Perform element-wise addition: C[index] = A[index] + B[index]
        C[index] = A[index] + B[index];
    }
}

int main() {
    // Define 3D tensor dimensions
    int depth = 32, height = 128, width = 128;
    int total_elements = depth * height * width;
    size_t size = total_elements * sizeof(float);

    std::cout << "3D Tensor Addition: " << depth << "x" << height << "x" << width
              << " = " << total_elements << " elements" << std::endl;

    // Allocate host (CPU) memory for input and output tensors
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C_cpu = (float*)malloc(size);
    float *h_C_gpu = (float*)malloc(size);

    // Initialize input tensors with test data
    for (int i = 0; i < total_elements; ++i) {
        h_A[i] = (float)i;        // Tensor A: sequential values
        h_B[i] = (float)(i * 2);  // Tensor B: doubled values
    }

    // Allocate device (GPU) memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Configure 3D kernel launch parameters
    dim3 threadsPerBlock(8, 8, 8);  // 8x8x8 = 512 threads per block
    dim3 blocksPerGrid(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,   // Blocks in x-dimension
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y,   // Blocks in y-dimension
        (depth + threadsPerBlock.z - 1) / threadsPerBlock.z     // Blocks in z-dimension
    );

    std::cout << "GPU: Launching " << blocksPerGrid.x << "x" << blocksPerGrid.y << "x" << blocksPerGrid.z
              << " blocks with " << threadsPerBlock.x << "x" << threadsPerBlock.y << "x" << threadsPerBlock.z
              << " threads per block" << std::endl;

    // Launch 3D CUDA kernel
    tensorAdd3D_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, depth, height, width);

    // Copy result back from device to host
    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    // Run CPU version for comparison and timing
    auto start = std::chrono::high_resolution_clock::now();
    tensorAdd3D_cpu(h_A, h_B, h_C_cpu, depth, height, width);
    auto end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Verify GPU results against CPU results
    bool success = true;
    for (int i = 0; i < total_elements && success; ++i) {
        if (std::abs(h_C_gpu[i] - h_C_cpu[i]) > 1e-6) {
            std::cout << "Mismatch at index " << i << ": GPU=" << h_C_gpu[i] << ", CPU=" << h_C_cpu[i] << std::endl;
            success = false;
        }
    }

    if (success) {
        std::cout << "Success! GPU and CPU results match." << std::endl;
        std::cout << "CPU computation took " << cpu_time.count() << " ms" << std::endl;
    }

    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
