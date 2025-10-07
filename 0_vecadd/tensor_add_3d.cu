
void tensorAdd3D_cpu(const float* A, const float* B, float* C, int depth, int height, int width) {
    for (int d = 0; d < depth; ++d) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int index = d * (height * width) + h * width + w;
                C[index] = A[index] + B[index];
            }
        }
    }
}

__global__ void tensorAdd3D_kernel(const float* A, const float* B, float* C, int depth, int height, int width) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.z * blockDim.z + threadIdx.z;

    if (d < depth && h < height && w < width) {
        int index = d * (height * width) + h * width + w;
        C[index] = A[index] + B[index];
    }
}

int main() {
    int depth = 32, height = 128, width = 128;
    int total_elements = depth * height * width;
    size_t size = total_elements * sizeof(float);

    std::cout << "3D Tensor Addition: " << depth << "x" << height << "x" << width
              << " = " << total_elements << " elements" << std::endl;

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C_cpu = (float*)malloc(size);
    float *h_C_gpu = (float*)malloc(size);

    for (int i = 0; i < total_elements; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(i * 2);
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(8, 8, 8);
    dim3 blocksPerGrid(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (depth + threadsPerBlock.z - 1) / threadsPerBlock.z
    );

    std::cout << "GPU: Launching " << blocksPerGrid.x << "x" << blocksPerGrid.y << "x" << blocksPerGrid.z
              << " blocks with " << threadsPerBlock.x << "x" << threadsPerBlock.y << "x" << threadsPerBlock.z
              << " threads per block" << std::endl;

    tensorAdd3D_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, depth, height, width);

    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    auto start = std::chrono::high_resolution_clock::now();
    tensorAdd3D_cpu(h_A, h_B, h_C_cpu, depth, height, width);
    auto end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

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

    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
