
__global__ void vectorAddScalable(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 1000000;
    size_t size = n * sizeof(float);

    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    for (int i = 0; i < n; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Launching " << blocksPerGrid << " blocks with " << threadsPerBlock
              << " threads each (total: " << blocksPerGrid * threadsPerBlock << " threads)" << std::endl;

    vectorAddScalable<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

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

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
