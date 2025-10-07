
__global__ void vectorAdd(float *a, float *b, float *c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main() {
    int n = 8;
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

    vectorAdd<<<1, 8>>>(d_a, d_b, d_c);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

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

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
