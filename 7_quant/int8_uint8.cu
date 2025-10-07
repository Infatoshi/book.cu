






__global__ void quantize_symmetric(float* input, int8_t* output, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    
    float scaled = input[idx] / scale;
    scaled = fmaxf(fminf(scaled, 127.0f), -127.0f);
    output[idx] = (int8_t)roundf(scaled);
}

__global__ void quantize_asymmetric(float* input, uint8_t* output, float scale, float zero_point, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    
    float scaled = (input[idx] - zero_point) / scale;
    scaled = fmaxf(fminf(scaled, 255.0f), 0.0f);
    output[idx] = (uint8_t)roundf(scaled);
}

__global__ void dequantize_symmetric(int8_t* input, float* output, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    
    output[idx] = (float)input[idx] * scale;
}

__global__ void dequantize_asymmetric(uint8_t* input, float* output, float scale, float zero_point, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    
    output[idx] = (float)input[idx] * scale + zero_point;
}

float rand_normal(float mean, float std) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.141592653589793f * u2);
    return z0 * std + mean;
}

int main() {
    const int SIZE = 512 * 1024;
    const int THREADS = 256;
    const int BLOCKS = (SIZE + THREADS - 1) / THREADS;

    float* h_input = (float*)malloc(SIZE * sizeof(float));
    srand(time(NULL));  

    float min_val = INFINITY, max_val = -INFINITY;
    for (int i = 0; i < SIZE; ++i) {
        h_input[i] = rand_normal(2.0f, 1.5f);
        min_val = fminf(min_val, h_input[i]);
        max_val = fmaxf(max_val, h_input[i]);
    }

    
    float symmetric_scale = fmaxf(fabsf(min_val), fabsf(max_val)) / 127.0f;
    float asymmetric_scale = (max_val - min_val) / 255.0f;
    float zero_point = min_val;

    printf("INT8 vs UINT8 Quantization Test\n");
    printf("Data range: [%f, %f]\n", min_val, max_val);
    printf("Data mean: %f (non-zero)\n\n", (min_val + max_val) / 2.0f);

    printf("Symmetric (INT8):\n");
    printf("  Scale: %f\n", symmetric_scale);
    printf("  Zero point: 0 (fixed)\n");
    printf("  Range: [-127, 127]\n\n");

    printf("Asymmetric (UINT8):\n");
    printf("  Scale: %f\n", asymmetric_scale);
    printf("  Zero point: %f\n", zero_point);
    printf("  Range: [0, 255]\n\n");

    
    float *d_input, *d_symmetric_output, *d_asymmetric_output;
    int8_t *d_symmetric_quantized;
    uint8_t *d_asymmetric_quantized;

    cudaMalloc(&d_input, SIZE * sizeof(float));
    cudaMalloc(&d_symmetric_output, SIZE * sizeof(float));
    cudaMalloc(&d_asymmetric_output, SIZE * sizeof(float));
    cudaMalloc(&d_symmetric_quantized, SIZE * sizeof(int8_t));
    cudaMalloc(&d_asymmetric_quantized, SIZE * sizeof(uint8_t));

    cudaMemcpy(d_input, h_input, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    quantize_symmetric<<<BLOCKS, THREADS>>>(d_input, d_symmetric_quantized, symmetric_scale, SIZE);
    dequantize_symmetric<<<BLOCKS, THREADS>>>(d_symmetric_quantized, d_symmetric_output, symmetric_scale, SIZE);

    quantize_asymmetric<<<BLOCKS, THREADS>>>(d_input, d_asymmetric_quantized, asymmetric_scale, zero_point, SIZE);
    dequantize_asymmetric<<<BLOCKS, THREADS>>>(d_asymmetric_quantized, d_asymmetric_output, asymmetric_scale, zero_point, SIZE);

    cudaDeviceSynchronize();

    float* h_symmetric_output = (float*)malloc(SIZE * sizeof(float));
    float* h_asymmetric_output = (float*)malloc(SIZE * sizeof(float));

    cudaMemcpy(h_symmetric_output, d_symmetric_output, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_asymmetric_output, d_asymmetric_output, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    
    float symmetric_mse = 0.0f, asymmetric_mse = 0.0f;
    float symmetric_mae = 0.0f, asymmetric_mae = 0.0f;

    for (int i = 0; i < SIZE; ++i) {
        float sym_error = h_input[i] - h_symmetric_output[i];
        float asym_error = h_input[i] - h_asymmetric_output[i];

        symmetric_mse += sym_error * sym_error;
        asymmetric_mse += asym_error * asym_error;
        symmetric_mae += fabsf(sym_error);
        asymmetric_mae += fabsf(asym_error);
    }

    symmetric_mse /= SIZE;
    asymmetric_mse /= SIZE;
    symmetric_mae /= SIZE;
    asymmetric_mae /= SIZE;

    printf("Accuracy Results:\n");
    printf("Symmetric (INT8):\n");
    printf("  MSE: %f\n", symmetric_mse);
    printf("  MAE: %f\n\n", symmetric_mae);

    printf("Asymmetric (UINT8):\n");
    printf("  MSE: %f\n", asymmetric_mse);
    printf("  MAE: %f\n\n", asymmetric_mae);

    printf("Asymmetric is %fx more accurate for this data\n\n", symmetric_mse / asymmetric_mse);

    printf("Key Insight:\n");
    printf("- Symmetric (INT8): Best for zero-mean data, simpler\n");
    printf("- Asymmetric (UINT8): Better for data with offset, uses full range\n");

    printf("\nSample comparisons:\n");
    printf("Original -> Symmetric -> Asymmetric\n");
    for (int i = 0; i < 3; ++i) {
        printf("%f -> %f -> %f\n",
               h_input[i], h_symmetric_output[i], h_asymmetric_output[i]);
    }

    cudaFree(d_input);
    cudaFree(d_symmetric_output);
    cudaFree(d_asymmetric_output);
    cudaFree(d_symmetric_quantized);
    cudaFree(d_asymmetric_quantized);
    free(h_input);
    free(h_symmetric_output);
    free(h_asymmetric_output);

    return 0;
}
