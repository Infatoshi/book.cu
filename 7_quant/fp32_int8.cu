

__global__ void quantize_fp32_to_int8(float* input, signed char* output, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float scaled = input[idx] / scale;
    scaled = fmaxf(fminf(scaled, 127.0f), -127.0f);
    output[idx] = (signed char)roundf(scaled);
}

__global__ void dequantize_int8_to_fp32(signed char* input, float* output, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    output[idx] = (float)input[idx] * scale;
}

float rand_normal(float mean, float std) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.141592653589793f * u2);
    return z0 * std + mean;
}

int main() {
    const int SIZE = 1024 * 1024;
    const int THREADS = 256;
    const int BLOCKS = (SIZE + THREADS - 1) / THREADS;

    srand(time(NULL));

    printf("FP32 -> INT8 Quantization Test\n");

    float* h_input = (float*)malloc(SIZE * sizeof(float));
    float* h_output = (float*)malloc(SIZE * sizeof(float));

    float max_abs = 0.0f;
    for (int i = 0; i < SIZE; ++i) {
        h_input[i] = rand_normal(0.0f, 2.0f);
        max_abs = fmaxf(max_abs, fabsf(h_input[i]));
    }

    float scale = max_abs / 127.0f;
    printf("Scale: %f\n", scale);
    printf("Memory reduction: 4x (FP32 -> INT8)\n\n");

    float *d_input, *d_output;
    signed char *d_quantized;

    cudaMalloc(&d_input, SIZE * sizeof(float));
    cudaMalloc(&d_output, SIZE * sizeof(float));
    cudaMalloc(&d_quantized, SIZE * sizeof(signed char));

    cudaMemcpy(d_input, h_input, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    quantize_fp32_to_int8<<<BLOCKS, THREADS>>>(d_input, d_quantized, scale, SIZE);
    cudaDeviceSynchronize();

    dequantize_int8_to_fp32<<<BLOCKS, THREADS>>>(d_quantized, d_output, scale, SIZE);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    float mse = 0.0f, mae = 0.0f, max_error = 0.0f;
    for (int i = 0; i < SIZE; ++i) {
        float error = h_input[i] - h_output[i];
        mse += error * error;
        mae += fabsf(error);
        max_error = fmaxf(max_error, fabsf(error));
    }
    mse /= SIZE;
    mae /= SIZE;

    printf("Accuracy Results:\n");
    printf("  MSE: %f\n", mse);
    printf("  MAE: %f\n", mae);
    printf("  Max Error: %f\n", max_error);

    printf("\nSample comparisons:\n");
    for (int i = 0; i < 5; ++i) {
        printf("  %f -> %f (error: %f)\n",
               h_input[i], h_output[i], h_input[i] - h_output[i]);
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_quantized);
    free(h_input);
    free(h_output);

    return 0;
}
