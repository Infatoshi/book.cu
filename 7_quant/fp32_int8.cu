

/**
 * FP32 to INT8 Quantization Kernel
 * Converts floating-point values to 8-bit signed integers
 * Uses symmetric quantization with scale factor for precision control
 * 
 * Quantization formula: q = clamp(round(x / scale), -127, 127)
 * Dequantization formula: x' = q * scale
 * 
 * @param input Input FP32 array (device memory)
 * @param output Output INT8 array (device memory)
 * @param scale Scale factor for quantization (max_value / 127)
 * @param size Number of elements to quantize
 */
__global__ void quantize_fp32_to_int8(float* input, signed char* output, float scale, int size) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Apply quantization: divide by scale and clamp to INT8 range
    float scaled = input[idx] / scale;
    scaled = fmaxf(fminf(scaled, 127.0f), -127.0f);  // Clamp to [-127, 127]
    output[idx] = (signed char)roundf(scaled);       // Round to nearest integer
}

/**
 * INT8 to FP32 Dequantization Kernel
 * Converts quantized 8-bit integers back to floating-point values
 * 
 * @param input Input INT8 array (device memory)
 * @param output Output FP32 array (device memory)
 * @param scale Scale factor used for quantization
 * @param size Number of elements to dequantize
 */
__global__ void dequantize_int8_to_fp32(signed char* input, float* output, float scale, int size) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Apply dequantization: multiply by scale factor
    output[idx] = (float)input[idx] * scale;
}

/**
 * Generate random numbers from normal distribution using Box-Muller transform
 * @param mean Mean of the normal distribution
 * @param std Standard deviation of the normal distribution
 * @return Random number from N(mean, std^2)
 */
float rand_normal(float mean, float std) {
    // Box-Muller transform for generating normal random numbers
    float u1 = (float)rand() / RAND_MAX;  // Uniform random [0,1]
    float u2 = (float)rand() / RAND_MAX;  // Uniform random [0,1]
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.141592653589793f * u2);
    return z0 * std + mean;  // Scale and shift to desired distribution
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
