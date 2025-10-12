

/**
 * FP32 to INT4 Quantization Kernel with Packing
 * Converts floating-point values to 4-bit signed integers and packs them
 * Two INT4 values are packed into one uint8_t byte for memory efficiency
 * 
 * Quantization formula: q = clamp(round(x / scale), -7, 7)
 * Packing: [q1_4bits][q2_4bits] -> uint8_t
 * 
 * @param input Input FP32 array (device memory)
 * @param output Output packed INT4 array (device memory, size/2 elements)
 * @param scale Scale factor for quantization (max_value / 7)
 * @param size Number of FP32 elements to quantize
 */
__global__ void quantize_fp32_to_int4_packed(float* input, uint8_t* output, float scale, int size) {
    // Each thread processes 2 elements (packs them into 1 byte)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size / 2) return;

    // Calculate indices for the two elements to pack
    int elem1_idx = idx * 2;      // First element index
    int elem2_idx = idx * 2 + 1;  // Second element index

    // Quantize first element
    float scaled1 = input[elem1_idx] / scale;
    scaled1 = fmaxf(fminf(scaled1, 7.0f), -7.0f);  // Clamp to [-7, 7]
    int8_t quant1 = (int8_t)roundf(scaled1);

    // Quantize second element (with bounds check)
    float scaled2 = (elem2_idx < size) ? input[elem2_idx] / scale : 0.0f;
    scaled2 = fmaxf(fminf(scaled2, 7.0f), -7.0f);
    int8_t quant2 = (elem2_idx < size) ? (int8_t)roundf(scaled2) : 0;

    // Pack two 4-bit values into one byte
    // Upper 4 bits: quant1, Lower 4 bits: quant2
    uint8_t packed = ((quant1 & 0xF) << 4) | (quant2 & 0xF);
    output[idx] = packed;
}

/**
 * INT4 to FP32 Dequantization Kernel with Unpacking
 * Unpacks packed INT4 values and converts them back to floating-point
 * Handles sign extension for negative 4-bit values
 * 
 * @param input Input packed INT4 array (device memory, size/2 elements)
 * @param output Output FP32 array (device memory)
 * @param scale Scale factor used for quantization
 * @param size Number of FP32 elements to produce
 */
__global__ void dequantize_int4_to_fp32_packed(uint8_t* input, float* output, float scale, int size) {
    // Each thread unpacks 2 elements from 1 byte
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size / 2) return;

    // Calculate indices for the two elements to unpack
    int elem1_idx = idx * 2;      // First element index
    int elem2_idx = idx * 2 + 1;  // Second element index

    // Extract packed byte
    uint8_t packed = input[idx];
    
    // Unpack two 4-bit values
    int8_t quant1 = (packed >> 4) & 0xF;  // Upper 4 bits
    int8_t quant2 = packed & 0xF;         // Lower 4 bits

    // Sign extension for negative values (4-bit -> 8-bit)
    // If MSB is 1, extend with 1s; otherwise extend with 0s
    if (quant1 & 0x8) quant1 |= 0xF0;  // Sign extend quant1
    if (quant2 & 0x8) quant2 |= 0xF0;  // Sign extend quant2

    // Dequantize and store results
    output[elem1_idx] = (float)quant1 * scale;
    if (elem2_idx < size) {
        output[elem2_idx] = (float)quant2 * scale;
    }
}

int main() {
    const int SIZE = 1024 * 1024;
    const int THREADS = 256;
    const int BLOCKS = (SIZE + THREADS - 1) / THREADS;

    float* h_input = (float*)malloc(SIZE * sizeof(float));
    srand(time(NULL));

    float max_abs = 0.0f;
    for (int i = 0; i < SIZE; ++i) {
        h_input[i] = ((float)rand() / RAND_MAX - 0.5f) * 4.0f;
        max_abs = fmaxf(max_abs, fabsf(h_input[i]));
    }

    float scale = max_abs / 7.0f;

    printf("FP32 -> INT4 Quantization Test\n");
    printf("Scale: %f\n", scale);
    printf("Memory reduction: 8x (FP32 -> INT4)\n");
    printf("Packing: 2 INT4 values per byte\n\n");

    float *d_input, *d_output;
    uint8_t *d_quantized;

    cudaMalloc(&d_input, SIZE * sizeof(float));
    cudaMalloc(&d_output, SIZE * sizeof(float));
    cudaMalloc(&d_quantized, (SIZE + 1) / 2 * sizeof(uint8_t));

    cudaMemcpy(d_input, h_input, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    quantize_fp32_to_int4_packed<<<BLOCKS, THREADS>>>(d_input, d_quantized, scale, SIZE);
    cudaDeviceSynchronize();

    dequantize_int4_to_fp32_packed<<<BLOCKS, THREADS>>>(d_quantized, d_output, scale, SIZE);
    cudaDeviceSynchronize();

    float* h_output = (float*)malloc(SIZE * sizeof(float));
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
